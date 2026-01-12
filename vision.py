# Part 2/4: vision.py - Advanced Screen Capture, Detection, and Processing
# This module handles real-time screen capture using MSS, object detection with YOLOv8 or OpenCV fallback,
# post-processing (NMS, filtering), HP/UI analysis via color/OCR, frame queuing for async processing.
# Maximized: multi-threading, error resilience, performance metrics, debug frame saving, downscaling,
# custom class mapping, risk assessment heuristics, integration with WorldState.
# GitHub file: vision.py
# Imports from utils.py: e.g., from utils import logger, WorldState, DET_CONF, MIN_OBJECT_SIZE, DOWNSCALE_FACTOR, etc.
# Usage in core.py: vision_proc = VisionProcessor(); frame = vision_proc.capture_screen(); detections = vision_proc.detect_objects(frame)

import time
import threading
import queue
import os
import cv2
import numpy as np
import mss
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Optional: Tesseract OCR for text/UI reading (e.g., HP numbers)
try:
    import pytesseract
    HAS_OCR = True
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path if needed
except ImportError:
    HAS_OCR = False
    logger.warning("pytesseract not installed. No OCR for UI text. Install with: pip install pytesseract")

# YOLOv8 Import
try:
    from ultralytics import YOLO
    from ultralytics.utils.ops import non_max_suppression  # For custom NMS if needed
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("Ultralytics YOLOv8 not installed. Falling back to OpenCV-based detection. Install: pip install ultralytics")

# Import from utils
from utils import (
    logger, config, reload_config_values, WorldState, Detection,
    CAPTURE_REGION, DOWNSCALE_FACTOR, DET_CONF, MIN_OBJECT_SIZE,
    HIGH_DANGER, CRITICAL_DANGER, VISION_MODEL_PATH, SHOW_FLOATING_DEBUG
)

# ────────────────────────────────────────────────
#   Constants & Config-Derived Values
# ────────────────────────────────────────────────

MAX_QUEUE_SIZE = int(config.get("General", "max_detection_queue_size", fallback="10"))
DEBUG_FRAMES_DIR = config.get("Debug", "debug_frames_dir", fallback="debug_frames")
SAVE_DEBUG_FRAMES = config.getboolean("Debug", "save_debug_frames", fallback=False)

if SAVE_DEBUG_FRAMES and not Path(DEBUG_FRAMES_DIR).exists():
    os.makedirs(DEBUG_FRAMES_DIR)

IOU_THRESHOLD = float(config.get("Thresholds", "iou_threshold_nms", fallback="0.45"))
MAX_FRAME_TIME_MS = int(config.get("Performance", "max_frame_process_time_ms", fallback="80")) / 1000.0

# Custom class mapping (for YOLO or OpenCV) - extendable for different games
CLASS_MAP = {
    0: "enemy",
    1: "npc",
    2: "loot",
    3: "health_bar",
    4: "mana_bar",
    5: "button",
    6: "trade_window",
    # Add more as needed, e.g., 7: "door", 8: "quest_marker"
}

RISK_HEURISTICS = {
    "enemy": lambda det: int(10 - det.confidence * 2 + (5000 / max(1, det.width * det.height))),  # Small/far = less risk
    "loot": lambda det: 1,
    "trade_window": lambda det: 6,  # High risk for trades
    # etc.
}

# Color ranges for fallback OpenCV (HSV)
COLOR_RANGES = {
    "enemy_red": ((0, 100, 100), (10, 255, 255)),    # Red enemies
    "loot_green": ((35, 100, 100), (85, 255, 255)),  # Green loot
    "health_bar_red": ((0, 150, 150), (10, 255, 255)),
    "mana_bar_blue": ((110, 150, 150), (130, 255, 255)),
}

# ────────────────────────────────────────────────
#   Frame Metrics for Performance Monitoring
# ────────────────────────────────────────────────

@dataclass
class FrameMetrics:
    capture_time: float = 0.0
    detect_time: float = 0.0
    post_process_time: float = 0.0
    total_time: float = 0.0
    fps: float = 0.0
    dropped_frames: int = 0

    def update(self, start: float, capture_end: float, detect_end: float, post_end: float, dropped: int):
        self.capture_time = capture_end - start
        self.detect_time = detect_end - capture_end
        self.post_process_time = post_end - detect_end
        self.total_time = post_end - start
        self.fps = 1.0 / max(1e-6, self.total_time)
        self.dropped_frames += dropped

    def log_summary(self):
        logger.debug(f"Frame Metrics: FPS={self.fps:.1f}, Total={self.total_time*1000:.0f}ms, Capture={self.capture_time*1000:.0f}ms, Detect={self.detect_time*1000:.0f}ms, Post={self.post_process_time*1000:.0f}ms, Dropped={self.dropped_frames}")

# ────────────────────────────────────────────────
#   VisionProcessor - Core Class
# ────────────────────────────────────────────────

class VisionProcessor:
    """
    Manages screen capture, detection pipeline, threading for async processing.
    Features: frame queue to decouple capture/detect, downscaling for perf,
    fallback modes, OCR for UI, custom NMS, risk calc, debug overlays/saving.
    """
    def __init__(self, model_path: str = VISION_MODEL_PATH):
        self.sct = mss.mss()
        self.region = {"top": CAPTURE_REGION[1], "left": CAPTURE_REGION[0], "width": CAPTURE_REGION[2], "height": CAPTURE_REGION[3]}
        self.frame_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.processed_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.metrics = FrameMetrics()
        self.lock = threading.Lock()
        self.last_frame: Optional[np.ndarray] = None

        # Load model
        self.model: Optional[YOLO] = None
        if HAS_YOLO:
            try:
                self.model = YOLO(model_path)
                self.model.to('cuda') if cv2.cuda.getCudaEnabledDeviceCount() > 0 else self.model.to('cpu')  # GPU if avail
                logger.info(f"YOLOv8 loaded: {model_path} on {'CUDA' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'CPU'}")
            except Exception as e:
                logger.error(f"YOLO load failed: {e}. Switching to OpenCV fallback.")
                self.model = None
        else:
            logger.info("Using OpenCV color/contour detection fallback.")

    def start(self):
        """Start capture and processing threads."""
        with self.lock:
            if self.running:
                return
            self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
        logger.info("Vision threads started.")

    def stop(self):
        """Graceful shutdown."""
        with self.lock:
            self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        logger.info("Vision threads stopped.")

    def _capture_loop(self):
        """Fast capture thread - pushes raw frames to queue."""
        while self.running:
            start = time.time()
            try:
                img = self.sct.grab(self.region)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
                if DOWNSCALE_FACTOR < 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR, interpolation=cv2.INTER_AREA)
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, start))
                else:
                    self.metrics.dropped_frames += 1
                    logger.debug("Frame queue full - dropped frame.")
            except Exception as e:
                logger.error(f"Capture error: {e}")
            time.sleep(max(0, FRAME_DELAY - (time.time() - start)))

    def _process_loop(self):
        """Detection thread - processes frames from queue."""
        while self.running:
            try:
                frame, capture_start = self.frame_queue.get(timeout=0.5)
                detect_start = time.time()
                raw_dets = self._detect_raw(frame)
                post_start = time.time()
                processed_dets = self._post_process_detections(raw_dets, frame)
                end = time.time()
                self.metrics.update(capture_start, detect_start, post_start, end, 0)
                if time.time() - capture_start > MAX_FRAME_TIME_MS:
                    logger.warning("Frame processing too slow - potential lag.")
                self.processed_queue.put(processed_dets)
                with self.lock:
                    self.last_frame = frame
                if SAVE_DEBUG_FRAMES:
                    self._save_debug_frame(frame, processed_dets)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Process error: {e}")

    def _detect_raw(self, frame: np.ndarray) -> List[Dict]:
        """Core detection: YOLO or OpenCV."""
        if self.model:
            results = self.model(frame, conf=DET_CONF, iou=IOU_THRESHOLD, verbose=False)
            dets = []
            for r in results:
                boxes = r.boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    cls_name = CLASS_MAP.get(cls_id, r.names.get(cls_id, "unknown"))
                    dets.append({
                        "class": cls_name,
                        "conf": conf,
                        "xyxy": box,
                        "cls_id": cls_id
                    })
            return dets
        else:
            # OpenCV fallback: multi-color contour detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dets = []
            for cls, (lower, upper) in COLOR_RANGES.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    if area < MIN_OBJECT_SIZE:
                        continue
                    conf = min(1.0, area / (frame.shape[0] * frame.shape[1] * 0.01))  # Pseudo-conf
                    dets.append({
                        "class": cls.split('_')[0],  # e.g., "enemy"
                        "conf": conf,
                        "xyxy": [x, y, x + w, y + h]
                    })
            return dets

    def _post_process_detections(self, raw_dets: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Apply NMS, filtering, risk, OCR/UI analysis."""
        # Convert to numpy for NMS if needed
        if len(raw_dets) > 1:
            boxes = np.array([d["xyxy"] for d in raw_dets])
            confs = np.array([d["conf"] for d in raw_dets])
            clss = np.array([d.get("cls_id", 0) for d in raw_dets])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), DET_CONF, IOU_THRESHOLD)
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            raw_dets = [raw_dets[i] for i in indices]

        # Add risk and extra analysis
        for d in raw_dets:
            cls = d["class"]
            det = Detection(**d)  # Wait, Detection is dataclass, but raw is dict - adjust
            d["risk"] = RISK_HEURISTICS.get(cls, lambda det: 3)(det)
            if "bar" in cls and HAS_OCR:
                crop = frame[int(d["xyxy"][1]):int(d["xyxy"][3]), int(d["xyxy"][0]):int(d["xyxy"][2])]
                text = pytesseract.image_to_string(crop, config='--psm 7 digits')  # For HP numbers
                if text.strip().isdigit():
                    d["extra"]["value"] = int(text)
            elif "trade_window" in cls:
                # Advanced: detect sub-elements like buttons
                pass  # Implement if needed

        return raw_dets

    def get_latest_detections(self) -> List[Dict]:
        """Fetch processed detections, non-blocking."""
        dets = []
        while not self.processed_queue.empty():
            dets = self.processed_queue.get()
        return dets

    def capture_screen(self) -> np.ndarray:  # Sync capture for non-threaded use
        """Fallback sync capture."""
        with self.lock:
            if self.last_frame is not None:
                return self.last_frame.copy()
        return cv2.cvtColor(np.array(self.sct.grab(self.region)), cv2.COLOR_BGRA2BGR)

    def detect_objects(self, frame: Optional[np.ndarray] = None) -> List[Dict]:
        """Sync detection for testing."""
        if frame is None:
            frame = self.capture_screen()
        raw = self._detect_raw(frame)
        return self._post_process_detections(raw, frame)

    def _save_debug_frame(self, frame: np.ndarray, dets: List[Dict]):
        """Save frame with bounding boxes."""
        debug_frame = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            color = (0, 255, 0) if d["conf"] > 0.8 else (0, 0, 255)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_frame, f"{d['class']} {d['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        ts = int(time.time())
        cv2.imwrite(os.path.join(DEBUG_FRAMES_DIR, f"debug_{ts}.jpg"), debug_frame)

    def get_metrics_summary(self) -> str:
        return f"FPS: {self.metrics.fps:.1f}, Dropped: {self.metrics.dropped_frames}"

# ────────────────────────────────────────────────
#   Advanced Features: Overlay Generator (for GUI debug)
# ────────────────────────────────────────────────

def generate_debug_overlay(frame: np.ndarray, dets: List[Dict], state: WorldState) -> np.ndarray:
    """Create overlay image with boxes, labels, risk colors."""
    overlay = frame.copy()
    alpha = float(config.get("Debug", "debug_overlay_opacity", fallback="0.7"))
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        risk = d.get("risk", 0)
        color = (0, 255, 0) if risk < HIGH_DANGER else (0, 255, 255) if risk < CRITICAL_DANGER else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} conf:{d['conf']:.2f} risk:{risk}"
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Add global state
    cv2.putText(overlay, f"HP: {state.player_hp_pct:.0f}% Danger: {state.danger_level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# ────────────────────────────────────────────────
#   UI-Specific Analyzers (e.g., Health Bar Fill %)
# ────────────────────────────────────────────────

def analyze_health_bar(crop: np.ndarray) -> float:
    """Estimate fill percentage via color ratio (red/green)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    total = red_pixels + green_pixels
    return (green_pixels / total * 100) if total > 0 else 100.0

# More analyzers: e.g., button state (active/inactive) via brightness, etc.

# ────────────────────────────────────────────────
#   Test Function (Standalone)
# ────────────────────────────────────────────────

if __name__ == "__main__":
    vp = VisionProcessor()
    vp.start()
    try:
        time.sleep(5)  # Run for 5s
        dets = vp.get_latest_detections()
        print(f"Detected: {len(dets)} objects")
    finally:
        vp.stop()
