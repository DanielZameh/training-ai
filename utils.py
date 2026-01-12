# Part 1/4: utils.py - Configuration, Logging, WorldState, ActionMemory, and Helpers
# This is the foundation module with all shared utilities, configs, state classes.
# Maximized content: detailed docstrings, type hints, error handling, more features.
# GitHub file: utils.py
# Other files will import like: from utils import config, WorldState, ActionMemory, logger, etc.

import json
import logging
import configparser
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional, Union
from collections import deque
import time
import threading
import os
import sys
from dataclasses import dataclass, field

# ────────────────────────────────────────────────
#   Enhanced Logging Setup with File + Console
# ────────────────────────────────────────────────

logger = logging.getLogger("ai_game_agent")
logger.setLevel(logging.DEBUG)  # More verbose for development

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)

# File handler - rotate if too big (simple version)
log_file = Path("agent_debug.log")
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(threadName)s %(message)s')
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("=== AI Game Agent Logger Initialized ===")

# ────────────────────────────────────────────────
#   Configuration Management - Robust & Thread-Safe
# ────────────────────────────────────────────────

CONFIG_FILE = Path("agent_config.ini")
DEFAULT_CONFIG = {
    "General": {
        "target_fps": "20",
        "capture_region": "0,0,1920,1080",      # left,top,width,height
        "reaction_delay_ms": "80",
        "vision_model_path": "yolov8n.pt",      # or your custom trained model
        "enable_voice": "false",
        "voice_rate": "150",
        "voice_volume": "0.9",
        "max_detection_queue_size": "10",
    },
    "Safety": {
        "enable_user_confirm_high_risk": "true",
        "forbidden_actions_file": "forbidden_actions.json",
        "emergency_key": "esc",
        "max_risk_without_confirm": "5",
        "auto_pause_on_danger_above": "8",
        "input_simulation_delay_min_ms": "50",
        "input_simulation_delay_max_ms": "150",
    },
    "Debug": {
        "show_floating_debug": "true",
        "log_last_n_actions": "100",
        "show_ai_avatar": "true",
        "debug_overlay_opacity": "0.7",
        "save_debug_frames": "false",
        "debug_frames_dir": "debug_frames",
    },
    "Thresholds": {
        "low_hp_threshold": "30.0",
        "critical_hp_threshold": "15.0",
        "danger_level_high": "7",
        "danger_level_critical": "9",
        "detection_confidence": "0.62",
        "iou_threshold_nms": "0.45",
        "min_object_size_pixels": "400",
    },
    "Performance": {
        "downscale_factor": "1.0",  # 0.5 = half size for faster processing
        "max_frame_process_time_ms": "80",
    }
}

config = configparser.ConfigParser()
config.optionxform = str  # Preserve case

def load_or_create_config():
    """Load config or create default with comments."""
    global config
    if CONFIG_FILE.exists():
        config.read(CONFIG_FILE)
        logger.info(f"Loaded existing config: {CONFIG_FILE}")
    else:
        config.read_dict(DEFAULT_CONFIG)
        # Add some explanatory comments
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            f.write("# AI Game Agent Configuration - Edit values carefully\n")
            f.write("# All values are strings, booleans use true/false\n\n")
            config.write(f)
        logger.info("Created new default configuration file")

load_or_create_config()

# Global parsed values (updated on reload if needed)
def reload_config_values():
    global TARGET_FPS, FRAME_DELAY, CAPTURE_REGION, REACTION_DELAY, VISION_MODEL_PATH
    global ENABLE_VOICE, ENABLE_USER_CONFIRM, EMERGENCY_KEY, SHOW_FLOATING_DEBUG
    global LOG_LAST_N, SHOW_AI_AVATAR, LOW_HP_THRESH, CRITICAL_HP_THRESH, HIGH_DANGER
    global CRITICAL_DANGER, DET_CONF, MIN_OBJECT_SIZE, DOWNSCALE_FACTOR

    TARGET_FPS = int(config.get("General", "target_fps", fallback="20"))
    FRAME_DELAY = 1.0 / TARGET_FPS
    CAPTURE_REGION = tuple(map(int, config.get("General", "capture_region", fallback="0,0,1920,1080").split(',')))
    REACTION_DELAY = float(config.get("General", "reaction_delay_ms", fallback="80")) / 1000.0
    VISION_MODEL_PATH = config.get("General", "vision_model_path", fallback="yolov8n.pt")
    ENABLE_VOICE = config.getboolean("General", "enable_voice", fallback=False)

    ENABLE_USER_CONFIRM = config.getboolean("Safety", "enable_user_confirm_high_risk", fallback=True)
    EMERGENCY_KEY = config.get("Safety", "emergency_key", fallback="esc").lower()
    MAX_RISK_NO_CONFIRM = int(config.get("Safety", "max_risk_without_confirm", fallback="5"))

    SHOW_FLOATING_DEBUG = config.getboolean("Debug", "show_floating_debug", fallback=True)
    LOG_LAST_N = int(config.get("Debug", "log_last_n_actions", fallback="100"))
    SHOW_AI_AVATAR = config.getboolean("Debug", "show_ai_avatar", fallback=True)

    LOW_HP_THRESH = float(config.get("Thresholds", "low_hp_threshold", fallback="30.0"))
    CRITICAL_HP_THRESH = float(config.get("Thresholds", "critical_hp_threshold", fallback="15.0"))
    HIGH_DANGER = int(config.get("Thresholds", "danger_level_high", fallback="7"))
    CRITICAL_DANGER = int(config.get("Thresholds", "danger_level_critical", fallback="9"))
    DET_CONF = float(config.get("Thresholds", "detection_confidence", fallback="0.62"))
    MIN_OBJECT_SIZE = int(config.get("Thresholds", "min_object_size_pixels", fallback="400"))

    DOWNSCALE_FACTOR = float(config.get("Performance", "downscale_factor", fallback="1.0"))

reload_config_values()

# Forbidden actions file
FORBIDDEN_FILE = Path(config.get("Safety", "forbidden_actions_file", fallback="forbidden_actions.json"))

# ────────────────────────────────────────────────
#   WorldState - Detailed Game World Representation
# ────────────────────────────────────────────────

@dataclass
class Detection:
    class_name: str
    confidence: float
    xyxy: List[float]           # [x1, y1, x2, y2]
    center: Tuple[float, float] = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)
    risk_level: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.center = ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2)
        self.width = self.xyxy[2] - self.xyxy[0]
        self.height = self.xyxy[3] - self.xyxy[1]

class WorldState:
    """
    Core game state representation - purely numerical/structural.
    Thread-safe updates via lock.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.timestamp: float = time.time()
        self.player_hp_pct: float = 100.0
        self.player_mana_pct: float = 100.0      # Optional, many games have mana
        self.player_position: Tuple[float, float] = (0.0, 0.0)  # Screen coords if known
        self.enemies: List[Detection] = []
        self.npcs: List[Detection] = []
        self.loot: List[Detection] = []
        self.ui_elements: List[Detection] = []
        self.interactives: List[Detection] = []
        self.danger_level: int = 0
        self.last_updated: float = 0.0

    def update_from_detections(self, raw_detections: List[Dict]):
        with self.lock:
            self.timestamp = time.time()
            self.last_updated = self.timestamp

            # Clear previous
            self.enemies.clear()
            self.npcs.clear()
            self.loot.clear()
            self.ui_elements.clear()

            # Process each detection
            for d in raw_detections:
                try:
                    cls = d.get("class", "unknown").lower()
                    conf = float(d.get("conf", 0.0))
                    box = d.get("xyxy", [0,0,0,0])
                    if sum(box) == 0 or conf < DET_CONF:
                        continue
                    det = Detection(
                        class_name=cls,
                        confidence=conf,
                        xyxy=box,
                        risk_level=d.get("risk", 5 if "enemy" in cls else 2)
                    )
                    if "enemy" in cls:
                        self.enemies.append(det)
                    elif "npc" in cls or "vendor" in cls:
                        self.npcs.append(det)
                    elif "loot" in cls or "chest" in cls or "item" in cls:
                        self.loot.append(det)
                    elif any(k in cls for k in ["button", "bar", "window", "dialog", "trade"]):
                        self.ui_elements.append(det)
                except Exception as e:
                    logger.debug(f"Invalid detection skipped: {e}")

            self.interactives = self.enemies + self.npcs + self.loot + self.ui_elements

            # Estimate HP if we have health bar (placeholder logic)
            health_bars = [ui for ui in self.ui_elements if "health" in ui.class_name]
            if health_bars:
                # In real: analyze fill color ratio
                self.player_hp_pct = 100.0 - (len(self.enemies) * 8)  # Dummy
                self.player_hp_pct = max(0.0, min(100.0, self.player_hp_pct))
            else:
                self.player_hp_pct = 100.0

            # Danger calculation - more sophisticated
            enemy_count = len(self.enemies)
            if enemy_count > 0:
                avg_conf = sum(e.confidence for e in self.enemies) / enemy_count
                proximity_bonus = 1 + (1 / (1 + enemy_count))  # closer = more dangerous
                self.danger_level = min(10, int(enemy_count * avg_conf * 8 * proximity_bonus))
            else:
                self.danger_level = max(0, self.danger_level - 2)  # Decay when safe

    def get_closest(self, category: str = "enemy") -> Optional[Detection]:
        """Get closest object to screen center (assuming player is center)"""
        items = getattr(self, category + "s", [])
        if not items:
            return None
        screen_center = (CAPTURE_REGION[2]/2, CAPTURE_REGION[3]/2)
        return min(items, key=lambda d: ((d.center[0] - screen_center[0])**2 + (d.center[1] - screen_center[1])**2)**0.5)

    def to_minimal_json(self) -> str:
        with self.lock:
            data = {
                "ts": round(self.timestamp, 3),
                "hp": round(self.player_hp_pct, 1),
                "danger": self.danger_level,
                "enemies": len(self.enemies),
                "loot": len(self.loot),
                "ui": len(self.ui_elements),
                "last_update_age": round(time.time() - self.last_updated, 2)
            }
            return json.dumps(data, separators=(',', ':'))

    def summary_for_llm(self) -> str:
        with self.lock:
            parts = [f"HP:{self.player_hp_pct:.0f}%", f"Danger:{self.danger_level}/10"]
            if self.enemies: parts.append(f"Enemies:{len(self.enemies)}")
            if self.loot: parts.append(f"Loot:{len(self.loot)}")
            if any("trade" in ui.class_name for ui in self.ui_elements):
                parts.append("TRADE_VISIBLE")
            if self.player_hp_pct < CRITICAL_HP_THRESH:
                parts.append("CRITICAL_HP")
            return " | ".join(parts) or "Calm state - nothing detected"

# ────────────────────────────────────────────────
#   ActionMemory - Learning from User Feedback
# ────────────────────────────────────────────────

class ActionMemory:
    """
    Persistent memory of user approvals/rejections + action history.
    Thread-safe with lock.
    """
    def __init__(self, filepath: Path = FORBIDDEN_FILE):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.forbidden: Set[str] = set()
        self.approved: Set[str] = set()
        self.action_history: deque = deque(maxlen=LOG_LAST_N)
        self.stats: Dict[str, int] = {}  # action_key -> count
        self.load()

    def load(self):
        with self.lock:
            if self.filepath.exists():
                try:
                    with self.filepath.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.forbidden = set(data.get("forbidden", []))
                    self.approved = set(data.get("approved", []))
                    self.stats = data.get("stats", {})
                    logger.info(f"Action memory loaded - {len(self.forbidden)} forbidden, {len(self.approved)} approved")
                except Exception as e:
                    logger.error(f"Failed to load action memory: {e}")

    def save(self):
        with self.lock:
            data = {
                "forbidden": list(self.forbidden),
                "approved": list(self.approved),
                "stats": self.stats,
                "last_saved": time.time()
            }
            try:
                with self.filepath.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save action memory: {e}")

    def record_action(self, action_key: str, executed: bool = True):
        with self.lock:
            if action_key not in self.stats:
                self.stats[action_key] = 0
            self.stats[action_key] += 1
            self.save()  # Save on every action for safety

    def is_forbidden(self, action_key: str) -> bool:
        with self.lock:
            return action_key in self.forbidden

    def is_approved(self, action_key: str) -> bool:
        with self.lock:
            return action_key in self.approved

    def mark_forbidden(self, action_key: str):
        with self.lock:
            self.forbidden.add(action_key)
            self.approved.discard(action_key)
            self.save()

    def mark_approved(self, action_key: str):
        with self.lock:
            self.approved.add(action_key)
            self.forbidden.discard(action_key)
            self.save()

    def log_execution(self, action: Dict):
        with self.lock:
            entry = {
                "ts": time.time(),
                "action": action,
                "key": action.get("key", "unknown"),
                "risk": action.get("risk", 0),
            }
            self.action_history.append(entry)

    def get_recent_actions(self, n: int = 10) -> List[Dict]:
        with self.lock:
            return list(self.action_history)[-n:]

    def get_most_used_actions(self, top_n: int = 5) -> List[Tuple[str, int]]:
        with self.lock:
            sorted_stats = sorted(self.stats.items(), key=lambda x: x[1], reverse=True)
            return sorted_stats[:top_n]
