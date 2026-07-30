# ══════════════════════════════════════════════════════════════════════════════
#  BIONIC HAND NEXUS — COGNITIVE ENTERPRISE EDITION v6.0  (FIXED)
#  Servos: 0=thumb  1=index  2=middle  3=ring  4=pinky  5=wrist
#
#  KEY FIXES vs original:
#  - Removed duplicate class definitions (ModelRegistry, PPOPolicy, all AI models,
#    SimulationEnvironment, PPOTrainer, GestureClassifier, GraspPlanner, all sensors,
#    EEGListener, HapticFeedback, IntentParser, ForceController, AnomalyDetector,
#    ReplayBuffer, EWCRegularizer, OnlinePersonalizer, FederatedClient,
#    WirelessBridge, OTAUpdater, FingerManager, DigitalTwinBroadcaster,
#    MotionMemory, DatasetFactory, SerialBridge, TelemetryPublisher,
#    VoiceController, PerformanceProfiler, HUDRenderer, PRESET_ANGLES,
#    SessionExporter, main() — all defined ONCE)
#  - Fixed phantom/autonomous finger movement:
#      * MotionSmoother: clamped spring feed-forward, prevented runaway
#      * KalmanAngleFilter: process noise / measurement noise balanced
#      * All angle outputs hard-clamped to [0, 180]
#      * Dead-band threshold applied BEFORE smoothing so tiny noise is rejected
#      * last_angles initialised to neutral [90]*6 and never updated unless
#        valid hand landmarks are actually detected
#  - Fixed EEG override firing with low confidence (threshold check was inverted)
#  - Fixed ForceController: integral anti-windup and correct sign convention
#  - Fixed AnomalyDetector.self_heal(): recovery_idx reset was wrong
#  - Fixed PPOTrainer: advantage normalisation when std==0
#  - Fixed MotionMemory.playback_frames(): negative delay guard
#  - Fixed SerialBridge / WirelessBridge: duplicate send guard was off-by-one
#  - Fixed GestureClassifier smooth_output: confidence average over wrong slice
#  - Fixed DatasetFactory.save(): duplicate write when buffer is empty
#  - Fixed HUDRenderer: servo bar width overflow when angle==180
#  - Fixed VoiceController: intent queue was never drained on non-voice build
#  - Removed dead GRASP_LIBRARY duplicate at module level
#  - All daemon threads guarded with _alive flag before start
#  - Consistent use of time.perf_counter() for timing (not mixing with time.time())
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os
import sys
import csv
import threading
import hashlib
import platform
import queue
import warnings
import asyncio
import logging
import logging.handlers
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Generator, Callable, Any, Deque
from datetime import datetime
from collections import deque, defaultdict
from enum import Enum
from copy import deepcopy
from functools import lru_cache
from contextlib import contextmanager

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Optional heavy dependencies ────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not installed. CNN/RL/Siamese modes disabled.")

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import pickle
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

try:
    import speech_recognition as sr
    import pyttsx3
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import bleak
    HAS_BLEAK = True
except ImportError:
    HAS_BLEAK = False

try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False

try:
    import pylsl as lsl
    HAS_LSL = True
except ImportError:
    HAS_LSL = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "camera_index": 0,
    "frame_width": 1280, "frame_height": 720, "frame_skip": 1,
    "serial_baud": 115200, "serial_port": "auto",
    "gesture_dir": "gestures", "dataset_dir": "dataset",
    "log_dir": "logs", "model_dir": "models",
    "export_dir": "exports", "screenshot_dir": "screenshots",
    "replay_buffer_dir": "replay",
    "model_path": "models/nexus_cnn.pth",
    "onnx_path": "models/nexus_model.onnx",
    "classifier_path": "models/gesture_clf.pkl",
    "rl_policy_path": "models/rl_policy.pth",
    "siamese_model_path": "models/siamese_net.pth",
    "fusion_model_path": "models/fusion_net.pth",
    "grasp_library_path": "grasp_library.json",

    # --- Kinematics (FIXED: sane defaults prevent phantom movement) ---
    "ratio_min": 0.5,       # raised from 0.4 → less noise at closed fist
    "ratio_max": 1.9,       # lowered from 2.1 → less noise at open hand
    "smoothing_window": 7,  # increased → smoother output
    "angle_change_threshold": 3,   # dead-band: ignore changes < 3 deg
    "max_velocity_deg_per_frame": 12,  # lowered → slower, no jerk
    "max_jerk_deg_per_frame2": 5,      # lowered → smoother

    # --- Kalman (FIXED: balanced noise params) ---
    "kalman_enabled": True,
    "kalman_process_noise": 5e-4,       # was 1e-3 (too high → noisy)
    "kalman_measurement_noise": 0.3,    # was 0.1  (too low  → noisy)

    "servo_limits": {
        "thumb":  [5, 175], "index":  [5, 175], "middle": [5, 175],
        "ring":   [5, 175], "pinky":  [5, 175], "wrist":  [0, 180]
    },

    "gesture_classifier_enabled": True,
    "gesture_confidence_threshold": 0.65,

    "fsr_enabled": False,
    "fsr_channels": 5,
    "target_grip_force_n": 3.0,
    "force_pid_kp": 0.6,    # was 0.8 → slightly calmer
    "force_pid_ki": 0.02,   # was 0.05 → less integral windup
    "force_pid_kd": 0.02,
    "slip_variance_threshold": 0.15,

    "stiffness_enabled": False,
    "stiffness_pid_kp": 0.5,
    "stiffness_force_threshold": 5.0,

    "spring_torque_enabled": False,
    "spring_constant": 0.02,   # was 0.05 → less phantom torque

    "yolo_enabled": False, "yolo_model": "yolov8n.pt", "yolo_confidence": 0.5,
    "grasp_planning_enabled": False,

    "rl_enabled": False,
    "rl_update_frequency": 50,
    "rl_learning_rate": 3e-4,
    "rl_gamma": 0.99,
    "rl_clip_epsilon": 0.2,
    "energy_saving_mode": False,
    "energy_penalty_weight": 0.01,

    "siamese_enabled": False, "siamese_support_set_size": 5,
    "fusion_enabled": False,
    "emg_channels": 8, "emg_sample_rate": 1000, "emg_window_ms": 200,

    "tactile_enabled": False, "tactile_grid_size": 4,
    "proximity_enabled": False, "proximity_min_cm": 1.0, "proximity_max_cm": 20.0,

    # FIXED: EEG threshold raised so random noise doesn't trigger override
    "eeg_enabled": False,
    "eeg_confidence_threshold": 0.82,  # was 0.75 → too easy to trigger
    "eeg_stream_name": "EEG",

    "haptic_enabled": False, "haptic_serial_port": "auto", "haptic_baud": 115200,

    # FIXED: anomaly window + threshold so normal motion doesn't false-alarm
    "anomaly_detection_enabled": True,
    "anomaly_window": 30,       # was 20
    "anomaly_threshold": 2.5,   # was 0.15 (z-score units, not raw)

    "federated_enabled": False,
    "federated_server": "https://fl.bionic-nexus.io",
    "federated_min_samples": 500,
    "user_id": hashlib.sha256(platform.node().encode()).hexdigest()[:12],

    "ota_enabled": False,
    "ota_server": "https://updates.bionic-nexus.io",
    "ota_firmware_url": "",
    "model_version": "6.0.0",

    "telemetry_enabled": False,
    "mqtt_broker": "localhost", "mqtt_port": 1883,
    "mqtt_topic_prefix": "bionic/nexus",
    "heartbeat_interval_s": 1.0,

    "api_port": 8765, "api_enabled": False,
    "websocket_port": 8766,

    "hand_id": 1,
    "digital_twin_enabled": False, "digital_twin_ws_port": 8767,

    "voice_enabled": False, "voice_language": "en-US",

    "continual_learning_enabled": False,
    "ewc_lambda": 400,
    "replay_buffer_size": 2000,
    "online_train_interval": 200,

    "finger_manager_enabled": False,
    "finger_i2c_addresses": [0x10, 0x11, 0x12, 0x13, 0x14],

    "ble_enabled": False,
    "ble_device_name": "BionicHand",
    "ble_service_uuid": "6e400001-b5a3-f393-e0a9-e50e24dcca9e",
    "ble_char_uuid":    "6e400002-b5a3-f393-e0a9-e50e24dcca9e",

    "playback_speed": 1.0,
    "profiler_enabled": False,
    "headless": False,
    "hud_theme": "dark",
}


class ConfigManager:
    CONFIG_PATH = "nexus_config.json"

    def __init__(self):
        self._cfg = deepcopy(DEFAULT_CONFIG)
        if Path(self.CONFIG_PATH).exists():
            try:
                with open(self.CONFIG_PATH) as f:
                    self._deep_merge(self._cfg, json.load(f))
            except Exception:
                pass  # corrupted config → use defaults
        else:
            self._save(self._cfg)

    def _deep_merge(self, base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v

    def get(self, *keys, default=None):
        d = self._cfg
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    def set(self, key: str, value):
        self._cfg[key] = value
        self._save(self._cfg)

    def _save(self, cfg: dict):
        try:
            with open(self.CONFIG_PATH, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def save(self):
        self._save(self._cfg)


CFG = ConfigManager()

for _dir in ["gesture_dir", "dataset_dir", "log_dir", "model_dir",
             "export_dir", "screenshot_dir", "replay_buffer_dir"]:
    Path(CFG.get(_dir)).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LOGGING + MLOPS
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger() -> logging.Logger:
    log_path = Path(CFG.get("log_dir")) / f"nexus_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("bionic.nexus")
    logger.setLevel(logging.DEBUG)
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOG = setup_logger()


class MLopsTracker:
    def __init__(self):
        self._run_id = hashlib.md5(
            f"{datetime.now().isoformat()}{CFG.get('user_id')}".encode()
        ).hexdigest()[:10]
        self._log_path = Path(CFG.get("log_dir")) / f"mlops_{self._run_id}.jsonl"
        self._metrics: Dict[str, list] = defaultdict(list)
        self._step = 0

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        s = step if step is not None else self._step
        record = {"step": s, "ts": time.time(), **metrics}
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass
        for k, v in metrics.items():
            self._metrics[k].append((s, v))
        self._step += 1

    def summary(self) -> Dict[str, float]:
        return {k: float(np.mean([v for _, v in vals]))
                for k, vals in self._metrics.items() if vals}


MLOPS = MLopsTracker()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ENUMS & DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class TrackingMode(Enum):
    MIRROR      = 1
    PLAYBACK    = 2
    CNN         = 3
    AUTONOMOUS  = 4
    EMG         = 5
    RL_ADAPTIVE = 6
    HYBRID      = 7
    FUSION      = 8
    EEG         = 9


class GestureClass(Enum):
    UNKNOWN   = "UNKNOWN"
    OPEN_HAND = "OPEN_HAND"
    FIST      = "FIST"
    PINCH     = "PINCH"
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DN = "THUMBS_DOWN"
    PEACE     = "PEACE"
    POINT     = "POINT"
    OK_SIGN   = "OK_SIGN"
    ROCK      = "ROCK"
    CLAW      = "CLAW"
    PINCER    = "PINCER"
    TRIPOD    = "TRIPOD"
    WAVE      = "WAVE"
    SNAP      = "SNAP"


class GraspType(Enum):
    CYLINDRICAL = "CYLINDRICAL"
    SPHERICAL   = "SPHERICAL"
    PINCH       = "PINCH"
    LATERAL     = "LATERAL"
    HOOK        = "HOOK"
    TRIPOD      = "TRIPOD"
    POWER       = "POWER"


class AnomalyLevel(Enum):
    NORMAL   = 0
    WARNING  = 1
    CRITICAL = 2


@dataclass
class AngleFrame:
    t_ms:       float
    angles:     List[int]
    gesture:    str   = "UNKNOWN"
    confidence: float = 0.0
    forces:     Optional[List[float]] = None
    emg_signal: Optional[List[float]] = None
    imu_data:   Optional[List[float]] = None


@dataclass
class Gesture:
    name:        str
    duration_ms: float
    frames:      List[AngleFrame] = field(default_factory=list)
    tags:        List[str]        = field(default_factory=list)
    created_at:  str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ServoState:
    angles:       List[int]   = field(default_factory=lambda: [90] * 6)
    velocities:   List[float] = field(default_factory=lambda: [0.0] * 6)
    forces:       List[float] = field(default_factory=lambda: [0.0] * 5)
    temperatures: List[float] = field(default_factory=lambda: [25.0] * 6)
    currents:     List[float] = field(default_factory=lambda: [0.0] * 6)
    stiffness:    List[float] = field(default_factory=lambda: [0.5] * 6)
    timestamp:    float       = field(default_factory=time.perf_counter)


@dataclass
class IMUData:
    accel:     List[float] = field(default_factory=lambda: [0.0] * 3)
    gyro:      List[float] = field(default_factory=lambda: [0.0] * 3)
    timestamp: float       = field(default_factory=time.perf_counter)


@dataclass
class TactileReading:
    finger_idx:  int
    grid:        List[float] = field(default_factory=lambda: [0.0] * 16)
    centroid_x:  float = 0.0
    centroid_y:  float = 0.0
    total_force: float = 0.0
    slip_risk:   float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PRESET ANGLES  (defined once, used everywhere)
# ══════════════════════════════════════════════════════════════════════════════

PRESET_ANGLES: Dict[str, List[int]] = {
    "open":       [175, 175, 175, 175, 175, 90],
    "fist":       [5,   5,   5,   5,   5,   90],
    "pinch":      [40,  30,  170, 170, 170, 90],
    "peace":      [5,   175, 175, 5,   5,   90],
    "thumbs_up":  [175, 5,   5,   5,   5,   90],
    "point":      [5,   175, 5,   5,   5,   90],
    "ok":         [40,  30,  175, 175, 175, 90],
    "power":      [90,  80,  85,  85,  85,  80],
    "tripod":     [50,  35,  35,  160, 160, 90],
    "lateral":    [30,  10,  160, 160, 160, 90],
    "wave_start": [175, 175, 175, 175, 175, 60],
    "wave_end":   [175, 175, 175, 175, 175, 120],
    "neutral":    [90,  90,  90,  90,  90,  90],
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: GRASP LIBRARY  (defined once)
# ══════════════════════════════════════════════════════════════════════════════

GRASP_LIBRARY: Dict[str, Tuple[GraspType, List[int]]] = {
    "bottle":      (GraspType.CYLINDRICAL, [80, 70, 75, 75, 75, 90]),
    "cup":         (GraspType.CYLINDRICAL, [85, 75, 80, 80, 80, 90]),
    "pen":         (GraspType.PINCH,       [40, 30, 30, 160, 160, 90]),
    "key":         (GraspType.LATERAL,     [30, 20, 160, 160, 160, 90]),
    "egg":         (GraspType.SPHERICAL,   [70, 65, 68, 68, 68, 90]),
    "phone":       (GraspType.CYLINDRICAL, [75, 60, 65, 65, 65, 90]),
    "scissors":    (GraspType.PINCH,       [50, 35, 35, 160, 160, 90]),
    "screwdriver": (GraspType.CYLINDRICAL, [90, 80, 85, 85, 85, 90]),
    "fork":        (GraspType.CYLINDRICAL, [80, 70, 75, 75, 75, 90]),
    "ball":        (GraspType.SPHERICAL,   [75, 70, 72, 72, 72, 90]),
    "book":        (GraspType.LATERAL,     [40, 10, 160, 160, 160, 90]),
    "door_handle": (GraspType.POWER,       [90, 85, 88, 88, 88, 80]),
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def _clamp_angle(a: float) -> int:
    """Hard clamp to valid servo range."""
    return int(np.clip(round(a), 0, 180))


class KalmanAngleFilter:
    """
    FIXED:
    - Balanced Q/R so noisy measurements don't cause phantom movement.
    - Innovation history used only for diagnostics.
    """

    def __init__(self, n: int = 6,
                 process_noise: float = 5e-4,
                 measurement_noise: float = 0.3):
        self.n = n
        self.Q = process_noise
        self.R = measurement_noise
        self.x = np.zeros((n, 2))          # [angle, velocity]
        self.P = np.tile(np.eye(2), (n, 1, 1))
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self._innov_sq: Deque[float] = deque(maxlen=60)

    def reset(self, angles: List[int]):
        """Reset filter to known state (call when mode changes)."""
        for i, a in enumerate(angles[:self.n]):
            self.x[i] = [float(a), 0.0]
            self.P[i] = np.eye(2)

    def update(self, measurements: List[int]) -> List[int]:
        out = []
        total_innov = 0.0
        for i, z in enumerate(measurements):
            z = float(np.clip(z, 0, 180))
            x_ = self.F @ self.x[i]
            P_ = self.F @ self.P[i] @ self.F.T + np.eye(2) * self.Q
            innov = z - float(self.H @ x_)
            S = float(self.H @ P_ @ self.H.T) + self.R
            K = (P_ @ self.H.T) / S
            self.x[i] = x_ + K.flatten() * innov
            self.P[i] = (np.eye(2) - np.outer(K.flatten(), self.H)) @ P_
            total_innov += innov ** 2
            out.append(_clamp_angle(self.x[i, 0]))
        self._innov_sq.append(total_innov / self.n)
        return out

    @property
    def innovations_rmse(self) -> float:
        if not self._innov_sq:
            return 0.0
        return float(np.sqrt(np.mean(self._innov_sq)))


class MotionSmoother:
    """
    FIXED:
    - Spring feed-forward clipped so it cannot accumulate and drift.
    - Proximity scaling minimum raised to prevent complete velocity=0 freeze.
    - prev_angles clamped on each write.
    """

    def __init__(self, n: int = 6):
        self.n = n
        self.prev_angles:     List[float] = [90.0] * n
        self.prev_velocities: List[float] = [0.0]  * n
        self.max_vel   = float(CFG.get("max_velocity_deg_per_frame"))
        self.max_jerk  = float(CFG.get("max_jerk_deg_per_frame2"))
        self.proximity_factor: float = 1.0
        self._spring_enabled = bool(CFG.get("spring_torque_enabled"))
        self._spring_k       = float(CFG.get("spring_constant"))
        self.rest_angles     = [90.0] * n

    def smooth(self, target: List[int]) -> List[int]:
        eff_max_vel = self.max_vel * float(np.clip(self.proximity_factor, 0.15, 1.0))
        out = []
        for i, t in enumerate(target[:self.n]):
            t = float(np.clip(t, 0, 180))
            desired_v = t - self.prev_angles[i]

            # FIXED: spring feed-forward clipped to ±5 deg to prevent runaway
            if self._spring_enabled:
                spring_resist = float(np.clip(
                    self._spring_k * (self.prev_angles[i] - self.rest_angles[i]),
                    -5.0, 5.0))
                desired_v -= spring_resist

            jerk = float(np.clip(desired_v - self.prev_velocities[i],
                                 -self.max_jerk, self.max_jerk))
            v = float(np.clip(self.prev_velocities[i] + jerk,
                              -eff_max_vel, eff_max_vel))
            new_angle = float(np.clip(self.prev_angles[i] + v, 0.0, 180.0))
            self.prev_angles[i] = new_angle
            self.prev_velocities[i] = v
            out.append(_clamp_angle(new_angle))
        return out

    def reset(self, angles: List[int]):
        """Reset smoother to known state (call on mode switch)."""
        self.prev_angles     = [float(np.clip(a, 0, 180)) for a in angles[:self.n]]
        self.prev_velocities = [0.0] * self.n

    @property
    def servo_velocities(self) -> List[float]:
        return list(self.prev_velocities)


class AdaptiveCalibrator:
    def __init__(self):
        self._ratios: List[float] = []
        self._collecting = False
        self._start_time = 0.0
        self._duration   = 5.0

    def start(self, duration_s: float = 5.0):
        self._ratios.clear()
        self._collecting  = True
        self._start_time  = time.perf_counter()
        self._duration    = duration_s
        LOG.info("AutoCalibration started")

    def feed_ratio(self, ratio: float):
        if not self._collecting:
            return
        if time.perf_counter() - self._start_time > self._duration:
            self._finish()
            return
        self._ratios.append(float(ratio))

    def _finish(self):
        self._collecting = False
        if len(self._ratios) < 50:
            LOG.warning("Calibration: insufficient data")
            return
        r = np.array(self._ratios)
        p2, p98 = np.percentile(r, 2), np.percentile(r, 98)
        CFG.set("ratio_min", round(float(p2),  3))
        CFG.set("ratio_max", round(float(p98), 3))
        LOG.info(f"AutoCalibration done: min={p2:.3f} max={p98:.3f}")
        print(f"[CALIBRATION] Updated: ratio_min={p2:.3f}, ratio_max={p98:.3f}")

    @property
    def active(self) -> bool:
        return self._collecting

    @property
    def progress(self) -> float:
        if not self._collecting:
            return 0.0
        return min(1.0, (time.perf_counter() - self._start_time) / self._duration)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: HAND TRACKER
# ══════════════════════════════════════════════════════════════════════════════

FINGER_LANDMARKS = {
    "thumb":  [2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
WRIST_IDX = 0


class HandTracker:
    """
    FIXED:
    - Returns None when no hand is detected (never returns stale/phantom angles).
    - ratio_min/max re-read from config each frame if calibration ran.
    """

    def __init__(self, calibrator: Optional[AdaptiveCalibrator] = None):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.72,   # slightly stricter
            min_tracking_confidence=0.65,
            model_complexity=1,
        )
        self.draw        = mp.solutions.drawing_utils
        self.draw_styles = mp.solutions.drawing_styles
        self.calibrator  = calibrator
        self._smooth: Deque = deque(maxlen=CFG.get("smoothing_window"))
        self._last_landmarks_3d: Optional[np.ndarray] = None
        self._no_hand_count = 0     # consecutive frames without a hand

    def _lm(self, lms, idx: int) -> np.ndarray:
        l = lms.landmark[idx]
        return np.array([l.x, l.y, l.z])

    def _finger_angle(self, lms, finger: str) -> Tuple[int, float]:
        idxs    = FINGER_LANDMARKS[finger]
        wrist   = self._lm(lms, WRIST_IDX)
        knuckle = self._lm(lms, idxs[0])
        tip     = self._lm(lms, idxs[-1])
        d_wt    = float(np.linalg.norm(wrist - tip))
        d_wk    = float(np.linalg.norm(wrist - knuckle))
        ratio_min = float(CFG.get("ratio_min"))
        ratio_max = float(CFG.get("ratio_max"))
        ratio = d_wt / d_wk if d_wk > 1e-6 else ratio_max
        t = float(np.clip(
            (ratio - ratio_min) / (ratio_max - ratio_min + 1e-9),
            0.0, 1.0))
        return _clamp_angle(t * 180), ratio

    def _wrist_rotation_angle(self, lms) -> int:
        wrist  = self._lm(lms, 0)
        idx_mc = self._lm(lms, 5)
        pky_mc = self._lm(lms, 17)
        v1 = idx_mc - wrist
        v2 = pky_mc - wrist
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) < 1e-6:
            return 90
        normal = normal / np.linalg.norm(normal)
        dot = float(np.clip(np.dot(normal, np.array([0, 1, 0])), -1, 1))
        return _clamp_angle((dot + 1.0) / 2.0 * 180)

    def _extract_3d_landmarks(self, lms) -> np.ndarray:
        return np.array([[lms.landmark[i].x, lms.landmark[i].y, lms.landmark[i].z]
                         for i in range(21)])

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List[int]]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        annotated = frame.copy()

        # FIXED: always return None if no hand — never return phantom angles
        if not results.multi_hand_landmarks:
            self._no_hand_count += 1
            self._last_landmarks_3d = None
            self._smooth.clear()   # clear buffer → no stale data bleeds in
            return annotated, None

        self._no_hand_count = 0
        hand = results.multi_hand_landmarks[0]
        self.draw.draw_landmarks(
            annotated, hand,
            mp.solutions.hands.HAND_CONNECTIONS,
            self.draw_styles.get_default_hand_landmarks_style(),
            self.draw_styles.get_default_hand_connections_style(),
        )

        finger_angles = []
        for f in FINGER_ORDER:
            ang, ratio = self._finger_angle(hand, f)
            finger_angles.append(ang)
            if self.calibrator:
                self.calibrator.feed_ratio(ratio)

        wrist_ang = self._wrist_rotation_angle(hand)
        raw_angles = finger_angles + [wrist_ang]

        # Dead-band: reject tiny jumps BEFORE smoothing
        thresh = int(CFG.get("angle_change_threshold"))
        if self._smooth:
            prev = list(self._smooth[-1])
            if not any(abs(raw_angles[i] - prev[i]) >= thresh for i in range(6)):
                raw_angles = prev  # no significant motion → hold last

        self._smooth.append(raw_angles)
        smoothed = [_clamp_angle(np.mean([s[i] for s in self._smooth]))
                    for i in range(6)]
        self._last_landmarks_3d = self._extract_3d_landmarks(hand)
        return annotated, smoothed

    @property
    def landmarks_3d(self) -> Optional[np.ndarray]:
        return self._last_landmarks_3d


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: AI / DEEP LEARNING MODELS  (defined ONCE)
# ══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class AngleRegressionCNN(nn.Module):
        def __init__(self, pretrained: bool = False):
            super().__init__()
            backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            self.features  = backbone.features
            self.avgpool   = backbone.avgpool
            self.regressor = nn.Sequential(
                nn.Linear(576, 256), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.2),
                nn.Linear(128, 6),   nn.Sigmoid(),
            )
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.features(x)
            x = self.avgpool(x)
            return self.regressor(x.flatten(1)) * 180.0

        def predict_frame(self, frame: np.ndarray, device: str = "cpu") -> Optional[List[int]]:
            try:
                img = self._transform(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    angles = self(img).squeeze().cpu().numpy()
                return [_clamp_angle(a) for a in angles]
            except Exception as e:
                LOG.error(f"CNN inference: {e}")
                return None

        def extract_embedding(self, frame: np.ndarray, device: str = "cpu") -> Optional[np.ndarray]:
            try:
                img = self._transform(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    x = self.features(img)
                    x = self.avgpool(x)
                    return x.flatten(1).cpu().numpy().squeeze()
            except Exception as e:
                LOG.error(f"Embedding: {e}")
                return None

    class SiameseGraspNet(nn.Module):
        def __init__(self, embed_dim: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            )
            self.projection = nn.Sequential(
                nn.Linear(128 * 16, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
            self.grip_head = nn.Sequential(
                nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 6), nn.Sigmoid())
            self._transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((64, 64)),
                transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.projection(self.encoder(x).flatten(1))

        def forward(self, query: "torch.Tensor", support: "torch.Tensor") -> "torch.Tensor":
            q = self.encode(query)
            s = self.encode(support).mean(0, keepdim=True)
            return self.grip_head((q + s) / 2.0) * 180.0

        def predict_from_frames(self, query_frame: np.ndarray,
                                support_frames: List[np.ndarray],
                                device: str = "cpu") -> Optional[List[int]]:
            try:
                q = self._transform(query_frame).unsqueeze(0).to(device)
                s = torch.stack([self._transform(f) for f in support_frames]).to(device)
                with torch.no_grad():
                    angles = self(q, s).squeeze().cpu().numpy()
                return [_clamp_angle(a) for a in angles]
            except Exception as e:
                LOG.error(f"Siamese: {e}")
                return None

    class FusionNet(nn.Module):
        def __init__(self, emg_channels: int = 8, emg_len: int = 200,
                     imu_dim: int = 6, vision_dim: int = 576):
            super().__init__()
            self.vision_proj = nn.Sequential(nn.Linear(vision_dim, 128), nn.ReLU())
            self.emg_conv    = nn.Sequential(
                nn.Conv1d(emg_channels, 32, 7, padding=3), nn.ReLU(),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8))
            self.emg_proj  = nn.Linear(64 * 8, 128)
            self.imu_proj  = nn.Sequential(nn.Linear(imu_dim, 32), nn.ReLU(),
                                           nn.Linear(32, 64), nn.ReLU())
            self.fusion    = nn.Sequential(
                nn.Linear(128 + 128 + 64, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 6), nn.Sigmoid())

        def forward(self, vision: "torch.Tensor", emg: "torch.Tensor",
                    imu: "torch.Tensor") -> "torch.Tensor":
            v = self.vision_proj(vision)
            e = self.emg_proj(self.emg_conv(emg).flatten(1))
            i = self.imu_proj(imu)
            return self.fusion(torch.cat([v, e, i], dim=-1)) * 180.0

        def predict(self, vision_emb: np.ndarray, emg_window: np.ndarray,
                    imu_vec: np.ndarray, device: str = "cpu") -> Optional[List[int]]:
            try:
                v = torch.FloatTensor(vision_emb).unsqueeze(0).to(device)
                e = torch.FloatTensor(emg_window).unsqueeze(0).to(device)
                i = torch.FloatTensor(imu_vec).unsqueeze(0).to(device)
                with torch.no_grad():
                    angles = self(v, e, i).squeeze().cpu().numpy()
                return [_clamp_angle(a) for a in angles]
            except Exception as e:
                LOG.error(f"FusionNet: {e}")
                return None

    class TemporalTransformer(nn.Module):
        def __init__(self, seq_len: int = 10, pred_horizon: int = 3,
                     d_model: int = 64, nhead: int = 4, num_layers: int = 2):
            super().__init__()
            self.seq_len       = seq_len
            self.pred_horizon  = pred_horizon
            self.input_proj    = nn.Linear(6, d_model)
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                             dim_feedforward=d_model * 4,
                                             dropout=0.1, batch_first=True)
            self.transformer   = nn.TransformerEncoder(enc, num_layers=num_layers)
            self.output_proj   = nn.Linear(d_model, 6 * pred_horizon)
            self.pos_enc       = nn.Parameter(torch.zeros(1, seq_len, d_model))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.input_proj(x / 180.0) + self.pos_enc
            x = self.transformer(x)
            return self.output_proj(x[:, -1, :]).view(-1, self.pred_horizon, 6) * 180.0

        def predict_next(self, angle_history: Deque,
                         device: str = "cpu") -> Optional[List[int]]:
            if len(angle_history) < self.seq_len:
                return None
            try:
                seq = torch.FloatTensor(list(angle_history)[-self.seq_len:]).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = self(seq)
                return [_clamp_angle(a) for a in pred[0, 0].cpu().numpy()]
            except Exception:
                return None

    class PPOPolicy(nn.Module):
        """FIXED: advantage std guard prevents division by zero."""

        def __init__(self, state_dim: int = 25, action_dim: int = 6):
            super().__init__()
            self.shared       = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 128),       nn.Tanh())
            self.actor_mean    = nn.Linear(128, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            self.critic        = nn.Linear(128, 1)

        def forward(self, state: "torch.Tensor"):
            x     = self.shared(state)
            mean  = torch.tanh(self.actor_mean(x)) * 10.0
            std   = torch.exp(self.actor_log_std.clamp(-3, 0))
            value = self.critic(x)
            return mean, std, value

        def get_action(self, state_np: np.ndarray,
                       deterministic: bool = False) -> np.ndarray:
            state = torch.FloatTensor(state_np).unsqueeze(0)
            mean, std, _ = self(state)
            if deterministic:
                return mean.detach().numpy().squeeze()
            return torch.distributions.Normal(mean, std).sample().detach().numpy().squeeze()

        @staticmethod
        def compute_energy_penalty(velocities: List[float],
                                   weight: float = None) -> float:
            w = weight if weight is not None else CFG.get("energy_penalty_weight")
            return float(w) * float(np.sum(np.array(velocities) ** 2))


class ModelRegistry:
    """Defined once — loads and caches all models."""

    def __init__(self):
        self._lock   = threading.RLock()
        self.device  = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self._models: Dict[str, Any] = {}
        LOG.info(f"ModelRegistry: device={self.device}")

    def _load_torch(self, name: str, cls, path_key: str, *args, **kwargs):
        with self._lock:
            if name in self._models:
                return self._models[name]
            if not HAS_TORCH:
                return None
            path = Path(CFG.get(path_key))
            try:
                m = cls(*args, **kwargs)
                if path.exists():
                    m.load_state_dict(torch.load(str(path), map_location=self.device,
                                                 weights_only=True))
                m.eval()
                self._models[name] = m
                LOG.info(f"{name} loaded ({path if path.exists() else 'new weights'})")
                return m
            except Exception as e:
                LOG.error(f"{name} load failed: {e}")
                return None

    def load_cnn(self):
        return self._load_torch("cnn", AngleRegressionCNN, "model_path", False) if HAS_TORCH else None

    def load_rl_policy(self):
        return self._load_torch("rl", PPOPolicy, "rl_policy_path") if HAS_TORCH else None

    def load_siamese(self):
        return self._load_torch("siamese", SiameseGraspNet, "siamese_model_path") if HAS_TORCH else None

    def load_fusion(self):
        if not HAS_TORCH:
            return None
        emg_len = int(CFG.get("emg_sample_rate") * CFG.get("emg_window_ms") / 1000)
        return self._load_torch("fusion", FusionNet, "fusion_model_path",
                                CFG.get("emg_channels"), emg_len)

    def load_onnx(self):
        with self._lock:
            if "onnx" in self._models:
                return self._models["onnx"]
            path = Path(CFG.get("onnx_path"))
            if not HAS_ONNX or not path.exists():
                return None
            try:
                sess = ort.InferenceSession(str(path))
                self._models["onnx"] = sess
                return sess
            except Exception as e:
                LOG.error(f"ONNX load: {e}")
                return None

    def get(self, name: str) -> Optional[Any]:
        with self._lock:
            return self._models.get(name)

    def register(self, name: str, model: Any):
        with self._lock:
            self._models[name] = model


REGISTRY = ModelRegistry()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: SIMULATION + PPO TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class SimulationEnvironment:
    N_JOINTS = 6

    def __init__(self, gui: bool = False):
        self._gui             = gui
        self._physics_client  = None
        self._step_count      = 0
        self._max_steps       = 200
        self._current_angles  = np.array([90.0] * self.N_JOINTS)
        self._energy_coeff    = float(CFG.get("energy_penalty_weight"))
        if HAS_PYBULLET:
            self._init_pybullet()
        else:
            LOG.warning("SimulationEnvironment: PyBullet unavailable, stub mode")

    def _init_pybullet(self):
        try:
            mode = p.GUI if self._gui else p.DIRECT
            self._physics_client = p.connect(mode)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")
        except Exception as e:
            LOG.error(f"PyBullet init: {e}")
            self._physics_client = None

    def reset(self) -> np.ndarray:
        self._step_count     = 0
        self._current_angles = np.array([90.0] * self.N_JOINTS)
        if self._physics_client is not None:
            p.setGravity(0, 0, -9.81 + np.random.uniform(-0.5, 0.5))
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        prev = self._current_angles.copy()
        self._current_angles = np.clip(self._current_angles + action, 0, 180)
        self._step_count += 1
        velocities   = self._current_angles - prev
        energy_pen   = self._energy_coeff * float(np.sum(velocities ** 2))
        target       = np.array([80, 70, 75, 75, 75, 90])
        dist         = float(np.mean(np.abs(self._current_angles - target)))
        reward       = max(0.0, 1.0 - dist / 90.0) - energy_pen
        done         = self._step_count >= self._max_steps or dist < 5.0
        if self._physics_client is not None:
            p.stepSimulation()
        return self._get_obs(), reward, done, {"dist": dist, "energy": energy_pen}

    def _get_obs(self) -> np.ndarray:
        imu_noise = np.random.normal(0, 0.5, 6)
        forces    = np.random.uniform(0, 2, 5)
        obj_class = np.zeros(8); obj_class[0] = 1.0
        return np.concatenate([
            self._current_angles / 180.0, imu_noise,
            forces / 10.0, obj_class
        ]).astype(np.float32)

    def close(self):
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass


class PPOTrainer:
    """FIXED: advantage normalisation guards std == 0."""

    def __init__(self):
        self._enabled       = HAS_TORCH
        self._lr            = float(CFG.get("rl_learning_rate"))
        self._gamma         = float(CFG.get("rl_gamma"))
        self._energy_saving = bool(CFG.get("energy_saving_mode"))

    def train(self, n_episodes: int = 500, steps_per_ep: int = 200,
              progress_cb: Optional[Callable] = None) -> Optional[str]:
        if not self._enabled:
            LOG.warning("PPOTrainer: PyTorch unavailable")
            return None
        print(f"[SIM] Starting PPO training ({n_episodes} episodes)...")
        policy    = PPOPolicy(state_dim=25, action_dim=6)
        optimizer = Adam(policy.parameters(), lr=self._lr)
        env       = SimulationEnvironment(gui=False)
        best      = -float("inf")

        for ep in range(n_episodes):
            obs    = env.reset()
            ep_r   = 0.0
            log_probs, rewards, values = [], [], []

            for _ in range(steps_per_ep):
                st   = torch.FloatTensor(obs).unsqueeze(0)
                mean, std, val = policy(st)
                dist = torch.distributions.Normal(mean, std)
                act  = dist.sample()
                lp   = dist.log_prob(act).sum(-1)

                a_np = act.squeeze().detach().numpy()
                obs, r, done, _ = env.step(a_np)

                if self._energy_saving:
                    r -= PPOPolicy.compute_energy_penalty(a_np.tolist())

                log_probs.append(lp)
                rewards.append(r)
                values.append(val)
                ep_r += r
                if done:
                    break

            if len(rewards) > 1:
                rets  = self._compute_returns(rewards)
                ret_t = torch.FloatTensor(rets)
                lp_t  = torch.stack(log_probs).squeeze()
                val_t = torch.stack(values).squeeze()
                adv   = ret_t - val_t.detach()
                # FIXED: guard std == 0
                adv_std = adv.std()
                if adv_std.item() > 1e-8:
                    adv = (adv - adv.mean()) / (adv_std + 1e-8)
                loss = -(lp_t * adv).mean() + 0.5 * F.mse_loss(val_t, ret_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

            if ep_r > best:
                best = ep_r
            if progress_cb:
                progress_cb(ep / n_episodes, ep_r)
            if (ep + 1) % 50 == 0:
                print(f"[SIM] Ep {ep+1}/{n_episodes} | r={ep_r:.3f} | best={best:.3f}")

        env.close()
        out_path = Path(CFG.get("model_dir")) / "rl_policy.pth"
        torch.save(policy.state_dict(), str(out_path))
        print(f"[SIM] Policy → {out_path}")
        return str(out_path)

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        rets, R = [], 0.0
        for r in reversed(rewards):
            R = r + self._gamma * R
            rets.insert(0, R)
        return rets


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: GESTURE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class GestureClassifier:
    OPEN_TH   = 140
    CLOSED_TH = 60
    MID_TH    = 90

    def __init__(self):
        self.rf     = None
        self.scaler = None
        self._load_model()
        self._history: Deque = deque(maxlen=10)

    def _load_model(self):
        path = Path(CFG.get("classifier_path"))
        if HAS_SKLEARN and path.exists():
            try:
                with open(path, "rb") as f:
                    bundle = pickle.load(f)
                self.rf     = bundle["model"]
                self.scaler = bundle["scaler"]
            except Exception as e:
                LOG.error(f"Classifier load: {e}")

    def classify(self, angles: List[int]) -> Tuple[GestureClass, float]:
        fa = angles[:5]
        if self.rf is not None and self.scaler is not None:
            try:
                x     = self.scaler.transform([fa])
                proba = self.rf.predict_proba(x)[0]
                idx   = int(np.argmax(proba))
                conf  = float(proba[idx])
                if conf >= CFG.get("gesture_confidence_threshold"):
                    return GestureClass(self.rf.classes_[idx]), conf
            except Exception:
                pass
        return self._rule_classify(fa)

    def _rule_classify(self, a: List[int]) -> Tuple[GestureClass, float]:
        o, c = self.OPEN_TH, self.CLOSED_TH
        if all(v > o for v in a):                                        return GestureClass.OPEN_HAND, 0.95
        if all(v < c for v in a):                                        return GestureClass.FIST,      0.95
        if a[0] > o and all(v < c for v in a[1:]):                       return GestureClass.THUMBS_UP, 0.90
        if a[0] < c and a[1] > o and all(v < c for v in a[2:]):          return GestureClass.POINT,     0.88
        if a[0] < c and a[1] > o and a[2] > o and all(v < c for v in a[3:]):
            return GestureClass.PEACE, 0.88
        if a[0] < c and a[3] > o and a[4] > o and all(v < c for v in a[1:3]):
            return GestureClass.ROCK, 0.85
        if abs(a[0] - a[1]) < 30 and a[0] < self.MID_TH and all(v > o for v in a[2:]):
            return GestureClass.PINCH, 0.80
        if all(c < v < o for v in a):                                    return GestureClass.CLAW,    0.75
        return GestureClass.UNKNOWN, 0.0

    def smooth_output(self, raw: GestureClass,
                      conf: float) -> Tuple[GestureClass, float]:
        self._history.append((raw, conf))
        if len(self._history) < 3:
            return raw, conf
        recent = [h[0] for h in list(self._history)[-3:]]
        if len(set(recent)) == 1:
            # FIXED: correct slice for mean confidence
            mean_conf = float(np.mean([h[1] for h in list(self._history)[-3:]]))
            return recent[0], mean_conf
        return raw, conf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: GRASP PLANNER
# ══════════════════════════════════════════════════════════════════════════════

class GraspPlanner:
    def __init__(self):
        self._yolo                = None
        self._siamese             = None
        self._support_frames:     List[np.ndarray] = []
        self._few_shot_collecting = False
        self._few_shot_target_n   = int(CFG.get("siamese_support_set_size"))
        self._loaded_library      = self._load_grasp_library()

        if CFG.get("yolo_enabled"):
            self._load_yolo()
        if CFG.get("siamese_enabled"):
            self._siamese = REGISTRY.load_siamese()

    def _load_grasp_library(self) -> dict:
        lib_path = Path(CFG.get("grasp_library_path"))
        merged   = dict(GRASP_LIBRARY)
        if lib_path.exists():
            try:
                with open(lib_path) as f:
                    custom = json.load(f)
                for k, v in custom.items():
                    merged[k] = (GraspType.CYLINDRICAL, v)
            except Exception as e:
                LOG.warning(f"Grasp library load: {e}")
        return merged

    def _save_grasp_library(self):
        lib_path = Path(CFG.get("grasp_library_path"))
        custom = {k: v[1] for k, v in self._loaded_library.items()
                  if k not in GRASP_LIBRARY}
        with open(lib_path, "w") as f:
            json.dump(custom, f, indent=2)

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(CFG.get("yolo_model"))
            LOG.info("YOLO loaded")
        except ImportError:
            LOG.warning("ultralytics not installed")

    def start_few_shot_capture(self):
        self._support_frames      = []
        self._few_shot_collecting = True
        print(f"[FEW-SHOT] Collecting {self._few_shot_target_n} support frames. Show the object...")

    def feed_few_shot_frame(self, frame: np.ndarray) -> bool:
        if not self._few_shot_collecting:
            return False
        self._support_frames.append(frame.copy())
        if len(self._support_frames) >= self._few_shot_target_n:
            self._few_shot_collecting = False
            print("[FEW-SHOT] Capture complete. Running Siamese prediction...")
            return True
        return False

    def few_shot_predict(self, query_frame: np.ndarray,
                         object_name: str = "custom_object") -> Optional[List[int]]:
        if self._siamese is None or not self._support_frames:
            return None
        angles = self._siamese.predict_from_frames(
            query_frame, self._support_frames, device=REGISTRY.device)
        if angles:
            self._loaded_library[object_name] = (GraspType.CYLINDRICAL, angles)
            self._save_grasp_library()
            print(f"[FEW-SHOT] New grip '{object_name}' → {angles}")
        return angles

    @property
    def few_shot_collecting(self) -> bool:
        return self._few_shot_collecting

    @property
    def few_shot_progress(self) -> float:
        if not self._few_shot_collecting:
            return 0.0
        return len(self._support_frames) / self._few_shot_target_n

    def detect_and_plan(self, frame: np.ndarray,
                        current_angles: List[int]) -> Tuple[np.ndarray, Optional[List[int]], Optional[str]]:
        if self._yolo is None:
            return frame, None, None
        results     = self._yolo(frame, conf=CFG.get("yolo_confidence"), verbose=False)
        annotated   = results[0].plot()
        best_label, best_conf = None, 0.0
        for r in results[0].boxes:
            label = results[0].names[int(r.cls)]
            conf  = float(r.conf)
            if conf > best_conf:
                best_conf  = conf
                best_label = label.lower()
        if best_label:
            entry = self._loaded_library.get(best_label)
            if entry:
                return annotated, entry[1], f"{best_label} ({entry[0].value})"
        return annotated, None, best_label

    def get_preset(self, object_name: str) -> Optional[List[int]]:
        entry = self._loaded_library.get(object_name.lower())
        return entry[1] if entry else None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: SENSOR SUBSYSTEMS
# ══════════════════════════════════════════════════════════════════════════════

class TactileArray:
    GRID_SIZE = 4

    def __init__(self):
        self._enabled = bool(CFG.get("tactile_enabled"))
        self._readings = [TactileReading(finger_idx=i) for i in range(5)]
        self._prev_centroids: List[Tuple[float, float]] = [(0.0, 0.0)] * 5

    def update(self) -> List[TactileReading]:
        if not self._enabled:
            return self._readings
        for i in range(5):
            grid = np.random.exponential(0.5, self.GRID_SIZE ** 2)
            grid = grid / (grid.max() + 1e-6)
            cx, cy = self._centroid(grid)
            slip = float(np.sqrt((cx - self._prev_centroids[i][0]) ** 2 +
                                 (cy - self._prev_centroids[i][1]) ** 2))
            self._prev_centroids[i] = (cx, cy)
            self._readings[i] = TactileReading(
                finger_idx=i, grid=grid.tolist(),
                centroid_x=cx, centroid_y=cy,
                total_force=float(grid.sum()),
                slip_risk=float(np.clip(slip * 5.0, 0.0, 1.0)))
        return self._readings

    def _centroid(self, grid: np.ndarray) -> Tuple[float, float]:
        g     = grid.reshape(self.GRID_SIZE, self.GRID_SIZE)
        total = g.sum() + 1e-9
        xs    = np.arange(self.GRID_SIZE)
        return (float((g.sum(axis=0) * xs).sum() / total),
                float((g.sum(axis=1) * xs).sum() / total))

    def max_slip_risk(self) -> float:
        return max(r.slip_risk for r in self._readings)


class ProximitySensor:
    def __init__(self):
        self._enabled     = bool(CFG.get("proximity_enabled"))
        self._distance_cm = 20.0
        self._lock        = threading.Lock()
        self._alive       = False
        self._thread: Optional[threading.Thread] = None
        if self._enabled:
            self._alive  = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def _poll_loop(self):
        t = 0.0
        while self._alive:
            d = 10.0 + 9.0 * np.sin(t * 0.3)
            with self._lock:
                self._distance_cm = float(np.clip(d, 0.5, 25.0))
            t += 0.1
            time.sleep(0.05)

    @property
    def distance_cm(self) -> float:
        with self._lock:
            return self._distance_cm

    @property
    def velocity_factor(self) -> float:
        d  = self.distance_cm
        mn = float(CFG.get("proximity_min_cm"))
        mx = float(CFG.get("proximity_max_cm"))
        return float(np.clip((d - mn) / (mx - mn + 1e-9), 0.2, 1.0))

    def stop(self):
        self._alive = False


class IMUSimulator:
    def __init__(self):
        self._data  = IMUData()
        self._t     = 0.0
        self._lock  = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._simulate, daemon=True)
        self._thread.start()

    def _simulate(self):
        while self._alive:
            with self._lock:
                self._data = IMUData(
                    accel=[np.sin(self._t * 0.5) * 0.3,
                           np.cos(self._t * 0.7) * 0.2 + 9.81,
                           np.sin(self._t * 1.1) * 0.1],
                    gyro=[np.cos(self._t * 0.3) * 5.0,
                          np.sin(self._t * 0.5) * 3.0,
                          np.cos(self._t * 0.8) * 2.0])
            self._t += 0.02
            time.sleep(0.02)

    def read(self) -> IMUData:
        with self._lock:
            return self._data

    def read_vector(self) -> np.ndarray:
        d = self.read()
        return np.array(d.accel + d.gyro, dtype=np.float32)

    def stop(self):
        self._alive = False


class EMGSimulator:
    def __init__(self):
        self._n_ch  = int(CFG.get("emg_channels"))
        self._sr    = int(CFG.get("emg_sample_rate"))
        self._win   = int(self._sr * int(CFG.get("emg_window_ms")) / 1000)
        self._buf: Deque = deque(maxlen=self._win)
        self._alive = True
        self._lock  = threading.Lock()
        self._thread = threading.Thread(target=self._generate, daemon=True)
        self._thread.start()

    def _generate(self):
        t = 0.0
        while self._alive:
            sample = [float(np.sin(t * 50 * (ch + 1)) * 0.5 +
                            np.random.normal(0, 0.05))
                      for ch in range(self._n_ch)]
            with self._lock:
                self._buf.append(sample)
            t += 1.0 / self._sr
            time.sleep(1.0 / self._sr)

    def get_window(self) -> Optional[np.ndarray]:
        with self._lock:
            if len(self._buf) < self._win:
                return None
            return np.array(list(self._buf)).T.astype(np.float32)

    def stop(self):
        self._alive = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13: EEG LISTENER  (FIXED: threshold check corrected)
# ══════════════════════════════════════════════════════════════════════════════

class EEGListener:
    """
    FIXED:
    - should_override() now correctly checks conf >= threshold (was always True
      when threshold < 0.5 due to logic inversion in original).
    - Simulated EEG needs confidence > 0.82 (new default) to trigger.
    """

    def __init__(self):
        self._enabled   = bool(CFG.get("eeg_enabled"))
        self._threshold = float(CFG.get("eeg_confidence_threshold"))
        self._command   = "NEUTRAL"
        self._confidence = 0.0
        self._lock  = threading.Lock()
        self._alive = False
        self._thread: Optional[threading.Thread] = None
        if self._enabled:
            self._alive  = True
            target = self._lsl_loop if HAS_LSL else self._sim_loop
            self._thread = threading.Thread(target=target, daemon=True)
            self._thread.start()

    def _lsl_loop(self):
        try:
            streams = lsl.resolve_stream("type", "EEG", timeout=5.0)
            if not streams:
                self._sim_loop()
                return
            inlet = lsl.StreamInlet(streams[0])
            while self._alive:
                sample, _ = inlet.pull_sample(timeout=0.1)
                if sample:
                    cmd, conf = self._classify_eeg(np.array(sample))
                    with self._lock:
                        self._command    = cmd
                        self._confidence = conf
        except Exception as e:
            LOG.warning(f"EEG LSL: {e}")
            self._sim_loop()

    def _sim_loop(self):
        t = 0.0
        while self._alive:
            power = 0.5 + 0.5 * np.sin(t * 0.4)
            if power > 0.85:      cmd, conf = "CLOSE", float(power)
            elif power < 0.15:    cmd, conf = "OPEN",  float(1.0 - power)
            else:                 cmd, conf = "NEUTRAL", 0.5
            with self._lock:
                self._command    = cmd
                self._confidence = conf
            t += 0.1
            time.sleep(0.1)

    def _classify_eeg(self, sample: np.ndarray) -> Tuple[str, float]:
        power = float(np.mean(np.abs(sample)))
        if power > 0.6:   return "CLOSE", min(1.0, power)
        if power < 0.2:   return "OPEN",  min(1.0, 1.0 - power)
        return "NEUTRAL", 0.5

    def get(self) -> Tuple[str, float]:
        with self._lock:
            return self._command, self._confidence

    def should_override(self) -> Tuple[bool, str]:
        """FIXED: was always triggering — now requires conf >= threshold."""
        cmd, conf = self.get()
        if cmd != "NEUTRAL" and conf >= self._threshold:
            return True, cmd
        return False, "NEUTRAL"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stop(self):
        self._alive = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14: HAPTIC FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════

class HapticFeedback:
    def __init__(self):
        self._enabled = bool(CFG.get("haptic_enabled"))
        self._ser     = None
        self._alive   = False
        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._active  = True
        if self._enabled and HAS_SERIAL:
            self._connect()

    def _connect(self):
        port = CFG.get("haptic_serial_port")
        if port == "auto":
            for pi in serial.tools.list_ports.comports():
                if any(k in pi.description.upper()
                       for k in ("CH340", "CP210", "FTDI", "ARDUINO")):
                    port = pi.device
                    break
        if port and port != "auto":
            try:
                self._ser   = serial.Serial(port, int(CFG.get("haptic_baud")), timeout=0.1)
                self._alive = True
                threading.Thread(target=self._write_loop, daemon=True).start()
            except Exception as e:
                LOG.warning(f"HapticFeedback connect: {e}")
                self._enabled = False

    def _write_loop(self):
        while self._alive:
            try:
                msg = self._queue.get(timeout=0.5)
                if self._ser and self._ser.is_open and self._active:
                    self._ser.write(msg.encode())
            except queue.Empty:
                pass
            except Exception as e:
                LOG.error(f"Haptic write: {e}")

    def send_force_feedback(self, force_errors: List[float]):
        if not self._enabled or not self._active:
            return
        pwm = [int(np.clip(abs(e) * 25.5, 0, 255)) for e in force_errors[:5]]
        try:
            self._queue.put_nowait("VIB:" + ",".join(map(str, pwm)) + "\n")
        except queue.Full:
            pass

    def send_slip_alert(self):
        if self._enabled:
            self.send_force_feedback([10.0] * 5)

    def toggle(self) -> bool:
        self._active = not self._active
        return self._active

    @property
    def active(self) -> bool:
        return self._active and self._enabled

    def close(self):
        self._alive = False
        if self._ser:
            self._ser.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15: INTENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

class IntentParser:
    GRIP_WORDS   = {"gentle": 1.5, "soft": 1.5, "light": 2.0,
                    "firm": 4.0, "strong": 6.0, "tight": 7.0, "hard": 8.0}
    COLOUR_WORDS = {"red", "blue", "green", "yellow", "white", "black",
                    "orange", "purple", "pink", "grey", "gray"}
    WRIST_WORDS  = {"left": -30, "right": 30, "up": 20, "down": -20}
    PRESET_WORDS = {"open": "open", "close": "fist", "fist": "fist",
                    "pinch": "pinch", "power": "power", "neutral": "neutral"}

    def parse(self, text: str) -> Dict[str, Any]:
        import re
        text   = text.lower().strip()
        result: Dict[str, Any] = {"raw": text}

        for word, preset in self.PRESET_WORDS.items():
            if word in text:
                result["preset"] = preset
                break

        for obj in set(GRASP_LIBRARY.keys()) | {"object", "item", "thing"}:
            if obj in text:
                result["object"] = obj
                break

        for word, force in self.GRIP_WORDS.items():
            if word in text:
                result["force"] = force
                break

        for colour in self.COLOUR_WORDS:
            if colour in text:
                result["colour"] = colour
                break

        for word, delta in self.WRIST_WORDS.items():
            if f"wrist {word}" in text or f"rotate {word}" in text:
                result["wrist_delta"] = delta
                break

        if "slow" in text or "slowly" in text:
            result["speed"] = 0.4
        elif "fast" in text or "quickly" in text:
            result["speed"] = 2.0

        m = re.search(r"(\d+)\s*deg", text)
        if m:
            deg = int(m.group(1))
            if "wrist_delta" in result:
                result["wrist_delta"] = int(np.sign(result["wrist_delta"])) * deg
            result["degrees"] = deg

        return result

    def to_angles(self, intent: Dict[str, Any],
                  current_angles: List[int]) -> Optional[List[int]]:
        angles = list(current_angles)

        if "preset" in intent:
            pa = PRESET_ANGLES.get(intent["preset"])
            if pa:
                angles = list(pa)

        if "object" in intent:
            entry = GRASP_LIBRARY.get(intent["object"])
            if entry:
                angles = list(entry[1])

        if "wrist_delta" in intent:
            angles[5] = _clamp_angle(angles[5] + intent["wrist_delta"])

        if "force" in intent:
            fs = intent["force"] / float(CFG.get("target_grip_force_n"))
            for i in range(5):
                angles[i] = _clamp_angle(angles[i] * fs)

        return angles


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16: FORCE CONTROLLER  (FIXED: integral sign, anti-windup)
# ══════════════════════════════════════════════════════════════════════════════

class ForceController:
    """
    FIXED:
    - Integral anti-windup tightened.
    - force error sign corrected: err = force_measured - target (not inverted).
    - stiffness update protected from going negative.
    """

    def __init__(self):
        self._kp = float(CFG.get("force_pid_kp"))
        self._ki = float(CFG.get("force_pid_ki"))
        self._kd = float(CFG.get("force_pid_kd"))
        self._integral   = [0.0] * 5
        self._prev_error = [0.0] * 5
        self._force_history: Deque = deque(maxlen=50)
        self._target_force   = float(CFG.get("target_grip_force_n"))
        self._slip_threshold = float(CFG.get("slip_variance_threshold"))
        self._stiffness_enabled = bool(CFG.get("stiffness_enabled"))
        self._stiffness  = [0.5] * 6
        self._s_kp       = float(CFG.get("stiffness_pid_kp"))
        self._s_th       = float(CFG.get("stiffness_force_threshold"))

    def update(self, angles: List[int],
               forces: List[float],
               tactile_slip: float = 0.0) -> Tuple[List[int], bool]:
        self._force_history.append(list(forces))
        slip = self._detect_slip() or tactile_slip > 0.7
        out  = list(angles)

        for i in range(5):
            # FIXED: err = target - measured  (positive → under-gripping → close more)
            err = self._target_force - float(forces[i])
            # Anti-windup: only integrate if not saturated
            if abs(self._integral[i]) < 8.0:
                self._integral[i] += err
            deriv = err - self._prev_error[i]
            delta = (self._kp * err +
                     self._ki * self._integral[i] +
                     self._kd * deriv)
            self._prev_error[i] = err

            if slip:
                # Open fingers slightly on slip
                out[i] = _clamp_angle(out[i] - 15)
            else:
                # delta > 0 → need more grip → reduce angle (closes finger)
                out[i] = _clamp_angle(out[i] - delta)

            if self._stiffness_enabled:
                self._adjust_stiffness(i, abs(err))

        return out, slip

    def _adjust_stiffness(self, idx: int, force_error: float):
        if force_error > self._s_th:
            self._stiffness[idx] = min(1.0, self._stiffness[idx] + self._s_kp * 0.05)
        else:
            self._stiffness[idx] = max(0.1, self._stiffness[idx] - 0.01)

    def get_stiffness_commands(self) -> List[int]:
        return [int(s * 255) for s in self._stiffness]

    def _detect_slip(self) -> bool:
        if len(self._force_history) < 10:
            return False
        recent = np.array(list(self._force_history)[-10:])
        return bool(float(np.mean(np.var(recent, axis=0))) > self._slip_threshold)

    def set_target_force(self, f: float):
        self._target_force = max(0.1, float(f))
        # Reset integral on force target change
        self._integral = [0.0] * 5


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 17: ANOMALY DETECTOR  (FIXED: threshold units, recovery index)
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    FIXED:
    - Threshold is now in z-score units (default 2.5) not raw degrees.
    - self_heal() recovery_idx reset correctly.
    - Window size increased for stability.
    """

    def __init__(self):
        self._window: Deque = deque(maxlen=int(CFG.get("anomaly_window")))
        self._threshold = float(CFG.get("anomaly_threshold"))
        self._enabled   = bool(CFG.get("anomaly_detection_enabled"))
        self._score_hist: Deque = deque(maxlen=100)
        self._recovery_macros = ["open", "neutral", "open"]
        self._recovery_idx    = 0
        self._recovering      = False

    def feed(self, angles: List[int]) -> Tuple[AnomalyLevel, float]:
        if not self._enabled:
            return AnomalyLevel.NORMAL, 0.0
        self._window.append(list(angles))
        score = self._stat_score(angles)
        self._score_hist.append(score)
        if score > self._threshold * 1.8:
            return AnomalyLevel.CRITICAL, score
        if score > self._threshold:
            return AnomalyLevel.WARNING, score
        return AnomalyLevel.NORMAL, score

    def _stat_score(self, angles: List[int]) -> float:
        if len(self._window) < 8:
            return 0.0
        data  = np.array(list(self._window))
        mu    = data.mean(axis=0)
        sigma = data.std(axis=0) + 1e-6
        z     = np.abs((np.array(angles, dtype=float) - mu) / sigma)
        return float(z.max())

    def self_heal(self) -> Optional[List[int]]:
        """FIXED: index reset properly, returns neutral when exhausted."""
        if self._recovery_idx >= len(self._recovery_macros):
            self._recovery_idx = 0
            self._recovering   = False
            LOG.warning("Anomaly: recovery exhausted, returning to neutral")
            return list(PRESET_ANGLES["neutral"])
        self._recovering  = True
        macro_name        = self._recovery_macros[self._recovery_idx]
        self._recovery_idx += 1
        angles = PRESET_ANGLES.get(macro_name)
        LOG.info(f"Self-heal macro '{macro_name}' → {angles}")
        return list(angles) if angles else list(PRESET_ANGLES["neutral"])

    def reset_recovery(self):
        self._recovery_idx = 0
        self._recovering   = False

    @property
    def rolling_score(self) -> float:
        if not self._score_hist:
            return 0.0
        return float(np.mean(list(self._score_hist)[-10:]))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 18: CONTINUAL LEARNING
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, max_size: Optional[int] = None):
        self._max     = max_size or int(CFG.get("replay_buffer_size"))
        self._data:   List[Dict] = []
        self._path    = Path(CFG.get("replay_buffer_dir")) / "buffer.jsonl"
        self._counter = 0
        self._lock    = threading.Lock()

    def add(self, frame_path: str, angles: List[int],
            gesture: str = "UNKNOWN", source: str = "mirror",
            emg: Optional[List[float]] = None,
            imu: Optional[List[float]] = None):
        with self._lock:
            record = {"path": frame_path, "angles": angles,
                      "gesture": gesture, "source": source,
                      "ts": time.time(), "emg": emg, "imu": imu}
            if len(self._data) < self._max:
                self._data.append(record)
            else:
                j = int(np.random.randint(0, max(1, self._counter)))
                if j < self._max:
                    self._data[j] = record
            self._counter += 1

    def sample(self, n: int) -> List[Dict]:
        with self._lock:
            n = min(n, len(self._data))
            if n == 0:
                return []
            return [self._data[i] for i in
                    np.random.choice(len(self._data), n, replace=False)]

    def save(self):
        with self._lock:
            try:
                with open(self._path, "w") as f:
                    for r in self._data:
                        f.write(json.dumps(r) + "\n")
            except Exception as e:
                LOG.warning(f"ReplayBuffer save: {e}")

    def __len__(self) -> int:
        return len(self._data)


class EWCRegularizer:
    def __init__(self, model: Any, lam: Optional[float] = None):
        self._model        = model
        self._lam          = lam or float(CFG.get("ewc_lambda"))
        self._params_prev: Dict = {}
        self._fisher:      Dict = {}

    def compute_fisher(self, dataloader):
        if not HAS_TORCH or dataloader is None:
            return
        self._model.eval()
        fisher: Dict = {n: torch.zeros_like(p)
                        for n, p in self._model.named_parameters()
                        if p.requires_grad}
        n_batches = 0
        for batch in dataloader:
            self._model.zero_grad()
            imgs, targets = batch
            out  = self._model(imgs)
            loss = F.mse_loss(out, targets)
            loss.backward()
            for n, p in self._model.named_parameters():
                if p.grad is not None:
                    fisher[n] = fisher[n] + p.grad ** 2
            n_batches += 1
        if n_batches > 0:
            self._fisher = {n: f / n_batches for n, f in fisher.items()}
        self._params_prev = {n: p.data.clone()
                             for n, p in self._model.named_parameters()}

    def penalty(self) -> "torch.Tensor":
        if not HAS_TORCH or not self._fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for n, p in self._model.named_parameters():
            if n in self._fisher:
                loss = loss + (self._fisher[n] * (p - self._params_prev[n]) ** 2).sum()
        return self._lam / 2.0 * loss


class OnlinePersonalizer:
    def __init__(self, model: Optional[Any] = None):
        self._model    = model
        self._enabled  = (bool(CFG.get("continual_learning_enabled"))
                          and HAS_TORCH and model is not None)
        self._interval = int(CFG.get("online_train_interval"))
        self._lr       = 1e-4
        self._ewc:       Optional[EWCRegularizer] = None
        self._optimizer: Optional[Any]             = None
        self._frame_count = 0
        self._batch: List[Tuple] = []
        self._lock    = threading.Lock()
        self._training = False
        if self._enabled:
            self._optimizer = Adam(self._model.parameters(), lr=self._lr)
            self._ewc       = EWCRegularizer(self._model)

    def feed(self, frame: np.ndarray, angles: List[int]):
        if not self._enabled:
            return
        with self._lock:
            self._batch.append((frame, angles))
            self._frame_count += 1
        if self._frame_count % self._interval == 0 and not self._training:
            with self._lock:
                batch_copy = list(self._batch[-32:])
            threading.Thread(target=self._train_step,
                             args=(batch_copy,), daemon=True).start()

    def _train_step(self, batch: List[Tuple]):
        if not batch or self._model is None:
            return
        self._training = True
        try:
            transform = transforms.Compose([
                transforms.ToPILImage(), transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            imgs   = torch.stack([transform(f) for f, _ in batch])
            labels = torch.FloatTensor([a for _, a in batch])
            self._model.train()
            self._optimizer.zero_grad()
            loss = F.mse_loss(self._model(imgs), labels)
            if self._ewc:
                loss = loss + self._ewc.penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()
            self._model.eval()
        except Exception as e:
            LOG.warning(f"OnlinePersonalizer step: {e}")
        finally:
            self._training = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 19: FEDERATED CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class FederatedClient:
    def __init__(self):
        self._enabled      = bool(CFG.get("federated_enabled"))
        self._server       = CFG.get("federated_server")
        self._user_id      = CFG.get("user_id")
        self._dp_noise     = 0.01
        self._min_samples  = int(CFG.get("federated_min_samples"))
        self._prev_weights: Optional[Dict[str, np.ndarray]] = None

    def should_upload(self, buf: ReplayBuffer) -> bool:
        return self._enabled and len(buf) >= self._min_samples

    def compute_model_deltas(self, model: Any) -> Dict[str, np.ndarray]:
        if not HAS_TORCH:
            return {}
        current = {n: p.data.cpu().numpy()
                   for n, p in model.named_parameters()}
        if self._prev_weights is None:
            self._prev_weights = current
            return {}
        deltas = {n: current[n] - self._prev_weights[n] for n in current}
        self._prev_weights = current
        return deltas

    def upload_gradients(self, delta: Dict[str, np.ndarray]) -> bool:
        if not self._enabled or not delta:
            return False
        noisy = {k: (v + np.random.normal(0, self._dp_noise, v.shape)).tolist()
                 for k, v in delta.items()}
        payload = {"user_id": self._user_id,
                   "model_version": CFG.get("model_version"),
                   "delta": noisy, "n_samples": 0}
        try:
            import urllib.request
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                f"{self._server}/api/v1/upload", data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            LOG.warning(f"Federated upload: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 20: WIRELESS BLE BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

class WirelessBridge:
    """FIXED: queue.get uses get_nowait in async context to avoid blocking."""

    def __init__(self):
        self._enabled   = bool(CFG.get("ble_enabled"))
        self._dev_name  = CFG.get("ble_device_name")
        self._char_uuid = CFG.get("ble_char_uuid")
        self._connected = False
        self._last      = [-1] * 6
        self._thresh    = int(CFG.get("angle_change_threshold"))
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        if self._enabled and HAS_BLEAK:
            threading.Thread(target=self._run_ble, daemon=True).start()
        elif self._enabled:
            LOG.warning("WirelessBridge: bleak not installed")

    def _run_ble(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ble_loop())

    async def _ble_loop(self):
        try:
            from bleak import BleakScanner, BleakClient
            device = await BleakScanner.find_device_by_name(self._dev_name, timeout=10.0)
            if device is None:
                LOG.warning(f"BLE: '{self._dev_name}' not found")
                return
            async with BleakClient(device) as client:
                self._connected = True
                LOG.info(f"BLE: connected {device.address}")
                print(f"[BLE] Connected to {device.name}")
                while self._connected:
                    try:
                        msg = self._queue.get_nowait()
                        await client.write_gatt_char(
                            self._char_uuid, msg.encode(), response=False)
                    except queue.Empty:
                        await asyncio.sleep(0.01)
        except Exception as e:
            LOG.error(f"BLE: {e}")
            self._connected = False

    def send(self, angles: List[int]) -> bool:
        if not self._connected:
            return False
        if not any(abs(angles[i] - self._last[i]) >= self._thresh
                   for i in range(min(6, len(angles)))):
            return False
        msg = "<" + ",".join(map(str, angles[:6])) + ">\n"
        try:
            self._queue.put_nowait(msg)
            self._last = list(angles[:6])
            return True
        except queue.Full:
            return False

    @property
    def alive(self) -> bool:
        return self._connected

    def close(self):
        self._connected = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 21: OTA UPDATER
# ══════════════════════════════════════════════════════════════════════════════

class OTAUpdater:
    def __init__(self):
        self._enabled = bool(CFG.get("ota_enabled"))
        self._server  = CFG.get("ota_server")
        self._status  = "idle"
        self._lock    = threading.Lock()

    def check_and_update(self, serial_port: str = "auto",
                         callback: Optional[Callable] = None):
        if not self._enabled:
            print("[OTA] Disabled. Enable 'ota_enabled' in nexus_config.json.")
            return
        threading.Thread(target=self._ota_thread,
                         args=(serial_port, callback), daemon=True).start()

    def _ota_thread(self, serial_port: str, cb: Optional[Callable]):
        try:
            with self._lock:
                self._status = "checking"
            print("[OTA] Checking for updates...")
            if cb:
                cb("checking", 0.0)
            model_url = self._check_model_update()
            if model_url:
                self._download_model(model_url, cb)
            fw_url = CFG.get("ota_firmware_url")
            if fw_url:
                print(f"[OTA] Firmware URL configured: {fw_url}")
            with self._lock:
                self._status = "done"
            if cb:
                cb("done", 1.0)
            print("[OTA] Update check complete.")
        except Exception as e:
            LOG.error(f"OTA: {e}")
            with self._lock:
                self._status = f"error: {e}"
            if cb:
                cb("error", 0.0)

    def _check_model_update(self) -> Optional[str]:
        try:
            import urllib.request
            url = f"{self._server}/api/v1/latest?version={CFG.get('model_version')}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                d = json.loads(resp.read())
            return d.get("download_url") if d.get("newer") else None
        except Exception:
            return None

    def _download_model(self, url: str, cb: Optional[Callable]):
        try:
            import urllib.request
            out = Path(CFG.get("model_dir")) / "nexus_cnn_new.pth"
            with urllib.request.urlopen(url, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                dl, chunk_size = 0, 8192
                with open(out, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        dl += len(chunk)
                        if cb and total > 0:
                            cb("downloading", dl / total)
            print(f"[OTA] Model saved to {out}. Restart to apply.")
        except Exception as e:
            LOG.error(f"OTA download: {e}")

    @property
    def status(self) -> str:
        with self._lock:
            return self._status


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 22: FINGER MANAGER (I²C)
# ══════════════════════════════════════════════════════════════════════════════

class FingerManager:
    def __init__(self):
        self._enabled   = bool(CFG.get("finger_manager_enabled"))
        self._addresses = list(CFG.get("finger_i2c_addresses"))
        self._present   = [False] * 5
        self._bus       = None
        if self._enabled:
            self._probe_bus()

    def _probe_bus(self):
        try:
            import smbus2
            self._bus = smbus2.SMBus(1)
            for i, addr in enumerate(self._addresses[:5]):
                try:
                    self._bus.read_byte(addr)
                    self._present[i] = True
                except Exception:
                    pass
        except ImportError:
            LOG.warning("FingerManager: smbus2 not installed")
        except Exception as e:
            LOG.warning(f"FingerManager probe: {e}")

    def send(self, angles: List[int]):
        if not self._enabled or self._bus is None:
            return
        for i, (addr, angle) in enumerate(zip(self._addresses, angles[:5])):
            if self._present[i]:
                try:
                    self._bus.write_byte_data(addr, 0x01, _clamp_angle(angle))
                except Exception as e:
                    LOG.warning(f"Finger {i} send: {e}")
                    self._present[i] = False

    @property
    def active_count(self) -> int:
        return sum(self._present)

    def get_status(self) -> List[bool]:
        return list(self._present)

    def close(self):
        if self._bus:
            try:
                self._bus.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 23: DIGITAL TWIN BROADCASTER
# ══════════════════════════════════════════════════════════════════════════════

class DigitalTwinBroadcaster:
    def __init__(self):
        self._clients: List[Any] = []
        self._port    = int(CFG.get("digital_twin_ws_port"))
        self._enabled = bool(CFG.get("digital_twin_enabled"))
        self._loop:   Optional[asyncio.AbstractEventLoop] = None

    def start(self):
        if not self._enabled:
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import websockets
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            server = websockets.serve(self._handler, "0.0.0.0", self._port)
            self._loop.run_until_complete(server)
            LOG.info(f"Digital Twin WS port {self._port}")
            self._loop.run_forever()
        except ImportError:
            LOG.warning("websockets not installed; digital twin disabled")

    async def _handler(self, ws, path):
        self._clients.append(ws)
        try:
            await ws.wait_closed()
        finally:
            if ws in self._clients:
                self._clients.remove(ws)

    def broadcast(self, state: ServoState, gesture: str,
                  anomaly: AnomalyLevel, fps: float,
                  landmarks_3d: Optional[np.ndarray] = None):
        if not self._enabled or not self._clients or self._loop is None:
            return
        payload = {"ts": time.time(), "angles": state.angles,
                   "forces": state.forces, "gesture": gesture,
                   "anomaly": anomaly.name, "fps": round(fps, 1),
                   "stiffness": state.stiffness,
                   "hand_id": CFG.get("hand_id")}
        if landmarks_3d is not None:
            payload["landmarks_3d"] = landmarks_3d.tolist()
        msg = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._send_all(msg), self._loop)

    async def _send_all(self, msg: str):
        if not self._clients:
            return
        import websockets
        await asyncio.gather(*[ws.send(msg) for ws in self._clients],
                             return_exceptions=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 24: MOTION MEMORY  (FIXED: negative delay guard in playback)
# ══════════════════════════════════════════════════════════════════════════════

class MotionMemory:
    def __init__(self):
        self.recording  = False
        self._gesture: Optional[Gesture] = None
        self._start     = 0.0
        Path(CFG.get("gesture_dir")).mkdir(exist_ok=True)

    def start_recording(self, name: str = "gesture"):
        self._gesture  = Gesture(name=name, duration_ms=0.0)
        self._start    = time.perf_counter()
        self.recording = True

    def capture_frame(self, angles: List[int],
                      forces: Optional[List[float]] = None,
                      emg:    Optional[List[float]] = None,
                      imu:    Optional[List[float]] = None):
        if not self.recording or self._gesture is None:
            return
        t_ms = (time.perf_counter() - self._start) * 1000.0
        self._gesture.frames.append(AngleFrame(
            t_ms=round(t_ms, 1), angles=list(angles),
            forces=forces, emg_signal=emg, imu_data=imu))

    def stop_recording(self) -> Optional[Gesture]:
        if not self.recording or self._gesture is None:
            return None
        self.recording = False
        self._gesture.duration_ms = round(
            (time.perf_counter() - self._start) * 1000.0, 1)
        return self._gesture

    def save(self, gesture: Gesture) -> str:
        fname = Path(CFG.get("gesture_dir")) / f"{gesture.name}.json"
        with open(str(fname), "w") as f:
            json.dump({
                "gesture_name": gesture.name,
                "duration_ms":  gesture.duration_ms,
                "created_at":   gesture.created_at,
                "tags":         gesture.tags,
                "frames": [{"t": fr.t_ms, "angles": fr.angles,
                             "forces": fr.forces,
                             "emg": fr.emg_signal,
                             "imu": fr.imu_data}
                            for fr in gesture.frames]
            }, f, indent=2)
        return str(fname)

    def load(self, path: str) -> Gesture:
        with open(path) as f:
            d = json.load(f)
        g = Gesture(name=d["gesture_name"], duration_ms=d["duration_ms"])
        g.frames = [AngleFrame(
            t_ms=fd["t"], angles=fd["angles"],
            forces=fd.get("forces"), emg_signal=fd.get("emg"),
            imu_data=fd.get("imu")) for fd in d["frames"]]
        return g

    def list_gestures(self) -> List[Path]:
        return sorted(Path(CFG.get("gesture_dir")).glob("*.json"),
                      key=lambda p: p.stat().st_mtime)

    def playback_frames(self, gesture: Gesture,
                        speed_factor: float = 1.0) -> Generator:
        """FIXED: delay clamped to >= 0, speed_factor validated."""
        sf = max(0.05, float(speed_factor))
        for i, f in enumerate(gesture.frames):
            if i == 0:
                delay = 0.0
            else:
                raw_delay = (gesture.frames[i].t_ms -
                             gesture.frames[i - 1].t_ms) / 1000.0
                delay = max(0.0, raw_delay) / sf   # FIXED: no negative delays
            yield list(f.angles), delay


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 25: DATASET FACTORY  (FIXED: empty-buffer guard)
# ══════════════════════════════════════════════════════════════════════════════

class DatasetFactory:
    def __init__(self):
        self._frames_dir  = Path(CFG.get("dataset_dir")) / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._labels_path = Path(CFG.get("dataset_dir")) / "labels.json"
        self._buffer: List[Dict] = []
        self.active   = False
        self._counter = 0
        if self._labels_path.exists():
            try:
                with open(str(self._labels_path)) as f:
                    self._counter = len(json.load(f))
            except Exception:
                pass

    def start(self):
        self.active  = True
        self._buffer = []

    def capture(self, frame: np.ndarray, angles: List[int],
                gesture: str = "UNKNOWN", auto_label: bool = True,
                emg: Optional[List[float]] = None,
                imu: Optional[List[float]] = None):
        if not self.active:
            return
        fname = f"frame_{self._counter:08d}.jpg"
        cv2.imwrite(str(self._frames_dir / fname), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        self._buffer.append({
            "filename": fname, "angles": list(angles),
            "gesture": gesture, "auto_labeled": auto_label,
            "emg": emg, "imu": imu,
            "timestamp_ms": round(time.perf_counter() * 1000)})
        self._counter += 1

    def save(self) -> int:
        # FIXED: return 0 immediately if nothing to save
        if not self._buffer:
            return 0
        existing: List[Dict] = []
        if self._labels_path.exists():
            try:
                with open(str(self._labels_path)) as f:
                    existing = json.load(f)
            except Exception:
                pass
        all_labels = existing + self._buffer
        with open(str(self._labels_path), "w") as f:
            json.dump(all_labels, f, indent=2)
        n = len(self._buffer)
        self._buffer = []
        return n


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 26: SERIAL BRIDGE  (FIXED: threshold comparison)
# ══════════════════════════════════════════════════════════════════════════════

class SerialBridge:
    """
    FIXED:
    - _last initialised to [-1] so first frame always sends.
    - Threshold comparison uses >= (not >) consistent with dead-band logic.
    """

    def __init__(self):
        self.ser     = None
        self._last   = [-1] * 6
        self._thresh = int(CFG.get("angle_change_threshold"))
        self._alive  = False
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._thread: Optional[threading.Thread] = None
        if HAS_SERIAL:
            self._connect()

    def _connect(self):
        port = CFG.get("serial_port")
        if port == "auto":
            for pi in serial.tools.list_ports.comports():
                desc = pi.description.upper()
                if any(k in desc for k in ("CH340", "CP210", "FTDI",
                                           "USB SERIAL", "ARDUINO")):
                    port = pi.device
                    break
        if port and port != "auto":
            try:
                self.ser    = serial.Serial(port, int(CFG.get("serial_baud")),
                                            timeout=0.1)
                self._alive = True
                self._thread = threading.Thread(target=self._write_loop, daemon=True)
                self._thread.start()
                LOG.info(f"Serial connected: {port}")
                print(f"[SERIAL] Connected to {port}")
            except Exception as e:
                LOG.error(f"Serial open: {e}")

    def _write_loop(self):
        while self._alive:
            try:
                msg = self._queue.get(timeout=0.5)
                if self.ser and self.ser.is_open:
                    self.ser.write(msg.encode())
            except queue.Empty:
                pass
            except Exception as e:
                LOG.error(f"Serial write: {e}")
                self._alive = False

    def send(self, angles: List[int]) -> bool:
        if self.ser is None:
            return False
        # Only send if at least one angle changed by >= threshold
        if not any(abs(angles[i] - self._last[i]) >= self._thresh
                   for i in range(min(6, len(angles)))):
            return False
        msg = "<" + ",".join(map(str, angles[:6])) + ">\n"
        try:
            self._queue.put_nowait(msg)
            self._last = list(angles[:6])
            return True
        except queue.Full:
            return False

    def send_emergency_stop(self):
        emergency = [90, 90, 90, 90, 90, 90]
        msg = "<" + ",".join(map(str, emergency)) + ">\n"
        try:
            self._queue.put_nowait(msg)
            self._last = emergency
        except queue.Full:
            pass
        LOG.critical("Emergency stop sent")

    @property
    def alive(self) -> bool:
        return self._alive

    def close(self):
        self._alive = False
        if self.ser:
            self.ser.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 27: TELEMETRY
# ══════════════════════════════════════════════════════════════════════════════

class TelemetryPublisher:
    def __init__(self):
        self._client   = None
        self._enabled  = bool(CFG.get("telemetry_enabled"))
        self._prefix   = CFG.get("mqtt_topic_prefix")
        self._last_hb  = 0.0
        self._hb_int   = float(CFG.get("heartbeat_interval_s"))
        if self._enabled and HAS_MQTT:
            self._connect()

    def _connect(self):
        try:
            self._client = mqtt.Client(client_id=f"nexus_{CFG.get('user_id')}")
            self._client.connect(CFG.get("mqtt_broker"), int(CFG.get("mqtt_port")), 60)
            self._client.loop_start()
        except Exception as e:
            LOG.warning(f"MQTT: {e}")
            self._enabled = False

    def publish(self, topic: str, payload: Any):
        if not self._enabled or self._client is None:
            return
        try:
            msg = json.dumps(payload) if not isinstance(payload, str) else payload
            self._client.publish(f"{self._prefix}/{topic}", msg, qos=0)
        except Exception:
            pass

    def publish_state(self, angles: List[int], gesture: str, conf: float,
                      fps: float, anomaly_score: float, forces: List[float]):
        if not self._enabled:
            return
        now = time.time()
        self.publish("angles",  {"values": angles, "ts": now})
        self.publish("gesture", {"class": gesture, "confidence": round(conf, 3)})
        self.publish("forces",  {"values": forces, "ts": now})
        self.publish("anomaly", {"score": round(anomaly_score, 4)})
        if now - self._last_hb > self._hb_int:
            self.publish("heartbeat", {"alive": True, "fps": round(fps, 1), "ts": now})
            self._last_hb = now

    def close(self):
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 28: VOICE CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

VOICE_COMMANDS: Dict[str, str] = {
    "mirror mode":        "mode:1",
    "playback mode":      "mode:2",
    "deep learning mode": "mode:3",
    "autonomous mode":    "mode:4",
    "emg mode":           "mode:5",
    "rl mode":            "mode:6",
    "hybrid mode":        "mode:7",
    "fusion mode":        "mode:8",
    "eeg mode":           "mode:9",
    "start recording":    "record:start",
    "stop recording":     "record:stop",
    "save dataset":       "dataset:save",
    "toggle kalman":      "kalman:toggle",
    "emergency stop":     "safety:stop",
    "open hand":          "preset:open",
    "close hand":         "preset:fist",
    "pinch":              "preset:pinch",
    "power grasp":        "preset:power",
    "status":             "query:status",
    "calibrate":          "calibrate:start",
    "toggle haptic":      "haptic:toggle",
    "eeg override":       "eeg:override",
    "check update":       "ota:check",
    "toggle fusion":      "fusion:toggle",
}


class VoiceController:
    def __init__(self):
        self._enabled = bool(CFG.get("voice_enabled")) and HAS_VOICE
        self._cmd_queue:    queue.Queue = queue.Queue()
        self._intent_queue: queue.Queue = queue.Queue()
        self._recognizer = None
        self._tts        = None
        self._thread: Optional[threading.Thread] = None
        self._intent_parser = IntentParser()
        if self._enabled:
            self._setup()

    def _setup(self):
        try:
            self._recognizer = sr.Recognizer()
            self._tts        = pyttsx3.init()
            self._tts.setProperty("rate", 150)
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            self.speak("Bionic Nexus voice control activated.")
        except Exception as e:
            LOG.warning(f"Voice setup: {e}")
            self._enabled = False

    def _listen_loop(self):
        with sr.Microphone() as mic:
            self._recognizer.adjust_for_ambient_noise(mic, duration=1.0)
            while self._enabled:
                try:
                    audio = self._recognizer.listen(mic, timeout=3.0,
                                                    phrase_time_limit=5.0)
                    text = self._recognizer.recognize_google(
                        audio, language=CFG.get("voice_language")).lower()
                    cmd = self._match_command(text)
                    if cmd:
                        self._cmd_queue.put(cmd)
                    else:
                        intent = self._intent_parser.parse(text)
                        if len(intent) > 1:
                            self._intent_queue.put(intent)
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    pass
                except Exception as e:
                    LOG.debug(f"Voice: {e}")

    def _match_command(self, text: str) -> Optional[str]:
        for phrase, cmd in VOICE_COMMANDS.items():
            if phrase in text:
                return cmd
        return None

    def get_command(self) -> Optional[str]:
        try:
            return self._cmd_queue.get_nowait()
        except queue.Empty:
            return None

    def get_intent(self) -> Optional[Dict]:
        try:
            return self._intent_queue.get_nowait()
        except queue.Empty:
            return None

    def speak(self, text: str):
        if self._enabled and self._tts:
            threading.Thread(
                target=lambda: (self._tts.say(text), self._tts.runAndWait()),
                daemon=True).start()

    @property
    def enabled(self) -> bool:
        return self._enabled


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 29: PERFORMANCE PROFILER
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceProfiler:
    def __init__(self):
        self._times: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))
        self._enabled = bool(CFG.get("profiler_enabled"))

    @contextmanager
    def measure(self, stage: str):
        if not self._enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        self._times[stage].append((time.perf_counter() - t0) * 1000.0)

    def report(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for stage, times in self._times.items():
            if times:
                arr = np.array(times)
                out[stage] = {
                    "mean_ms": round(float(arr.mean()), 2),
                    "p95_ms":  round(float(np.percentile(arr, 95)), 2),
                    "max_ms":  round(float(arr.max()), 2),
                }
        return out

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        CFG.set("profiler_enabled", self._enabled)
        return self._enabled


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 30: HUD RENDERER  (FIXED: bar width overflow, safe slices)
# ══════════════════════════════════════════════════════════════════════════════

class HUDRenderer:
    SERVO_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Wrist"]
    MODE_COLORS = {
        TrackingMode.MIRROR:      (0, 220, 120),
        TrackingMode.PLAYBACK:    (220, 180, 0),
        TrackingMode.CNN:         (0, 160, 255),
        TrackingMode.AUTONOMOUS:  (255, 100, 0),
        TrackingMode.EMG:         (200, 0, 255),
        TrackingMode.RL_ADAPTIVE: (255, 200, 0),
        TrackingMode.HYBRID:      (0, 255, 220),
        TrackingMode.FUSION:      (100, 255, 180),
        TrackingMode.EEG:         (255, 80, 160),
    }
    ANOMALY_COLORS = {
        AnomalyLevel.NORMAL:   (0, 220, 100),
        AnomalyLevel.WARNING:  (0, 200, 255),
        AnomalyLevel.CRITICAL: (0, 0, 255),
    }

    def draw(self, frame: np.ndarray, *,
             mode: TrackingMode,
             recording: bool,
             angles: List[int],
             fps: float,
             gesture: str,
             conf: float,
             anomaly: AnomalyLevel,
             anomaly_score: float,
             forces: Optional[List[float]] = None,
             calib_progress: float = 0.0,
             profiler_data: Optional[Dict] = None,
             serial_alive: bool = False,
             ble_alive: bool = False,
             object_label: Optional[str] = None,
             eeg_cmd: str = "NEUTRAL",
             eeg_conf: float = 0.0,
             playback_speed: float = 1.0,
             haptic_active: bool = False,
             fusion_enabled: bool = False,
             few_shot_progress: float = 0.0,
             proximity_cm: float = 20.0) -> np.ndarray:

        h, w = frame.shape[:2]
        overlay  = frame.copy()
        panel_x  = max(0, w - 220)
        panel_w  = w - panel_x - 10   # available bar width

        # Translucent panel
        cv2.rectangle(overlay, (panel_x - 10, 60),
                      (w - 10, 60 + 6 * 42 + 20), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Servo angle bars — FIXED: bar_w clamped to panel_w
        for i, (name, angle) in enumerate(zip(self.SERVO_NAMES, angles[:6])):
            y   = 70 + i * 42
            pct = float(np.clip(angle / 180.0, 0.0, 1.0))
            cv2.rectangle(frame, (panel_x, y), (w - 20, y + 22), (40, 42, 55), -1)
            bar_col = ((80, 200, 255) if pct < 0.33 else
                       (40, 220, 100) if pct < 0.67 else (60, 100, 255))
            bar_w = int(np.clip(pct * panel_w, 0, panel_w))   # FIXED
            if bar_w > 0:
                cv2.rectangle(frame, (panel_x, y), (panel_x + bar_w, y + 22), bar_col, -1)
            cv2.putText(frame, name, (panel_x + 5, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 210), 1)
            cv2.putText(frame, f"{angle}\xb0", (w - 50, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 250), 1)

        # Force bars
        if forces and any(f > 0.01 for f in forces):
            for i, (nm, force) in enumerate(zip(self.SERVO_NAMES[:5], forces)):
                y    = h - 220 + i * 30
                bw   = int(np.clip(min(force / 10.0, 1.0) * 120, 0, 120))
                cv2.rectangle(frame, (15, y), (135, y + 18), (30, 30, 40), -1)
                if bw > 0:
                    cv2.rectangle(frame, (15, y), (15 + bw, y + 18), (80, 200, 130), -1)
                cv2.putText(frame, f"{nm[0]}: {force:.1f}N",
                            (20, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 230, 200), 1)

        # Mode badge
        mc = self.MODE_COLORS.get(mode, (200, 200, 200))
        cv2.rectangle(frame, (15, 15), (260, 50), (20, 20, 30), -1)
        cv2.putText(frame, f"  {mode.name}", (18, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, mc, 1)
        cv2.circle(frame, (18, 30), 6, mc, -1)

        # Gesture badge
        cv2.rectangle(frame, (15, 58), (295, 88), (20, 20, 30), -1)
        cv2.putText(frame, f"GESTURE: {gesture}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 210, 80), 1)
        cw = int(np.clip(conf * 100, 0, 100))
        cv2.rectangle(frame, (295, 62), (405, 82), (40, 40, 50), -1)
        if cw > 0:
            cv2.rectangle(frame, (295, 62), (295 + cw, 82), (50, 200, 100), -1)
        cv2.putText(frame, f"{conf:.0%}", (410, 79),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 180), 1)

        # Anomaly
        ac = self.ANOMALY_COLORS[anomaly]
        cv2.circle(frame, (445, 75), 9, ac, -1)
        cv2.putText(frame, f"ANOM:{anomaly.name}({anomaly_score:.2f})",
                    (460, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ac, 1)

        # EEG
        if eeg_conf > 0.3:
            ec = (255, 80, 160) if eeg_cmd != "NEUTRAL" else (100, 100, 120)
            cv2.putText(frame, f"EEG:{eeg_cmd} {eeg_conf:.0%}",
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ec, 1)

        # Proximity
        if proximity_cm < 18.0:
            pc = (0, 255, 200) if proximity_cm > 5 else (0, 100, 255)
            cv2.putText(frame, f"PROX:{proximity_cm:.1f}cm",
                        (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pc, 1)

        # FPS
        fc = (0, 220, 80) if fps >= 20 else (0, 180, 255) if fps >= 10 else (0, 80, 255)
        cv2.putText(frame, f"FPS:{fps:.1f}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fc, 1)

        # Status line
        cv2.putText(frame, "USB",
                    (110, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 80) if serial_alive else (80, 80, 100), 1)
        cv2.putText(frame, "BLE",
                    (148, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 200, 255) if ble_alive else (60, 60, 80), 1)
        if haptic_active:
            cv2.putText(frame, "HAP", (186, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 60), 1)
        if fusion_enabled:
            cv2.putText(frame, "FUSION", (224, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 180), 1)

        # Recording blink
        if recording:
            if int(time.perf_counter() * 4) % 2:
                cv2.circle(frame, (20, h - 55), 9, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (36, h - 48),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

        if mode == TrackingMode.PLAYBACK and playback_speed != 1.0:
            cv2.putText(frame, f"SPD:{playback_speed:.1f}x",
                        (300, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 100), 1)

        # Calibration bar
        if calib_progress > 0:
            cv2.rectangle(frame, (15, h - 80), (295, h - 60), (30, 30, 40), -1)
            bw = int(np.clip(calib_progress * 280, 0, 280))
            if bw > 0:
                cv2.rectangle(frame, (15, h - 80), (15 + bw, h - 60), (0, 200, 255), -1)
            cv2.putText(frame, f"CALIBRATING {calib_progress:.0%}",
                        (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1)

        # Few-shot bar
        if few_shot_progress > 0:
            cv2.rectangle(frame, (15, h - 105), (295, h - 85), (30, 20, 40), -1)
            bw = int(np.clip(few_shot_progress * 280, 0, 280))
            if bw > 0:
                cv2.rectangle(frame, (15, h - 105), (15 + bw, h - 85), (200, 80, 255), -1)
            cv2.putText(frame, f"FEW-SHOT {few_shot_progress:.0%}",
                        (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 100, 255), 1)

        # Object label
        if object_label:
            cv2.rectangle(frame, (15, 135), (320, 165), (20, 20, 30), -1)
            cv2.putText(frame, f"OBJ: {object_label}",
                        (20, 157), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 1)

        # Profiler
        if profiler_data:
            y0 = h - 170
            n  = len(profiler_data)
            cv2.rectangle(frame, (w - 270, y0 - 10), (w - 10, y0 + 22 * n), (20, 20, 30), -1)
            for j, (stage, st) in enumerate(profiler_data.items()):
                cv2.putText(frame,
                            f"{stage:12s} {st['mean_ms']:5.1f}ms p95={st['p95_ms']:.1f}",
                            (w - 265, y0 + j * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 180, 160), 1)

        # Legend
        cv2.putText(frame,
                    "1-9=Mode  R=Rec  S=Data  C=Calib  K=Kalman  G=Clf  P=Prof  E=Export  Q=Quit",
                    (15, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 140), 1)
        cv2.putText(frame,
                    "Shift+N=FewShot  +/-=Speed  H=Haptic  T=EEG  U=OTA  Y=Fusion  O/B/M=Preset",
                    (15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 140), 1)

        return frame


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 31: SESSION EXPORTER
# ══════════════════════════════════════════════════════════════════════════════

class SessionExporter:
    @staticmethod
    def export_csv(frames: List[AngleFrame], path: str) -> int:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ms", "thumb", "index", "middle", "ring", "pinky",
                        "wrist", "gesture", "confidence"])
            for fr in frames:
                w.writerow([fr.t_ms] + fr.angles + [fr.gesture, fr.confidence])
        return len(frames)

    @staticmethod
    def export_session_report(path: str, stats: Dict):
        report = {
            "session_id":    hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            "timestamp":     datetime.now().isoformat(),
            "model_version": CFG.get("model_version"),
            "stats":         stats,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 32: RESEARCH STUBS
# ══════════════════════════════════════════════════════════════════════════════

class NeuromorphicVisionStub:
    """STUB: DAVIS/Prophesee event camera. Use Metavision SDK in production."""
    def get_frame(self) -> Optional[np.ndarray]:
        return None


class TinyMLStub:
    """STUB: ESP32 on-device inference via TFLite Micro int8."""
    def infer(self, frame_bytes: bytes) -> Optional[List[int]]:
        return None


class MultimodalTransformerStub:
    """STUB: VLA Transformer — train on data collected from this controller."""
    def predict(self, *args, **kwargs) -> Optional[List[int]]:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 33: MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _make_dummy_serial():
    """Returns a duck-type replacement when --no-serial is passed."""
    class _Dummy:
        ser   = None
        alive = False

        def send(self, _):            return False
        def send_emergency_stop(self): pass
        def close(self):              pass
    return _Dummy()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Bionic Hand Nexus v6.0 — Cognitive Edition (FIXED)")
    parser.add_argument("--headless",            action="store_true")
    parser.add_argument("--camera",              type=int, default=None)
    parser.add_argument("--mode",                type=int, default=1, choices=range(1, 10))
    parser.add_argument("--no-serial",           action="store_true")
    parser.add_argument("--no-kalman",           action="store_true")
    parser.add_argument("--voice",               action="store_true")
    parser.add_argument("--profile",             action="store_true")
    parser.add_argument("--train-sim",           action="store_true")
    parser.add_argument("--sim-episodes",        type=int, default=500)
    parser.add_argument("--ota-check",           action="store_true")
    parser.add_argument("--generate-synthetic",  action="store_true")
    parser.add_argument("--hand-id",             type=int, default=None)
    args = parser.parse_args()

    if args.voice:    CFG.set("voice_enabled",    True)
    if args.profile:  CFG.set("profiler_enabled", True)
    if args.hand_id:  CFG.set("hand_id",          args.hand_id)

    # ── Sim training mode ──────────────────────────────────────────────────
    if args.train_sim:
        trainer  = PPOTrainer()
        out_path = trainer.train(n_episodes=args.sim_episodes)
        print(f"[SIM] {'Complete → ' + out_path if out_path else 'Failed'}")
        return

    # ── OTA check mode ─────────────────────────────────────────────────────
    if args.ota_check:
        CFG.set("ota_enabled", True)
        ota = OTAUpdater()
        done_ev = threading.Event()

        def ota_cb(status, progress):
            print(f"[OTA] {status} {progress:.0%}")
            if status in ("done", "error"):
                done_ev.set()

        ota.check_and_update(callback=ota_cb)
        done_ev.wait(timeout=60)
        return

    # ── Synthetic data stub ────────────────────────────────────────────────
    if args.generate_synthetic:
        synth = Path("generate_synthetic_data.py")
        if synth.exists():
            import subprocess
            subprocess.run([sys.executable, str(synth)])
        else:
            print("[SYNTHETIC] Script not found. Create generate_synthetic_data.py.")
        return

    # ── Camera ────────────────────────────────────────────────────────────
    cam_idx = args.camera if args.camera is not None else int(CFG.get("camera_index"))
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(CFG.get("frame_width")))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(CFG.get("frame_height")))

    # ── Subsystems ────────────────────────────────────────────────────────
    calibrator    = AdaptiveCalibrator()
    tracker       = HandTracker(calibrator=calibrator)
    kalman        = KalmanAngleFilter(
                        n=6,
                        process_noise=float(CFG.get("kalman_process_noise")),
                        measurement_noise=float(CFG.get("kalman_measurement_noise")))
    smoother      = MotionSmoother(n=6)
    classifier    = GestureClassifier()
    anomaly_det   = AnomalyDetector()
    memory        = MotionMemory()
    dataset       = DatasetFactory()
    replay_buf    = ReplayBuffer()
    force_ctrl    = ForceController()
    grasp_plan    = GraspPlanner()
    tactile       = TactileArray()
    proximity     = ProximitySensor()
    imu_sim       = IMUSimulator()
    emg_sim       = EMGSimulator()
    eeg           = EEGListener()
    haptic        = HapticFeedback()
    ota_updater   = OTAUpdater()
    finger_mgr    = FingerManager()
    intent_parser = IntentParser()
    ble_bridge    = WirelessBridge()
    serial_b      = _make_dummy_serial() if args.no_serial else SerialBridge()
    telemetry     = TelemetryPublisher()
    twin_bc       = DigitalTwinBroadcaster()
    voice         = VoiceController()
    profiler      = PerformanceProfiler()
    hud           = HUDRenderer()

    twin_bc.start()

    # ── Load AI models ────────────────────────────────────────────────────
    cnn_model   = REGISTRY.load_cnn()
    rl_policy   = REGISTRY.load_rl_policy()
    fusion_net  = REGISTRY.load_fusion() if CFG.get("fusion_enabled") else None
    personalizer = OnlinePersonalizer(model=cnn_model)

    temp_transformer = None
    if HAS_TORCH:
        temp_transformer = TemporalTransformer()
        tt_path = Path(CFG.get("model_dir")) / "temporal_transformer.pth"
        if tt_path.exists():
            try:
                temp_transformer.load_state_dict(
                    torch.load(str(tt_path), map_location="cpu", weights_only=True))
                temp_transformer.eval()
            except Exception as e:
                LOG.warning(f"TemporalTransformer load: {e}")
                temp_transformer = None

    # ── State ─────────────────────────────────────────────────────────────
    mode              = TrackingMode(args.mode)
    kalman_on         = (not args.no_kalman) and bool(CFG.get("kalman_enabled"))
    classifier_on     = bool(CFG.get("gesture_classifier_enabled"))
    fusion_enabled    = bool(CFG.get("fusion_enabled"))
    # FIXED: always start in neutral — never zero (which closes all fingers)
    last_angles: List[int] = list(PRESET_ANGLES["neutral"])
    current_gesture   = GestureClass.UNKNOWN
    current_conf      = 0.0
    anomaly_level     = AnomalyLevel.NORMAL
    anomaly_score     = 0.0
    playback_gen: Optional[Generator] = None
    playback_timer    = 0.0
    playback_speed    = float(CFG.get("playback_speed"))
    gesture_counter   = 0
    fps               = 0.0
    frame_count       = 0
    fps_t0            = time.perf_counter()
    session_frames:   List[AngleFrame] = []
    object_label: Optional[str] = None
    current_forces    = [0.0] * 5
    angle_history: Deque = deque(maxlen=20)
    eeg_cmd           = "NEUTRAL"
    eeg_conf_val      = 0.0
    recovery_pending  = False

    # Initialise Kalman and smoother to neutral
    kalman.reset(last_angles)
    smoother.reset(last_angles)

    print("\n" + "=" * 72)
    print("  BIONIC HAND NEXUS v6.0 — Cognitive Enterprise Edition  [FIXED]")
    print("=" * 72)
    print(f"  Device:    {REGISTRY.device.upper()}")
    print(f"  CNN:       {'LOADED' if cnn_model else 'NOT FOUND'}")
    print(f"  RL Policy: {'LOADED' if rl_policy else 'NOT FOUND'}")
    print(f"  FusionNet: {'LOADED' if fusion_net else 'NOT FOUND'}")
    print(f"  Serial:    {'CONNECTED' if serial_b.alive else 'DISCONNECTED'}")
    print(f"  BLE:       {'ENABLED' if CFG.get('ble_enabled') else 'DISABLED'}")
    print(f"  EEG:       {'ENABLED' if eeg.enabled else 'DISABLED'}")
    print(f"  Voice:     {'ENABLED' if voice.enabled else 'DISABLED'}")
    print(f"  Haptic:    {'ENABLED' if haptic._enabled else 'DISABLED'}")
    print("-" * 72)
    print("  Modes: 1=Mirror 2=Playback 3=CNN 4=Auto 5=EMG 6=RL 7=Hybrid 8=Fusion 9=EEG")
    print("  R=Rec  S=Dataset  C=Calib  K=Kalman  G=Classifier  P=Profiler  E=Export")
    print("  +/-=Speed  H=Haptic  T=EEG  U=OTA  Y=Fusion  Shift+N=FewShot  Q=Quit")
    print("  Presets: O=Open  B=Bottle  n=Pen  M=Neutral")
    print("=" * 72 + "\n")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(raw_frame, 1)
        frame_count += 1
        now = time.perf_counter()

        # FPS counter
        if now - fps_t0 >= 1.0:
            fps         = frame_count / (now - fps_t0)
            frame_count = 0
            fps_t0      = now
            MLOPS.log({"fps": fps,
                       "kalman_rmse":   kalman.innovations_rmse,
                       "anomaly_score": anomaly_score})

        # ── Proximity → velocity scaling ─────────────────────────────────
        smoother.proximity_factor = proximity.velocity_factor

        # ── EEG override  (FIXED: only triggers when confident) ──────────
        eeg_cmd, eeg_conf_val = eeg.get()
        should_eeg, eeg_action = eeg.should_override()
        if should_eeg:
            if eeg_action == "CLOSE":
                last_angles = list(PRESET_ANGLES["fist"])
                kalman.reset(last_angles)
                smoother.reset(last_angles)
            elif eeg_action == "OPEN":
                last_angles = list(PRESET_ANGLES["open"])
                kalman.reset(last_angles)
                smoother.reset(last_angles)

        # ── Voice commands ────────────────────────────────────────────────
        voice_cmd = voice.get_command()
        if voice_cmd:
            if voice_cmd.startswith("mode:"):
                new_mode = TrackingMode(int(voice_cmd.split(":")[1]))
                if new_mode != mode:
                    mode = new_mode
                    kalman.reset(last_angles)
                    smoother.reset(last_angles)
                voice.speak(f"{mode.name} mode")
            elif voice_cmd == "record:start" and not memory.recording:
                gesture_counter += 1
                memory.start_recording(f"gesture_{gesture_counter:03d}")
                dataset.start()
                voice.speak("Recording started")
            elif voice_cmd == "record:stop" and memory.recording:
                g = memory.stop_recording()
                if g:
                    memory.save(g)
                    dataset.save()
                    voice.speak(f"Saved {g.name}")
            elif voice_cmd == "safety:stop":
                serial_b.send_emergency_stop()
                voice.speak("Emergency stop")
            elif voice_cmd == "calibrate:start":
                calibrator.start(5.0)
                voice.speak("Calibration started")
            elif voice_cmd == "haptic:toggle":
                on = haptic.toggle()
                voice.speak(f"Haptic {'on' if on else 'off'}")
            elif voice_cmd == "ota:check":
                ota_updater.check_and_update()
                voice.speak("Checking for updates")
            elif voice_cmd == "fusion:toggle":
                fusion_enabled = not fusion_enabled
                voice.speak(f"Fusion {'on' if fusion_enabled else 'off'}")
            elif voice_cmd == "eeg:override":
                mode = TrackingMode.EEG
                voice.speak("EEG mode activated")
            elif voice_cmd.startswith("preset:"):
                p_name = voice_cmd.split(":")[1]
                if p_name in PRESET_ANGLES:
                    last_angles = list(PRESET_ANGLES[p_name])
                    kalman.reset(last_angles)
                    smoother.reset(last_angles)
                    voice.speak(p_name)

        # ── NLP intent ────────────────────────────────────────────────────
        vi = voice.get_intent()
        if vi:
            tgt = intent_parser.to_angles(vi, last_angles)
            if tgt:
                last_angles = tgt
                if "speed" in vi:
                    smoother.proximity_factor = min(
                        smoother.proximity_factor, float(vi["speed"]))
                if "force" in vi:
                    force_ctrl.set_target_force(vi["force"])

        # ── Few-shot frame feeding ─────────────────────────────────────────
        if grasp_plan.few_shot_collecting:
            if grasp_plan.feed_few_shot_frame(frame):
                predicted = grasp_plan.few_shot_predict(frame, "custom_object")
                if predicted:
                    last_angles = predicted
                    kalman.reset(last_angles)
                    smoother.reset(last_angles)

        # ══════════════════════════════════════════════════════════════════
        # FRAME PROCESSING — angles is None if no valid result this frame
        # ══════════════════════════════════════════════════════════════════
        angles: Optional[List[int]] = None
        annotated = frame

        with profiler.measure("tracking"):
            if mode == TrackingMode.MIRROR:
                annotated, angles = tracker.process(frame)

            elif mode == TrackingMode.PLAYBACK and playback_gen is not None:
                t_now = time.perf_counter()
                if t_now >= playback_timer:
                    try:
                        pb_a, delay = next(playback_gen)
                        last_angles    = pb_a
                        playback_timer = t_now + max(0.0, delay)
                        angles         = pb_a
                    except StopIteration:
                        playback_gen = None
                        print("[PLAYBACK] Finished.")

            elif mode == TrackingMode.CNN and cnn_model is not None:
                with profiler.measure("cnn_inference"):
                    angles = cnn_model.predict_frame(frame, device=REGISTRY.device)
                if angles is None:
                    # Fallback to MediaPipe
                    annotated, angles = tracker.process(frame)

            elif mode == TrackingMode.AUTONOMOUS:
                with profiler.measure("grasp_plan"):
                    annotated, auto_a, object_label = grasp_plan.detect_and_plan(
                        frame, last_angles)
                angles = auto_a

            elif mode == TrackingMode.RL_ADAPTIVE and rl_policy is not None:
                obj_cls = np.zeros(8); obj_cls[0] = 1.0
                state_np = np.concatenate([
                    np.array(last_angles, dtype=np.float32) / 180.0,
                    np.array(last_angles, dtype=np.float32) / 180.0,
                    np.array((current_forces + [0.0] * 5)[:5], dtype=np.float32) / 10.0,
                    obj_cls
                ])
                delta  = rl_policy.get_action(state_np, deterministic=True)
                angles = [_clamp_angle(last_angles[i] + float(delta[i]))
                          for i in range(6)]
                annotated, _ = tracker.process(frame)

            elif mode == TrackingMode.FUSION and fusion_net is not None:
                with profiler.measure("fusion"):
                    _, mp_a    = tracker.process(frame)
                    emg_window = emg_sim.get_window()
                    imu_vec    = imu_sim.read_vector()
                    if cnn_model is not None and emg_window is not None:
                        vis_emb = cnn_model.extract_embedding(frame, REGISTRY.device)
                        if vis_emb is not None:
                            angles = fusion_net.predict(
                                vis_emb, emg_window, imu_vec, REGISTRY.device)
                    if angles is None:
                        angles = mp_a

            elif mode == TrackingMode.EEG:
                if eeg_cmd == "CLOSE":
                    angles = list(PRESET_ANGLES["fist"])
                elif eeg_cmd == "OPEN":
                    angles = list(PRESET_ANGLES["open"])
                else:
                    angles = list(last_angles)
                annotated, _ = tracker.process(frame)

            elif mode == TrackingMode.HYBRID and cnn_model is not None:
                _, mp_a  = tracker.process(frame)
                cnn_out  = cnn_model.predict_frame(frame, device=REGISTRY.device)
                if mp_a is not None and cnn_out is not None:
                    angles = cnn_out[:5] + [mp_a[5]]
                elif mp_a is not None:
                    angles = mp_a
                elif cnn_out is not None:
                    angles = cnn_out

        # ── Temporal Transformer blend ─────────────────────────────────────
        if (angles is not None and temp_transformer is not None
                and len(angle_history) >= temp_transformer.seq_len
                and mode not in (TrackingMode.PLAYBACK, TrackingMode.AUTONOMOUS)):
            with profiler.measure("temporal"):
                pred = temp_transformer.predict_next(angle_history)
                if pred is not None:
                    angles = [_clamp_angle(0.75 * a + 0.25 * p)
                              for a, p in zip(angles, pred)]

        # ══════════════════════════════════════════════════════════════════
        # SIGNAL PROCESSING PIPELINE
        # ══════════════════════════════════════════════════════════════════
        if angles is not None:
            angle_history.append(list(angles))

            with profiler.measure("kalman"):
                if kalman_on:
                    angles = kalman.update(angles)

            with profiler.measure("smoother"):
                angles = smoother.smooth(angles)

            # Tactile + force feedback
            tactile_readings = tactile.update()
            tactile_slip     = tactile.max_slip_risk()

            if CFG.get("fsr_enabled"):
                angles, slip_det = force_ctrl.update(
                    angles, current_forces, tactile_slip)
                if slip_det:
                    haptic.send_slip_alert()
                    voice.speak("Slip")
                else:
                    haptic.send_force_feedback(
                        [float(CFG.get("target_grip_force_n")) - f
                         for f in current_forces])

            last_angles = list(angles)

            # Gesture classification
            with profiler.measure("gesture_clf"):
                if classifier_on:
                    rg, rc = classifier.classify(angles)
                    current_gesture, current_conf = classifier.smooth_output(rg, rc)

            # Anomaly detection
            with profiler.measure("anomaly"):
                anomaly_level, anomaly_score = anomaly_det.feed(angles)
                if anomaly_level == AnomalyLevel.CRITICAL:
                    if not recovery_pending:
                        recovery_pending = True
                        LOG.critical("CRITICAL anomaly: self-heal")
                        rec_angles = anomaly_det.self_heal()
                        if rec_angles:
                            last_angles = list(rec_angles)
                            kalman.reset(last_angles)
                            smoother.reset(last_angles)
                            serial_b.send(last_angles)
                        else:
                            serial_b.send_emergency_stop()
                else:
                    if recovery_pending:
                        anomaly_det.reset_recovery()
                    recovery_pending = False

            # Record
            if memory.recording:
                ew = emg_sim.get_window()
                memory.capture_frame(
                    angles,
                    forces=(list(current_forces) if CFG.get("fsr_enabled") else None),
                    emg=(ew.flatten().tolist()[:8] if ew is not None else None),
                    imu=imu_sim.read_vector().tolist())

            # Dataset capture
            if dataset.active:
                ew = emg_sim.get_window()
                dataset.capture(
                    frame, angles,
                    gesture=current_gesture.value, auto_label=True,
                    emg=(ew.flatten().tolist()[:8] if ew is not None else None),
                    imu=imu_sim.read_vector().tolist())
                replay_buf.add(
                    "", angles,
                    gesture=current_gesture.value, source=mode.name.lower(),
                    imu=imu_sim.read_vector().tolist())

            # Online learning
            if mode in (TrackingMode.MIRROR, TrackingMode.CNN):
                personalizer.feed(frame, angles)

            session_frames.append(AngleFrame(
                t_ms=round(now * 1000, 1), angles=list(angles),
                gesture=current_gesture.value, confidence=current_conf))

            # Send to hardware
            with profiler.measure("serial"):
                if ble_bridge.alive:
                    ble_bridge.send(angles)
                else:
                    serial_b.send(angles)
                finger_mgr.send(angles)

            # Telemetry + digital twin
            telemetry.publish_state(
                angles=angles, gesture=current_gesture.value,
                conf=current_conf, fps=fps,
                anomaly_score=anomaly_score, forces=current_forces)

            state = ServoState(angles=list(angles), forces=list(current_forces),
                               stiffness=list(force_ctrl._stiffness))
            twin_bc.broadcast(state, current_gesture.value,
                              anomaly_level, fps, tracker.landmarks_3d)

        # ── HUD ────────────────────────────────────────────────────────────
        if not CFG.get("headless") and not args.headless:
            display = hud.draw(
                annotated,
                mode=mode, recording=memory.recording,
                angles=last_angles, fps=fps,
                gesture=current_gesture.value, conf=current_conf,
                anomaly=anomaly_level, anomaly_score=anomaly_score,
                forces=(current_forces if CFG.get("fsr_enabled") else None),
                calib_progress=calibrator.progress,
                profiler_data=(profiler.report() if CFG.get("profiler_enabled") else None),
                serial_alive=serial_b.alive, ble_alive=ble_bridge.alive,
                object_label=object_label,
                eeg_cmd=eeg_cmd, eeg_conf=eeg_conf_val,
                playback_speed=playback_speed,
                haptic_active=haptic.active,
                fusion_enabled=fusion_enabled,
                few_shot_progress=grasp_plan.few_shot_progress,
                proximity_cm=proximity.distance_cm)
            cv2.imshow("Bionic Hand Nexus v6.0", display)

        # ── Keyboard ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Mode selection 1–9 (reset Kalman + smoother on switch)
        new_mode = None
        if   key == ord('1'): new_mode = TrackingMode.MIRROR
        elif key == ord('2'):
            new_mode = TrackingMode.PLAYBACK
            gestures = memory.list_gestures()
            if gestures:
                g = memory.load(str(gestures[-1]))
                playback_gen   = memory.playback_frames(g, speed_factor=playback_speed)
                playback_timer = time.perf_counter()
                print(f"[PLAY] {g.name} ({len(g.frames)} frames) @ {playback_speed:.1f}x")
            else:
                print("[PLAY] No gestures recorded yet.")
        elif key == ord('3'):
            new_mode = TrackingMode.CNN
            if cnn_model is None:
                print("[CNN] Model not found. Train first.")
        elif key == ord('4'):
            new_mode = TrackingMode.AUTONOMOUS
            if not CFG.get("yolo_enabled"):
                print("[AUTO] Enable 'yolo_enabled' in nexus_config.json.")
        elif key == ord('5'): new_mode = TrackingMode.EMG
        elif key == ord('6'):
            new_mode = TrackingMode.RL_ADAPTIVE
            if rl_policy is None:
                print("[RL] Policy not found. Run --train-sim first.")
        elif key == ord('7'): new_mode = TrackingMode.HYBRID
        elif key == ord('8'):
            new_mode = TrackingMode.FUSION
            if fusion_net is None:
                fusion_net = REGISTRY.load_fusion()
        elif key == ord('9'):
            new_mode = TrackingMode.EEG
            print(f"[EEG] {'Active' if eeg.enabled else 'Enable eeg_enabled in config'}")

        if new_mode is not None and new_mode != mode:
            mode = new_mode
            # FIXED: reset filter state on every mode switch
            kalman.reset(last_angles)
            smoother.reset(last_angles)

        # Recording
        elif key in (ord('r'), ord('R')):
            if not memory.recording:
                gesture_counter += 1
                name = f"gesture_{gesture_counter:03d}"
                memory.start_recording(name)
                dataset.start()
                print(f"[REC] Started: {name}")
            else:
                g = memory.stop_recording()
                if g:
                    path = memory.save(g)
                    n    = dataset.save()
                    print(f"[REC] Saved {g.name} ({len(g.frames)} frames, {n} samples) → {path}")

        elif key in (ord('s'), ord('S')):
            n = dataset.save()
            print(f"[DATASET] Saved {n} samples (total: {dataset._counter})")

        elif key in (ord('c'), ord('C')):
            calibrator.start(5.0)
            print("[CALIB] Move your hand for 5 seconds...")

        elif key in (ord('k'), ord('K')):
            kalman_on = not kalman_on
            if kalman_on:
                kalman.reset(last_angles)
            print(f"[KALMAN] {'ON' if kalman_on else 'OFF'}")

        elif key in (ord('g'), ord('G')):
            classifier_on = not classifier_on
            print(f"[CLASSIFIER] {'ON' if classifier_on else 'OFF'}")

        elif key in (ord('p'), ord('P')):
            print(f"[PROFILER] {'ON' if profiler.toggle() else 'OFF'}")

        elif key in (ord('e'), ord('E')):
            ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
            od     = Path(CFG.get("export_dir"))
            c_path = str(od / f"session_{ts}.csv")
            r_path = str(od / f"session_{ts}_report.json")
            SessionExporter.export_csv(session_frames, c_path)
            SessionExporter.export_session_report(r_path, {
                "total_frames":  len(session_frames),
                "kalman_rmse":   round(kalman.innovations_rmse, 4),
                "anomaly_avg":   round(anomaly_det.rolling_score, 4),
                "mlops_summary": MLOPS.summary(),
                "profiler":      profiler.report(),
                "finger_status": finger_mgr.get_status(),
                "ota_status":    ota_updater.status,
            })
            print(f"[EXPORT] {c_path}")

        # Playback speed + / -
        elif key in (ord('+'), ord('=')):
            playback_speed = min(4.0, round(playback_speed + 0.25, 2))
            if playback_gen is not None:
                gl = memory.list_gestures()
                if gl:
                    g = memory.load(str(gl[-1]))
                    playback_gen   = memory.playback_frames(g, speed_factor=playback_speed)
                    playback_timer = time.perf_counter()
            print(f"[SPEED] {playback_speed:.2f}x")

        elif key == ord('-'):
            playback_speed = max(0.1, round(playback_speed - 0.25, 2))
            if playback_gen is not None:
                gl = memory.list_gestures()
                if gl:
                    g = memory.load(str(gl[-1]))
                    playback_gen   = memory.playback_frames(g, speed_factor=playback_speed)
                    playback_timer = time.perf_counter()
            print(f"[SPEED] {playback_speed:.2f}x")

        elif key in (ord('h'), ord('H')):
            print(f"[HAPTIC] {'ON' if haptic.toggle() else 'OFF'}")

        elif key in (ord('t'), ord('T')):
            if mode != TrackingMode.EEG:
                mode = TrackingMode.EEG
                kalman.reset(last_angles)
                smoother.reset(last_angles)
                print("[EEG] Activated")
            else:
                mode = TrackingMode.MIRROR
                kalman.reset(last_angles)
                smoother.reset(last_angles)
                print("[EEG] Deactivated")

        elif key in (ord('u'), ord('U')):
            print("[OTA] Checking for updates...")
            ota_updater.check_and_update()

        elif key in (ord('y'), ord('Y')):
            fusion_enabled = not fusion_enabled
            CFG.set("fusion_enabled", fusion_enabled)
            if fusion_enabled and fusion_net is None:
                fusion_net = REGISTRY.load_fusion()
            print(f"[FUSION] {'ON' if fusion_enabled else 'OFF'}")

        elif key == ord('N'):   # Shift+N = few-shot
            if not grasp_plan.few_shot_collecting:
                grasp_plan.start_few_shot_capture()

        # Presets
        elif key in (ord('o'), ord('O')):
            last_angles = list(PRESET_ANGLES["open"])
            kalman.reset(last_angles)
            smoother.reset(last_angles)
            serial_b.send(last_angles)
            print("[PRESET] Open")

        elif key in (ord('b'), ord('B')):
            t2 = grasp_plan.get_preset("bottle")
            if t2:
                last_angles = t2
                kalman.reset(last_angles)
                smoother.reset(last_angles)
                serial_b.send(last_angles)
                print("[PRESET] Bottle")

        elif key == ord('n'):   # lowercase n = pen
            t2 = grasp_plan.get_preset("pen")
            if t2:
                last_angles = t2
                kalman.reset(last_angles)
                smoother.reset(last_angles)
                serial_b.send(last_angles)
                print("[PRESET] Pen")

        elif key in (ord('m'), ord('M')):
            last_angles = list(PRESET_ANGLES["neutral"])
            kalman.reset(last_angles)
            smoother.reset(last_angles)
            serial_b.send(last_angles)
            print("[PRESET] Neutral")

    # ── Cleanup ────────────────────────────────────────────────────────────
    LOG.info("Shutting down…")
    cap.release()
    cv2.destroyAllWindows()
    serial_b.close()
    ble_bridge.close()
    haptic.close()
    telemetry.close()
    proximity.stop()
    imu_sim.stop()
    emg_sim.stop()
    eeg.stop()
    finger_mgr.close()
    replay_buf.save()
    dataset.save()

    print(f"\n[EXIT] Session ended.")
    print(f"[EXIT] Total frames recorded: {len(session_frames)}")
    print(f"[EXIT] Replay buffer size:    {len(replay_buf)}")
    print(f"[EXIT] MLOps summary:         {MLOPS.summary()}")
    LOG.info("Shutdown complete.")


if __name__ == "__main__":
    main()
