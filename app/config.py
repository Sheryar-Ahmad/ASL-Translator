import os
from copy import deepcopy
from typing import Any, Dict

import yaml


DEFAULTS: Dict[str, Any] = {
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "mirror": True,
    },
    "asl": {
        "model_path": "models/asl/asl_landmarks.onnx",
        "labels_path": "models/asl/labels.txt",
        "input_type": "landmarks",
        "use_z": True,
        "image_normalization": "auto",
        "image_input_size": 224,
        "image_margin": 0.25,
        "draw_landmarks": True,
        "max_hands": 1,
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.6,
        "confidence_threshold": 0.65,
        "commit_mode": "manual",
        "labels": {
            "space_label": "space",
            "delete_label": "delete",
            "nothing_label": "nothing",
        },
        "auto": {
            "vote_window": 9,
            "stable_ms": 900,
            "reset_ms": 1400,
        },
    },
    "tts": {
        "engine": "piper",
        "piper_executable": "piper",
        "model_path": "models/tts/en_US-lessac-low.onnx",
        "output_dir": ".cache/tts",
        "autoplay": True,
    },
    "sentence": {
        "auto_speak": True,
        "clear_after_speak": False,
        "complete_pause_ms": 2500,
        "min_chars": 3,
        "max_sentence_length": 200,
        "speak_on_space": True,
        "stable_frames": 15,
        "cooldown_frames": 45,
    },
    "ui": {
        "window_name": "ASL Translator",
        "show_fps": True,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        return deepcopy(DEFAULTS)

    with open(path, "r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    if not isinstance(user_config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return deep_merge(DEFAULTS, user_config)
