import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


CHECKS = []


def add_check(name, ok, critical=True, message=""):
    CHECKS.append(
        {
            "name": name,
            "ok": bool(ok),
            "critical": bool(critical),
            "message": message,
        }
    )


def check_import(module_name, critical=True):
    try:
        importlib.import_module(module_name)
        add_check(f"import {module_name}", True, critical)
    except Exception as exc:
        add_check(f"import {module_name}", False, critical, str(exc))


def check_piper():
    piper_path = shutil.which("piper")

    if piper_path:
        add_check("piper executable", True, True, piper_path)
        return

    try:
        process = subprocess.run(
            [sys.executable, "-m", "piper", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

        ok = process.returncode == 0
        message = process.stderr.decode("utf-8", errors="ignore")[:200]

        add_check("python -m piper", ok, True, message)

    except Exception as exc:
        add_check("python -m piper", False, True, str(exc))


def get_camera_index_from_config():
    config_path = Path("config.yaml")

    if not config_path.exists():
        return 0

    try:
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return int(cfg.get("camera", {}).get("index", 0))

    except Exception:
        return 0


def check_camera():
    try:
        import cv2
    except Exception as exc:
        add_check("camera read", False, False, str(exc))
        return

    camera_index = get_camera_index_from_config()

    try:
        capture = cv2.VideoCapture(camera_index)
        opened = capture.isOpened()

        read_ok = False

        if opened:
            ret, _ = capture.read()
            read_ok = bool(ret)

        capture.release()

        add_check(
            f"camera read index {camera_index}",
            opened and read_ok,
            False,
            "Check camera index, privacy settings, or try another camera index.",
        )

    except Exception as exc:
        add_check(f"camera read index {camera_index}", False, False, str(exc))


def main():
    version_ok = (3, 10) <= sys.version_info[:2] < (3, 12)

    add_check(
        "Python 3.10 or 3.11",
        version_ok,
        True,
        f"Found Python {sys.version.split()[0]}",
    )

    required_modules = [
        "cv2",
        "mediapipe",
        "onnxruntime",
        "numpy",
        "pygame",
        "PIL",
        "yaml",
    ]

    for module_name in required_modules:
        check_import(module_name, critical=True)

    check_import("huggingface_hub", critical=False)

    check_piper()

    add_check(
        "config.yaml exists",
        Path("config.yaml").exists(),
        False,
        "Create config.yaml from the project files.",
    )

    add_check(
        "TTS model exists",
        Path("models/tts/en_US-lessac-low.onnx").exists(),
        True,
        "Run: python scripts/download_models.py",
    )

    add_check(
        "TTS JSON config exists",
        Path("models/tts/en_US-lessac-low.onnx.json").exists(),
        False,
        "Run: python scripts/download_models.py",
    )

    asl_dir = Path("models/asl")
    asl_model_found = False

    if asl_dir.exists():
        asl_model_found = Path("models/asl/asl_landmarks.onnx").exists() or any(
            asl_dir.rglob("*.onnx")
        )

    add_check(
        "ASL model exists",
        asl_model_found,
        False,
        "App runs in dummy mode until you download or add a real ASL ONNX model.",
    )

    add_check(
        "ASL labels exist",
        Path("models/asl/labels.txt").exists(),
        False,
        "Run: python scripts/download_models.py",
    )

    if "--camera" in sys.argv:
        check_camera()

    failed_critical = [
        check for check in CHECKS if not check["ok"] and check["critical"]
    ]

    print("\nEnvironment check results:\n")

    for check in CHECKS:
        if check["ok"]:
            status = "PASS"
        elif check["critical"]:
            status = "FAIL"
        else:
            status = "WARN"

        line = f"[{status}] {check['name']}"

        if not check["ok"] and check["message"]:
            line += f" - {check['message']}"

        print(line)

    if failed_critical:
        print("\nCritical environment checks failed.")
        sys.exit(1)

    print("\nEnvironment OK.")


if __name__ == "__main__":
    main()
