import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config

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


def check_piper(executable="piper"):
    piper_path = shutil.which(executable)

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


def check_camera(camera_index=0):
    try:
        import cv2
    except Exception as exc:
        add_check("camera read", False, False, str(exc))
        return

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
    parser = argparse.ArgumentParser(description="Check configured ASL runtime dependencies and models.")
    parser.add_argument("--config", default="config.yaml", help="Runtime configuration path.")
    parser.add_argument("--camera", action="store_true", help="Also test webcam capture.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    CHECKS.clear()
    version_ok = sys.version_info[:2] == (3, 12)

    add_check(
        "Python 3.12",
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

    check_piper(cfg["tts"].get("piper_executable", "piper"))

    add_check(
        "Configuration exists",
        Path(args.config).is_file(),
        False,
        "Create config.yaml from the project files.",
    )

    add_check(
        "TTS model exists",
        Path(cfg["tts"]["model_path"]).is_file(),
        True,
        "Run: python scripts/download_models.py",
    )

    add_check(
        "TTS JSON config exists",
        Path(cfg["tts"]["model_path"] + ".json").is_file(),
        True,
        "Run: python scripts/download_models.py",
    )

    asl_model_found = Path(cfg["asl"]["model_path"]).is_file()

    add_check(
        "ASL model exists",
        asl_model_found,
        False,
        "Recognition is disabled until you add a model or run scripts/train_from_csv.py.",
    )

    add_check(
        "ASL labels exist",
        Path(cfg["asl"]["labels_path"]).is_file(),
        asl_model_found,
        "Train the ASL model or restore its matching labels.txt.",
    )

    if args.camera:
        check_camera(int(cfg["camera"]["index"]))

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
