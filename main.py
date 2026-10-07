#!/usr/bin/env python3
import argparse
import logging
import platform
import time

import cv2

from app.asl_engine import ASLEngine
from app.config import load_config
from app.text_buffer import AutoCommitController, TextBuffer
from app.tts_engine import TTSEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("main")


def draw_hud(frame, cfg, prediction, text_buffer, fps):
    height, width = frame.shape[:2]

    label = prediction.get("label", "")
    confidence = float(prediction.get("confidence", 0.0))
    sentence = text_buffer.text

    cv2.rectangle(frame, (0, 0), (width, 70), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Sign: {label} ({confidence:.2f})",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Text: {sentence[:60]}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if cfg.get("ui", {}).get("show_fps", True):
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (width - 130, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        "c:commit space:space .:period s/enter:speak x:clear q:quit",
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )


def run_speak(cfg, text):
    tts = TTSEngine(cfg.get("tts", {}))
    tts.start()

    try:
        tts.speak(text)
        tts.wait_idle()
        if tts.last_error:
            raise RuntimeError(tts.last_error)
    finally:
        tts.stop()


def run_camera(cfg):
    tts = TTSEngine(cfg.get("tts", {}))

    asl = ASLEngine(cfg.get("asl", {}))

    labels_cfg = cfg.get("asl", {}).get("labels", {})

    text_buffer = TextBuffer(
        space_label=labels_cfg.get("space_label", "space"),
        delete_label=labels_cfg.get("delete_label", "delete"),
        nothing_label=labels_cfg.get("nothing_label", "nothing"),
    )

    auto_commit = AutoCommitController(cfg.get("asl", {}))

    sentence_cfg = cfg.get("sentence", {})
    auto_speak = bool(sentence_cfg.get("auto_speak", True))
    complete_pause_ms = int(sentence_cfg.get("complete_pause_ms", 2500))
    min_chars = int(sentence_cfg.get("min_chars", 3))
    clear_after_speak = bool(sentence_cfg.get("clear_after_speak", False))

    camera_cfg = cfg.get("camera", {})
    camera_index = int(camera_cfg.get("index", 0))

    if platform.system() == "Windows":
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        capture.release()
        asl.close()
        tts.stop()
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Try changing camera.index in config.yaml."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg.get("width", 640)))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg.get("height", 480)))
    capture.set(cv2.CAP_PROP_FPS, int(camera_cfg.get("fps", 30)))

    window_name = cfg.get("ui", {}).get("window_name", "ASL Translator")
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except Exception:
        capture.release()
        asl.close()
        tts.stop()
        raise
    tts.start()

    last_time = time.time()
    fps = 0.0

    last_commit_ms = 0
    spoken_text = None

    def commit_label(label, confidence, now_ms):
        nonlocal last_commit_ms, spoken_text

        if text_buffer.commit_label(label, confidence):
            last_commit_ms = now_ms
            spoken_text = None
            return True

        return False

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                logger.error("Camera frame read failed.")
                break

            if camera_cfg.get("mirror", True):
                frame = cv2.flip(frame, 1)

            prediction = asl.predict(frame)

            now_ms = int(time.time() * 1000)

            commit_mode = str(cfg.get("asl", {}).get("commit_mode", "manual")).lower()

            if commit_mode == "auto":
                committed_label = auto_commit.update(
                    prediction.get("label"),
                    float(prediction.get("confidence", 0.0)),
                    now_ms,
                )

                if committed_label:
                    commit_label(
                        committed_label,
                        float(prediction.get("confidence", 0.0)),
                        now_ms,
                    )

            draw_hud(frame, cfg, prediction, text_buffer, fps)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit.
            if key in (ord("q"), 27):
                break

            # Commit current ASL prediction.
            elif key == ord("c"):
                commit_label(
                    prediction.get("label"),
                    float(prediction.get("confidence", 0.0)),
                    now_ms,
                )

            # Keyboard space inserts space.
            elif key == 32:
                commit_label(text_buffer.space_label, 1.0, now_ms)

            # Period key inserts sentence-ending punctuation.
            elif key == ord("."):
                commit_label(".", 1.0, now_ms)

            # Speak manually.
            elif key in (ord("s"), 13):
                if text_buffer.text:
                    tts.speak(text_buffer.text)
                    spoken_text = text_buffer.text

            # Delete last character/sign.
            elif key == 8:
                if text_buffer.backspace():
                    last_commit_ms = now_ms
                    spoken_text = None

            # Clear sentence.
            elif key == ord("x"):
                text_buffer.clear()
                spoken_text = None
                last_commit_ms = 0

            # Automatic sentence speaking.
            current_text = text_buffer.text

            if auto_speak and current_text and spoken_text != current_text:
                if len(current_text) >= min_chars:
                    ends_sentence = current_text[-1] in ".!?"

                    paused_long_enough = (
                        last_commit_ms > 0
                        and (now_ms - last_commit_ms) >= complete_pause_ms
                    )

                    if ends_sentence or paused_long_enough:
                        tts.speak(current_text)
                        spoken_text = current_text

                        if clear_after_speak:
                            text_buffer.clear()
                            spoken_text = None
                            last_commit_ms = 0

            current_time = time.time()
            delta = current_time - last_time

            if delta > 0:
                fps = 1.0 / delta

            last_time = current_time

    finally:
        asl.close()
        tts.stop()
        capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Offline ASL to text and speech MVP")

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )

    parser.add_argument(
        "--speak",
        default="",
        help="Speak the given text and exit.",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical user interface.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.gui:
        from app.gui_app import run_gui

        run_gui(cfg, args.config)
    elif args.speak:
        run_speak(cfg, args.speak)
    else:
        run_camera(cfg)


if __name__ == "__main__":
    main()
