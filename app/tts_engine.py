import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    import pygame
except Exception:
    pygame = None


logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.model_path = str(self.cfg.get("model_path", ""))
        self.output_dir = str(self.cfg.get("output_dir", ".cache/tts"))
        self.autoplay = bool(self.cfg.get("autoplay", True))
        self.piper_executable = str(self.cfg.get("piper_executable", "piper"))

        os.makedirs(self.output_dir, exist_ok=True)

        self.queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self._mixer_ready = False

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def speak(self, text: str) -> None:
        cleaned = (text or "").strip()
        if cleaned:
            self.queue.put(cleaned)

    def wait_idle(self) -> None:
        self.queue.join()

    def stop(self, timeout: float = 10.0) -> None:
        self.queue.put(None)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def _worker(self) -> None:
        while True:
            text = self.queue.get()

            try:
                if text is None:
                    break

                wav_path = self._synthesize(text)

                if wav_path and self.autoplay:
                    self._play(wav_path)

            except Exception:
                logger.exception("TTS worker failed")

            finally:
                self.queue.task_done()

    def _base_command(self) -> List[str]:
        found = shutil.which(self.piper_executable)

        if found:
            return [found]

        return [sys.executable, "-m", "piper"]

    def _synthesize(self, text: str) -> str:
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"TTS model not found: {self.model_path}. "
                "Run: python scripts/download_models.py"
            )

        output_path = os.path.join(
            self.output_dir,
            f"tts_{uuid.uuid4().hex}.wav",
        )

        command = self._base_command() + [
            "--model",
            self.model_path,
            "--output_file",
            output_path,
        ]

        logger.info("Synthesizing speech: %s", text)

        process = subprocess.run(
            command,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )

        if process.returncode != 0:
            stderr = process.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(
                "Piper TTS failed. "
                "Check Piper installation and model path. "
                f"Error: {stderr[:500]}"
            )

        if not os.path.exists(output_path):
            raise RuntimeError("Piper TTS did not produce an output WAV file.")

        return output_path

    def _play(self, path: str) -> None:
        if pygame is None:
            logger.warning("pygame is not available. Cannot play audio.")
            return

        try:
            if not self._mixer_ready:
                pygame.mixer.init()
                self._mixer_ready = True

            pygame.mixer.music.load(path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            try:
                pygame.mixer.music.unload()
            except Exception:
                pass

        except Exception:
            logger.exception("Audio playback failed")

        finally:
            try:
                os.remove(path)
            except OSError:
                pass
