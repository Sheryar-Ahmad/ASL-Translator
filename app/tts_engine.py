import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
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
        self.queue: "queue.Queue[Optional[tuple[int, str]]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self._mixer_ready = False
        self._stopping = threading.Event()
        self._cancel = threading.Event()
        self._process = None
        self._state_lock = threading.RLock()
        self._generation = 0
        self.status = "Idle"
        self.last_error = None

    @property
    def available(self) -> bool:
        return bool(self.model_path and os.path.isfile(self.model_path)
                    and os.path.isfile(self.model_path + ".json")
                    and (not self.autoplay or pygame is not None))

    @property
    def busy(self) -> bool:
        return self.queue.unfinished_tasks > 0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self._stopping.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def speak(self, text: str) -> None:
        cleaned = (text or "").strip()
        if cleaned and not self._stopping.is_set():
            with self._state_lock:
                if self._stopping.is_set():
                    return
                self.start()
                self.queue.put((self._generation, cleaned))

    def wait_idle(self) -> None:
        self.queue.join()

    def cancel(self) -> None:
        """Discard pending speech without terminating the reusable worker."""
        with self._state_lock:
            self._generation += 1
            self._cancel.set()
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass

    def stop(self, timeout: float = 10.0) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        self.cancel()
        if self.thread and self.thread.is_alive():
            self.queue.put(None)
            self.thread.join(timeout=timeout)

    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            wav_path = None
            try:
                if item is None:
                    break
                generation, text = item
                with self._state_lock:
                    if self._stopping.is_set() or generation != self._generation:
                        continue
                    self._cancel.clear()
                self.last_error = None
                self.status = "Preparing speech…"
                wav_path = self._synthesize(text)
                if self.autoplay and not self._cancel.is_set():
                    self.status = "Speaking…"
                    self._play(wav_path)
                self.status = "Idle" if self._cancel.is_set() else "Completed"
            except Exception as exc:
                if self._cancel.is_set() or self._stopping.is_set():
                    self.status = "Idle"
                else:
                    self.last_error = str(exc)
                    self.status = "Speech failed"
                    logger.exception("TTS worker failed")
            finally:
                if wav_path and self.autoplay:
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                self.queue.task_done()
        if self._mixer_ready:
            pygame.mixer.quit()
            self._mixer_ready = False

    def _base_command(self) -> List[str]:
        found = shutil.which(self.piper_executable)
        return [found] if found else [sys.executable, "-m", "piper"]

    def _synthesize(self, text: str) -> str:
        if not self.available:
            raise FileNotFoundError(
                "Speech requires the Piper .onnx model and matching .onnx.json file "
                "(and pygame for playback). Run: python scripts/download_models.py"
            )
        output_path = os.path.join(self.output_dir, f"tts_{uuid.uuid4().hex}.wav")
        command = self._base_command() + ["--model", self.model_path,
                                         "--output_file", output_path]
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            if self._cancel.is_set() or self._stopping.is_set():
                self._process.terminate()
            try:
                _, stderr = self._process.communicate(text.encode("utf-8"), timeout=120)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.communicate()
                raise RuntimeError("Piper speech synthesis timed out.")
            if self._process.returncode != 0:
                raise RuntimeError("Piper TTS failed: " + stderr.decode("utf-8", errors="replace")[:500])
            if not os.path.exists(output_path):
                raise RuntimeError("Piper did not produce an output WAV file.")
            return output_path
        except Exception:
            try:
                os.remove(output_path)
            except OSError:
                pass
            raise
        finally:
            self._process = None

    def _play(self, path: str) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for audio playback.")
        if not self._mixer_ready:
            pygame.mixer.init()
            self._mixer_ready = True
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._cancel.wait(0.05) or self._stopping.is_set():
                    pygame.mixer.music.stop()
                    break
        finally:
            pygame.mixer.music.unload()
