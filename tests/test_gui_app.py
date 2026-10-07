import threading
import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from app import gui_app
from app.config import load_config


@pytest.fixture
def app(tmp_path, monkeypatch):
    captures = []
    class Capture:
        def __init__(self, *args):
            self.index = args[0]
            self.opened = True
            self.reading = False
            self.release_during_read = False
            captures.append(self)
        def isOpened(self):
            return self.opened
        def set(self, *args):
            pass
        def read(self):
            self.reading = True
            time.sleep(.01)
            self.reading = False
            return True, np.zeros((480, 640, 3), dtype=np.uint8)
        def release(self):
            self.release_during_read = self.reading
            self.opened = False
    class ASL:
        dummy = False
        def __init__(self, cfg):
            self.closed = False
        def predict(self, frame):
            return {"label": "a", "confidence": 1., "hand_present": True}
        def close(self):
            self.closed = True
    monkeypatch.setattr(gui_app.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(gui_app, "ASLEngine", ASL)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display is unavailable")
    root.withdraw()
    path = tmp_path / "custom.yaml"
    path.write_text("sentence:\n  auto_speak: false\n", encoding="utf-8")
    cfg = load_config(str(path))
    cfg["tts"]["output_dir"] = str(tmp_path / "tts")
    cfg["tts"]["model_path"] = str(tmp_path / "missing.onnx")
    instance = gui_app.ASLGuiApp(root, cfg, str(path))
    instance.test_captures = captures
    yield instance
    instance.on_close()
    assert not instance.camera_thread.is_alive()
    assert instance.asl.closed
    assert not any(c.release_during_read for c in captures)


def pump(app, duration=.1):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        app.root.update()
        time.sleep(.005)


def test_switch_camera_is_owned_by_worker(app):
    pump(app)
    app.camera_index_var.set("2")
    app._on_camera_change()
    pump(app)
    assert app.test_captures[-1].index == 2
    assert all(not c.opened for c in app.test_captures[:-1])
    assert not any(c.release_during_read for c in app.test_captures)


def test_pause_and_clear_reject_stale_predictions(app):
    pump(app)
    app.start_recognition()
    epoch = app._epoch
    app.stop_recognition()
    app._packets = gui_app.queue.Queue(maxsize=1)
    app._packets.put((np.zeros((10, 10, 3), dtype=np.uint8),
                      {"label": "a", "confidence": 1.}, epoch, 30, ""))
    app._update_ui()
    assert not app.text_buffer.text
    assert not app.latest_prediction
    app._commit("a")
    app.clear_sentence()
    assert not app.text_buffer.text
    assert app.auto_commit.candidate is None


def test_settings_use_selected_config_and_save_toggles(app):
    app.auto_speak_var.set(False)
    app.speak_on_space_var.set(False)
    app._on_conf_change("0.8")
    pump(app, .5)
    saved = yaml.safe_load(app.config_path.read_text(encoding="utf-8"))
    assert saved["asl"]["confidence_threshold"] == .8
    assert saved["sentence"]["speak_on_space"] is False
    assert saved["sentence"]["auto_speak"] is False


def test_commit_runs_on_main_thread_and_honors_length_limit(app):
    assert threading.current_thread() is threading.main_thread()
    app.max_sentence_length = 2
    assert app._commit("a")
    assert app._commit("b")
    assert not app._commit("c")
    assert app.text_buffer.text == "ab"
    assert app._commit(app.text_buffer.delete_label)
    assert app.text_buffer.text == "a"


@pytest.mark.parametrize("size", ["860x560", "1024x680", "1380x860"])
def test_preview_preserves_aspect_and_controls_fit(app, size):
    app.root.geometry(size)
    app.root.deiconify()
    pump(app)
    app._render_frame()
    photo = app._photo
    assert photo is not None
    assert abs(photo.width() / photo.height() - 640 / 480) < .02
    assert photo.width() <= app.cam_canvas.winfo_width()
    assert photo.height() <= app.cam_canvas.winfo_height()
    for widget in [app.btn_start, app.btn_stop, app.sent_text, *app.speech_buttons]:
        if widget in app.speech_buttons:
            widget.focus_force()
            pump(app, .02)
        assert widget.winfo_width() > 30
        assert widget.winfo_height() > 15
        assert widget.winfo_rooty() + widget.winfo_height() <= app.root.winfo_rooty() + app.root.winfo_height() + 2
