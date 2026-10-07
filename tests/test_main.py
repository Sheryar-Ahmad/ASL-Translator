from types import SimpleNamespace
import sys
import pytest
import main
from app import gui_app


def test_cli_speech_failure_propagates(monkeypatch):
    calls = []
    tts = SimpleNamespace(start=lambda: calls.append("start"), speak=lambda text: None,
                          wait_idle=lambda: None, stop=lambda: calls.append("stop"),
                          last_error="Synthesis failed")
    monkeypatch.setattr(main, "TTSEngine", lambda cfg: tts)
    with pytest.raises(RuntimeError, match="Synthesis failed"):
        main.run_speak({}, "hello")
    assert calls == ["start", "stop"]


@pytest.mark.parametrize("failure", ["camera", "window"])
def test_camera_initialization_failure_closes_resources(monkeypatch, failure):
    calls = []
    tts = SimpleNamespace(stop=lambda: calls.append("tts-stop"))
    asl = SimpleNamespace(close=lambda: calls.append("asl-close"))
    capture = SimpleNamespace(isOpened=lambda: failure != "camera", set=lambda *args: None,
                              release=lambda: calls.append("capture-release"))
    monkeypatch.setattr(main, "TTSEngine", lambda cfg: tts)
    monkeypatch.setattr(main, "ASLEngine", lambda cfg: asl)
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda *args: capture)
    def fail_window(*args):
        raise RuntimeError("Window unavailable")
    monkeypatch.setattr(main.cv2, "namedWindow", fail_window)
    with pytest.raises(RuntimeError):
        main.run_camera({})
    assert set(calls) == {"capture-release", "asl-close", "tts-stop"}


def test_gui_receives_custom_config_path(monkeypatch):
    calls = []
    cfg = {"camera": {"index": 2}}
    monkeypatch.setattr(sys, "argv", ["main.py", "--gui", "--config", "custom.yaml"])
    monkeypatch.setattr(main, "load_config", lambda path: cfg)
    monkeypatch.setattr(gui_app, "run_gui", lambda config, path: calls.append((config, path)))
    main.main()
    assert calls == [(cfg, "custom.yaml")]
