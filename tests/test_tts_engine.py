from pathlib import Path
import threading
from app.tts_engine import TTSEngine


def make_engine(tmp_path, **cfg):
    return TTSEngine(dict(output_dir=str(tmp_path), **cfg))


def test_voice_requires_matching_json(tmp_path):
    model = tmp_path / "voice.onnx"
    model.touch()
    engine = make_engine(tmp_path, model_path=str(model), autoplay=False)
    assert not engine.available
    Path(str(model) + ".json").write_text("{}")
    assert engine.available


def test_worker_reports_failure_and_cleans_audio(tmp_path, monkeypatch):
    engine = make_engine(tmp_path)
    wav = tmp_path / "speech.wav"
    def synth(text):
        wav.touch()
        return str(wav)
    def fail(path):
        raise RuntimeError("No audio device")
    monkeypatch.setattr(engine, "_synthesize", synth)
    monkeypatch.setattr(engine, "_play", fail)
    try:
        engine.speak("hello")
        engine.wait_idle()
        assert engine.last_error == "No audio device"
        assert engine.status == "Speech failed"
        assert not wav.exists()
        assert not engine.busy
    finally:
        engine.stop()


def test_cancel_keeps_worker_usable(tmp_path, monkeypatch):
    engine = make_engine(tmp_path)
    entered = threading.Event()
    seen = []
    def synth(text):
        seen.append(text)
        if text == "first":
            entered.set()
            assert engine._cancel.wait(2)
        return None
    monkeypatch.setattr(engine, "_synthesize", synth)
    engine.autoplay = False
    try:
        engine.speak("first")
        assert entered.wait(2)
        engine.speak("discard")
        engine.cancel()
        engine.wait_idle()
        engine.speak("next")
        engine.wait_idle()
        assert seen == ["first", "next"]
        assert engine.thread.is_alive()
    finally:
        engine.stop()
    assert not engine.thread.is_alive()


def test_stop_without_start_does_not_leave_unfinished_tasks(tmp_path):
    engine = make_engine(tmp_path)
    engine.stop()
    engine.stop()
    engine.speak("ignored")
    assert not engine.busy
