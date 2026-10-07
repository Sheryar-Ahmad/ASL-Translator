import sys
import pytest
import yaml
from scripts import check_env


@pytest.fixture
def configured_checker(tmp_path, monkeypatch):
    voice = tmp_path / "custom_voice.onnx"
    voice.touch()
    voice_json = tmp_path / "custom_voice.onnx.json"
    voice_json.write_text("{}")
    model = tmp_path / "classifier.onnx"
    model.touch()
    labels = tmp_path / "classes.txt"
    labels.write_text("a")
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump({
        "tts": {"model_path": str(voice)},
        "asl": {"model_path": str(model), "labels_path": str(labels)},
        "camera": {"index": 3},
    }))
    monkeypatch.setattr(sys, "argv", ["check_env.py", "--config", str(config)])
    monkeypatch.setattr(check_env, "check_import", lambda *a, **k: None)
    monkeypatch.setattr(check_env, "check_piper", lambda *a, **k: None)
    return voice_json


def test_checker_uses_configured_models(configured_checker):
    check_env.main()
    assert all(check["ok"] for check in check_env.CHECKS)
    check_env.main()
    assert len(check_env.CHECKS) == 6


def test_missing_voice_json_is_critical(configured_checker):
    configured_checker.unlink()
    with pytest.raises(SystemExit) as error:
        check_env.main()
    assert error.value.code == 1
    assert any(check["name"] == "TTS JSON config exists" and not check["ok"] and check["critical"]
               for check in check_env.CHECKS)


def test_unsupported_python_fails(configured_checker, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 13, 0))
    with pytest.raises(SystemExit):
        check_env.main()


def test_camera_check_uses_configured_index(configured_checker, monkeypatch):
    called = []
    monkeypatch.setattr(sys, "argv", sys.argv + ["--camera"])
    monkeypatch.setattr(check_env, "check_camera", lambda index: called.append(index))
    check_env.main()
    assert called == [3]
