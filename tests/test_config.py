import pytest
from app.config import DEFAULTS, deep_merge, load_config


def test_missing_config_is_independent(tmp_path):
    first = load_config(str(tmp_path / "missing.yaml"))
    first["camera"]["index"] = 99
    assert load_config(str(tmp_path / "missing.yaml"))["camera"]["index"] == DEFAULTS["camera"]["index"]


def test_merge_does_not_alias_inputs():
    override = {"extra": {"values": [1]}}
    merged = deep_merge(DEFAULTS, override)
    merged["extra"]["values"].append(2)
    merged["tts"]["autoplay"] = False
    assert override["extra"]["values"] == [1]
    assert DEFAULTS["tts"]["autoplay"] is True


def test_reject_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- camera", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(str(path))
