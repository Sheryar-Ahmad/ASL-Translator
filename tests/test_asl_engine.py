from types import SimpleNamespace
import numpy as np
import pytest
from app.asl_engine import ASLEngine


def engine_for(outputs):
    engine = ASLEngine.__new__(ASLEngine)
    engine.labels = ["a", "b"]
    engine.input_name = "input"
    engine.session = SimpleNamespace(run=lambda *args: outputs)
    return engine


@pytest.mark.parametrize("outputs", [
    [np.array([1]), [{0: .2, 1: .8}]],
    [np.array(["b"]), [{"a": .2, "b": .8}]],
    [np.array([1]), np.array([[.2, .8]])],
    [np.array([[.2, .8]])],
])
def test_supported_onnx_outputs(outputs):
    probs = engine_for(outputs)._infer(np.zeros((1, 210)))
    np.testing.assert_allclose(probs, [[.2, .8]])


def test_logits_are_normalized():
    probs = engine_for([np.array([[-2., 2.]])])._infer(np.zeros((1, 210)))
    assert probs.argmax() == 1
    np.testing.assert_allclose(probs.sum(), 1)


@pytest.mark.parametrize("outputs", [[np.array([[.1, .2, .7]])], [np.array([[np.nan, .2]])],
                                      [None, [{99: 1.0}]]])
def test_reject_incompatible_outputs(outputs):
    with pytest.raises(ValueError):
        engine_for(outputs)._infer(np.zeros((1, 210)))


def test_missing_model_never_fabricates_a_letter():
    engine = ASLEngine.__new__(ASLEngine)
    engine.hands = SimpleNamespace(process=lambda image: SimpleNamespace(multi_hand_landmarks=[object()]))
    engine.draw_landmarks = False
    engine.dummy = True
    engine.nothing_label = "nothing"
    prediction = engine.predict(np.zeros((20, 20, 3), dtype=np.uint8))
    assert prediction == {"label": "nothing", "confidence": 0., "hand_present": True}
