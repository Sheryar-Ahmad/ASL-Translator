from app.text_buffer import AutoCommitController, TextBuffer


def test_commit_letters():
    buffer = TextBuffer()

    buffer.commit_label("a")
    buffer.commit_label("b")
    buffer.commit_label("c")

    assert buffer.text == "abc"


def test_space_and_delete():
    buffer = TextBuffer()

    buffer.commit_label("h")
    buffer.commit_label("i")
    buffer.commit_label("space")
    buffer.commit_label("o")
    buffer.commit_label("k")

    assert buffer.text == "hi ok"

    buffer.backspace()
    assert buffer.text == "hi o"

    buffer.commit_label("delete")
    assert buffer.text == "hi"


def test_clear():
    buffer = TextBuffer()

    buffer.commit_label("a")
    buffer.clear()

    assert buffer.text == ""


def test_word_label_appends_word_and_space():
    buffer = TextBuffer()

    buffer.commit_label("hello")
    buffer.commit_label("world")

    assert buffer.text == "hello world"


def test_auto_commit_controller():
    cfg = {
        "confidence_threshold": 0.5,
        "labels": {
            "nothing_label": "nothing",
        },
        "auto": {
            "vote_window": 5,
            "stable_ms": 100,
            "reset_ms": 100,
        },
    }

    controller = AutoCommitController(cfg)

    assert controller.update("a", 0.9, 0) is None
    assert controller.update("a", 0.9, 200) == "a"

    controller.update("nothing", 0.0, 400)
    assert controller.update("b", 0.9, 600) is None
    assert controller.update("b", 0.9, 750) is None
    assert controller.update("b", 0.9, 1000) == "b"
