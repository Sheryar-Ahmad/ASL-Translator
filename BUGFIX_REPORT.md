# Project audit and fixes

The initial suite passed five text-buffer tests. This audit found the following additional issues through code inspection and regression checks.

| Issue | Fix and validation |
| --- | --- |
| Fixed 1200 × 780 minimum window, tiny settings, stretched preview | Flexible camera/sentence grid, readable settings tabs with scrolling and keyboard access; real Tk layout checks at 860 × 560, 1024 × 680 and 1380 × 860; aspect-ratio assertions. |
| Capture released while another thread reads it | Camera worker exclusively opens, reads, switches and releases capture; switching and teardown tests assert no concurrent release. |
| Text buffer modified by both UI and worker | Worker sends only the latest frame/prediction through a bounded queue; all voting, commits and text actions run on Tk's thread. |
| Stale votes survive pause/start/clear | Controller reset and recognition epochs reject obsolete predictions. |
| Missing model produces fabricated A predictions | Missing models return the configured nothing label at zero confidence; GUI disables recognition until a model is available. |
| Duplicate/empty spaces trigger speech | Spaces report a commit only when a separator is actually inserted. |
| Settings ignore --config and toggle changes | Selected config path is forwarded; toggles and sliders save with a 350 ms debounce; configuration defaults are copied rather than shared. |
| Sentence length setting unused; correction control absent | Enforce the configured maximum for new labels; add Delete and keyboard correction. |
| ONNX inference assumes integer-keyed ZipMap only | Support integer/string ZipMap, dense probabilities and logits; reject bad shapes, non-finite values and unknown classes. |
| Voice download omits Piper JSON configuration | Download both .onnx and .onnx.json; show unavailable voice assets in the GUI. |
| Speech errors hidden; cache WAVs leak after playback failure | Publish worker status/error; clean playback WAVs on success and failure; CLI speech failures propagate as errors. |
| No usable speech cancellation | Cancel terminates synthesis/playback and discards queued speech while preserving the reusable worker; stop is idempotent. |
| Dependency/setup inconsistencies | Declare Pillow in pyproject, pin legacy-compatible MediaPipe in both install paths, restrict the project to Python 3.12, refresh uv.lock and correct README setup commands. |
| CLI camera initialization leaks resources on failure | Close the camera and recognition engine when opening the camera/window fails; correct default capture height and use configured space labels. |

## Validation

- 35 regression tests, including real Tk layout and worker lifecycle checks.
- Real bundled ONNX model inference plus MediaPipe processing of a no-hand image.
- Real Piper synthesis: 73,984 WAV frames at 22,050 Hz; temporary audio removed afterward.
- Dependency lock consistency and Git whitespace checks.

## Limits

Camera lifecycle tests use simulated cameras; live signing accuracy and physical speaker playback need a check on the user's hardware. Screen capture is unavailable in the test environment, so layout verification uses actual Tk geometry and widget visibility rather than screenshots. MediaPipe emits upstream protobuf deprecation warnings on Python 3.12. This audit does not establish recognition accuracy for arbitrary signers or ASL vocabulary beyond the trained labels.
