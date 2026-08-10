# Project Structure

```text
asl-translator/
│
├── main.py                     # Application entry point
├── config.yaml                 # Runtime configuration
├── requirements.txt            # Python dependencies
├── prd.md                      # Product requirements document
├── instructions.md             # Setup and usage instructions
├── workflow.md                 # Runtime and development workflow
├── work-phases.md              # Phase plan and exit gates
├── project-structure.md        # This file
│
├── app/
│   ├── __init__.py
│   ├── config.py               # Config loading and defaults
│   ├── asl_engine.py           # MediaPipe + ONNX ASL inference
│   ├── tts_engine.py           # Piper TTS background worker
│   └── text_buffer.py          # Sentence building and auto commit
│
├── scripts/
│   ├── download_models.py      # Download TTS model and scaffold ASL labels
│   └── check_env.py            # Environment and model checker
│
├── models/
│   ├── asl/
│   │   ├── asl_landmarks.onnx  # Your pretrained ASL ONNX model
│   │   ├── labels.txt          # Label order matching model output
│   │   └── README.md           # Model contract notes
│   │
│   └── tts/
│       ├── en_US-lessac-low.onnx
│       └── en_US-lessac-low.onnx.json
│
├── tests/
│   └── test_text_buffer.py     # Unit tests for sentence logic
│
└── .cache/
    └── tts/                    # Temporary generated speech files
```

## File Responsibilities

### `main.py`

Starts the application.

Responsibilities:

- parse arguments,
- load config,
- start TTS worker,
- start ASL engine,
- run camera loop,
- handle keyboard input,
- draw HUD.

### `app/config.py`

Loads `config.yaml` and merges with defaults.

### `app/asl_engine.py`

Handles:

- MediaPipe hand detection,
- landmark normalization,
- ONNX inference,
- label mapping,
- dummy mode when ASL model is missing.

### `app/tts_engine.py`

Handles:

- Piper TTS synthesis,
- background speech queue,
- WAV playback,
- temporary file cleanup.

### `app/text_buffer.py`

Handles:

- sentence construction,
- space handling,
- delete handling,
- clear,
- auto commit stabilization.

### `scripts/download_models.py`

Downloads TTS model and creates ASL label scaffolding.

### `scripts/check_env.py`

Checks:

- Python version,
- dependencies,
- Piper availability,
- TTS model,
- ASL model,
- optional camera read.

### `models/asl/`

Stores your pretrained ASL model and labels.

### `models/tts/`

Stores the pretrained Piper TTS voice.


Create only this file structure with all it's files , empty in a zip folder 

All files should be empty , no code