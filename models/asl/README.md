# ASL model

The application uses these versioned files together:

- `asl_landmarks.onnx`: the trained ONNX classifier.
- `labels.txt`: one label per line, in model class-index order.

The bundled model takes float32 input of shape `[batch, 210]`. Each row contains the pairwise Euclidean distances between the 21 MediaPipe hand landmarks using x, y and z coordinates. The configured runtime processes one hand per frame.

To retrain from the included dataset, run this from the repository root:

```bash
uv run scripts/train_from_csv.py --csv archive/sign_data.csv
```

Training exports the model and labels into this directory and updates `config.yaml`. Replace both files together when changing models. Missing models disable GUI recognition.

`scripts/download_models.py` downloads the Piper voice into `models/tts/`; it does not download or train the ASL classifier.
