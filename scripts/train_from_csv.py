#!/usr/bin/env python3
"""
Train an ASL classifier directly from sign_data.csv.

CSV format:

Distance_0 ... Distance_209 Sign

Example:

Distance_0,Distance_1,...,Distance_209,Sign
0.073,0.122,...,0.042,A

Usage:

python scripts/train_from_csv.py --csv archive/sign_data.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models/asl")
MODEL_OUT = MODELS_DIR / "asl_landmarks.onnx"
LABELS_OUT = MODELS_DIR / "labels.txt"
CONFIG_PATH = Path("config.yaml")


def load_csv(csv_path: Path):
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pip install pandas")

    df = pd.read_csv(csv_path)

    if "Sign" not in df.columns:
        sys.exit("CSV must contain a 'Sign' column.")

    feature_columns = [c for c in df.columns if c != "Sign"]

    X = df[feature_columns].astype(np.float32).values
    y = df["Sign"].astype(str).str.strip().str.lower().values

    print("=" * 60)
    print("ASL Landmark Distance Trainer")
    print("=" * 60)
    print(f"CSV File : {csv_path}")
    print(f"Samples  : {len(X)}")
    print(f"Features : {X.shape[1]}")
    print()

    return X, y


def train(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_enc,
        test_size=0.15,
        random_state=42,
        stratify=y_enc,
    )

    print(f"Training on {len(X_train)} samples")
    print(f"Validating on {len(X_val)} samples\n")

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True,
    )

    clf.fit(X_train, y_train)

    acc = clf.score(X_val, y_val)

    print(f"\nValidation Accuracy: {acc * 100:.2f}%")

    return clf, le


def export_onnx(clf, le, feature_count):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        sys.exit("pip install skl2onnx")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    onnx_model = convert_sklearn(
        clf,
        initial_types=[("float_input", FloatTensorType([None, feature_count]))],
        target_opset=17,
    )

    with open(MODEL_OUT, "wb") as f:
        f.write(onnx_model.SerializeToString())

    labels = list(le.classes_)

    LABELS_OUT.write_text(
        "\n".join(labels),
        encoding="utf-8",
    )

    print("\nSaved model  :", MODEL_OUT)
    print("Saved labels :", LABELS_OUT)
    print("Classes      :", labels)

    return labels


def patch_config():
    try:
        import yaml
    except ImportError:
        return

    if not CONFIG_PATH.exists():
        return

    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}

    asl = cfg.setdefault("asl", {})

    asl["input_type"] = "landmarks"
    asl["feature_type"] = "pairwise_distance"
    asl["feature_size"] = 210
    asl["model_path"] = str(MODEL_OUT)
    asl["labels_path"] = str(LABELS_OUT)

    CONFIG_PATH.write_text(
        yaml.dump(cfg, allow_unicode=True),
        encoding="utf-8",
    )

    print("\nconfig.yaml updated.")


def verify(feature_count, labels):
    try:
        import onnxruntime as ort
    except ImportError:
        return

    session = ort.InferenceSession(
        str(MODEL_OUT),
        providers=["CPUExecutionProvider"],
    )

    dummy = np.zeros((1, feature_count), dtype=np.float32)

    outputs = session.run(
        None,
        {session.get_inputs()[0].name: dummy},
    )

    probs = outputs[1][0]
    best = max(probs.items(), key=lambda x: x[1])

    print(f"ONNX verification OK -> {labels[best[0]]} ({best[1]:.3f})")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="archive/sign_data.csv",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    X, y = load_csv(csv_path)

    clf, le = train(X, y)

    labels = export_onnx(
        clf,
        le,
        X.shape[1],
    )

    patch_config()

    verify(
        X.shape[1],
        labels,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
