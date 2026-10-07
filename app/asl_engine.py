import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


logger = logging.getLogger(__name__)


DEFAULT_LABELS: List[str] = [chr(ord("a") + i) for i in range(26)] + [
    "space",
    "delete",
    "nothing",
]


class ASLEngine:
    def __init__(self, cfg: Dict[str, Any]):
        if cv2 is None:
            raise ImportError("opencv-python is required.")

        if mp is None:
            raise ImportError("mediapipe is required.")

        self.cfg = cfg or {}
        self.labels = self._load_labels()
        self.nothing_label = str(
            self.cfg.get("labels", {}).get("nothing_label", "nothing")
        )

        self.model_path = str(self.cfg.get("model_path", ""))
        self.model_dir = (
            os.path.dirname(os.path.abspath(self.model_path))
            if self.model_path
            else "models/asl"
        )
        self.preprocess_cfg = self._load_preprocessor_config()

        self.dummy = not (self.model_path and os.path.exists(self.model_path))

        self.session = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[List[Any]] = None
        self._warned_label_mismatch = False

        if not self.dummy:
            if ort is None:
                raise ImportError(
                    "onnxruntime is required when using a real ASL ONNX model."
                )

            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )

            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape

            logger.info(
                "ASL ONNX input: %s shape=%s",
                self.input_name,
                self.input_shape,
            )
        else:
            logger.warning(
                "ASL model not found at '%s'. Running in dummy mode. "
                "Place a pretrained ONNX model there to enable real recognition.",
                self.model_path,
            )

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=int(self.cfg.get("max_hands", 1)),
            min_detection_confidence=float(
                self.cfg.get("min_detection_confidence", 0.6)
            ),
            min_tracking_confidence=float(self.cfg.get("min_tracking_confidence", 0.6)),
        )

        self.draw_landmarks = bool(self.cfg.get("draw_landmarks", True))

    def _load_labels(self) -> List[str]:
        labels_path = str(self.cfg.get("labels_path", ""))

        if labels_path and os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as handle:
                labels = [line.strip() for line in handle if line.strip()]

            if not labels:
                raise ValueError(f"Labels file is empty: {labels_path}")

            return labels

        model_path = str(self.cfg.get("model_path", ""))

        if not model_path or not os.path.exists(model_path):
            return DEFAULT_LABELS.copy()

        raise FileNotFoundError(
            f"ASL labels file not found: {labels_path}. "
            "Create labels.txt with one class name per line."
        )

    def _load_preprocessor_config(self) -> Dict[str, Any]:
        candidates = []

        if hasattr(self, "model_dir"):
            candidates.append(os.path.join(self.model_dir, "preprocessor_config.json"))

        candidates.append("models/asl/preprocessor_config.json")

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        return json.load(handle)
                except Exception:
                    logger.warning("Could not parse preprocessor config: %s", path)

        return {}

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return {
                "label": self.nothing_label,
                "confidence": 0.0,
                "hand_present": False,
            }

        hand = result.multi_hand_landmarks[0]

        if self.draw_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

        if self.dummy:
            return {
                "label": self.nothing_label,
                "confidence": 0.0,
                "hand_present": True,
            }

        features = self._features(frame, hand)
        probabilities = self._infer(features)

        class_index = int(np.argmax(probabilities[0]))
        confidence = float(probabilities[0][class_index])

        if class_index < len(self.labels):
            label = self.labels[class_index]
        else:
            label = str(class_index)

        return {
            "label": label,
            "confidence": confidence,
            "hand_present": True,
        }

    def _features(self, frame: np.ndarray, hand: Any) -> np.ndarray:
        input_type = str(self.cfg.get("input_type", "landmarks")).lower()

        if input_type == "landmarks":
            return self._landmark_features(hand)

        if input_type == "image":
            return self._image_features(frame, hand)

        raise ValueError("asl.input_type must be either 'landmarks' or 'image'.")

    def _landmark_features(self, hand: Any) -> np.ndarray:
        points = []
        for landmark in hand.landmark:
            points.append([landmark.x, landmark.y, landmark.z])

        points_array = np.array(points, dtype=np.float32)
        n_landmarks = points_array.shape[0]

        diff = points_array[:, None, :] - points_array[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        i, j = np.triu_indices(n_landmarks, k=1)
        features = dist_matrix[i, j].astype(np.float32).reshape(1, -1)

        if not getattr(self, "_features_debug_printed", False):
            logger.info(
                "ASL feature vector shape: %s, ONNX input shape: %s",
                features.shape,
                self.input_shape,
            )
            self._features_debug_printed = True

        return features

    def _expected_landmark_size(self) -> Optional[int]:
        if not self.input_shape:
            return None

        if len(self.input_shape) == 2:
            dimension = self.input_shape[1]

            if isinstance(dimension, int) and dimension > 0:
                return dimension

        if len(self.input_shape) == 3:
            dim_a = self.input_shape[1]
            dim_b = self.input_shape[2]

            if isinstance(dim_a, int) and isinstance(dim_b, int):
                if {dim_a, dim_b} == {21, 3}:
                    return 63

        return None

    def _image_features(self, frame: np.ndarray, hand: Any) -> np.ndarray:
        """
        Replicate the training image format exactly:
        1. Square crop around the hand with margin
        2. Draw WHITE connection lines over the crop
        3. Draw RED filled circles at each landmark joint
        4. Draw PURPLE rectangle border around the crop
        5. Resize to 64x64, convert BGR->RGB, normalise [0,1], flatten
        """
        input_size = int(self.cfg.get("image_input_size", 64))
        height, width = frame.shape[:2]

        # ── 1. Compute square crop around hand ───────────────────────────────
        xs = np.array([lm.x for lm in hand.landmark], dtype=np.float32) * width
        ys = np.array([lm.y for lm in hand.landmark], dtype=np.float32) * height

        margin = float(self.cfg.get("image_margin", 0.25))
        x1 = int(max(0, xs.min() - margin * width))
        y1 = int(max(0, ys.min() - margin * height))
        x2 = int(min(width, xs.max() + margin * width))
        y2 = int(min(height, ys.max() + margin * height))

        sq = max(max(1, x2 - x1), max(1, y2 - y1))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1 = max(0, cx - sq // 2)
        y1 = max(0, cy - sq // 2)
        x2 = min(width, x1 + sq)
        y2 = min(height, y1 + sq)

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            img = np.zeros((input_size, input_size, 3), dtype=np.float32)
            return img.flatten().reshape(1, -1).astype(np.float32)

        ch, cw = crop.shape[:2]

        # Convert landmark coords to crop-local pixel coords
        def to_crop(lm):
            px = int(lm.x * width) - x1
            py = int(lm.y * height) - y1
            px = max(0, min(cw - 1, px))
            py = max(0, min(ch - 1, py))
            return (px, py)

        # ── 2. Draw WHITE connection lines ────────────────────────────────────
        for conn in mp.solutions.hands.HAND_CONNECTIONS:
            pt1 = to_crop(hand.landmark[conn[0]])
            pt2 = to_crop(hand.landmark[conn[1]])
            cv2.line(crop, pt1, pt2, (255, 255, 255), 2)

        # ── 3. Draw RED dots at each landmark ─────────────────────────────────
        for lm in hand.landmark:
            pt = to_crop(lm)
            cv2.circle(crop, pt, 4, (0, 0, 255), -1)

        # ── 4. Draw PURPLE border ─────────────────────────────────────────────
        cv2.rectangle(crop, (0, 0), (cw - 1, ch - 1), (255, 0, 255), 3)

        # ── 5. Resize → RGB → normalise → flatten ─────────────────────────────
        img = cv2.resize(crop, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        return img.flatten().reshape(1, -1).astype(np.float32)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        mode = str(self.cfg.get("image_normalization", "auto")).lower()
        pre = getattr(self, "preprocess_cfg", {}) or {}

        default_mean = [0.485, 0.456, 0.406]
        default_std = [0.229, 0.224, 0.225]

        mean = pre.get("image_mean", default_mean)
        std = pre.get("image_std", default_std)
        rescale_factor = float(pre.get("rescale_factor", 1.0 / 255.0))

        def to_unit(img: np.ndarray) -> np.ndarray:
            return img / 255.0

        def normalize_imagenet(img: np.ndarray) -> np.ndarray:
            img = to_unit(img)
            mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
            std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
            return (img - mean_arr) / (std_arr + 1e-9)

        if mode == "none":
            return image

        if mode == "unit":
            return to_unit(image)

        if mode == "imagenet":
            return normalize_imagenet(image)

        # Auto mode.
        if pre:
            if pre.get("do_rescale", True):
                image = image * rescale_factor
            else:
                if float(image.max()) > 1.5:
                    image = to_unit(image)

            if pre.get("do_normalize", False):
                mean_arr = np.array(
                    pre.get("image_mean", default_mean),
                    dtype=np.float32,
                ).reshape(1, 1, 3)

                std_arr = np.array(
                    pre.get("image_std", default_std),
                    dtype=np.float32,
                ).reshape(1, 1, 3)

                image = (image - mean_arr) / (std_arr + 1e-9)

            return image

        return to_unit(image)

    def _infer(self, features: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            None,
            {self.input_name: features.astype(np.float32)},
        )

        # Accept sklearn ZipMap exports and classifiers with dense scores.
        scores = outputs[1] if len(outputs) > 1 else outputs[0]
        if isinstance(scores, list) and scores and isinstance(scores[0], dict):
            probabilities = np.zeros((len(scores), len(self.labels)), dtype=np.float32)
            for row, probability_map in enumerate(scores):
                for key, probability in probability_map.items():
                    if isinstance(key, str):
                        if key not in self.labels:
                            raise ValueError(f"Unknown ONNX class label: {key}")
                        index = self.labels.index(key)
                    else:
                        index = int(key)
                    if not 0 <= index < len(self.labels):
                        raise ValueError(f"ONNX class index out of range: {index}")
                    probabilities[row, index] = float(probability)
        else:
            probabilities = np.asarray(scores, dtype=np.float32)
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(1, -1)
        if probabilities.shape != (features.shape[0], len(self.labels)):
            raise ValueError("ONNX output shape does not match labels.txt.")
        if not np.isfinite(probabilities).all():
            raise ValueError("ONNX output contains non-finite scores.")
        if (probabilities < 0).any() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-3
        ):
            probabilities = self._softmax(probabilities)
        return probabilities

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        exp_values = np.exp(values - np.max(values, axis=-1, keepdims=True))
        return exp_values / (np.sum(exp_values, axis=-1, keepdims=True) + 1e-9)

    def close(self) -> None:
        try:
            self.hands.close()
        except Exception:
            pass
