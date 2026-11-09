"""Load the trained Keras model once and run it on PIL images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASSES, IMAGE_SIZE, MODELS_DIR
from .model import cxcywh_to_corners

MODEL_PATH = MODELS_DIR / "detector.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"


class Detector:
    def __init__(self, model_path: Path = MODEL_PATH):
        import tensorflow as tf  # imported lazily so the app starts fast without a model

        self.model = tf.keras.models.load_model(model_path, compile=False)
        classes_file = model_path.parent / "classes.json"
        self.classes = json.loads(classes_file.read_text()) if classes_file.exists() else CLASSES

    @classmethod
    def load_if_available(cls) -> Detector | None:
        return cls() if MODEL_PATH.exists() else None

    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        w, h = image.size
        arr = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        out = self.model.predict(arr[None], verbose=0)
        probs = out["vehicle_class"][0]
        x1, y1, x2, y2 = (float(v) for v in cxcywh_to_corners(out["bounding_box"])[0])
        idx = int(probs.argmax())
        top = sorted(((self.classes[i], float(p)) for i, p in enumerate(probs)), key=lambda t: -t[1])[:3]
        return {
            "class": self.classes[idx],
            "confidence": round(float(probs[idx]), 4),
            "box": [round(x1 * w, 1), round(y1 * h, 1), round(x2 * w, 1), round(y2 * h, 1)],
            "box_normalized": [round(v, 4) for v in (x1, y1, x2, y2)],
            "image_size": [w, h],
            "top3": [{"class": c, "confidence": round(p, 4)} for c, p in top],
        }


def load_metrics() -> dict | None:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
