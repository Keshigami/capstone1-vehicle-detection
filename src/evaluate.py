"""Compute validation metrics for a trained detector; also refreshes models/metrics.json on its own:

uv run python -m src.evaluate          # re-evaluates models/detector.keras on the quick split
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import CLASSES, IMAGE_SIZE, IMAGES_DIR, MODELS_DIR, Dataset, build_dataset, load_labels
from .model import box_iou, cxcywh_to_corners


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s)) if f1s else 0.0


def all_boxes_normalised(image_ids: list[str]) -> dict[str, np.ndarray]:
    """Every labelled box per image, normalised by that image's size."""
    labels = load_labels()
    labels = labels[labels["image_id"].isin(set(image_ids))]
    out: dict[str, np.ndarray] = {}
    for image_id, grp in labels.groupby("image_id"):
        with Image.open(IMAGES_DIR / f"{image_id}.jpg") as im:
            w, h = im.size
        out[image_id] = (grp[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32) / [w, h, w, h]).astype(np.float32)
    return out


def compute_metrics(model, ds: Dataset, tr: np.ndarray, va: np.ndarray) -> dict:
    pred = model.predict(ds.images[va], batch_size=64, verbose=0)
    pred_cls = pred["vehicle_class"].argmax(axis=1)
    pred_box = cxcywh_to_corners(pred["bounding_box"])
    y_va = ds.labels[va]
    ious = box_iou(pred_box, ds.boxes[va])
    mae_norm = float(np.abs(pred_box - ds.boxes[va]).mean())

    majority = int(np.bincount(ds.labels[tr], minlength=len(CLASSES)).argmax())
    mean_box = ds.boxes[tr].mean(axis=0)
    baseline_iou = float(box_iou(np.tile(mean_box, (len(va), 1)), ds.boxes[va]).mean())

    # IoU against whichever labelled vehicle the predicted box overlaps most: the target is
    # "the largest box", but a box around another real vehicle in the frame is not a miss.
    va_ids = [ds.image_ids[i] for i in va]
    boxes_by_image = all_boxes_normalised(va_ids)
    best = np.array(
        [
            box_iou(np.tile(pred_box[k], (len(boxes_by_image[i]), 1)), boxes_by_image[i]).max()
            for k, i in enumerate(va_ids)
        ]
    )

    per_class = {}
    for i, name in enumerate(CLASSES):
        mask = y_va == i
        if mask.any():
            per_class[name] = {"val_images": int(mask.sum()), "recall": round(float((pred_cls[mask] == i).mean()), 3)}
    return {
        "val_accuracy": round(float((pred_cls == y_va).mean()), 4),
        "val_majority_baseline_accuracy": round(float((y_va == majority).mean()), 4),
        "val_macro_f1": round(macro_f1(y_va, pred_cls, len(CLASSES)), 4),
        "val_box_mean_iou": round(float(ious.mean()), 4),
        "val_mean_box_baseline_iou": round(baseline_iou, 4),
        "val_box_iou_at_0.5": round(float((ious >= 0.5).mean()), 4),
        "val_box_best_match_mean_iou": round(float(best.mean()), 4),
        "val_box_best_match_iou_at_0.5": round(float((best >= 0.5).mean()), 4),
        "val_box_mae_normalised": round(mae_norm, 4),
        "val_box_mae_px_at_224": round(mae_norm * IMAGE_SIZE, 2),
        "per_class": per_class,
    }


def main(metrics_path: Path = MODELS_DIR / "metrics.json") -> None:
    import tensorflow as tf

    from .train import train_val_split

    metrics = json.loads(metrics_path.read_text())
    model = tf.keras.models.load_model(MODELS_DIR / "detector.keras", compile=False)
    ds = build_dataset(None if metrics["mode"] == "full" else metrics["images_used"], seed=42)
    tr, va = train_val_split(len(ds.labels), seed=42)
    metrics.update(compute_metrics(model, ds, tr, va))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("history", "per_class")}, indent=2))


if __name__ == "__main__":
    main()
