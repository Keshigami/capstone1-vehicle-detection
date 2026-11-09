"""Train the detector.

  uv run python -m src.train --quick   # 4,500 images, 3 + 12 epochs, about 7 minutes on an M4 Max (CPU)
  uv run python -m src.train --full    # all 5,626 images on disk, 3 + 30 epochs with early stopping
  add --backbone custom for the notebook's from-scratch CNN

Writes models/detector.keras, models/classes.json and models/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime

import numpy as np
import tensorflow as tf

from .data import CLASSES, IMAGE_SIZE, MODELS_DIR, build_dataset
from .evaluate import compute_metrics
from .model import build_model, compile_model, set_backbone_trainable

# epochs = (frozen-backbone epochs, fine-tuning epochs); for the custom backbone both phases just train
QUICK = {"images": 4500, "epochs": (3, 12)}
FULL = {"images": None, "epochs": (3, 30)}


def train_val_split(n: int, val_fraction: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = round(n * val_fraction)
    return idx[n_val:], idx[:n_val]


def flip_augment(
    images: np.ndarray, labels: np.ndarray, boxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append a left-right mirrored copy of every image; x1' = 1 - x2, x2' = 1 - x1."""
    flipped = images[:, :, ::-1, :]
    fb = boxes.copy()
    fb[:, 0], fb[:, 2] = 1.0 - boxes[:, 2], 1.0 - boxes[:, 0]
    return np.concatenate([images, flipped]), np.concatenate([labels, labels]), np.concatenate([boxes, fb])


def class_sample_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """sqrt inverse-frequency weights, capped at 5: lifts minority classes without wrecking accuracy."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = np.sqrt(counts.max() / np.maximum(counts, 1))
    w = np.minimum(w, 5.0)
    return w[labels].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="subsample, few epochs (default)")
    mode.add_argument("--full", action="store_true", help="all images on disk, 20 epochs")
    parser.add_argument("--backbone", choices=["mobilenetv2", "custom"], default="mobilenetv2")
    parser.add_argument("--images", type=int, help="override image count")
    parser.add_argument("--epochs", type=int, help="override fine-tuning epoch count")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = dict(FULL if args.full else QUICK)
    if args.images:
        cfg["images"] = args.images
    if args.epochs:
        cfg["epochs"] = (cfg["epochs"][0], args.epochs)
    warmup_epochs, tune_epochs = cfg["epochs"]
    mode_name = "full" if args.full else "quick"

    tf.keras.utils.set_random_seed(args.seed)
    t0 = time.time()
    print(f"[train] mode={mode_name} backbone={args.backbone} images={cfg['images'] or 'all'} epochs={cfg['epochs']}")
    ds = build_dataset(cfg["images"], seed=args.seed)
    print(f"[train] loaded {len(ds.labels)} images in {time.time() - t0:.1f}s")
    tr, va = train_val_split(len(ds.labels), seed=args.seed)

    model = build_model(args.backbone)
    x_tr, l_tr, b_tr = flip_augment(ds.images[tr], ds.labels[tr], ds.boxes[tr])
    y_tr = {"vehicle_class": l_tr, "bounding_box": b_tr}
    y_va = {"vehicle_class": ds.labels[va], "bounding_box": ds.boxes[va]}
    sw_tr = {"vehicle_class": class_sample_weights(l_tr, len(CLASSES)), "bounding_box": np.ones(len(l_tr), np.float32)}
    fit_kwargs = dict(sample_weight=sw_tr, validation_data=(ds.images[va], y_va), batch_size=args.batch_size, verbose=2)
    t_fit = time.time()
    history: dict[str, list[float]] = {}

    def record(h) -> None:
        for k, v in h.history.items():
            history.setdefault(k, []).extend(float(x) for x in v)

    if args.backbone == "mobilenetv2":
        # phase 1: frozen ImageNet features, train the two heads
        set_backbone_trainable(model, False)
        compile_model(model, lr=5e-4)
        record(model.fit(x_tr, y_tr, epochs=warmup_epochs, **fit_kwargs))
        # phase 2: unfreeze the top blocks at a low learning rate, keep the best epoch
        n = set_backbone_trainable(model, True, from_block=10)
        print(f"[train] fine-tuning {n} backbone layers")
        compile_model(model, lr=1e-4)
    else:
        compile_model(model, lr=1e-3)
        tune_epochs = warmup_epochs + tune_epochs
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ]
    record(model.fit(x_tr, y_tr, epochs=tune_epochs, callbacks=callbacks, **fit_kwargs))
    fit_seconds = time.time() - t_fit
    epochs_run = len(history["loss"])

    evaluation = compute_metrics(model, ds, tr, va)

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "detector.keras")
    (MODELS_DIR / "classes.json").write_text(json.dumps(CLASSES, indent=2))
    metrics = {
        "mode": mode_name,
        "backbone": args.backbone,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": platform.machine() + " / " + platform.platform(),
        "tensorflow": tf.__version__,
        "images_used": len(ds.labels),
        "train_images": len(tr),
        "val_images": len(va),
        "epochs": epochs_run,
        "epochs_planned": [warmup_epochs, tune_epochs],
        "input_size": IMAGE_SIZE,
        "box_target": "largest box per image; model predicts normalised cx,cy,w,h, reported as x1,y1,x2,y2",
        **{k: v for k, v in evaluation.items() if k != "per_class"},
        "final_train_loss": round(history["loss"][-1], 4),
        "best_val_loss": round(min(history["val_loss"]), 4),
        "fit_seconds": round(fit_seconds, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "per_class": evaluation["per_class"],
        "history": {k: [round(x, 4) for x in v] for k, v in history.items()},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
