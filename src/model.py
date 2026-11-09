"""Two detectors with the notebook's two-head shape (softmax class + 4-number box):

- "custom": the notebook's family, a small conv stack trained from scratch;
- "mobilenetv2" (default): an ImageNet-pretrained MobileNetV2 backbone with the same two heads,
  which is what makes a few minutes of CPU training produce a usable detector.

Differences from the notebook, each on purpose:
- inputs scaled to [0, 1] and boxes normalised to [0, 1] (the notebook regressed raw
  pixel coordinates against unscaled images, which is why its box MAE sat near 200 px);
- the box head predicts (cx, cy, w, h) through sigmoids, so a box is always well-formed and inside
  the image; corners are derived from it in the loss, the metrics and the detector;
- the box head has its own dense branch without dropout and an L1 loss, weighted up so it is
  not drowned by the classification loss;
- training mirrors every image left-right (with its box) to double the data.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import CLASSES, IMAGE_SIZE


def build_custom_model(num_classes: int = len(CLASSES), size: int = IMAGE_SIZE) -> keras.Model:
    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    features = layers.Flatten()(x)
    # class head: dense + dropout, as in the notebook
    c = layers.Dense(128, activation="relu")(features)
    c = layers.BatchNormalization()(c)
    c = layers.Dropout(0.5)(c)
    vehicle_class = layers.Dense(num_classes, activation="softmax", name="vehicle_class")(c)
    # box head: its own dense branch, no dropout (dropout adds noise to a regression target)
    b = layers.Dense(128, activation="relu")(features)
    b = layers.BatchNormalization()(b)
    bounding_box = layers.Dense(4, activation="sigmoid", name="bounding_box")(b)  # cx, cy, w, h
    return keras.Model(inputs=inputs, outputs={"vehicle_class": vehicle_class, "bounding_box": bounding_box})


def build_mobilenet_model(num_classes: int = len(CLASSES), size: int = IMAGE_SIZE) -> keras.Model:
    """ImageNet MobileNetV2 features (7x7x1280 at 224 px) with a pooled class head and a
    spatial box head (1x1 conv to 64 channels, flatten, dense) so the box keeps position info."""
    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)  # [0,1] -> [-1,1], MobileNetV2's expected range
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(size, size, 3))
    feats = base(x)  # layer name "mobilenetv2_1.00_224"; set_backbone_trainable finds it by that prefix
    c = layers.GlobalAveragePooling2D()(feats)
    c = layers.Dropout(0.3)(c)
    vehicle_class = layers.Dense(num_classes, activation="softmax", name="vehicle_class")(c)
    b = layers.Conv2D(64, 1, use_bias=False)(feats)
    b = layers.BatchNormalization()(b)
    b = layers.ReLU()(b)
    b = layers.Flatten()(b)
    b = layers.Dense(256, use_bias=False)(b)
    b = layers.BatchNormalization()(b)
    b = layers.ReLU()(b)
    b = layers.Dropout(0.2)(b)
    bounding_box = layers.Dense(4, activation="sigmoid", name="bounding_box")(b)  # cx, cy, w, h
    return keras.Model(inputs=inputs, outputs={"vehicle_class": vehicle_class, "bounding_box": bounding_box})


def build_model(backbone: str = "mobilenetv2", num_classes: int = len(CLASSES), size: int = IMAGE_SIZE) -> keras.Model:
    if backbone == "custom":
        return build_custom_model(num_classes, size)
    if backbone == "mobilenetv2":
        return build_mobilenet_model(num_classes, size)
    raise ValueError(f"unknown backbone {backbone}")


def set_backbone_trainable(model: keras.Model, trainable: bool, from_block: int = 13) -> int:
    """Freeze or unfreeze the MobileNetV2 backbone. When unfreezing, only blocks >= from_block
    train and BatchNorm layers stay frozen (small batches would wreck their statistics).
    Returns the number of trainable backbone layers. No-op for the custom model."""
    bases = [layer for layer in model.layers if isinstance(layer, keras.Model) and layer.name.startswith("mobilenetv2")]
    if not bases:
        return 0
    base = bases[0]
    base.trainable = trainable
    if not trainable:
        return 0
    n = 0
    for layer in base.layers:
        name = layer.name
        block = int(name.split("_")[1]) if name.startswith("block_") and name.split("_")[1].isdigit() else -1
        is_top = name.startswith("Conv_1") or name.startswith("out_relu")
        layer.trainable = (block >= from_block or is_top) and not isinstance(layer, layers.BatchNormalization)
        n += int(layer.trainable)
    return n


def cxcywh_to_corners(b):
    """(cx, cy, w, h) -> (x1, y1, x2, y2); works on tf tensors and numpy arrays of shape (n, 4)."""
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    if isinstance(b, np.ndarray):
        return np.clip(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), 0.0, 1.0)
    return tf.clip_by_value(tf.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1), 0.0, 1.0)


def giou_l1_loss(y_true, y_pred):
    """y_true: corners (x1, y1, x2, y2); y_pred: (cx, cy, w, h). 1 - GIoU plus L1 on the corners:
    GIoU moves boxes that do not overlap yet, L1 sharpens corners once they do."""
    y_true = tf.cast(y_true, tf.float32)
    pred = cxcywh_to_corners(tf.cast(y_pred, tf.float32))
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    inter = tf.nn.relu(tf.minimum(px2, tx2) - tf.maximum(px1, tx1)) * tf.nn.relu(
        tf.minimum(py2, ty2) - tf.maximum(py1, ty1)
    )
    area_p = (px2 - px1) * (py2 - py1)
    area_t = (tx2 - tx1) * (ty2 - ty1)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union
    hull = (tf.maximum(px2, tx2) - tf.minimum(px1, tx1)) * (tf.maximum(py2, ty2) - tf.minimum(py1, ty1)) + 1e-7
    giou = iou - (hull - union) / hull
    l1 = tf.reduce_mean(tf.abs(y_true - pred), axis=1)
    return (1.0 - giou) + l1


def compile_model(model: keras.Model, lr: float = 1e-3, box_weight: float = 2.0) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss={"vehicle_class": "sparse_categorical_crossentropy", "bounding_box": giou_l1_loss},
        loss_weights={"vehicle_class": 1.0, "bounding_box": box_weight},
        metrics={"vehicle_class": ["accuracy"]},
    )
    return model


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise IoU for (n, 4) arrays of x1, y1, x2, y2."""
    ix1 = np.maximum(a[:, 0], b[:, 0])
    iy1 = np.maximum(a[:, 1], b[:, 1])
    ix2 = np.minimum(a[:, 2], b[:, 2])
    iy2 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)
