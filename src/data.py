"""Load the MIO-TCD style vehicle dataset: labels.csv (no header) and Images/.

labels.csv has one row per bounding box: image_id, class, x_min, y_min, x_max, y_max,
in pixel coordinates of the original image. The CSV covers 110,593 image ids but
Images.zip ships only 5,626 files, so everything is filtered to the images on disk.

Box selection: the notebook took the first 2,000 CSV rows (one row per box, so an
image appeared several times with different targets). Here each image contributes one
training target, the LARGEST box, which is the vehicle a single-box regressor can
realistically localise. Boxes are normalised to [0, 1] by the original width and
height so they map straight onto the 224x224 network input.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR
DATA_DIR = Path(os.environ.get("CAPSTONE1_DATA", REPO_DIR / "data" / "capstone1"))
IMAGES_DIR = DATA_DIR / "Images"
LABELS_CSV = DATA_DIR / "labels.csv"
MODELS_DIR = PROJECT_DIR / "models"
IMAGE_SIZE = 224

CLASSES = [
    "articulated_truck",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "motorized_vehicle",
    "non-motorized_vehicle",
    "pedestrian",
    "pickup_truck",
    "single_unit_truck",
    "work_van",
]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class Dataset:
    images: np.ndarray  # (n, 224, 224, 3) float32 in [0, 1]
    labels: np.ndarray  # (n,) int
    boxes: np.ndarray  # (n, 4) float32 normalised x1, y1, x2, y2
    image_ids: list[str]


def load_labels() -> pd.DataFrame:
    """All boxes for images that exist on disk, with area and normalised coordinates."""
    df = pd.read_csv(LABELS_CSV, header=None, names=["image_id", "cls", "x1", "y1", "x2", "y2"])
    df["image_id"] = df["image_id"].map(lambda i: f"{int(i):08d}")
    on_disk = {p.stem for p in IMAGES_DIR.glob("*.jpg")}
    df = df[df["image_id"].isin(on_disk)].copy()
    df["area"] = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
    return df


def select_boxes(df: pd.DataFrame, strategy: str = "largest") -> pd.DataFrame:
    """One row per image. 'largest' keeps the biggest box, 'first' the first CSV row."""
    if strategy == "largest":
        return df.sort_values("area", ascending=False).groupby("image_id").head(1).sort_values("image_id")
    if strategy == "first":
        return df.groupby("image_id").head(1).sort_values("image_id")
    raise ValueError(f"unknown strategy {strategy}")


def load_image(path: Path, size: int = IMAGE_SIZE) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (resized float array in [0,1], (orig_w, orig_h))."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        orig = im.size
        arr = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    return arr, orig


def build_dataset(n_images: int | None = None, seed: int = 42, strategy: str = "largest") -> Dataset:
    """Load up to n_images (random subset with a fixed seed, all when None)."""
    rows = select_boxes(load_labels(), strategy)
    if n_images is not None and n_images < len(rows):
        rows = rows.sample(n=n_images, random_state=seed).sort_values("image_id")
    images, labels, boxes, ids = [], [], [], []
    for row in rows.itertuples(index=False):
        arr, (w, h) = load_image(IMAGES_DIR / f"{row.image_id}.jpg")
        images.append(arr)
        labels.append(CLASS_TO_INDEX[row.cls])
        boxes.append([row.x1 / w, row.y1 / h, row.x2 / w, row.y2 / h])
        ids.append(row.image_id)
    return Dataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int32),
        boxes=np.asarray(boxes, dtype=np.float32),
        image_ids=ids,
    )


def sample_image_path() -> Path:
    """A deterministic sample image for demos and tests."""
    return IMAGES_DIR / "00000001.jpg"
