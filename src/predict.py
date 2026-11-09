"""Run the detector on one image:  uv run python -m src.predict path/to/image.jpg"""

from __future__ import annotations

import argparse
import json

from PIL import Image

from .data import sample_image_path
from .detector import MODEL_PATH, Detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default=None, help="image path (default: a dataset sample)")
    parser.add_argument("--json", action="store_true", help="print the full JSON result")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"no model at {MODEL_PATH}; run `just train` first")
    path = args.image or str(sample_image_path())
    result = Detector().predict(Image.open(path))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    x1, y1, x2, y2 = result["box"]
    print(f"{path}")
    print(f"class       {result['class']}")
    print(f"confidence  {result['confidence']:.3f}")
    print(f"box         x1={x1} y1={y1} x2={x2} y2={y2}  (image {result['image_size'][0]}x{result['image_size'][1]})")


if __name__ == "__main__":
    main()
