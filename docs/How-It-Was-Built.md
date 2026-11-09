# How it was built

1. Read the notebook: first 2,000 CSV rows, 224 px resize, a five-block CNN with a softmax head and
   a 4-number box head, `sparse_categorical_crossentropy + mse`, 10 epochs, 64.5% val accuracy and a
   box MAE of 202 px because it regressed raw pixel coordinates without scaling anything.
2. Inspected the data: the CSV lists 110,593 image ids but only 5,626 images exist; filtered to
   those. The notebook's row-per-box setup repeats an image with different targets, so each image
   now contributes one target, its largest box, normalised by its own width and height.
3. Kept the notebook's two-head shape but put an ImageNet MobileNetV2 under it (authorised for the
   10-minute budget), with the box head predicting centre and size through sigmoids so the box is
   always valid, and `1 - GIoU + L1` instead of MSE. Left-right flip augmentation and sqrt
   inverse-frequency class weights, because car is 59% of the targets.
4. Training in two phases (frozen backbone, then blocks 10-16 unfrozen at a low learning rate) with
   early stopping; `--quick` uses 4,500 images, `--full` all 5,626.
5. Two bugs found by looking at raw outputs, not just the loss: a sorted-corner GIoU let the model
   learn y1 and y2 swapped (IoU 0.0 with a plausible MAE), and an unnormalised head saturated its
   sigmoids in the frozen phase. Fixed by the centre/size parametrisation and BatchNorm in the head.
6. Evaluation writes `models/metrics.json` with baselines next to every number (majority class,
   mean box) and a second IoU against the best-matching labelled box, to separate "wrong vehicle"
   from "loose box". `src/evaluate.py` recomputes it from the saved model.
7. Part 2: stripped column names, dropped the 13 footer rows of totals, turned '-' into 0,
   re-derived Year from Date, folded Holland into Netherlands, then grouped by year, country, state,
   model and weekday. The recomputed totals match the sheet's own footer row exactly.
8. FastAPI service with the shared dark UI, one-click samples, canvas box drawing, `/health`,
   `/metrics`, `/safety`; justfile; tests; this brief.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | TensorFlow 2.21 runs natively on Apple silicon |
| Model or core logic | Keras: MobileNetV2 (ImageNet) + class head + box head; GIoU + L1 box loss | Pretrained features are what make a five-minute CPU run useful; the notebook's from-scratch CNN is kept as `--backbone custom` |
| Interface | FastAPI + one HTML page per part (plain JS canvas draws the box) | No build step, scriptable ids `#input #run #result` |
| Data | 5,626 traffic-camera images with 17,967 boxes (MIO-TCD style, 142 MB zip); Tesla sheet 294 events | From the Caltech dataset folder, read only; images extracted to `../data/` |
| Tests | pytest + FastAPI TestClient, 8 tests, offline | Health, metrics, predict on a real image, bad upload, page ids, safety tables and totals |

## Libraries
`tensorflow` (Keras model, training, saved `.keras` file), `numpy` (arrays, IoU), `pandas` (labels
and the Tesla sheet), `pillow` (image decode and resize), `fastapi` + `uvicorn` (HTTP service),
`jinja2` (the two pages), `python-multipart` (file upload), `matplotlib` (three PNG charts on the
safety page), `pytest` + `httpx` (tests), `ruff` (lint and format), `opencv-python-headless`
(installed for parity with the notebook; the code path uses Pillow).
