---
title: Vehicle detection and Tesla autopilot safety
group: capstone
role: Computer Vision
port: 8101
status: working
one_liner: A Keras model that names the main vehicle in a traffic-camera frame and boxes it, plus the data analysis of Tesla's fatal-crash sheet.
---
# Vehicle detection and Tesla autopilot safety

## Demo
<!-- demo:auto -->
![demo](demo/demo.gif)

Full walkthrough with captions: [demo/demo.mp4](demo/demo.mp4) (84 s, 1280x800, animated intro and end card). Run it yourself: `just setup && just up`, then open http://localhost:8101/.

Two-minute walkthrough:
1. Capstone 1, part 1: pick a sample traffic-camera frame
2. Detect vehicle: the Keras CNN names the vehicle and draws its box
3. Top-3 classes with their confidence
4. A second frame, same model
5. Drop your own image here: any road photo works
6. Model card: numbers from the quick training run on this machine
7. Part 2: Tesla autopilot safety analysis
8. Headline numbers from the Tesla deaths sheet, 2013 to early 2023
9. Deaths per year climb with the fleet; 2023 is only January
10. Autopilot claimed (amber) versus verified (teal): the gap is the story
11. By model: unknown for most events; S and 3 lead among the known ones
<!-- /demo:auto -->

## Brief
Intelligent transport systems need to count, track and classify vehicles from roadside cameras in
real time. Part 1 asks for a deep-learning model that says what kind of vehicle is in a frame
(car, bus, pickup, truck, bicycle and so on) and where it is, as a rectangle. Part 2 asks what the
public "Tesla Deaths" sheet says about Autopilot and road safety: events by year, country and
model, who died, and how many Autopilot deaths are claimed versus verified. Solved means a page
where you drop a camera frame and get a labelled box, and a page that answers the safety questions
with numbers that can be checked.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language and runtime | Python 3.11 via uv | TensorFlow 2.21 runs natively on Apple silicon |
| Model or core logic | Keras: MobileNetV2 (ImageNet) + class head + box head; GIoU + L1 box loss | Pretrained features are what make a five-minute CPU run useful; the notebook's from-scratch CNN is kept as `--backbone custom` |
| Interface | FastAPI + one HTML page per part (plain JS canvas draws the box) | No build step, scriptable ids `#input #run #result` |
| Data | 5,626 traffic-camera images with 17,967 boxes (MIO-TCD style, 142 MB zip); Tesla sheet 294 events | From the Caltech dataset folder, read only; images extracted to `../data/` |
| Tests | pytest + FastAPI TestClient, 8 tests, offline | Health, metrics, predict on a real image, bad upload, page ids, safety tables and totals |

## Libraries used
`tensorflow` (Keras model, training, saved `.keras` file), `numpy` (arrays, IoU), `pandas` (labels
and the Tesla sheet), `pillow` (image decode and resize), `fastapi` + `uvicorn` (HTTP service),
`jinja2` (the two pages), `python-multipart` (file upload), `matplotlib` (three PNG charts on the
safety page), `pytest` + `httpx` (tests), `ruff` (lint and format), `opencv-python-headless`
(installed for parity with the notebook; the code path uses Pillow).

## How it was built
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

## Results
Quick mode on an M4 Max (CPU), 4,500 images, 11 epochs, 307 s:

| metric | value | what it means |
|---|---|---|
| val accuracy | 73.0% | majority-class baseline 58.7%; the notebook reported 64.5% |
| val macro F1 | 0.47 | rare classes drag it down; bus 0.90, car 0.86, pickup 0.48 recall |
| mean box IoU | 0.245 | mean-box baseline 0.086; a real detector would be 0.6+ |
| boxes with IoU >= 0.5 | 11% | the box is on the right vehicle but loose |
| best-match IoU | 0.249 | almost the same, so misses are imprecision, not another vehicle |
| box MAE | 33 px at 224 | notebook: 202 px |

Part 2: 294 fatal events, 353 deaths, 2013 to January 2023; 73% in the USA, California 91 events;
2022 the peak with 93 events; Autopilot claimed in 35 events (38 deaths), verified in 16 (19
deaths, 5.4% of all deaths), 27 deaths once NHTSA SGO reports are added; model known for only 38%
of events.

## Demo details
- Run: `just setup && just train && just up`, open `http://localhost:8101/`.
- Two-minute walkthrough: (1) click **Detect vehicle** with the preselected sample: "bus 100%" and
  a box around the articulated bus; (2) click the other thumbnails or drop your own road photo;
  (3) open **Part 2: Tesla safety analysis** and look at the amber (claimed) versus teal (verified)
  chart and the headline tiles; (4) `just demo` prints the JSON for one sample.
- Video clip: `demo/clips/capstone1-vehicle-detection.webm` (recorded by `just demo-video`).
- Sample inputs: six dataset frames served at `/sample/{id}` and shown as thumbnails; sample
  `00000001` gives bus 100%, `00000007` and `00000123` give car 98%, `00000042` (a near-empty road)
  gives car 34%, which is the model saying it is unsure.

## Limits and next steps
- One box per image, the largest vehicle; the data has up to a dozen vehicles per frame and a
  production model would predict all of them (anchor-based or DETR-style head, 512 px input).
- IoU 0.25 is honest for five minutes of CPU training; `just train-full` adds the remaining images
  and up to 30 fine-tune epochs, which should give a few points, not a new regime.
- Classes with fewer than a dozen validation images (motorcycle, single-unit truck, non-motorized
  vehicle, pedestrian) have near-zero recall; they need oversampling or more data.
- The MobileNetV2 weights download once from Keras on the first `just train`; after that the whole
  project is offline. Tests never touch the network.
- The Tesla sheet counts events with no miles-driven denominator, so it supports "how many" and
  "where", not "is Autopilot safer".
