<!-- public-top -->
# Vehicle detection and Tesla autopilot safety

A Keras model that names the main vehicle in a traffic-camera frame and boxes it, plus the data analysis of Tesla's fatal-crash sheet. Caltech CTME PGC AIML, capstone.

**Watch it first:** [the 90-second film](demo/film.mp4) (what it does, how it is built, the proof, the answer) or the [60-second cut](demo/film-60s.mp4). The plain walkthrough with captions is [demo/demo.mp4](demo/demo.mp4).

## Try it in three commands

```bash
git clone https://github.com/Keshigami/capstone1-vehicle-detection.git && cd capstone1-vehicle-detection
just setup        # creates the venv with uv (Python 3.11)
just up           # http://127.0.0.1:8101/  (just down to stop)
```

You need [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) (`brew install uv just` on a Mac). The trained model is in `models/`, so the app runs as is. Retraining needs the Caltech `Capstone 1` dataset (`DATA_ROOT`), which is not in the repo. No API key is needed anywhere in this project. `just test` runs the lint and the offline tests.

Step by step: [docs/Getting-Started.md](docs/Getting-Started.md), then [how it was built](docs/How-It-Was-Built.md), [results](docs/Results.md), [limits](docs/Limits-and-Next-Steps.md) and [the film](docs/The-Film.md). The same pages are on the [wiki](https://github.com/Keshigami/capstone1-vehicle-detection/wiki).
<!-- /public-top -->
# Capstone 1: vehicle detection and Tesla autopilot safety

## Demo walkthrough
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

Port 8101. Part 1 is a Keras model that names the main vehicle in a traffic-camera frame and draws a
bounding box around it, behind a FastAPI upload page. Part 2 is the exploratory analysis of the
"Tesla - Deaths" sheet the problem statement asks for, served as a page with tables and charts.

## Data

- `Capstone 1/Part 1/Images.zip`: 5,626 traffic-camera JPEGs (MIO-TCD style), mostly 720x480, some
  342x228. Extracted by `just setup` into `../data/capstone1/Images/` (git-ignored).
- `Capstone 1/Part 1/labels.csv`: 351,549 boxes, no header, columns `image_id, class, x1, y1, x2, y2`
  in pixels of the original image. The CSV covers 110,593 image ids but only 5,626 images ship, so
  17,967 boxes are usable. Eleven classes, `car` is 58-66% depending on how you count.
- `Capstone 1/Part 2/Tesla - Deaths.csv`: 307 rows, 24 columns with padded names, 13 footer rows of
  totals, '-' for zero, one `Year=202` typo. Read in place from `DATA_ROOT`, never copied.

Box target: one box per image, the largest one (`src/data.py`, `select_boxes(strategy="largest")`).
The notebook used the first 2,000 CSV rows, so the same image appeared several times with different
targets; a single-box regressor cannot learn that. Coordinates are normalised by each image's own
width and height, so they map directly onto the 224x224 network input and back.

## Model

`src/model.py`, two builders with the same two-head shape (softmax over 11 classes, 4 box numbers):

- `mobilenetv2` (default): ImageNet-pretrained MobileNetV2 at 224 px. Class head: global average
  pooling, dropout, dense. Box head: 1x1 conv to 64 channels on the 7x7 feature map, flatten, dense
  256, sigmoid over `(cx, cy, w, h)`, so a box is always well-formed. Loss: sparse cross-entropy
  plus `2 x (1 - GIoU + L1)` on the corners. Two phases: 3 epochs with the backbone frozen at lr 1e-3
  (heads only), then blocks 10-16 unfrozen at lr 1e-4 with BatchNorm kept frozen, ReduceLROnPlateau,
  early stopping on val loss with the best weights restored.
- `custom`: the notebook's family, five conv blocks from scratch (`--backbone custom`). Kept for
  comparison; it stays at the majority-class baseline in quick mode.

Training mirrors every image left-right (box mirrored with it), and weights the class loss by
sqrt inverse frequency (capped at 5) so the model does not simply answer "car".

Why the box head went through three versions: corners regressed directly with a sorted-corner GIoU
let the network converge to swapped y1/y2 (IoU 0.0 while MAE looked fine), and an unnormalised head
at lr 1e-3 saturated its sigmoids. Predicting centre and size, with BatchNorm in the head, fixed both.

## Numbers (quick mode, this machine)

From `models/metrics.json`, Apple M4 Max, CPU only, TensorFlow 2.21:

| metric | value | reference |
|---|---|---|
| images used | 4,500 (3,600 train, 900 val), 7,200 after flip | |
| epochs run | 11 (3 frozen + 8 fine-tune, best at fine-tune epoch 5) | |
| wall time | 307 s | budget 600 s |
| val accuracy | **73.0%** | majority class 58.7%; notebook 64.5% on 2,000 rows |
| val macro F1 | 0.47 | |
| mean box IoU vs largest box | **0.245** | mean-box baseline 0.086 |
| boxes with IoU >= 0.5 | 11.0% | |
| mean IoU vs best-matching labelled box | 0.249 | so it is not "wrong vehicle", it is imprecision |
| box MAE | 0.148 normalised, 33 px at 224 | notebook 202 px (raw-pixel regression) |

Per-class recall on the 900 validation images: bus 0.90 (72), car 0.86 (528), bicycle 0.88 (8),
articulated truck 0.59 (34), pickup truck 0.48 (171), motorized vehicle 0.42 (36), work van 0.38 (32),
motorcycle 0.17 (6), single-unit truck 0.0 (11), non-motorized vehicle 0.0 (2).

Honest reading: classification is useful (bus, car, trucks); the box lands on the right vehicle
and is usually too loose or too tight rather than elsewhere. A quarter of mean IoU is well above
chance but below what a real detector (anchors or a DETR-style head, higher input resolution) gets.
`just train-full` uses all 5,626 images and up to 30 fine-tune epochs; expect a few points more
accuracy and some IoU, not a different regime.

Part 2 totals reproduce the sheet's own footer row exactly (353 deaths, 117 drivers, 48 occupants,
130 other-vehicle, 46 cyclists/pedestrians, 38 autopilot claimed, 19 verified, 27 with NHTSA SGO),
which is the check that the cleaning is right.

## Run

```
just setup        # uv sync, copy common/ui.css, extract Images.zip if missing
just train        # quick: 4,500 images, 3 + 12 epochs, ~5 min   (just train-full for everything)
just test         # pytest, 8 tests, offline; trains first if there is no model
just up           # background on :8101 with .server.pid; just down stops it
just demo         # curl a dataset sample through /predict
just predict ../data/capstone1/Images/00000007.jpg   # CLI, no server
uv run python -m src.evaluate   # recompute metrics.json from the saved model
```

Endpoints: `GET /` upload page, `POST /predict` (multipart `file`) -> `{class, confidence, box
[x1,y1,x2,y2] px, box_normalized, image_size, top3}`, `GET /health` ->
`{ok, service, model_loaded}`, `GET /metrics` (metrics.json), `GET /samples`, `GET /sample/{id}`,
`GET /safety` (Part 2 page), `GET /safety.json`.

Data location defaults to `../data/capstone1`; override with `CAPSTONE1_DATA`. The Tesla CSV is
read from `DATA_ROOT` (default `~/Caltech CTME/PGC AIML - Capstone/Datasets`).

## Two minutes for an interviewer

1. Open http://localhost:8101/. The first sample is preselected; click **Detect vehicle**. A box
   with "bus 100%" appears on the articulated bus in about 60 ms.
2. Click the other thumbnails: samples 7, 123 and 1234 give "car" at 98-99.8%, sample 500 gives
   car at 59%, and the near-empty road (42) gives the model's honest 34% "car". Drop any road photo
   to try your own.
3. Open **Part 2: Tesla safety analysis**: 294 events, 353 deaths, Autopilot claimed in 35 events
   but verified in 16; the amber-vs-teal chart is the point of the analysis.
4. `curl -F file=@image.jpg localhost:8101/predict` for the JSON, `/metrics` for the numbers.

## Known limits

- One box per image, the largest vehicle; frames with several similar vehicles only get one.
- Quick mode trains on 4,500 of 5,626 images for a few minutes on CPU; IoU 0.25 means boxes are
  approximate. A production detector would use anchors or a transformer head, 512 px input, and
  all boxes per image.
- Rare classes (motorcycle, single-unit truck, non-motorized vehicle, pedestrian) have under a dozen
  validation images and near-zero recall.
- The ImageNet weights download once into `~/.keras/models/` the first time `just train` runs; after
  that everything is offline. Tests never need the network.
- Part 2 counts events; the sheet has no miles-driven or fleet-size denominator, so it cannot say
  whether Autopilot is safer or riskier than human driving.

## Standalone repo

Exported from Max's `caltech-capstone` workspace. `common/` is vendored (LLM door, UI tokens). Run: `just setup && just test && just up`.
LLM-backed features read `ANTHROPIC_API_KEY` (or `LLM_PROVIDER` + `OPENAI_API_KEY` / `GOOGLE_API_KEY`); with no key they run in a deterministic offline mode.
Dataset: the Caltech CTME capstone data is not in this repo. Point `DATA_ROOT` at the `Datasets` folder (`Capstone 1`, `Capstone 2`) before `just train`.
Film: `demo/film.mp4` (90 s, what it does, how it is built, the proof, the answer) and `demo/film-60s.mp4` (60 s cut); `demo/demo.mp4` is the plain walkthrough.
