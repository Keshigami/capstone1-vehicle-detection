# Getting started

## What you need
- [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). On a Mac: `brew install uv just`.
- Python 3.11 is installed by uv.
- No API key is needed anywhere in this project.

## Three commands
```bash
git clone https://github.com/Keshigami/capstone1-vehicle-detection.git && cd capstone1-vehicle-detection
just setup
just up           # http://127.0.0.1:8101/  (just down to stop)
```
The trained model is in `models/`, so the app runs as is. Retraining needs the Caltech `Capstone 1` dataset (`DATA_ROOT`), which is not in the repo.

## Commands
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

## What you see
- Run: `just setup && just train && just up`, open `http://localhost:8101/`.
- Two-minute walkthrough: (1) click **Detect vehicle** with the preselected sample: "bus 100%" and
  a box around the articulated bus; (2) click the other thumbnails or drop your own road photo;
  (3) open **Part 2: Tesla safety analysis** and look at the amber (claimed) versus teal (verified)
  chart and the headline tiles; (4) `just demo` prints the JSON for one sample.
- Video clip: `demo/clips/capstone1-vehicle-detection.webm` (recorded by `just demo-video`).
- Sample inputs: six dataset frames served at `/sample/{id}` and shown as thumbnails; sample
  `00000001` gives bus 100%, `00000007` and `00000123` give car 98%, `00000042` (a near-empty road)
  gives car 34%, which is the model saying it is unsure.

## Try this
Try it: sample 42, the near-empty road.
