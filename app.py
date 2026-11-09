"""FastAPI service on port 8101: vehicle detection (Part 1) and Tesla safety analysis (Part 2)."""

from __future__ import annotations

import io
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from src.data import IMAGES_DIR, PROJECT_DIR, REPO_DIR
from src.detector import Detector, load_metrics
from src.safety import get_report

SERVICE = "capstone1-vehicle-detection"
PORT = 8101
STATIC_DIR = PROJECT_DIR / "static"
TEMPLATES = Jinja2Templates(directory=str(PROJECT_DIR / "templates"))
SAMPLE_IDS = ["00000001", "00000007", "00000042", "00000123", "00000500", "00001234"]

STATIC_DIR.mkdir(exist_ok=True)
shared_css = REPO_DIR / "common" / "ui.css"
if shared_css.exists():
    shutil.copy(shared_css, STATIC_DIR / "ui.css")

app = FastAPI(title="Capstone 1: vehicle detection", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.state.detector = Detector.load_if_available()


def _detector() -> Detector:
    if app.state.detector is None:
        raise HTTPException(503, "model not trained yet: run `just train`")
    return app.state.detector


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": SERVICE, "model_loaded": app.state.detector is not None, "port": PORT}


@app.get("/metrics")
def metrics() -> JSONResponse:
    m = load_metrics()
    if m is None:
        raise HTTPException(404, "models/metrics.json not found: run `just train`")
    return JSONResponse(m)


@app.get("/samples")
def samples() -> dict:
    return {"samples": [s for s in SAMPLE_IDS if (IMAGES_DIR / f"{s}.jpg").exists()]}


@app.get("/sample/{image_id}")
def sample(image_id: str) -> FileResponse:
    if not image_id.isdigit():
        raise HTTPException(400, "image id is 8 digits")
    path = IMAGES_DIR / f"{int(image_id):08d}.jpg"
    if not path.exists():
        raise HTTPException(404, "no such sample; run `just setup` to extract the images")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    det = _detector()
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"not an image: {e}") from e
    result = det.predict(image)
    result["filename"] = file.filename
    return result


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    m = load_metrics() or {}
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "port": PORT,
            "model_loaded": app.state.detector is not None,
            "metrics": m,
            "samples": samples()["samples"],
        },
    )


@app.get("/safety", response_class=HTMLResponse)
def safety(request: Request) -> HTMLResponse:
    try:
        report = get_report()
    except FileNotFoundError as e:
        raise HTTPException(503, f"Tesla CSV not found: {e}") from e
    tables = {k: v.to_dict(orient="records") for k, v in report["tables"].items()}
    return TEMPLATES.TemplateResponse(
        request,
        "safety.html",
        {
            "port": PORT,
            "model_loaded": app.state.detector is not None,
            "summary": report["summary"],
            "inspection": report["inspection"],
            "tables": tables,
            "charts": report["charts"],
            "narrative": report["narrative"],
        },
    )


@app.get("/safety.json")
def safety_json() -> dict:
    report = get_report()
    return {
        "inspection": report["inspection"],
        "summary": report["summary"],
        "tables": {k: v.to_dict(orient="records") for k, v in report["tables"].items()},
        "charts": report["charts"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=False, app_dir=str(Path(__file__).parent))
