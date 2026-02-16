from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from .ml import predict_topk

app = FastAPI(title="ML Image Classifier", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Please upload an image file."}, status_code=400)

    raw = await file.read()
    try:
        img = Image.open(BytesIO(raw))
    except Exception:
        return JSONResponse({"error": "Invalid image."}, status_code=400)

    preds = predict_topk(img, k=5)
    return {"predictions": [{"label": l, "prob": p} for l, p in preds]}
