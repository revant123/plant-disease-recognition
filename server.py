"""Inference API for the Plant Disease Recognition model (deployed on Railway).

Wraps the same model, preprocessing and Grad-CAM code as the Streamlit app (app.py)
in a small FastAPI service, so a separate web front end (deployed on Vercel) can call it.

Run locally:
    pip install -r requirements-api.txt
    uvicorn server:app --port 8000
"""
import base64
import io
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app import analyse, load_model, pretty

MAX_BYTES = 10 * 1024 * 1024  # 10 MB upload limit
STATE = {}


@asynccontextmanager
async def lifespan(_app):
    # Load once at start-up and run one warm-up prediction so the first user isn't slow.
    model, meta = load_model()
    analyse(model, meta, Image.new("RGB", (meta["img_size"], meta["img_size"]), (90, 140, 60)))
    STATE["model"], STATE["meta"] = model, meta
    yield
    STATE.clear()


app = FastAPI(title="Plant Disease Recognition API", version="1.0", lifespan=lifespan)

# ALLOWED_ORIGINS: comma-separated list, e.g. "https://plant-disease.vercel.app". Default "*".
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=os.getenv("ALLOWED_ORIGIN_REGEX") or None,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "plant-disease-recognition", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok" if "model" in STATE else "loading"}


@app.get("/classes")
def classes():
    names = STATE["meta"]["class_names"]
    return {"count": len(names), "classes": [{"id": n, "label": pretty(n)} for n in names]}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the image. Upload a JPG or PNG leaf photo.")

    preds, overlay = analyse(STATE["model"], STATE["meta"], img)

    buf = io.BytesIO()
    Image.fromarray(np.asarray(overlay)).save(buf, format="PNG")
    name, conf = preds[0]
    crop, disease = pretty(name).split(": ", 1)
    return {
        "prediction": {"id": name, "label": pretty(name), "crop": crop, "condition": disease,
                       "confidence": conf, "healthy": "healthy" in name.lower()},
        "top3": [{"id": n, "label": pretty(n), "confidence": c} for n, c in preds],
        "gradcam_png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
    }
