"""
FastAPI backend that serves a YOLO object detection model.

Endpoints:
    GET  /         -> welcome message
    GET  /health   -> is the API running and is the model loaded?
    POST /predict  -> upload an image, get detected objects back
"""

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image

from app.model import load_model, run_detection

# Path to the model weights (set in the Dockerfile)
MODEL_PATH = os.getenv("MODEL_PATH", "yolo11n.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs once when the server starts.
    Code after `yield` runs once when the server stops.

    We load the model here, ONE time, instead of inside /predict.
    Loading a model is slow, so doing it on every request would
    make the API very slow.
    """
    print(f"Loading model from {MODEL_PATH} ...")
    app.state.model = load_model(MODEL_PATH)
    print("Model loaded!")
    yield
    print("Shutting down ...")
    app.state.model = None


app = FastAPI(
    title="YOLO Object Detection API",
    description="Upload an image and get the detected objects back.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {"message": "YOLO Object Detection API. Go to /docs to try it."}


@app.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), confidence: float = 0.25):
    """
    Detect objects in an uploaded image.

    - **file**: an image file (jpg, png, ...)
    - **confidence**: only keep objects the model is at least this sure about (0 to 1)
    """
    # 1) Make sure the user sent an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # 2) Read the uploaded bytes and open them as a PIL image
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image.")

    # 3) Run the model (it was loaded at startup)
    model = request.app.state.model
    detections, image_base64 = run_detection(model, image, confidence)

    # 4) Send the results back as JSON
    return {
        "filename": file.filename,
        "count": len(detections),
        "detections": detections,
        "image_base64": image_base64,
    }
