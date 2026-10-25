"""
Helper functions for the YOLO model.

We keep all the model code in this file so that main.py only
has to deal with the web API part.
"""

import base64
import io

from PIL import Image
from ultralytics import YOLO


def load_model(model_path: str) -> YOLO:
    """Load a pretrained YOLO model from a .pt file."""
    return YOLO(model_path)


def run_detection(model: YOLO, image: Image.Image, confidence: float = 0.25):
    """
    Run object detection on one image.

    Returns:
        detections:   a list of dicts, one per detected object
        image_base64: the image with boxes drawn on it, as a base64 JPEG string
    """
    # model.predict() returns a list of results (one per image).
    # We send only one image, so we take the first result.
    results = model.predict(image, conf=confidence, device="cpu", verbose=False)
    result = results[0]

    # 1) Turn every detected box into a simple dictionary
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class_name": result.names[class_id],  # e.g. "person", "car"
                "confidence": round(float(box.conf[0]), 3),
                "box": [round(x1), round(y1), round(x2), round(y2)],
            }
        )

    # 2) Draw the boxes on the image.
    # result.plot() returns a numpy array in BGR color order (OpenCV style),
    # so we flip it to RGB before making a PIL image.
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    annotated_image = Image.fromarray(annotated_rgb)

    # 3) Convert the image to a base64 string so it can be sent inside JSON
    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return detections, image_base64
