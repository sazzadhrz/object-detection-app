"""
Streamlit frontend for the YOLO object detection API.

It does three things:
    1. Lets the user upload an image
    2. Sends the image to the FastAPI backend (/predict)
    3. Shows the results that come back
"""

import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

# Where the backend is running.
# In Docker Compose this is set to http://backend:8000 (the service name).
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="YOLO Object Detection", page_icon="🔍")

st.title("🔍 YOLO Object Detection")
st.write(
    "Upload an image. It is sent to a **FastAPI** backend, which runs a "
    "pretrained **YOLO11n** model and sends back the objects it found."
)

# ---------- Sidebar: settings and backend status ----------
st.sidebar.header("Settings")
confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Only show objects the model is at least this sure about.",
)

st.sidebar.header("Backend status")
try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    if health.get("model_loaded"):
        st.sidebar.success("Backend is running and the model is loaded")
    else:
        st.sidebar.warning("Backend is running but the model is not loaded")
except requests.exceptions.RequestException:
    st.sidebar.error(f"Cannot reach the backend at {BACKEND_URL}")

# ---------- Main page: upload and detect ----------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Original image", width="stretch")

    if st.button("Detect objects", type="primary"):
        # Send the image to the backend
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        params = {"confidence": confidence}

        with st.spinner("Running the model..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict", files=files, params=params, timeout=60
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.RequestException as error:
                st.error(f"Something went wrong while calling the backend: {error}")
                st.stop()

        # Show the image with boxes drawn on it
        image_bytes = base64.b64decode(result["image_base64"])
        annotated_image = Image.open(io.BytesIO(image_bytes))
        st.image(annotated_image, caption="Detected objects", width="stretch")

        # Show the list of detections as a table
        st.subheader(f"Found {result['count']} object(s)")
        if result["count"] > 0:
            table = [
                {
                    "Object": d["class_name"],
                    "Confidence": d["confidence"],
                    "Box (x1, y1, x2, y2)": str(d["box"]),
                }
                for d in result["detections"]
            ]
            st.dataframe(table, width="stretch")

        # Show the raw JSON so students can see what the API returns
        with st.expander("Raw API response (JSON)"):
            raw = dict(result)
            raw["image_base64"] = raw["image_base64"][:50] + "... (shortened)"
            st.json(raw)
