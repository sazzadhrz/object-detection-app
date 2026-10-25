# YOLO Object Detection with FastAPI, Streamlit and Docker

A small demo project that shows how to:

1. Serve a **pretrained machine learning model** (YOLO11n) through a **REST API** (FastAPI)
2. Use that API from a **web frontend** (Streamlit)
3. Run everything with **Docker** and **Docker Compose**

The model runs on the **CPU**, so no GPU is needed.

## How it works

```
 ┌──────────┐   upload image   ┌─────────────┐  POST /predict  ┌──────────────────────┐
 │ Browser  │ ───────────────> │  Streamlit  │ ──────────────> │  FastAPI + YOLO11n   │
 │          │ <─────────────── │  (frontend) │ <────────────── │  (backend)           │
 └──────────┘   show results   └─────────────┘   JSON result   └──────────────────────┘
                                  port 8501                         port 8000
```

- The **backend** loads the YOLO model **once at startup** and keeps it in memory.
- The **frontend** sends the uploaded image to the backend and shows what comes back.
- The two containers talk to each other over the Docker Compose network,
  using the service name: `http://backend:8000`.

## Project structure

```
fastapi_docker_application/
├── docker-compose.yml      # Runs both containers together
├── backend/
│   ├── Dockerfile          # Builds the API image (downloads the model weights)
│   ├── requirements.txt
│   └── app/
│       ├── main.py         # FastAPI app and endpoints
│       └── model.py        # Load the model and run detection
└── frontend/
    ├── Dockerfile          # Builds the Streamlit image
    ├── requirements.txt
    └── app.py              # Streamlit user interface
```

## Run it

You only need Docker and Docker Compose installed.

```bash
docker compose up --build
```

The first build takes a few minutes because it downloads PyTorch and the model.
After that, open:

| What                | URL                          |
| ------------------- | ---------------------------- |
| Frontend (Streamlit)| http://localhost:8501        |
| API docs (Swagger)  | http://localhost:8000/docs   |
| Health check        | http://localhost:8000/health |

To stop everything:

```bash
docker compose down
```

## API endpoints

| Method | Path       | What it does                                     |
| ------ | ---------- | ------------------------------------------------ |
| GET    | `/`        | Welcome message                                  |
| GET    | `/health`  | Says if the API is running and the model loaded  |
| POST   | `/predict` | Upload an image, get the detected objects back   |

### Try `/predict` from the terminal

```bash
curl -F "file=@bus.jpg" "http://localhost:8000/predict?confidence=0.25"
```

Example response (the image string is shortened):

```json
{
  "filename": "bus.jpg",
  "count": 5,
  "detections": [
    {"class_name": "bus", "confidence": 0.94, "box": [4, 229, 796, 728]},
    {"class_name": "person", "confidence": 0.888, "box": [671, 395, 810, 879]}
  ],
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQ..."
}
```

- `box` is `[x1, y1, x2, y2]`: the top-left and bottom-right corners in pixels.
- `image_base64` is the image with boxes drawn on it, encoded as text so it fits in JSON.

## Code walk-through

1. **`backend/app/model.py`**
   - `load_model()` loads the YOLO weights file.
   - `run_detection()` runs the model on one image, turns the results into a list of
     dictionaries, and draws the boxes on the image.
2. **`backend/app/main.py`**
   - The `lifespan` function runs when the server starts. It loads the model once and
     saves it in `app.state.model`. Loading the model on every request would be slow.
   - `/predict` reads the uploaded file, calls `run_detection()`, and returns JSON.
3. **`frontend/app.py`**
   - Shows an upload box and a confidence slider.
   - Sends the image to the backend with `requests.post(...)`.
   - Decodes the base64 image and shows it, plus a table of the detections.
4. **`backend/Dockerfile`**
   - Installs the **CPU-only** version of PyTorch (much smaller than the GPU version).
   - Downloads `yolo11n.pt` during the build, so the container starts fast and works
     without internet.
5. **`docker-compose.yml`**
   - Starts both containers.
   - The frontend waits until the backend's **healthcheck** passes before it starts.

## Ideas to extend

- Try a bigger model, like `yolo11s.pt`, and compare speed and accuracy.
- Add webcam support in the frontend with `st.camera_input`.
- Return only some classes (for example, only `person`).
