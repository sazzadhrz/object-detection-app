FROM python:3.11-slim

# Print Python logs right away (useful for `docker compose logs`)
ENV PYTHONUNBUFFERED=1

# System libraries that OpenCV (used by ultralytics) needs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only version of PyTorch first.
# Without this, pip would download the much bigger GPU (CUDA) version.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the YOLO11n weights at build time and run one test prediction.
# This also downloads anything else ultralytics needs (like the font for
# drawing labels), so the container can start without internet.
WORKDIR /app/models
RUN python -c "\
from PIL import Image; \
from ultralytics import YOLO; \
model = YOLO('yolo11n.pt'); \
model.predict(Image.new('RGB', (640, 640)), device='cpu')[0].plot(); \
print('Model downloaded and tested')"

# Copy our application code
WORKDIR /app
COPY app ./app

ENV MODEL_PATH=/app/models/yolo11n.pt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
