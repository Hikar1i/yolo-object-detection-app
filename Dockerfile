# YOLOv8 Object Detection — Streamlit app container
# Build:  docker build -t yolov8-detection-app .
# Run:    docker run -p 8501:8501 yolov8-detection-app
FROM python:3.10-slim

# System deps required by OpenCV / video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY assets/ ./assets/
COPY .streamlit/ ./.streamlit/

RUN mkdir -p models outputs logs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", \
    "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
