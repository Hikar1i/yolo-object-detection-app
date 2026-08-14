# Usage Guide

## Running locally

```bash
streamlit run src/app.py
```

Optional flags:
```bash
streamlit run src/app.py --server.port 8502   # custom port
streamlit run src/app.py --server.headless true  # no auto browser open
```

### Sidebar settings (apply globally)

| Setting | Effect |
|---|---|
| Detection Model | Switches between nano/small/medium/large/x-large — reloaded and cached on change |
| Confidence Threshold | Minimum score for a box to be kept (default `0.25`) |
| IoU Threshold | Non-Maximum Suppression aggressiveness (default `0.45`) |
| Inference Image Size | Resize before inference; larger = better small-object recall, slower |

## Running with Docker

```bash
docker compose up --build
```

Or without compose:
```bash
docker build -t yolov8-detection-app .
docker run -p 8501:8501 -v $(pwd)/outputs:/app/outputs yolov8-detection-app
```

Then open **http://localhost:8501**.

## Command-line / programmatic usage

The detection engine is fully usable outside Streamlit:

```python
from PIL import Image
from src.detector import YOLODetector, InferenceConfig

detector = YOLODetector("yolov8n.pt", device="auto")
image = Image.open("assets/sample_images/bus.jpg")

result = detector.predict_image(image, InferenceConfig(confidence=0.3))
print(f"{result.num_detections} objects in {result.inference_time_ms:.1f} ms")
for label, count in result.class_distribution.items():
    print(f"  {label}: {count}")
```

Batch inference:
```python
from pathlib import Path

images = [Image.open(p) for p in Path("assets/sample_images").glob("*.jpg")]
results = detector.predict_batch(images)
```

Video / webcam streaming with tracking:
```python
for frame, result in detector.predict_video_stream(
    source=0,               # webcam index, or a file path / RTSP URL
    use_tracking=True,
    tracker="bytetrack.yaml",
):
    print(result.num_detections, result.track_ids)
```

See **`notebooks/tutorial.ipynb`** for a fully worked, runnable walkthrough,
including a custom-training example.

## Configuration options

All tunables live in `src/config.py`:

- `AVAILABLE_MODELS` — add/remove models offered in the UI.
- `DEFAULT_CONFIDENCE`, `DEFAULT_IOU`, `DEFAULT_IMAGE_SIZE` — inference defaults.
- `MAX_UPLOAD_SIZE_MB` — upload size cap.
- `SUPPORTED_IMAGE_TYPES` / `SUPPORTED_VIDEO_TYPES` — accepted file extensions.
- `BOX_COLOR_PALETTE` — bounding box colors, cycled by class id.

Streamlit-level settings (theme, max upload size, CORS) live in
`.streamlit/config.toml`.

## REST API usage

```bash
uvicorn src.api:app --reload --port 8000
```

```bash
curl -X POST "http://localhost:8000/predict?model=yolov8n.pt&confidence=0.3" \
  -F "file=@assets/sample_images/bus.jpg"
```

Full reference: **[API.md](API.md)**.
