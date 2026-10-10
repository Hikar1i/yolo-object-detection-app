# REST API 接口

The optional FastAPI service in `src/api.py` exposes headless detection for
non-Streamlit clients (scripts, mobile apps, other backends).

## Running the API

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs** (Swagger UI, auto-generated
from the Pydantic models below).

## Endpoints

### `GET /`
Basic service info.

```json
{
  "name": "YOLOv8 Object Detection API",
  "docs": "/docs",
  "predict_endpoint": "/predict",
  "health_endpoint": "/health"
}
```

### `GET /health`
Liveness + device info.

```json
{
  "status": "ok",
  "gpu_available": false,
  "device": {"device": "CPU", "name": "N/A", "memory_gb": "N/A"}
}
```

### `GET /models`
内置官方模型的显示名称与权重标识。本地上传模型仅供 Web 页面使用，不加入此 API 列表。

```json
{
  "YOLOv8 Nano（速度最快，精度较低）": "yolov8n.pt",
  "YOLOv8 Small": "yolov8s.pt",
  "YOLOv8 Medium": "yolov8m.pt",
  "YOLOv8 Large": "yolov8l.pt",
  "YOLOv8 X-Large（速度最慢，精度较高）": "yolov8x.pt"
}
```

### `POST /predict`

Runs detection on an uploaded image.

**Query parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `model` | string | `yolov8n.pt` | Model weight identifier (see `/models`) |
| `confidence` | float | `0.25` | Minimum confidence, `0.0`–`1.0` |
| `iou` | float | `0.45` | NMS IoU threshold, `0.0`–`1.0` |

**Body** — `multipart/form-data`，字段名为 `file`。服务通过 Pillow 解码图片；代码未按扩展名限制上传类型，无法解码时返回 `400`。

**Response** — `200 OK`

```json
{
  "model": "yolov8n.pt",
  "image_width": 810,
  "image_height": 1080,
  "inference_time_ms": 42.7,
  "num_detections": 5,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "box": [123.4, 56.7, 340.1, 480.9]
    }
  ]
}
```

注意：`/predict` 仅接受 `/models` 中的官方权重标识，不接受本地路径。

**Errors**

| Status | Cause |
|---|---|
| `400` | Unknown model, oversized file (`> MAX_UPLOAD_SIZE_MB`), or unreadable image |
| `500` | Model failed to load or inference raised an unexpected error |

### Examples

**curl**
```bash
curl -X POST "http://localhost:8000/predict?model=yolov8n.pt&confidence=0.3" \
  -F "file=@assets/sample_images/bus.jpg"
```

**Python (requests)**
```python
import requests

with open("assets/sample_images/bus.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        params={"model": "yolov8n.pt", "confidence": 0.3},
        files={"file": f},
    )
response.raise_for_status()
print(response.json())
```

**JavaScript (fetch)**
```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);

const res = await fetch("http://localhost:8000/predict?model=yolov8n.pt", {
  method: "POST",
  body: form,
});
console.log(await res.json());
```
