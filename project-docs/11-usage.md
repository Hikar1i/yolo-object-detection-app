# 页面与程序使用指南

启动及安装见 [环境与安装](08-environments.md)，容器与云部署见 [部署指南](07-deployment.md)。

### Sidebar settings (apply globally)

| Setting | Effect |
|---|---|
| 检测模型 | 选择官方 Nano/Small/Medium/Large/X-Large 或 `models/` 内的本地权重，按路径缓存加载 |
| 置信度阈值 | Minimum score for a box to be kept (default `0.25`) |
| IoU 阈值 | Non-Maximum Suppression aggressiveness (default `0.45`) |
| 推理图像尺寸 | Resize before inference; larger = better small-object recall, slower |

## 模型上传与页面流程

在侧栏展开 **上传模型**，选择可信来源的 YOLO `.pt` 目标检测权重，点击 **保存模型**。
最大 200 MB，保存到 `models/`；拒绝空文件、目录路径及同名覆盖。保存成功后自动选中，
图片、视频、实时检测与模型对比共用本地模型列表。文件保存时不加载模型，兼容性在推理时检查。
不要上传不可信的权重，加载 `.pt` 文件可能执行其中的代码。

- **图片检测**：上传一张或多张图片，或选择 bus.jpg/zidane.jpg；点击 **开始检测**，查看原图、标注图、指标及详情，下载标注图片。
- **视频检测**：上传视频，设置跟踪、帧间隔和最多处理帧数；点击 **开始检测**，下载标注视频。
- **实时检测**：需要 `streamlit-webrtc`/`av` 与摄像头权限，访问地址需为 HTTPS 或 localhost。
- **模型对比**：选择多个模型与同一张图片，点击 **开始对比**；自定义模型没有官方 COCO 参考指标，显示“暂无数据”。
- **统计分析**：查看图片和视频检测历史，按来源、模型、类别汇总，可清空历史。
- **使用说明**：查看参数建议和模型参考指标。

页面标题、帮助与表格使用中文；模型原始类别名和必要技术名保留英文。
模型文件保存与发现机制见 [架构](02-architecture.md)。

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

- `AVAILABLE_MODELS` — 内置官方模型列表；本地 `.pt` 权重由 `get_available_models()` 扫描 `MODELS_DIR` 加入 Web 页面。
- `DEFAULT_CONFIDENCE`, `DEFAULT_IOU`, `DEFAULT_IMAGE_SIZE` — inference defaults.
- `MAX_UPLOAD_SIZE_MB` — upload size cap.
- `SUPPORTED_IMAGE_TYPES` / `SUPPORTED_VIDEO_TYPES` — accepted file extensions.
- `BOX_COLOR_PALETTE` — bounding box colors, cycled by class id.

程序可通过 `InferenceConfig(half_precision=True)` 请求 FP16；安装库支持时传入 `quantize=16`，否则使用旧版 `half=True`。默认 FP32。

Streamlit-level settings (theme, max upload size, file watcher) live in
`.streamlit/config.toml`.

## REST API

启动命令、接口契约及 curl/Python/JavaScript 示例统一见 [REST API 接口](04-api.md)。
