"""YOLOv8 detection engine.

Wraps ``ultralytics.YOLO`` with a small, application-friendly API: cached
model loading, device auto-selection (CUDA/MPS/CPU), single-image and
batch inference, video/webcam streaming with optional multi-object
tracking, and structured, serializable results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """Structured, framework-agnostic view of a single inference call."""

    boxes: np.ndarray  # (N, 4) xyxy pixel coords
    confidences: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,)
    class_names: dict[int, str]
    track_ids: np.ndarray | None
    inference_time_ms: float
    image_shape: tuple[int, int]  # (height, width)

    @property
    def num_detections(self) -> int:
        return int(self.boxes.shape[0])

    @property
    def detected_class_labels(self) -> list[str]:
        return [self.class_names.get(int(c), "unknown") for c in self.class_ids]

    @property
    def class_distribution(self) -> dict[str, int]:
        """Count of detections per class label."""
        distribution: dict[str, int] = {}
        for label in self.detected_class_labels:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


@dataclass
class InferenceConfig:
    """Bundles the tunable knobs passed to every ``model.predict``/``track`` call."""

    confidence: float = config.DEFAULT_CONFIDENCE
    iou: float = config.DEFAULT_IOU
    image_size: int = config.DEFAULT_IMAGE_SIZE
    max_detections: int = config.MAX_DETECTIONS
    classes: list[int] | None = None  # None = all classes
    device: str = "auto"
    half_precision: bool = False

    @property
    def precision_kwargs(self) -> dict[str, int | bool | None]:
        """Use the precision option supported by the installed Ultralytics."""
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16 if self.half_precision else None}
        return {"half": self.half_precision}


def get_best_device(preferred: str = "auto") -> str:
    """Resolve the best available inference device.

    Args:
        preferred: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or ``"mps"``.

    Returns:
        A device string accepted by Ultralytics/PyTorch.
    """
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_gpu_available() -> bool:
    """Whether a CUDA-capable GPU is visible to PyTorch."""
    return torch.cuda.is_available()


def get_device_info() -> dict[str, str]:
    """Human-readable summary of the active compute device, for the UI."""
    if torch.cuda.is_available():
        return {
            "device": "GPU (CUDA)",
            "name": torch.cuda.get_device_name(0),
            "memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        }
    return {"device": "CPU", "name": "N/A", "memory_gb": "N/A"}


class ModelCache:
    """Process-wide cache so repeated tab switches don't reload weights.

    Streamlit re-executes the script top-to-bottom on every interaction;
    without caching, each rerun would reload a multi-hundred-MB model.
    Callers should generally use ``@st.cache_resource`` on top of this,
    but the cache is also useful outside a Streamlit context (tests, CLI).
    """

    _cache: dict[str, "YOLODetector"] = {}

    @classmethod
    def get(cls, model_name: str, device: str = "auto") -> "YOLODetector":
        key = f"{model_name}:{device}"
        if key not in cls._cache:
            logger.info("Loading model %s onto device=%s (cache miss)", model_name, device)
            cls._cache[key] = YOLODetector(model_name, device=device)
        else:
            logger.debug("Reusing cached model %s", key)
        return cls._cache[key]

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


class YOLODetector:
    """A single loaded YOLOv8 model, ready for inference."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL_NAME, device: str = "auto") -> None:
        """Load YOLOv8 weights.

        Args:
            model_name: Ultralytics weight identifier, e.g. ``"yolov8n.pt"``.
                Custom checkpoint paths are also accepted.
            device: ``"auto"``, ``"cpu"``, or a CUDA device string.

        Raises:
            RuntimeError: If the model file cannot be loaded (corrupt
                weights, unsupported architecture, missing dependency).
        """
        self.model_name = model_name
        self.device = get_best_device(device)

        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
        except Exception as exc:  # noqa: BLE001 - surface a friendly error upstream
            logger.exception("Failed to load model %s", model_name)
            raise RuntimeError(
                f"无法加载模型 '{model_name}'。首次下载官方权重时请检查网络连接，"
                f"并确认模型文件存在且与当前环境兼容。"
                f"原始错误：{exc}"
            ) from exc

        self.class_names: dict[int, str] = self.model.names
        logger.info("Loaded %s on %s | %d classes", model_name, self.device, len(self.class_names))

    # ------------------------------------------------------------------ #
    # Single-image inference
    # ------------------------------------------------------------------ #
    def predict_image(self, image: Image.Image | np.ndarray, cfg: InferenceConfig | None = None) -> DetectionResult:
        """Run detection on a single image.

        Args:
            image: A PIL image or BGR/RGB numpy array.
            cfg: Inference parameters; defaults are used if omitted.

        Returns:
            A populated :class:`DetectionResult`.
        """
        cfg = cfg or InferenceConfig()
        start = time.perf_counter()

        results = self.model.predict(
            source=image,
            conf=cfg.confidence,
            iou=cfg.iou,
            imgsz=cfg.image_size,
            max_det=cfg.max_detections,
            classes=cfg.classes,
            device=self.device,
            **cfg.precision_kwargs,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return self._to_detection_result(results[0], elapsed_ms)

    def predict_batch(
        self, images: Iterable[Image.Image | np.ndarray], cfg: InferenceConfig | None = None
    ) -> list[DetectionResult]:
        """Run detection on a batch of images in a single forward pass.

        Args:
            images: Iterable of PIL images or numpy arrays.
            cfg: Inference parameters; defaults are used if omitted.

        Returns:
            One :class:`DetectionResult` per input image, in order.
        """
        cfg = cfg or InferenceConfig()
        images = list(images)
        start = time.perf_counter()

        results = self.model.predict(
            source=images,
            conf=cfg.confidence,
            iou=cfg.iou,
            imgsz=cfg.image_size,
            max_det=cfg.max_detections,
            classes=cfg.classes,
            device=self.device,
            **cfg.precision_kwargs,
            verbose=False,
        )
        total_elapsed_ms = (time.perf_counter() - start) * 1000
        per_image_ms = total_elapsed_ms / max(len(images), 1)

        return [self._to_detection_result(r, per_image_ms) for r in results]

    # ------------------------------------------------------------------ #
    # Video / webcam streaming
    # ------------------------------------------------------------------ #
    def predict_video_stream(
        self,
        source: str | int | Path,
        cfg: InferenceConfig | None = None,
        use_tracking: bool = False,
        tracker: str = config.DEFAULT_TRACKER,
        frame_skip: int = config.VIDEO_FRAME_SKIP,
    ) -> Generator[tuple[np.ndarray, DetectionResult], None, None]:
        """Stream detections frame-by-frame from a video file, RTSP URL, or webcam index.

        Args:
            source: File path, URL, or integer webcam index.
            cfg: Inference parameters; defaults are used if omitted.
            use_tracking: If True, run ByteTrack/BoT-SORT for persistent
                object ids instead of plain per-frame detection.
            tracker: Tracker config name, used only when ``use_tracking``.
            frame_skip: Process every Nth frame (>=1) to trade accuracy for speed.

        Yields:
            ``(annotated_frame_source, DetectionResult)`` pairs. The frame
            returned is the *original* BGR frame (unannotated) — callers
            draw boxes themselves via ``utils.draw_detections`` so the UI
            can toggle overlays independently.
        """
        cfg = cfg or InferenceConfig()
        frame_skip = max(1, frame_skip)

        predict_fn = self.model.track if use_tracking else self.model.predict
        extra_kwargs = {"tracker": tracker, "persist": True} if use_tracking else {}

        frame_index = 0
        stream = predict_fn(
            source=source,
            conf=cfg.confidence,
            iou=cfg.iou,
            imgsz=cfg.image_size,
            max_det=cfg.max_detections,
            classes=cfg.classes,
            device=self.device,
            **cfg.precision_kwargs,
            stream=True,
            verbose=False,
            **extra_kwargs,
        )

        for result in stream:
            frame_index += 1
            if frame_index % frame_skip != 0:
                continue
            detection = self._to_detection_result(result, result.speed.get("inference", 0.0))
            yield result.orig_img, detection

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _to_detection_result(self, result, inference_time_ms: float) -> DetectionResult:
        boxes_obj = result.boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4)),
                confidences=np.empty((0,)),
                class_ids=np.empty((0,), dtype=int),
                class_names=self.class_names,
                track_ids=None,
                inference_time_ms=inference_time_ms,
                image_shape=result.orig_shape,
            )

        track_ids = None
        if boxes_obj.id is not None:
            track_ids = boxes_obj.id.cpu().numpy().astype(int)

        return DetectionResult(
            boxes=boxes_obj.xyxy.cpu().numpy(),
            confidences=boxes_obj.conf.cpu().numpy(),
            class_ids=boxes_obj.cls.cpu().numpy().astype(int),
            class_names=self.class_names,
            track_ids=track_ids,
            inference_time_ms=inference_time_ms,
            image_shape=result.orig_shape,
        )


def compare_models(
    image: Image.Image | np.ndarray,
    model_names: list[str],
    cfg: InferenceConfig | None = None,
) -> dict[str, DetectionResult]:
    """Run the same image through several models for side-by-side comparison.

    Args:
        image: Input image shared across all models.
        model_names: Ultralytics weight identifiers to compare.
        cfg: Shared inference parameters.

    Returns:
        Mapping of model name -> its :class:`DetectionResult`.
    """
    results: dict[str, DetectionResult] = {}
    for name in model_names:
        detector = ModelCache.get(name)
        results[name] = detector.predict_image(image, cfg)
    return results
