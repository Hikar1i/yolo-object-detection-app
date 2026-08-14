"""YOLOv8 Object Detection — main Streamlit application.

Run with:
    streamlit run src/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow `streamlit run src/app.py` to resolve `from src import ...` regardless
# of the working directory Streamlit was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from src import config
from src.detector import (
    InferenceConfig,
    YOLODetector,
    compare_models,
    get_device_info,
    is_gpu_available,
)
from src.utils import (
    DetectionRecord,
    bgr_to_pil,
    draw_detections,
    get_logger,
    image_to_bytes,
    load_history,
    pil_to_bgr,
    save_history_record,
    validate_file_size,
)

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Page configuration & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF9D9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #8a8a8a;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(151, 166, 195, 0.10);
        border: 1px solid rgba(151, 166, 195, 0.25);
        border-radius: 0.6rem;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 1.6rem; }
    .metric-card p { margin: 0; color: #8a8a8a; font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1rem;
    }
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_detector(model_name: str, device: str = "auto") -> YOLODetector:
    """Load (and cache) a YOLOv8 detector for the lifetime of the server process."""
    return YOLODetector(model_name, device=device)


def get_inference_config() -> InferenceConfig:
    """Build an InferenceConfig from the current sidebar widget state."""
    return InferenceConfig(
        confidence=st.session_state.get("confidence", config.DEFAULT_CONFIDENCE),
        iou=st.session_state.get("iou", config.DEFAULT_IOU),
        image_size=st.session_state.get("image_size", config.DEFAULT_IMAGE_SIZE),
    )


def render_metric_row(metrics: dict[str, str]) -> None:
    """Render a row of styled metric cards."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{value}</h3><p>{label}</p></div>',
                unsafe_allow_html=True,
            )


def log_detection(source_type: str, model_name: str, result, fps: float = 0.0) -> None:
    """Persist a detection event to the analytics history."""
    save_history_record(
        DetectionRecord(
            timestamp=time.time(),
            source_type=source_type,
            model_name=model_name,
            num_detections=result.num_detections,
            inference_time_ms=result.inference_time_ms,
            fps=fps,
            classes_detected=result.detected_class_labels,
        )
    )


# --------------------------------------------------------------------------- #
# Sidebar — global settings
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    """Render the global settings sidebar and return the selected model name."""
    with st.sidebar:
        st.markdown(f"## {config.APP_ICON} Settings")

        model_label = st.selectbox(
            "Detection Model",
            options=list(config.AVAILABLE_MODELS.keys()),
            index=list(config.AVAILABLE_MODELS.keys()).index(config.DEFAULT_MODEL_KEY),
            help="Larger models are more accurate but slower.",
        )
        model_name = config.AVAILABLE_MODELS[model_label]

        st.slider(
            "Confidence Threshold", config.MIN_CONFIDENCE, config.MAX_CONFIDENCE,
            config.DEFAULT_CONFIDENCE, 0.01, key="confidence",
            help="Minimum score for a detection to be kept.",
        )
        st.slider(
            "IoU Threshold (NMS)", config.MIN_IOU, config.MAX_IOU,
            config.DEFAULT_IOU, 0.01, key="iou",
            help="Lower values remove more overlapping boxes.",
        )
        st.selectbox(
            "Inference Image Size", config.SUPPORTED_IMAGE_SIZES,
            index=config.SUPPORTED_IMAGE_SIZES.index(config.DEFAULT_IMAGE_SIZE), key="image_size",
        )

        st.divider()
        device_info = get_device_info()
        if is_gpu_available():
            st.success(f"GPU: {device_info['name']} ({device_info['memory_gb']})")
        else:
            st.info("Running on CPU — GPU not detected.")

        st.divider()
        st.caption(f"YOLOv8 Object Detection v{__import__('src').__version__}")
        st.caption("Built with Ultralytics YOLOv8 + Streamlit")

    return model_name


# --------------------------------------------------------------------------- #
# Tab 1 — Image Detection
# --------------------------------------------------------------------------- #
def tab_image_detection(model_name: str) -> None:
    st.subheader("🖼️ Image Object Detection")
    st.write("Upload one or more images, or try a bundled sample, to run YOLOv8 detection.")

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload image(s)", type=config.SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=True, key="image_uploader",
        )
    with col_sample:
        use_sample = st.selectbox(
            "Or use a sample", ["None", "bus.jpg", "zidane.jpg"], key="sample_image"
        )

    images: list[tuple[str, Image.Image]] = []
    if uploaded_files:
        for f in uploaded_files:
            valid, msg = validate_file_size(f.size)
            if not valid:
                st.error(f"{f.name}: {msg}")
                continue
            images.append((f.name, Image.open(f)))
    if use_sample != "None":
        sample_path = config.SAMPLE_IMAGES_DIR / use_sample
        if sample_path.exists():
            images.append((use_sample, Image.open(sample_path)))

    if not images:
        st.info("👆 Upload an image or select a sample to get started.")
        return

    if st.button("🚀 Run Detection", type="primary", key="run_image_detection"):
        detector = load_detector(model_name)
        cfg = get_inference_config()

        for name, image in images:
            with st.spinner(f"Detecting objects in {name}..."):
                result = detector.predict_image(image, cfg)
                bgr = pil_to_bgr(image)
                annotated = draw_detections(
                    bgr, result.boxes, result.class_ids, result.confidences, result.class_names
                )
                annotated_pil = bgr_to_pil(annotated)

            log_detection("image", model_name, result)

            st.markdown(f"#### Results — `{name}`")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(annotated_pil, caption="Detected", use_container_width=True)

            fps_equiv = 1000 / result.inference_time_ms if result.inference_time_ms > 0 else 0
            render_metric_row({
                "Objects Detected": str(result.num_detections),
                "Inference Time": f"{result.inference_time_ms:.1f} ms",
                "Effective FPS": f"{fps_equiv:.1f}",
                "Unique Classes": str(len(set(result.class_ids.tolist()))),
            })

            if result.num_detections > 0:
                df = pd.DataFrame({
                    "Class": result.detected_class_labels,
                    "Confidence": [f"{c:.2%}" for c in result.confidences],
                    "Box (x1,y1,x2,y2)": [tuple(round(v, 1) for v in b) for b in result.boxes],
                })
                with st.expander("📋 Detection details"):
                    st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇️ Download annotated image",
                data=image_to_bytes(annotated_pil),
                file_name=f"detected_{Path(name).stem}.png",
                mime="image/png",
                key=f"download_{name}",
            )
            st.divider()


# --------------------------------------------------------------------------- #
# Tab 2 — Video Detection
# --------------------------------------------------------------------------- #
def tab_video_detection(model_name: str) -> None:
    st.subheader("🎬 Video Object Detection")
    st.write("Upload a video file to run frame-by-frame detection with optional object tracking.")

    uploaded_video = st.file_uploader("Upload a video", type=config.SUPPORTED_VIDEO_TYPES, key="video_uploader")

    col1, col2, col3 = st.columns(3)
    with col1:
        use_tracking = st.checkbox("Enable object tracking (ByteTrack)", value=True)
    with col2:
        frame_skip = st.slider("Process every Nth frame", 1, 10, 1, help="Higher = faster, less smooth.")
    with col3:
        max_frames = st.slider("Max frames to process", 30, 1000, 300, step=10)

    if not uploaded_video:
        st.info("👆 Upload a video file to get started.")
        return

    valid, msg = validate_file_size(uploaded_video.size)
    if not valid:
        st.error(msg)
        return

    if st.button("🚀 Run Detection", type="primary", key="run_video_detection"):
        temp_path = config.OUTPUTS_DIR / f"upload_{uploaded_video.name}"
        temp_path.write_bytes(uploaded_video.getvalue())

        detector = load_detector(model_name)
        cfg = get_inference_config()

        cap = cv2.VideoCapture(str(temp_path))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        output_path = config.OUTPUTS_DIR / f"annotated_{Path(uploaded_video.name).stem}.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))

        frame_placeholder = st.empty()
        progress_bar = st.progress(0, text="Processing video...")
        metric_placeholder = st.empty()

        total_detections = 0
        total_time_ms = 0.0
        frame_count = 0
        all_classes_detected: list[str] = []
        start_wall = time.perf_counter()

        try:
            stream = detector.predict_video_stream(
                str(temp_path), cfg, use_tracking=use_tracking, frame_skip=frame_skip
            )
            for frame_bgr, result in stream:
                frame_count += 1
                annotated = draw_detections(
                    frame_bgr, result.boxes, result.class_ids, result.confidences,
                    result.class_names, track_ids=result.track_ids,
                )
                writer.write(annotated)
                frame_placeholder.image(bgr_to_pil(annotated), caption=f"Frame {frame_count}", use_container_width=True)

                total_detections += result.num_detections
                total_time_ms += result.inference_time_ms
                all_classes_detected.extend(result.detected_class_labels)
                progress_bar.progress(min(frame_count / max_frames, 1.0), text=f"Frame {frame_count}/{max_frames}")

                if frame_count % 5 == 0:
                    elapsed = time.perf_counter() - start_wall
                    live_fps = frame_count / elapsed if elapsed > 0 else 0
                    with metric_placeholder.container():
                        render_metric_row({
                            "Frames Processed": str(frame_count),
                            "Total Detections": str(total_detections),
                            "Processing FPS": f"{live_fps:.1f}",
                            "Avg Inference": f"{total_time_ms / frame_count:.1f} ms",
                        })

                if frame_count >= max_frames:
                    break
        finally:
            writer.release()

        progress_bar.progress(1.0, text="Done!")
        st.success(f"✅ Processed {frame_count} frames — {total_detections} total detections.")

        save_history_record(
            DetectionRecord(
                timestamp=time.time(),
                source_type="video",
                model_name=model_name,
                num_detections=total_detections,
                inference_time_ms=total_time_ms / max(frame_count, 1),
                fps=frame_count / max(time.perf_counter() - start_wall, 1e-6),
                classes_detected=all_classes_detected,
            )
        )

        if output_path.exists():
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download annotated video", data=f.read(),
                    file_name=output_path.name, mime="video/mp4",
                )


# --------------------------------------------------------------------------- #
# Tab 3 — Real-time Detection (browser webcam via streamlit-webrtc)
# --------------------------------------------------------------------------- #
def tab_realtime_detection(model_name: str) -> None:
    st.subheader("📹 Real-time Webcam Detection")
    st.write("Live object detection streamed directly from your browser's webcam.")

    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        import av
    except ImportError:
        st.warning(
            "Real-time webcam detection requires the `streamlit-webrtc` and `av` packages. "
            "Install them with:\n\n```bash\npip install streamlit-webrtc av\n```"
        )
        st.info(
            "Alternatively, use the **Video Detection** tab to upload a pre-recorded clip, "
            "or run `python -m src.detector` locally with `source=0` for an OpenCV window."
        )
        return

    detector = load_detector(model_name)
    cfg = get_inference_config()

    class DetectionProcessor:
        def __init__(self) -> None:
            self.frame_count = 0
            self.last_result = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result = detector.predict_image(img, cfg)
            self.last_result = result
            annotated = draw_detections(
                img, result.boxes, result.class_ids, result.confidences, result.class_names
            )
            self.frame_count += 1
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="realtime-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DetectionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.caption(
        "⚠️ Webcam access requires HTTPS (or `localhost`) and browser permission. "
        "Detection runs server-side on each incoming frame."
    )


# --------------------------------------------------------------------------- #
# Tab 4 — Model Comparison
# --------------------------------------------------------------------------- #
def tab_model_comparison() -> None:
    st.subheader("⚖️ Model Comparison")
    st.write("Run the same image through multiple YOLOv8 variants side-by-side.")

    selected_labels = st.multiselect(
        "Models to compare", options=list(config.AVAILABLE_MODELS.keys()),
        default=list(config.AVAILABLE_MODELS.keys())[:2],
    )
    uploaded_file = st.file_uploader(
        "Upload an image", type=config.SUPPORTED_IMAGE_TYPES, key="compare_uploader"
    )
    use_sample = st.checkbox("Use bundled sample (bus.jpg) instead", value=not bool(uploaded_file))

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
    elif use_sample:
        sample_path = config.SAMPLE_IMAGES_DIR / "bus.jpg"
        if sample_path.exists():
            image = Image.open(sample_path)

    if not image or not selected_labels:
        st.info("👆 Select at least one model and provide an image.")
        return

    if st.button("🚀 Compare Models", type="primary"):
        model_names = [config.AVAILABLE_MODELS[label] for label in selected_labels]
        cfg = get_inference_config()

        with st.spinner(f"Running {len(model_names)} model(s)..."):
            results = compare_models(image, model_names, cfg)

        cols = st.columns(len(results))
        bgr = pil_to_bgr(image)
        summary_rows = []

        for col, (name, result) in zip(cols, results.items()):
            annotated = draw_detections(bgr, result.boxes, result.class_ids, result.confidences, result.class_names)
            with col:
                st.image(bgr_to_pil(annotated), caption=name, use_container_width=True)
                st.metric("Detections", result.num_detections)
                st.metric("Inference Time", f"{result.inference_time_ms:.1f} ms")

            benchmarks = config.MODEL_BENCHMARKS.get(name, {})
            summary_rows.append({
                "Model": name,
                "Detections": result.num_detections,
                "Inference Time (ms)": round(result.inference_time_ms, 1),
                "Params (M)": benchmarks.get("params_m", "N/A"),
                "COCO mAP50-95": benchmarks.get("map50_95", "N/A"),
            })

        st.markdown("#### Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 5 — Statistics & Analytics
# --------------------------------------------------------------------------- #
def tab_statistics() -> None:
    st.subheader("📊 Statistics & Analytics")

    history = load_history()
    if not history:
        st.info("No detection history yet. Run some detections in the other tabs first.")
        return

    df = pd.DataFrame(history)
    df["num_detections"] = pd.to_numeric(df["num_detections"], errors="coerce")
    df["inference_time_ms"] = pd.to_numeric(df["inference_time_ms"], errors="coerce")
    df["fps"] = pd.to_numeric(df["fps"], errors="coerce")

    render_metric_row({
        "Total Runs": str(len(df)),
        "Total Objects Detected": str(int(df["num_detections"].sum())),
        "Avg Inference Time": f"{df['inference_time_ms'].mean():.1f} ms",
        "Avg Detections/Run": f"{df['num_detections'].mean():.1f}",
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Detections per source type")
        st.bar_chart(df.groupby("source_type")["num_detections"].sum())
    with col2:
        st.markdown("##### Inference time by model")
        st.bar_chart(df.groupby("model_name")["inference_time_ms"].mean())

    all_classes = [c for row in df["classes_detected"].dropna() for c in row.split("|") if c]
    if all_classes:
        class_counts = pd.Series(all_classes).value_counts().head(15)
        st.markdown("##### Top detected classes")
        st.bar_chart(class_counts)

    with st.expander("📄 Raw history"):
        st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Clear history"):
        from src.utils import clear_history
        clear_history()
        st.rerun()


# --------------------------------------------------------------------------- #
# Tab 6 — Documentation
# --------------------------------------------------------------------------- #
def tab_documentation() -> None:
    st.subheader("📖 Documentation")

    st.markdown("""
### What is this?
A production-style demo application wrapping **Ultralytics YOLOv8** for real-time
object detection, built with **Streamlit**. It supports image, video, and live
webcam detection, multi-model comparison, and a lightweight analytics dashboard.

### Quick start
1. **Image Detection** — upload a photo (or use a sample) and click *Run Detection*.
2. **Video Detection** — upload an `.mp4`/`.avi` file; enable tracking to keep
   consistent IDs across frames; download the annotated result.
3. **Real-time Detection** — grant webcam access in your browser for live inference.
4. **Model Comparison** — see how nano vs. large models trade off speed and accuracy.
5. **Statistics & Analytics** — review aggregate history of everything you've run.

### Tuning tips
- **Confidence threshold**: raise it to cut false positives; lower it to catch
  more (but noisier) detections.
- **IoU threshold**: controls how aggressively overlapping boxes for the same
  object are merged (Non-Maximum Suppression).
- **Image size**: larger sizes improve small-object recall at the cost of speed.
""")

    st.markdown("##### Model reference")
    bench_df = pd.DataFrame(config.MODEL_BENCHMARKS).T
    bench_df.index.name = "model"
    st.dataframe(bench_df, use_container_width=True)

    st.markdown("""
### Troubleshooting
| Issue | Fix |
|---|---|
| Model download fails | Check internet access; weights auto-download from Ultralytics on first use. |
| Webcam tab shows a warning | `pip install streamlit-webrtc av`. |
| Slow inference on CPU | Use the *Nano* or *Small* model and/or a smaller image size. |
| CUDA not detected | Install a CUDA-matched PyTorch build — see `docs/INSTALLATION.md`. |

Full docs: `README.md`, `docs/INSTALLATION.md`, `docs/USAGE.md`,
`docs/DEPLOYMENT.md`, `docs/API.md`.
""")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(f'<div class="main-header">{config.APP_ICON} YOLOv8 Object Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Real-time object detection with Ultralytics YOLOv8 & Streamlit</div>',
        unsafe_allow_html=True,
    )

    model_name = render_sidebar()

    tabs = st.tabs([
        "🖼️ Image Detection", "🎬 Video Detection", "📹 Real-time Detection",
        "⚖️ Model Comparison", "📊 Statistics & Analytics", "📖 Documentation",
    ])

    with tabs[0]:
        tab_image_detection(model_name)
    with tabs[1]:
        tab_video_detection(model_name)
    with tabs[2]:
        tab_realtime_detection(model_name)
    with tabs[3]:
        tab_model_comparison()
    with tabs[4]:
        tab_statistics()
    with tabs[5]:
        tab_documentation()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level UI safety net
        logger.exception("Unhandled application error")
        st.error(f"An unexpected error occurred: {exc}")
        st.info("Check the logs in the `logs/` directory for details, or reload the page.")
