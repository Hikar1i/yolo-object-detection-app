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
    get_available_models,
    save_uploaded_model,
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
        st.markdown(f"## {config.APP_ICON} 设置")

        with st.expander("上传模型"):
            st.caption(f"仅上传可信来源的 YOLO .pt 目标检测权重，最大 {config.MAX_UPLOAD_SIZE_MB} MB。加载不可信权重可能执行恶意代码。")
            uploaded_model = st.file_uploader(
                "选择模型权重", type=["pt"], key="model_uploader",
            )
            if st.button("保存模型", key="save_model", disabled=uploaded_model is None):
                try:
                    saved_path = save_uploaded_model(uploaded_model.name, uploaded_model.getvalue())
                except FileExistsError:
                    st.error("同名模型已存在，请重命名后上传；不会覆盖已有模型。")
                except (ValueError, OSError) as exc:
                    st.error(f"模型保存失败：{exc}")
                else:
                    st.session_state["selected_model"] = f"本地模型：{saved_path.name}"
                    st.success(f"已保存并选中模型：{saved_path.name}")

        models = get_available_models()
        if st.session_state.get("selected_model") not in models:
            st.session_state["selected_model"] = config.DEFAULT_MODEL_KEY
        model_label = st.selectbox(
            "检测模型", options=list(models), key="selected_model",
            help="较大的模型通常精度更高，但推理速度更慢。本地权重会在推理时检查兼容性。",
        )
        model_name = models[model_label]

        st.slider(
            "置信度阈值", config.MIN_CONFIDENCE, config.MAX_CONFIDENCE,
            config.DEFAULT_CONFIDENCE, 0.01, key="confidence",
            help="仅保留置信度达到此阈值的检测结果。",
        )
        st.slider(
            "IoU 阈值（NMS）", config.MIN_IOU, config.MAX_IOU,
            config.DEFAULT_IOU, 0.01, key="iou",
            help="阈值越低，剔除的重叠检测框越多。",
        )
        st.selectbox(
            "推理图像尺寸", config.SUPPORTED_IMAGE_SIZES,
            index=config.SUPPORTED_IMAGE_SIZES.index(config.DEFAULT_IMAGE_SIZE), key="image_size",
        )

        st.divider()
        device_info = get_device_info()
        if is_gpu_available():
            st.success(f"GPU: {device_info['name']} ({device_info['memory_gb']})")
        else:
            st.info("未检测到可用 GPU，当前使用 CPU 推理。")

        st.divider()
        st.caption(f"YOLOv8 目标检测 v{__import__('src').__version__}")
        st.caption("基于 Ultralytics YOLOv8 + Streamlit")

    return model_name


# --------------------------------------------------------------------------- #
# Tab 1 — 图片检测
# --------------------------------------------------------------------------- #
def tab_image_detection(model_name: str) -> None:
    st.subheader("🖼️ 图片目标检测")
    st.write("上传一张或多张图片，或选择内置示例，进行 YOLOv8 目标检测。")

    col_upload, col_sample = st.columns([3, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "上传图片（支持多选）", type=config.SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=True, key="image_uploader",
        )
    with col_sample:
        use_sample = st.selectbox(
            "或选择示例图片", ["不使用示例", "bus.jpg", "zidane.jpg"], key="sample_image"
        )

    images: list[tuple[str, Image.Image]] = []
    if uploaded_files:
        for f in uploaded_files:
            valid, msg = validate_file_size(f.size)
            if not valid:
                st.error(f"{f.name}: {msg}")
                continue
            images.append((f.name, Image.open(f)))
    if use_sample != "不使用示例":
        sample_path = config.SAMPLE_IMAGES_DIR / use_sample
        if sample_path.exists():
            images.append((use_sample, Image.open(sample_path)))

    if not images:
        st.info("👆 请上传图片或选择示例图片。")
        return

    if st.button("🚀 开始检测", type="primary", key="run_image_detection"):
        detector = load_detector(model_name)
        cfg = get_inference_config()

        for name, image in images:
            with st.spinner(f"正在检测 {name} 中的目标…"):
                result = detector.predict_image(image, cfg)
                bgr = pil_to_bgr(image)
                annotated = draw_detections(
                    bgr, result.boxes, result.class_ids, result.confidences, result.class_names
                )
                annotated_pil = bgr_to_pil(annotated)

            log_detection("image", model_name, result)

            st.markdown(f"#### 检测结果 — `{name}`")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="原图", width="stretch")
            with col2:
                st.image(annotated_pil, caption="检测结果", width="stretch")

            fps_equiv = 1000 / result.inference_time_ms if result.inference_time_ms > 0 else 0
            render_metric_row({
                "检测目标数": str(result.num_detections),
                "推理耗时": f"{result.inference_time_ms:.1f} ms",
                "等效 FPS": f"{fps_equiv:.1f}",
                "类别数": str(len(set(result.class_ids.tolist()))),
            })

            if result.num_detections > 0:
                df = pd.DataFrame({
                    "类别": result.detected_class_labels,
                    "置信度": [f"{c:.2%}" for c in result.confidences],
                    "检测框 (x1,y1,x2,y2)": [tuple(round(v, 1) for v in b) for b in result.boxes],
                })
                with st.expander("📋 检测详情"):
                    st.dataframe(df, width="stretch")

            st.download_button(
                "⬇️ 下载标注图片",
                data=image_to_bytes(annotated_pil),
                file_name=f"detected_{Path(name).stem}.png",
                mime="image/png",
                key=f"download_{name}",
            )
            st.divider()


# --------------------------------------------------------------------------- #
# Tab 2 — 视频检测
# --------------------------------------------------------------------------- #
def tab_video_detection(model_name: str) -> None:
    st.subheader("🎬 视频目标检测")
    st.write("上传视频进行逐帧检测，可启用目标跟踪以保持目标编号连续。")

    uploaded_video = st.file_uploader("上传视频", type=config.SUPPORTED_VIDEO_TYPES, key="video_uploader")

    col1, col2, col3 = st.columns(3)
    with col1:
        use_tracking = st.checkbox("启用目标跟踪（ByteTrack）", value=True)
    with col2:
        frame_skip = st.slider("每隔 N 帧处理一帧", 1, 10, 1, help="间隔越大，处理速度越快，但输出视频越不连贯。")
    with col3:
        max_frames = st.slider("最多处理帧数", 30, 1000, 300, step=10)

    if not uploaded_video:
        st.info("👆 请先上传视频文件。")
        return

    valid, msg = validate_file_size(uploaded_video.size)
    if not valid:
        st.error(msg)
        return

    if st.button("🚀 开始检测", type="primary", key="run_video_detection"):
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
        progress_bar = st.progress(0, text="正在处理视频…")
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
                frame_placeholder.image(bgr_to_pil(annotated), caption=f"第 {frame_count} 帧", width="stretch")

                total_detections += result.num_detections
                total_time_ms += result.inference_time_ms
                all_classes_detected.extend(result.detected_class_labels)
                progress_bar.progress(min(frame_count / max_frames, 1.0), text=f"第 {frame_count} 帧/{max_frames}")

                if frame_count % 5 == 0:
                    elapsed = time.perf_counter() - start_wall
                    live_fps = frame_count / elapsed if elapsed > 0 else 0
                    with metric_placeholder.container():
                        render_metric_row({
                            "已处理帧数": str(frame_count),
                            "检测目标总数": str(total_detections),
                            "处理速度（FPS）": f"{live_fps:.1f}",
                            "平均推理耗时": f"{total_time_ms / frame_count:.1f} ms",
                        })

                if frame_count >= max_frames:
                    break
        finally:
            writer.release()

        progress_bar.progress(1.0, text="处理完成！")
        st.success(f"✅ 已处理 {frame_count} 帧，共检测到 {total_detections} 个目标。")

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
                    "⬇️ 下载标注视频", data=f.read(),
                    file_name=output_path.name, mime="video/mp4",
                )


# --------------------------------------------------------------------------- #
# Tab 3 — 实时检测 (browser webcam via streamlit-webrtc)
# --------------------------------------------------------------------------- #
def tab_realtime_detection(model_name: str) -> None:
    st.subheader("📹 摄像头实时检测")
    st.write("通过浏览器摄像头采集视频，实时显示目标检测结果。")

    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        import av
    except ImportError:
        st.warning(
            "摄像头实时检测需要 `streamlit-webrtc` 和 `av` 软件包。"
            "安装命令：\n\n```bash\npip install streamlit-webrtc av\n```"
        )
        st.info("也可在 **视频检测** 页面上传已录制的视频。")
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
        "⚠️ 使用摄像头需要通过 HTTPS（或 `localhost`）访问，并授予浏览器摄像头权限。"
        "服务器对接收到的每一帧进行推理。"
    )


# --------------------------------------------------------------------------- #
# Tab 4 — 模型对比
# --------------------------------------------------------------------------- #
def tab_model_comparison() -> None:
    st.subheader("⚖️ 模型对比")
    st.write("使用多个 YOLO 模型检测同一张图片，对比检测结果与推理耗时。")
    models = get_available_models()

    selected_labels = st.multiselect(
        "选择要对比的模型", options=list(models),
        default=list(config.AVAILABLE_MODELS.keys())[:2],
    )
    uploaded_file = st.file_uploader(
        "上传图片", type=config.SUPPORTED_IMAGE_TYPES, key="compare_uploader"
    )
    use_sample = st.checkbox("使用内置示例图片（bus.jpg）", value=not bool(uploaded_file))

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file)
    elif use_sample:
        sample_path = config.SAMPLE_IMAGES_DIR / "bus.jpg"
        if sample_path.exists():
            image = Image.open(sample_path)

    if not image or not selected_labels:
        st.info("👆 请至少选择一个模型，并上传图片或使用示例。")
        return

    if st.button("🚀 开始对比", type="primary"):
        model_names = [models[label] for label in selected_labels]
        cfg = get_inference_config()

        with st.spinner(f"正在运行 {len(model_names)} 个模型…"):
            results = compare_models(image, model_names, cfg)

        cols = st.columns(len(results))
        bgr = pil_to_bgr(image)
        summary_rows = []

        for col, (name, result) in zip(cols, results.items()):
            annotated = draw_detections(bgr, result.boxes, result.class_ids, result.confidences, result.class_names)
            with col:
                st.image(bgr_to_pil(annotated), caption=name, width="stretch")
                st.metric("检测目标数", result.num_detections)
                st.metric("推理耗时", f"{result.inference_time_ms:.1f} ms")

            benchmarks = config.MODEL_BENCHMARKS.get(name, {})
            summary_rows.append({
                "模型": name,
                "检测目标数": result.num_detections,
                "推理耗时（ms）": round(result.inference_time_ms, 1),
                "参数量（百万）": benchmarks.get("params_m", "暂无数据"),
                "COCO mAP50-95": benchmarks.get("map50_95", "暂无数据"),
            })

        st.markdown("#### 对比汇总")
        st.dataframe(pd.DataFrame(summary_rows), width="stretch")


# --------------------------------------------------------------------------- #
# Tab 5 — 统计分析
# --------------------------------------------------------------------------- #
def tab_statistics() -> None:
    st.subheader("📊 统计分析")

    history = load_history()
    if not history:
        st.info("暂无检测历史。请先在图片或视频页面完成检测。")
        return

    df = pd.DataFrame(history)
    df["num_detections"] = pd.to_numeric(df["num_detections"], errors="coerce")
    df["inference_time_ms"] = pd.to_numeric(df["inference_time_ms"], errors="coerce")
    df["fps"] = pd.to_numeric(df["fps"], errors="coerce")

    render_metric_row({
        "检测次数": str(len(df)),
        "检测目标总数": str(int(df["num_detections"].sum())),
        "平均推理耗时": f"{df['inference_time_ms'].mean():.1f} ms",
        "每次平均目标数": f"{df['num_detections'].mean():.1f}",
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 各来源的检测目标数")
        counts = df.groupby("source_type")["num_detections"].sum()
        counts = counts.rename(index={"image": "图片", "video": "视频", "webcam": "摄像头"})
        st.bar_chart(counts.rename_axis("来源").rename("检测目标数"))
    with col2:
        st.markdown("##### 各模型平均推理耗时")
        timings = df.groupby("model_name")["inference_time_ms"].mean()
        st.bar_chart(timings.rename_axis("模型").rename("平均推理耗时（ms）"))

    all_classes = [c for row in df["classes_detected"].dropna() for c in row.split("|") if c]
    if all_classes:
        class_counts = pd.Series(all_classes).value_counts().head(15)
        st.markdown("##### 检测次数最多的类别")
        st.bar_chart(class_counts.rename_axis("类别").rename("检测次数"))

    with st.expander("📄 检测历史明细"):
        display_df = df.assign(source_type=df["source_type"].replace({
            "image": "图片", "video": "视频", "webcam": "摄像头",
        })).rename(columns={
            "timestamp": "时间戳", "source_type": "来源", "model_name": "模型",
            "num_detections": "检测目标数", "inference_time_ms": "推理耗时（ms）",
            "fps": "FPS", "classes_detected": "检测类别",
        })
        st.dataframe(display_df, width="stretch")

    if st.button("🗑️ 清空检测历史"):
        from src.utils import clear_history
        clear_history()
        st.rerun()


# --------------------------------------------------------------------------- #
# Tab 6 — 使用说明
# --------------------------------------------------------------------------- #
def tab_documentation() -> None:
    st.subheader("📖 使用说明")

    st.markdown("""
### 应用介绍
本应用基于 **Ultralytics YOLOv8** 和 **Streamlit**，用于在线测试目标检测模型。
支持图片检测、视频检测、摄像头实时检测、模型对比和检测历史统计。

### 模型上传
在侧栏展开 **上传模型**，选择可信的 YOLO `.pt` 目标检测权重并点击 **保存模型**。
文件保存到 `models/`，同名文件不会覆盖。保存成功后自动选中，可用于图片、视频与模型对比。
上传仅保存文件，兼容性在实际推理时检查；自定义模型没有官方 COCO 参考指标。

### 快速开始
1. **图片检测**：上传图片或选择示例，点击 **开始检测**，查看并下载标注结果。
2. **视频检测**：上传 `.mp4`、`.avi` 等视频，可启用目标跟踪，处理完成后下载标注视频。
3. **实时检测**：允许浏览器访问摄像头，查看实时检测画面。
4. **模型对比**：选择多个模型，对比同一张图片的检测结果和推理耗时。
5. **统计分析**：查看图片与视频检测的历史记录和汇总指标。

### 参数调整建议
- **置信度阈值**：调高可减少误检；调低可检出更多目标，也可能增加误检。
- **IoU 阈值**：控制非极大值抑制（NMS）对重叠检测框的剔除程度。
- **推理图像尺寸**：较大的尺寸通常有利于检测小目标，但推理耗时更长。
""")

    st.markdown("##### 模型参考指标")
    bench_df = pd.DataFrame(config.MODEL_BENCHMARKS).T
    bench_df.index.name = "模型"
    bench_df = bench_df.rename(columns={
        "params_m": "参数量（百万）", "map50_95": "COCO mAP50-95",
        "speed_ms_cpu": "CPU 耗时（ms）", "speed_ms_gpu": "GPU 耗时（ms）",
    })
    st.dataframe(bench_df, width="stretch")

    st.markdown("""
### 常见问题
| 问题 | 处理建议 |
|---|---|
| 模型下载失败 | 检查网络连接；官方权重会在首次使用时自动下载。 |
| 摄像头页面提示缺少依赖 | 当前环境未提供摄像头组件，可先使用图片或视频检测。 |
| CPU 推理较慢 | 选择 Nano 或 Small 模型，或降低推理图像尺寸。 |
| 未检测到 CUDA | 检查 GPU 驱动和 PyTorch CUDA 支持，参阅 `project-docs/08-environments.md`。 |

项目文档：`README.md`、`project-docs/08-environments.md`、`project-docs/11-usage.md`、
`project-docs/07-deployment.md`、`project-docs/04-api.md`。
""")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(f'<div class="main-header">{config.APP_ICON} YOLOv8 目标检测</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">基于 Ultralytics YOLOv8 与 Streamlit 的在线目标检测</div>',
        unsafe_allow_html=True,
    )

    model_name = render_sidebar()

    tabs = st.tabs([
        "🖼️ 图片检测", "🎬 视频检测", "📹 实时检测",
        "⚖️ 模型对比", "📊 统计分析", "📖 使用说明",
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
        st.error(f"运行出错：{exc}")
        st.info("请查看 `logs/` 目录下的日志了解详情，或刷新页面重试。")
