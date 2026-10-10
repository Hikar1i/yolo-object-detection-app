# 测试

使用已安装依赖的 `cu12-dev` 环境，不需安装新依赖：

```bash
conda run -n cu12-dev python -m pytest -m "not integration" -q
```

- `tests/test_utils.py`：图片转换、绘图、历史记录和文件大小；上传文件名、非空、后缀、大小限制、禁止覆盖、失败清理与模型发现。
- `tests/test_app.py`：Streamlit AppTest 检查上传保存、自动选择、模型对比列表与同名错误；使用临时目录并禁用摄像头组件，不加载测试权重。该上传模拟使用本机 Streamlit 1.65.0 的 AppTest 文件上传能力。
- `tests/test_detector.py`：检测结果、设备选择、模型缓存和真实图片推理。

完整测试与覆盖率命令：

```bash
conda run -n cu12-dev python -m pytest
conda run -n cu12-dev python -m pytest --cov=src
```

注意：现有 `test_model_cache_reuses_instance` 未标注 integration，会加载官方模型，首次运行可能下载权重。因此 `not integration` 当前并不保证完全离线。

2026-10-09 验证结果：34 项通过，1 项 integration 测试未执行；GPU 自动选择新增 PyTorch 设备名称回归检查。

本次额外验证：以真实 YOLOv8 Nano 权重执行页面上传、自动选择、bus.jpg 推理及临时历史保存。未验证摄像头采集和真实视频处理。


2026-10-10 精度参数修复：50 项测试通过，1 项 integration 测试未执行。新增检查覆盖图片、批量、视频与跟踪的 FP32/FP16 参数，以及新旧 Ultralytics 配置字段；旧版行为使用配置模拟，未另建旧版环境。额外使用真实 GPU 权重验证图片、批量、短视频流与 FP16 推理，确认未产生 half 弃用警告。
