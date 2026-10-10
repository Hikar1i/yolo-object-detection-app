# 环境与安装

## 本机已有环境

本机 `cu12-dev` 已具备图片和视频推理依赖，直接启动即可，无需重复安装：

```bash
conda run -n cu12-dev streamlit run src/app.py
```

访问 **http://localhost:8501**。摄像头组件的现状见 [已知问题](09-known-issues.md)。
运行用户须能写入 `models/`、`outputs/`、`logs/` 及已有日志文件。
本机实际库版本与 `requirements.txt` 的锁定版本不同，详见 [测试策略](06-testing-strategy.md)。

以下步骤用于新建环境；部署方式见 [部署指南](07-deployment.md)，页面操作见 [使用指南](11-usage.md)。

## Requirements

- **Python 3.10+**（Docker 使用 Python 3.10；本机 `cu12-dev` 使用 Python 3.12）
- **pip** (bundled with Python)
- ~2 GB free disk space for model weights + dependencies
- Optional: an NVIDIA GPU with CUDA 11.8+ for accelerated inference

## 1. Clone the repository

```bash
git clone https://github.com/yazenemino/yolov8-object-detection-app.git
cd yolov8-object-detection-app
```

## 2. Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs `ultralytics`, `torch`/`torchvision` (CPU build by default on
most platforms), `streamlit`, `streamlit-webrtc`, `opencv-python-headless`,
`fastapi`, and testing/notebook tooling.

## 4. GPU setup (optional but recommended)

PyTorch 的 CUDA 支持取决于安装的构建版本、驱动和硬件。需要 GPU 加速时：

1. Confirm your NVIDIA driver and GPU support CUDA:
   ```bash
   nvidia-smi
   ```
2. Uninstall the CPU-only torch build and install a CUDA-matched one from the
   [PyTorch install matrix](https://pytorch.org/get-started/locally/). Example
   for CUDA 12.1:
   ```bash
   pip uninstall -y torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. Verify:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```
4. No app code changes are needed — `src/detector.py::get_best_device()`
   auto-selects CUDA when available, otherwise falls back to Apple MPS, then
   CPU.

### cuDNN

cuDNN ships bundled with the PyTorch CUDA wheels above — a separate install
is only needed if you're building PyTorch from source or using a system CUDA
toolkit directly.

## 5. Run the app

```bash
streamlit run src/app.py
```

可按需指定端口或无浏览器启动：

```bash
streamlit run src/app.py --server.port 8502
streamlit run src/app.py --server.headless true
```

Visit **http://localhost:8501**（默认端口）. The default model (YOLOv8-Nano) downloads
automatically on first use (~6 MB). 本项目传入官方权重文件名，下载位置由
Ultralytics 处理；本机运行时已在仓库根目录生成 `yolov8n.pt`。手动或页面上传的
权重位于 `models/`，通过侧栏“本地模型”选项使用。

## 验证与故障排查

测试命令、网络限制和既有验证记录统一见 [测试策略](06-testing-strategy.md)。
安装与运行故障统一见 [已知问题](09-known-issues.md)。
