# Installation Guide

## Requirements

- **Python 3.10+** (3.8/3.9 also work; the codebase uses `from __future__ import annotations` so newer type-hint syntax stays compatible)
- **pip** (bundled with Python)
- ~2 GB free disk space for model weights + dependencies
- Optional: an NVIDIA GPU with CUDA 11.8+ for accelerated inference

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/yolov8-object-detection-app.git
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

The CPU wheels installed above work everywhere but are slower. For CUDA
acceleration:

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
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
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

Visit **http://localhost:8501**. The default model (YOLOv8-Nano) downloads
automatically on first use (~6 MB) and is cached under `~/.cache` /
Ultralytics' default weights directory for subsequent runs.

## 6. Verify the install

```bash
pytest -m "not integration"
```

All tests should pass without network access. To also exercise real
inference (downloads `yolov8n.pt` if not already cached):

```bash
pytest
```

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Running from the wrong directory | Run commands from the repo root, not `src/` |
| `ImportError: libGL.so.1` (Linux) | Missing OpenCV system dependency | `sudo apt-get install libgl1 libglib2.0-0` |
| Model download hangs/fails | No internet access on first run | Manually download the `.pt` file from the [Ultralytics releases page](https://github.com/ultralytics/assets/releases) into `models/` and point `AVAILABLE_MODELS` in `src/config.py` at the local path |
| `torch.cuda.is_available()` returns `False` on a GPU machine | CPU-only torch wheel installed | Follow the GPU setup steps above |
| Webcam tab errors on import | `streamlit-webrtc`/`av` not installed | `pip install streamlit-webrtc av` (already in `requirements.txt`; reinstall if you used a partial install) |
