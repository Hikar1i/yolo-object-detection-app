# 部署指南

## 资源选型参考

以下为原指南的选型建议，尚未进行对应硬件的性能验证，实际需求取决于模型、分辨率和并发。

| Tier | CPU | RAM | GPU | Notes |
|---|---|---|---|---|
| Minimum | 2 vCPU | 4 GB | none | YOLOv8-Nano only, expect 1–3 FPS video |
| Recommended | 4 vCPU | 8 GB | none | Small/Medium models, image + short video use |
| GPU (best) | 4 vCPU | 16 GB | NVIDIA, 4GB+ VRAM | Real-time video/webcam, larger models |

## 1. Local deployment

启动命令与本机 Conda 环境统一见 [环境与安装](08-environments.md)。页面及程序操作见 [使用指南](11-usage.md)。

## 2. Docker deployment

### CPU (any host with Docker)

```bash
docker compose up --build -d
docker compose logs -f
```

也可不使用 Compose：

```bash
docker build -t yolov8-detection-app .
docker run -p 8501:8501 -v "$(pwd)/models:/app/models" -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/logs:/app/logs" yolov8-detection-app
```

访问 **http://localhost:8501**。Compose 已挂载 `models/`、`outputs/` 与 `logs/`。

### GPU (NVIDIA host)

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.
2. Verify: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
3. Uncomment the `deploy.resources.reservations.devices` block in
   `docker-compose.yml`.
4. Rebuild the image with a CUDA-enabled torch (edit `requirements.txt` to
   pull from `https://download.pytorch.org/whl/cu121`, or install it as an
   extra `RUN pip install` step in the `Dockerfile`).
5. `docker compose up --build -d`

## 3. Cloud deployment

以下为原指南提供的部署路径，本项目未在这些云平台进行验证；平台当前规格、价格及控制台流程需部署时核对。

### Streamlit Community Cloud

1. Push this repo to GitHub (public or connected private repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, main file path `src/app.py`.
4. Deploy. First load will be slower while YOLOv8 weights download.

> Note: Community Cloud is CPU-only and has no persistent disk across
> restarts — expect model re-downloads after a cold start, and treat
> `outputs/`/history as ephemeral.

### Railway

1. `railway init` in the repo root, or connect the GitHub repo via the
   Railway dashboard.
2. Railway auto-detects the `Dockerfile` — no extra config needed.
3. Set the exposed port to `8501` in the service settings.
4. Deploy; Railway provides a public HTTPS URL (required for webcam access).

### Render

1. New **Web Service** → connect the repo → environment: **Docker**.
2. Render builds from the root `Dockerfile` automatically.
3. Set the health check path to `/_stcore/health`.
4. Instance type: at least 1 GB RAM for the Small model; 512 MB is workable
   for Nano only.

### Hugging Face Spaces

1. Create a new **Space** → SDK: **Docker**.
2. Push this repo's contents to the Space's git remote.
3. Spaces builds the included `Dockerfile` directly; no changes required.

## 4. Production notes

- **HTTPS is required for webcam access** (the *Real-time Detection* tab) on
  every platform except `localhost` — all cloud options above provide it by
  default.
- Mount `models/`, `outputs/` and `logs/` as persistent volumes if you need detection
  history to survive restarts (see `docker-compose.yml`).
- For multi-user traffic, front the FastAPI service (`src/api.py`) with a
  process manager (`gunicorn -k uvicorn.workers.UvicornWorker`) instead of
  `uvicorn --reload`.
- Rotate `logs/app.log` (e.g. via `logrotate` or a sidecar) in long-running
  deployments — the app does not rotate logs itself.
