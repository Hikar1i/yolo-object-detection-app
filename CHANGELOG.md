# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.0] - 2026-08-14

### Added
- Initial release: Streamlit application with six tabs — Image Detection,
  Video Detection, Real-time Detection, Model Comparison, Statistics &
  Analytics, and Documentation.
- `src/detector.py`: cached YOLOv8 inference engine supporting nano/small/
  medium/large/x-large models, batch inference, video/webcam streaming, and
  ByteTrack/BoT-SORT object tracking.
- `src/api.py`: optional FastAPI REST endpoint (`/predict`, `/health`,
  `/models`) for headless integration.
- Detection history logging and an analytics dashboard (class distribution,
  per-model inference time, per-source detection counts).
- Docker and docker-compose support, including a documented GPU passthrough
  path.
- Unit tests for `detector.py` and `utils.py` (`pytest`), with model-download
  tests isolated behind an `integration` marker.
- Jupyter tutorial notebook covering programmatic inference, batch
  processing, and a custom-training walkthrough.
- Full documentation set: `README.md`, `docs/INSTALLATION.md`,
  `docs/USAGE.md`, `docs/DEPLOYMENT.md`, `docs/API.md`.
