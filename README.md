# VAR Molten Pool Analysis System - AI Processor

[简体中文](README.zh.md) | English

> Flask-based AI service for preprocessing, YOLO detection, tracking, and result export.

## Responsibilities

The AI processor is responsible for:

- consuming analysis messages from RabbitMQ
- optional video preprocessing
- YOLO11 detection and BoT-SORT tracking
- anomaly event generation and metrics calculation
- exporting annotated result videos
- calling back the backend with progress, result data, model version, and generated file paths

## Runtime Model

This service supports two local modes:

1. start the Flask application and health endpoint through `app.py`
2. start the RabbitMQ consumer in the same runtime, so analysis tasks can be consumed asynchronously

In practice, local integration testing should run the full `app.py` entrypoint.

## Current Configuration

The AI processor uses `config.py` as the configuration source of truth.

Important facts:

- `config.py` loads the root repository `.env`, not only `ai-processor/.env`
- backend callback configuration uses `BACKEND_BASE_URL`
- default model path is `weights/best.pt`
- default queue name is `video_analysis_queue`

Important variables include:

- `AI_PROCESSOR_HOST`
- `AI_PROCESSOR_PORT`
- `AI_PROCESSOR_DEBUG`
- `BACKEND_BASE_URL`
- `YOLO_MODEL_PATH`
- `YOLO_DEVICE`
- `DEFAULT_CONFIDENCE_THRESHOLD`
- `DEFAULT_IOU_THRESHOLD`
- `TRACKER_CONFIG`
- `PROGRESS_UPDATE_INTERVAL`
- `RABBITMQ_*`
- `STORAGE_*`

Generate environment files from the main repository root:

```bash
./scripts/use-env.sh dev
```

## Local Development

Install dependencies:

```bash
cd ai-processor
pip install -r requirements.txt
```

If you intentionally want a CPU-only dependency set, use:

```bash
pip install -r requirements-cpu.txt
```

Start the service:

```bash
python app.py
```

Default local URL:

- `http://localhost:5000`

Health check:

```bash
curl http://localhost:5000/health
```

## Model and Device

Before local testing or deployment, confirm:

- `ai-processor/weights/best.pt` exists
- `YOLO_DEVICE` matches your environment, or is intentionally left empty for auto selection

Typical device choices:

- `cuda`: NVIDIA GPU
- `mps`: Apple Silicon
- `cpu`: CPU only
- empty string: auto select

## Important Files

- `app.py`: Flask entrypoint and health endpoint
- `mq_consumer.py`: RabbitMQ consumer and task dispatch
- `config.py`: configuration source of truth
- `analyzer/video_processor.py`: preprocessing, analysis, export flow
- `preprocessor/video_preprocessor.py`: CPU-based video preprocessing

## Local Testing Expectations

Minimum checks before deployment:

- `GET /health` returns healthy status
- model loads successfully
- the selected device is reported correctly
- RabbitMQ connection succeeds
- a real analysis task can be consumed and completed

Be honest about the current state: this module does not have a strong automated test suite. Manual integration testing is mandatory.

## Docker Notes

- GPU production uses the main repository `docker-compose.prod.yml`
- CPU production uses `docker-compose.prod.cpu.yml`
- CPU production image is built from `Dockerfile.cpu`
- production deployment still depends on `weights/best.pt` being available

## What to Read Next

- Main repository overview:
  `https://github.com/jjhhyyg/VAR-melting-defect-detection-source-code.git`
- Main handover guide in the root repository:
  `docs/项目接手、开发测试与部署指南.md`
