# VAR Molten Pool Analysis Desktop Worker

[简体中文](README.zh.md) | English

> The current AI module serves the local desktop analysis pipeline. It no longer provides the legacy Flask/RabbitMQ service entrypoint or Docker image entrypoints.

## Responsibilities

- Read job JSON files created by the Tauri/Rust desktop core
- Run video preprocessing, YOLO detection, BoT-SORT tracking, and result video export
- Report status, progress, model version, and generated file paths through stdout NDJSON events
- Provide the worker entrypoint and minimal dependency files for macOS desktop packaging

## Entrypoint

```bash
python desktop_worker.py /path/to/job.json
python desktop_worker.py --self-check
```

The desktop app normally creates the job file and launches the worker from `frontend/src-tauri/src/lib.rs`. Developers usually do not need to write job files manually unless debugging the worker itself.

## Dependencies

```bash
pip install -r requirements-desktop-macos.txt
pip install -r requirements.txt
```

## Model Weights

Model weights are not committed to Git. The default build script reads `weights/best.pt`; place that file locally or set `YOLO_MODEL_PATH` to another local model path.

## Key Files

- `desktop_worker.py`: desktop worker entrypoint
- `analyzer/video_processor.py`: preprocessing, detection, tracking, and result export flow
- `preprocessor/video_preprocessor.py`: video preprocessing
- `utils/callback.py`: stdout/HTTP callback abstraction; desktop uses `stdout://`
- `requirements-desktop-macos.txt`: minimal macOS worker dependencies

## Testing

- `python3 -m py_compile desktop_worker.py utils/callback.py analyzer/video_processor.py`
- `python desktop_worker.py --self-check`
- Import and analyze a real video from the macOS desktop UI

## Note

`BackendCallback` still keeps the HTTP callback branch to reduce risk in the analysis pipeline. The current desktop path uses `stdout://` and does not depend on the legacy backend service.
