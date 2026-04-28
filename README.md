# VAR Molten Pool Analysis Desktop Worker

[简体中文](README.zh.md) | English

> The current AI module serves the local desktop analysis pipeline. The desktop entrypoint is `desktop_worker.py`.

## Responsibilities

- Read job JSON files created by the Tauri/Rust desktop core
- Call `var-gpu-preprocessor` to create the preprocessed video
- Call `var-video-analyzer` with `best.onnx` and ONNX Runtime CUDA EP to emit per-frame detections
- Generate anomaly events, dynamic metrics, and result summaries in Python
- Report status, progress, model version, and generated file paths through stdout NDJSON events

## Entrypoint

```bash
python desktop_worker.py /path/to/job.json
python desktop_worker.py --self-check
```

The desktop app normally creates the job file and launches the worker from `frontend/src-tauri/src/lib.rs`. Developers usually do not need to write job files manually unless debugging the worker itself.

## Dependencies

```bash
pip install -r requirements-desktop-windows-cuda.txt
```

The Windows worker package is built by `frontend/scripts/build-desktop-worker.mjs`, which creates and maintains `frontend/.desktop-worker-venv` with only the lightweight worker dependencies. ONNX export still uses `TAURI_ONNX_EXPORT_PYTHON` / `TAURI_WORKER_PYTHON` / conda `var-env` with `torch + ultralytics`.

## Model Weights

Model weights are not committed to Git. The source model is `weights/best.pt`; the first step of the Windows runtime build runs `npm run desktop:export-onnx` to create `weights/best.onnx`. Runtime analysis uses `best.onnx` only and no longer has a Python PT/ONNX fallback.

## Key Files

- `desktop_worker.py`: desktop worker entrypoint
- `analyzer/video_processor.py`: GPU preprocessing sidecar, C++ ONNX analyzer, event generation, and result aggregation
- `utils/callback.py`: stdout event emission for the desktop worker
- `requirements-desktop-windows-cuda.txt`: minimal Windows worker dependencies

## Testing

- `python3 -m py_compile desktop_worker.py utils/callback.py analyzer/video_processor.py`
- `python desktop_worker.py --self-check`
- `desktop_worker.py --self-check --backend cpp-onnx --model weights/best.onnx`
- Import the Windows runtime zip, then import and analyze a real video from the desktop UI

## Note

`BackendCallback` emits desktop NDJSON events through `stdout://` and does not depend on a legacy backend service. The default flow no longer creates `result.mp4`; the desktop app plays the original/preprocessed video and overlays bboxes from `output/detections.json`.
