"""Detector backend selection for Phase 1."""
from pathlib import Path
from typing import Any


def create_detector(model_path: str, device: str = "") -> Any:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        from .onnx_detector import ONNXDetector

        return ONNXDetector(model_path=model_path, device=device)

    from .yolo_tracker import YOLOTracker

    return YOLOTracker(model_path=model_path, device=device)
