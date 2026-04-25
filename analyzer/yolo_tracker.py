"""
YOLO detect-only module.

The class name is kept as YOLOTracker for import compatibility, but Phase 0
only runs Ultralytics predict and does not emit object IDs.
"""
import logging
from typing import Any, Dict, List

import numpy as np
import torch
from ultralytics import YOLO

from config import Config

logger = logging.getLogger(__name__)


class YOLOTracker:
    """YOLO detect-only wrapper."""

    def __init__(self, model_path: str, device: str = ""):
        self.model_path = model_path
        self.device = Config.auto_select_device(device)
        self.class_names = Config.CLASS_NAMES

        logger.info("Loading YOLO model from %s", model_path)

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.model_version = self._get_model_version(model_path)
            logger.info("YOLO model loaded successfully, version=%s", self.model_version)
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise

    def _get_model_version(self, model_path: str) -> str:
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if "train_args" in ckpt and "model" in ckpt["train_args"]:
                model_version = ckpt["train_args"]["model"]
                if model_version.endswith(".pt"):
                    model_version = model_version[:-3]
                return model_version
            logger.warning("Model version not found in checkpoint, using default 'yolo11n'")
            return "yolo11n"
        except Exception as e:
            logger.warning("Failed to read model version from checkpoint: %s, using default 'yolo11n'", e)
            return "yolo11n"

    def detect_frame(self, frame: np.ndarray, conf: float = 0.4, iou: float = 0.4) -> List[Dict[str, Any]]:
        """Run YOLO predict on a single frame and return plain detections."""
        try:
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                verbose=Config.VERBOSE,
            )

            detections: List[Dict[str, Any]] = []
            if not results:
                return detections

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return detections

            boxes_xyxy = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else np.array(result.boxes.xyxy)
            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, "cpu") else np.array(result.boxes.conf)
            class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, "cpu") else np.array(result.boxes.cls)
            class_names = [result.names.get(int(cls_id), f"Unknown_{int(cls_id)}") for cls_id in class_ids]

            for index, box in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                detections.append(
                    {
                        "class_id": int(class_ids[index]),
                        "class_name": class_names[index],
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "width": float(width),
                        "height": float(height),
                        "confidence": float(confidences[index]),
                    }
                )

            return detections
        except Exception as e:
            logger.error("Error during YOLO detect-only inference: %s", e, exc_info=True)
            raise RuntimeError(f"YOLO检测失败: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_version": self.model_version,
            "device": str(self.device),
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
        }
