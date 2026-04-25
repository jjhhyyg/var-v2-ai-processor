"""
ONNX Runtime detect-only module.

This mirrors the public detection shape returned by YOLOTracker while avoiding
Ultralytics/PyTorch imports at runtime.
"""
import logging
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from config import Config
from utils.performance_trace import PerformanceTrace, record_elapsed_since

logger = logging.getLogger(__name__)


class ONNXDetector:
    """YOLO detect-only wrapper backed by ONNX Runtime."""

    def __init__(self, model_path: str, device: str = "", require_cuda: bool | None = None):
        _ = device
        self.model_path = model_path
        self.class_names = Config.CLASS_NAMES
        self.model_version = Path(model_path).stem
        self.require_cuda = self._resolve_require_cuda(require_cuda)
        self.performance_trace: Optional[PerformanceTrace] = None

        try:
            self._preload_dlls_silently()

            available_providers = ort.get_available_providers()
            if self.require_cuda and "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError(f"CUDAExecutionProvider unavailable: {available_providers}")

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available_providers
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.active_providers = self.session.get_providers()
            if self.require_cuda and "CUDAExecutionProvider" not in self.active_providers:
                raise RuntimeError(f"CUDAExecutionProvider not active: {self.active_providers}")

            self.input_meta = self.session.get_inputs()[0]
            self.input_name = self.input_meta.name
            self.input_height, self.input_width = self._resolve_input_size(self.input_meta.shape)
            self.model_version = f"onnxruntime:{Path(model_path).name}"
            logger.info(
                "ONNX model loaded successfully, providers=%s, input=%s %sx%s",
                self.active_providers,
                self.input_name,
                self.input_width,
                self.input_height,
            )
        except Exception as e:
            logger.error("Failed to load ONNX model: %s", e)
            raise

    def set_performance_trace(self, trace: PerformanceTrace) -> None:
        self.performance_trace = trace

    @staticmethod
    def _resolve_require_cuda(require_cuda: bool | None) -> bool:
        if require_cuda is not None:
            return require_cuda
        raw = os.getenv("ONNX_REQUIRE_CUDA")
        if raw is not None:
            return raw.lower() not in ("0", "false", "no", "off")
        return os.name == "nt"

    @staticmethod
    def _preload_dlls_silently() -> None:
        try:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    ort.preload_dlls()
        except AttributeError:
            pass
        except Exception as exc:
            logger.warning("ONNX Runtime DLL preload failed: %s", exc)

    @staticmethod
    def _resolve_input_size(shape) -> Tuple[int, int]:
        if len(shape) != 4:
            raise ValueError(f"ONNX input must be NCHW, got shape={shape}")

        height = shape[2]
        width = shape[3]
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(f"ONNX input must have static H/W for Phase 1, got shape={shape}")
        return height, width

    def detect_frame(self, frame: np.ndarray, conf: float = 0.4, iou: float = 0.4) -> List[Dict[str, Any]]:
        try:
            preprocess_start = time.perf_counter_ns()
            input_tensor, scale, pad = self._preprocess(frame)
            record_elapsed_since(self.performance_trace, "detectorPreprocess", preprocess_start)

            inference_start = time.perf_counter_ns()
            outputs = self.session.run(None, {self.input_name: input_tensor})
            record_elapsed_since(self.performance_trace, "detectorInference", inference_start)

            postprocess_start = time.perf_counter_ns()
            try:
                return self._postprocess(
                    outputs[0],
                    original_shape=frame.shape[:2],
                    scale=scale,
                    pad=pad,
                    conf_threshold=conf,
                    iou_threshold=iou,
                )
            finally:
                record_elapsed_since(self.performance_trace, "detectorPostprocess", postprocess_start)
        except Exception as e:
            logger.error("Error during ONNX detect-only inference: %s", e, exc_info=True)
            raise RuntimeError(f"ONNX检测失败: {e}") from e

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        letterboxed, scale, pad = self.letterbox(
            frame,
            new_shape=(self.input_height, self.input_width),
        )
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(tensor), scale, pad

    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        original_height, original_width = image.shape[:2]
        target_height, target_width = new_shape
        scale = min(target_height / original_height, target_width / original_width)

        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))
        pad_width = target_width - resized_width
        pad_height = target_height - resized_height
        half_pad_width = pad_width / 2
        half_pad_height = pad_height / 2

        if (original_width, original_height) != (resized_width, resized_height):
            image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        top = int(round(half_pad_height - 0.1))
        bottom = int(round(half_pad_height + 0.1))
        left = int(round(half_pad_width - 0.1))
        right = int(round(half_pad_width + 0.1))
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, scale, (left, top)

    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[float, float],
        conf_threshold: float,
        iou_threshold: float,
        max_det: int = 300,
    ) -> List[Dict[str, Any]]:
        predictions = np.asarray(output)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise ValueError(f"Unexpected ONNX output shape: {output.shape}")

        expected_rows = 4 + len(self.class_names)
        if predictions.shape[0] == expected_rows:
            predictions = predictions.T

        if predictions.shape[1] < 5:
            raise ValueError(f"Unexpected ONNX output columns: {predictions.shape}")

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = confidences >= conf_threshold

        if not np.any(keep):
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh[keep])
        boxes_xyxy = self._scale_boxes_to_original(boxes_xyxy, original_shape, scale, pad)
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        nms_start = time.perf_counter_ns()
        try:
            keep_indices = self._nms_class_aware(
                boxes_xyxy,
                confidences,
                class_ids,
                iou_threshold=iou_threshold,
                max_det=max_det,
            )
        finally:
            record_elapsed_since(getattr(self, "performance_trace", None), "detectorNms", nms_start)

        detections: List[Dict[str, Any]] = []
        for index in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[index]
            class_id = int(class_ids[index])
            confidence = float(confidences[index])
            width = float(x2 - x1)
            height = float(y2 - y1)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, f"Unknown_{class_id}"),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center_x": float((x1 + x2) / 2),
                    "center_y": float((y1 + y2) / 2),
                    "width": width,
                    "height": height,
                    "confidence": confidence,
                }
            )

        return detections

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        converted = boxes.astype(np.float32).copy()
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return converted

    @staticmethod
    def _scale_boxes_to_original(
        boxes: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[float, float],
    ) -> np.ndarray:
        pad_x, pad_y = pad
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, :4] /= scale

        original_height, original_width = original_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_height)
        return boxes

    @staticmethod
    def _nms_class_aware(
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
        max_det: int,
    ) -> List[int]:
        selected: List[int] = []
        for class_id in np.unique(class_ids):
            class_indices = np.where(class_ids == class_id)[0]
            class_order = class_indices[np.argsort(scores[class_indices])[::-1]]
            while len(class_order) > 0:
                current = int(class_order[0])
                selected.append(current)
                if len(selected) >= max_det or len(class_order) == 1:
                    break
                ious = ONNXDetector._box_iou(boxes[current], boxes[class_order[1:]])
                class_order = class_order[1:][ious <= iou_threshold]
            if len(selected) >= max_det:
                break

        selected.sort(key=lambda index: float(scores[index]), reverse=True)
        return selected[:max_det]

    @staticmethod
    def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_version": self.model_version,
            "device": ",".join(self.active_providers),
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
        }

    def cleanup(self) -> None:
        self.session = None
