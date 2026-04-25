"""
视频处理主逻辑模块。

Phase 0 使用 YOLO detect-only 链路，不再产生追踪对象或 track ID。
"""
import cv2
import time
import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image, ImageDraw, ImageFont

from .detector_factory import create_detector
from .metrics_calculator import MetricsCalculator
from .anomaly_event_generator import AnomalyEventGenerator
from utils.callback import BackendCallback
from utils.video_storage import VideoStorageManager
from utils.atomic_write import atomic_write_json, safe_read_json
from utils.performance_trace import PerformanceTrace
from config import Config
from preprocessor import OptimizedVideoPreprocessor

logger = logging.getLogger(__name__)


def cv2_add_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = None
    try:
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    continue

        if font is None:
            import platform

            if platform.system() == "Windows":
                for font_name in ("msyh.ttc", "simhei.ttf"):
                    try:
                        font = ImageFont.truetype(font_name, font_size)
                        break
                    except Exception:
                        continue

        if font is None:
            font = ImageFont.load_default()
    except Exception as e:
        logger.warning("加载字体时发生异常: %s, 使用默认字体", e)
        font = ImageFont.load_default()

    color_rgb = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class VideoAnalyzer:
    """视频分析器。"""

    def __init__(self, model_path: str, device: str = ""):
        self.model_path = model_path
        self.yolo_tracker = create_detector(model_path, device)
        self.device = getattr(self.yolo_tracker, "device", "")
        self.performance_trace = PerformanceTrace()
        self._last_analyzed_frame_count = 0
        self._attach_performance_trace()
        self.metrics_calculator = MetricsCalculator()
        self._is_cleaned_up = False
        self.preprocessor = OptimizedVideoPreprocessor()
        self.storage_manager = VideoStorageManager()

        logger.info("VideoAnalyzer initialized with device: %s", self.device)

    def analyze_video_task(
        self,
        task_id: int,
        video_path: str,
        video_duration: int,
        timeout_threshold: int,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        enable_preprocessing: bool = False,
        preprocessing_strength: str = "moderate",
        preprocessing_enhance_pool: bool = False,
        enable_dynamic_metrics: bool = True,
        callback_url: Optional[str] = None,
        frame_rate: float = 25.0,
        preprocessed_output_path: Optional[str] = None,
    ) -> tuple[str, str]:
        _ = video_duration
        callback = BackendCallback(task_id, callback_url)
        final_video_path = None
        preprocessing_duration = 0
        preprocessing_average_fps: Optional[float] = None
        self.performance_trace = PerformanceTrace()
        self._last_analyzed_frame_count = 0
        self._attach_performance_trace()

        try:
            all_metrics: List[Dict[str, Any]] = []
            all_detections: List[List[Dict[str, Any]]] = []

            preprocessed_video_path = video_path
            if enable_preprocessing:
                logger.info(
                    "Task %s: Starting video preprocessing (strength=%s, enhance_pool=%s)",
                    task_id,
                    preprocessing_strength,
                    preprocessing_enhance_pool,
                )
                callback.notify_preprocessing(f"正在预处理视频（强度：{preprocessing_strength}）...")

                if preprocessed_output_path:
                    preprocessed_video_path = preprocessed_output_path
                    Path(preprocessed_video_path).parent.mkdir(parents=True, exist_ok=True)
                else:
                    preprocessed_dir = Path(Config.get_storage_path(Config.STORAGE_PREPROCESSED_VIDEOS_SUBDIR))
                    preprocessed_dir.mkdir(parents=True, exist_ok=True)
                    from utils.filename_utils import add_or_update_timestamp, extract_base_name

                    video_stem = Path(video_path).stem
                    base_name = extract_base_name(video_stem)
                    base_filename = f"{base_name}_preprocessed.mp4"
                    preprocessed_filename = Path(add_or_update_timestamp(base_filename, update_existing=True)).name
                    preprocessed_video_path = str(preprocessed_dir / preprocessed_filename)

                preprocessing_start = time.time()

                def preprocessing_progress_callback(current_frame, total_frames, elapsed_time):
                    progress = current_frame / total_frames if total_frames > 0 else 0
                    callback.update_progress(
                        {
                            "status": "PREPROCESSING",
                            "phase": f"预处理视频中（{preprocessing_strength}）",
                            "progress": round(progress, 4),
                            "currentFrame": current_frame,
                            "totalFrames": total_frames,
                            "preprocessingDuration": int(elapsed_time),
                        }
                    )

                self.preprocessor.process_video(
                    input_path=Config.resolve_path(video_path),
                    output_path=preprocessed_video_path,
                    frame_rate=frame_rate,
                    strength=preprocessing_strength,
                    enhance_pool=preprocessing_enhance_pool,
                    progress_callback=preprocessing_progress_callback,
                )
                preprocessing_elapsed = time.time() - preprocessing_start
                preprocessing_duration = int(preprocessing_elapsed)

                if callback.is_stdout_mode():
                    callback.emit_event(
                        "preprocessed_video_ready",
                        {"path": os.path.abspath(preprocessed_video_path)},
                    )

            logger.info("Task %s: Reading video metadata", task_id)
            callback.notify_preprocessing("正在读取视频元数据...")

            final_video_path = preprocessed_video_path if enable_preprocessing else Config.resolve_path(preprocessed_video_path)
            cap = cv2.VideoCapture(final_video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {final_video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = frame_rate
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if enable_preprocessing and preprocessing_duration > 0:
                preprocessing_average_fps = round(total_frames / preprocessing_duration, 3)

            logger.info(
                "Task %s: Video info - %s frames, %s fps, %sx%s",
                task_id,
                total_frames,
                fps,
                width,
                height,
            )

            logger.info("Task %s: Starting detect-only analysis", task_id)
            callback.notify_analyzing_start(total_frames, preprocessing_duration)

            analyzing_start = time.time()
            frame_index = 0

            self.metrics_calculator.reset()

            while cap.isOpened():
                with self.performance_trace.measure("videoRead"):
                    ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_index / fps
                with self.performance_trace.measure("detectorTotal"):
                    detections = self.yolo_tracker.detect_frame(
                        frame,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                    )
                all_detections.append(detections)

                if enable_dynamic_metrics:
                    with self.performance_trace.measure("metrics"):
                        metrics = self.metrics_calculator.calculate_metrics(frame_index, timestamp, frame)
                    all_metrics.append(metrics)

                processed_count = frame_index + 1
                if processed_count % Config.PROGRESS_UPDATE_INTERVAL == 0:
                    analyzing_elapsed = int(time.time() - analyzing_start)
                    total_elapsed = preprocessing_duration + analyzing_elapsed
                    is_timeout = total_elapsed > timeout_threshold
                    timeout_warning = total_elapsed > (timeout_threshold * 0.8)
                    progress = processed_count / total_frames if total_frames > 0 else 0

                    with self.performance_trace.measure("progressEmit"):
                        callback.update_progress(
                            {
                                "status": "ANALYZING",
                                "phase": "视频分析中",
                                "progress": round(progress, 4),
                                "currentFrame": frame_index,
                                "totalFrames": total_frames,
                                "preprocessingDuration": preprocessing_duration,
                                "analyzingElapsedTime": analyzing_elapsed,
                                "isTimeout": is_timeout,
                                "timeoutWarning": timeout_warning,
                            }
                        )

                frame_index += 1

            cap.release()
            self._last_analyzed_frame_count = frame_index

            analyzing_elapsed_seconds = time.time() - analyzing_start
            analyzing_duration = int(analyzing_elapsed_seconds)
            total_duration = preprocessing_duration + analyzing_duration
            is_timeout = total_duration > timeout_threshold
            detection_average_fps = (
                round(frame_index / analyzing_elapsed_seconds, 3)
                if analyzing_elapsed_seconds > 0
                else None
            )

            logger.info(
                "Task %s: Detect-only analysis completed - frames=%s, analyzing=%ss, timeout=%s",
                task_id,
                frame_index,
                analyzing_duration,
                is_timeout,
            )

            global_analysis = None
            if enable_dynamic_metrics:
                with self.performance_trace.measure("metrics"):
                    global_analysis = self.metrics_calculator.analyze_all(fps)

            with self.performance_trace.measure("eventGeneration"):
                anomaly_generator = AnomalyEventGenerator(fps=fps, total_frames=total_frames)
                anomaly_events = anomaly_generator.generate_events(all_detections)
            logger.info("Task %s: Generated %s anomaly events", task_id, len(anomaly_events))

            detections_file = Path(Config.get_detection_results_path(task_id))
            detections_file.parent.mkdir(parents=True, exist_ok=True)
            with self.performance_trace.measure("detectionsPersist"):
                atomic_write_json(
                    str(detections_file),
                    all_detections,
                    indent=2,
                    ensure_ascii=False,
                    use_lock=True,
                )

            result_status = "COMPLETED_TIMEOUT" if is_timeout else "COMPLETED"
            result_data = {
                "status": result_status,
                "isTimeout": is_timeout,
                "videoInfo": {
                    "sourceVideoFps": fps,
                    "totalFrames": total_frames,
                    "width": width,
                    "height": height,
                },
                "performance": {
                    "preprocessingAverageFps": preprocessing_average_fps,
                    "defectDetectionAverageFps": detection_average_fps,
                    "preprocessingDurationSeconds": preprocessing_duration,
                    "defectDetectionDurationSeconds": analyzing_duration,
                    "detectionBackend": self._detection_backend(),
                    "timingSummary": self.get_timing_summary(),
                },
                "dynamicMetrics": all_metrics,
                "globalAnalysis": global_analysis,
                "anomalyEvents": anomaly_events,
            }

            with self.performance_trace.measure("resultEmit"):
                callback.submit_result(result_data)
            logger.info("Task %s: Result submitted successfully", task_id)
            return result_status, final_video_path

        except Exception as e:
            logger.error("Task %s: Failed with error: %s", task_id, e, exc_info=True)
            try:
                callback.submit_result({"status": "FAILED", "failureReason": str(e)})
            except Exception as submit_error:
                logger.error("Task %s: Failed to submit error result: %s", task_id, submit_error)
            return "FAILED", final_video_path if final_video_path else Config.resolve_path(video_path)

    def _attach_performance_trace(self) -> None:
        if hasattr(self.yolo_tracker, "set_performance_trace"):
            self.yolo_tracker.set_performance_trace(self.performance_trace)

    def get_timing_summary(self) -> Dict[str, Any]:
        return self.performance_trace.summary(self._last_analyzed_frame_count)

    def _detection_backend(self) -> str:
        if str(self.model_path).lower().endswith(".onnx"):
            providers = getattr(self.yolo_tracker, "active_providers", [])
            if "CUDAExecutionProvider" in providers:
                return "onnxruntime-cuda"
            return "onnxruntime-cpu"

        device = str(self.device).lower()
        if device.startswith("cuda") or device.isdigit():
            return "pytorch-cuda"
        if device.startswith("mps"):
            return "pytorch-mps"
        return "pytorch-cpu"

    def get_info(self) -> Dict[str, Any]:
        return {
            "yolo_model": self.yolo_tracker.get_model_info(),
            "device": str(self.device),
        }

    def cleanup(self):
        if self._is_cleaned_up:
            return

        try:
            if hasattr(self, "yolo_tracker") and self.yolo_tracker is not None:
                if hasattr(self.yolo_tracker, "cleanup"):
                    self.yolo_tracker.cleanup()
                    self._is_cleaned_up = True
                    return

                if hasattr(self.yolo_tracker, "model") and self.yolo_tracker.model is not None:
                    try:
                        import torch

                        model_device = next(self.yolo_tracker.model.parameters()).device
                        if model_device.type != "cpu":
                            self.yolo_tracker.model.to("cpu")

                        if self.device == "cuda" and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif self.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                    except StopIteration:
                        pass
                    except Exception as e:
                        logger.warning("Failed to clear device cache: %s", e, exc_info=False)

            self._is_cleaned_up = True
        except Exception as e:
            logger.error("Error during cleanup: %s", e)

    def __del__(self):
        if not self._is_cleaned_up:
            try:
                self.cleanup()
            except Exception as e:
                logger.error("Error in destructor cleanup: %s", e)

    def export_annotated_video(
        self,
        task_id: int,
        video_path: str,
        output_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        callback_url: Optional[str] = None,
        frame_rate: float = 25.0,
        progress_status: str = "COMPLETED",
    ) -> bool:
        _ = confidence_threshold
        _ = iou_threshold
        callback = BackendCallback(task_id, callback_url)

        try:
            detections_file = Path(Config.get_detection_results_path(task_id))
            if not detections_file.exists():
                raise FileNotFoundError(f"Detection results not found: {detections_file}. Please run analyze_video_task first.")

            all_detections = safe_read_json(str(detections_file), use_lock=True, lock_timeout=30.0)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = frame_rate
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if len(all_detections) != total_frames:
                logger.warning(
                    "Task %s: Frame count mismatch - video has %s frames, detections have %s frames",
                    task_id,
                    total_frames,
                    len(all_detections),
                )

            estimate_size_mb = self.storage_manager.estimate_video_size(width, height, total_frames, fps)
            out, actual_output_path, finalize = self.storage_manager.create_video_writer(
                output_path,
                fps,
                width,
                height,
                estimate_size_mb=estimate_size_mb,
            )
            _ = actual_output_path

            frame_index = 0
            export_start = time.time()
            success = False

            with self.performance_trace.measure("resultVideoExport"):
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        detections = all_detections[frame_index] if frame_index < len(all_detections) else []
                        annotated_frame = self._draw_detections(frame, detections)
                        out.write(annotated_frame)

                        if frame_index % Config.PROGRESS_UPDATE_INTERVAL == 0:
                            progress = (frame_index + 1) / total_frames if total_frames > 0 else 0
                            with self.performance_trace.measure("progressEmit"):
                                callback.update_progress(
                                    {
                                        "status": progress_status,
                                        "phase": "生成结果视频",
                                        "progress": round(progress, 4),
                                        "currentFrame": frame_index,
                                        "totalFrames": total_frames,
                                    }
                                )

                        frame_index += 1

                    success = True
                finally:
                    cap.release()
                    finalize(success=success)

            validation_result = self.storage_manager.validate_video_file(output_path, check_frames=False)
            export_duration = int(time.time() - export_start)
            logger.info(
                "Task %s: Export completed in %ss, output=%s, size=%.2fMB",
                task_id,
                export_duration,
                output_path,
                validation_result["size_mb"],
            )
            return True

        except Exception as e:
            logger.error("Task %s: Failed to export video: %s", task_id, e, exc_info=True)
            return False

    def _draw_detections(self, frame, detections: List[Dict[str, Any]]) -> Any:
        annotated_frame = frame.copy()

        colors = {
            "熔池未到边": (0, 100, 0),
            "电极粘连物": (0, 0, 255),
            "锭冠": (255, 0, 0),
            "辉光": (255, 255, 0),
            "边弧（侧弧）": (128, 0, 128),
            "爬弧": (0, 165, 255),
        }

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            category = det.get("class_name", "Unknown")
            confidence = det.get("confidence", 0.0)
            color = colors.get(category, (0, 255, 0))

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{category} {confidence:.2f}"
            text_bg_height = 25
            text_bg_width = max(80, len(label) * 12)
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_bg_height),
                (x1 + text_bg_width, y1),
                color,
                -1,
            )
            annotated_frame = cv2_add_chinese_text(
                annotated_frame,
                label,
                (x1 + 2, y1 - text_bg_height + 2),
                font_size=16,
                color=(255, 255, 255),
            )

        return annotated_frame
