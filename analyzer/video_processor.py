"""
视频处理主逻辑模块。

Phase 0 使用 YOLO detect-only 链路，不再产生追踪对象或 track ID。
"""
import cv2
import time
import logging
import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from .metrics_calculator import MetricsCalculator
from .anomaly_event_generator import AnomalyEventGenerator
from utils.callback import BackendCallback
from utils.atomic_write import atomic_write_json
from utils.performance_trace import PerformanceTrace
from config import Config

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """视频分析器。"""

    def __init__(self, model_path: str, device: str = ""):
        self.model_path = model_path
        self.requested_device = device
        self.device = ""
        self.performance_trace = PerformanceTrace()
        self._last_analyzed_frame_count = 0
        self.metrics_calculator = MetricsCalculator()
        self._is_cleaned_up = False

        if not self._should_use_cpp_video_analyzer():
            raise RuntimeError("Windows CUDA worker requires best.onnx and C++ ONNX GPU analyzer sidecar")

        self.device = "cuda"
        logger.info("VideoAnalyzer initialized in C++ ONNX CUDA sidecar mode")

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
        preprocessing_benchmark: Optional[Dict[str, Any]] = None
        self.performance_trace = PerformanceTrace()
        self._last_analyzed_frame_count = 0

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
                if not callback.notify_preprocessing(f"正在预处理视频（强度：{preprocessing_strength}）..."):
                    raise RuntimeError("worker stdout 已关闭，停止视频预处理")

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

                preprocessing_benchmark = self._run_gpu_preprocessor(
                    input_path=Config.resolve_path(video_path),
                    output_path=preprocessed_video_path,
                    frame_rate=frame_rate,
                    callback=callback,
                )
                preprocessing_elapsed = time.time() - preprocessing_start
                preprocessing_duration = int(preprocessing_elapsed)

                if callback.is_stdout_mode():
                    if not callback.emit_event(
                        "preprocessed_video_ready",
                        {"path": os.path.abspath(preprocessed_video_path)},
                    ):
                        raise RuntimeError("worker stdout 已关闭，无法通知预处理视频已生成")

            logger.info("Task %s: Reading video metadata", task_id)
            if not callback.notify_preprocessing("正在读取视频元数据..."):
                raise RuntimeError("worker stdout 已关闭，停止视频分析")

            final_video_path = preprocessed_video_path if enable_preprocessing else Config.resolve_path(preprocessed_video_path)
            cap = cv2.VideoCapture(final_video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {final_video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = frame_rate
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if enable_preprocessing:
                if preprocessing_benchmark:
                    total_fps = preprocessing_benchmark.get("totalFps")
                    if isinstance(total_fps, (int, float)):
                        preprocessing_average_fps = round(float(total_fps), 3)
                elif preprocessing_duration > 0:
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
            if not callback.notify_analyzing_start(total_frames, preprocessing_duration):
                raise RuntimeError("worker stdout 已关闭，停止视频分析")

            analyzing_start = time.time()
            frame_index = 0
            detection_benchmark: Optional[Dict[str, Any]] = None

            self.metrics_calculator.reset()
            use_cpp_video_analyzer = self._should_use_cpp_video_analyzer()
            detections_file = Path(Config.get_detection_results_path(task_id))
            detections_file.parent.mkdir(parents=True, exist_ok=True)

            if not use_cpp_video_analyzer:
                raise RuntimeError("Python ONNX/PT fallback has been removed; C++ ONNX GPU analyzer is required")

            cap.release()
            cpp_result = self._run_cpp_video_analyzer(
                input_path=final_video_path,
                output_detections_path=str(detections_file),
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                callback=callback,
                preprocessing_duration=preprocessing_duration,
                timeout_threshold=timeout_threshold,
                total_frames=total_frames,
                started_at=analyzing_start,
            )
            all_detections = cpp_result["detections"]
            detection_benchmark = cpp_result.get("detectionBenchmark")
            frame_index = len(all_detections)

            if enable_dynamic_metrics:
                cap = cv2.VideoCapture(final_video_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot reopen video file for metrics: {final_video_path}")
                metric_frame_index = 0
                while cap.isOpened():
                    with self.performance_trace.measure("videoRead"):
                        ret, frame = cap.read()
                    if not ret:
                        break
                    timestamp = metric_frame_index / fps
                    with self.performance_trace.measure("metrics"):
                        metrics = self.metrics_calculator.calculate_metrics(metric_frame_index, timestamp, frame)
                    all_metrics.append(metrics)
                    metric_frame_index += 1
                cap.release()

            if detection_benchmark:
                total_fps = detection_benchmark.get("totalFps")
                if isinstance(total_fps, (int, float)):
                    detection_average_fps = round(float(total_fps), 3)
                else:
                    detection_average_fps = None
            else:
                detection_average_fps = None

            self._last_analyzed_frame_count = frame_index

            analyzing_elapsed_seconds = time.time() - analyzing_start
            analyzing_duration = int(analyzing_elapsed_seconds)
            total_duration = preprocessing_duration + analyzing_duration
            is_timeout = total_duration > timeout_threshold
            if detection_average_fps is None:
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
                    "detectionBackend": self._detection_backend(use_cpp_video_analyzer),
                    "preprocessingBenchmark": preprocessing_benchmark,
                    "detectionBenchmark": detection_benchmark,
                    "timingSummary": self.get_timing_summary(),
                },
                "dynamicMetrics": all_metrics,
                "globalAnalysis": global_analysis,
                "anomalyEvents": anomaly_events,
            }

            with self.performance_trace.measure("resultEmit"):
                if not callback.submit_result(result_data):
                    raise RuntimeError("worker stdout 已关闭，无法提交分析结果")
            logger.info("Task %s: Result submitted successfully", task_id)
            return result_status, final_video_path

        except Exception as e:
            logger.error("Task %s: Failed with error: %s", task_id, e, exc_info=True)
            try:
                callback.submit_result({"status": "FAILED", "failureReason": str(e)})
            except Exception as submit_error:
                logger.error("Task %s: Failed to submit error result: %s", task_id, submit_error)
            return "FAILED", final_video_path if final_video_path else Config.resolve_path(video_path)

    def _terminate_sidecar(self, process: subprocess.Popen, label: str, reason: str) -> None:
        if process.poll() is not None:
            return
        logger.error("%s: terminating sidecar because %s", label, reason)
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            logger.exception("%s: graceful sidecar termination failed; killing", label)
            try:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                logger.exception("%s: forced sidecar kill failed", label)

    def _ensure_detector(self):
        raise RuntimeError("Python ONNX/PT fallback has been removed; C++ ONNX GPU analyzer is required")

    def _resolve_gpu_preprocessor_bin(self) -> str:
        configured = getattr(Config, "GPU_PREPROCESSOR_BIN", "")
        if configured:
            resolved = Config.resolve_path(configured)
            if os.path.exists(resolved):
                return resolved
            if os.path.exists(configured):
                return configured

        exe_name = "var-gpu-preprocessor.exe" if os.name == "nt" else "var-gpu-preprocessor"
        candidates: List[Path] = []

        executable_path = Path(getattr(sys, "executable", "")).resolve()
        if executable_path.name:
            for parent in [executable_path.parent, *executable_path.parents]:
                candidates.append(parent / "tools" / exe_name)
                candidates.append(parent.parent / "tools" / exe_name)

        module_path = Path(__file__).resolve()
        for parent in [module_path.parent, *module_path.parents]:
            candidates.append(parent / "tools" / exe_name)
            candidates.append(parent / "frontend" / "src-tauri" / "resources" / "runtime" / "windows-x64" / "tools" / exe_name)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            "未找到 GPU 预处理 sidecar var-gpu-preprocessor.exe；"
            "请先运行 npm run desktop:build-gpu-sidecars，或设置 GPU_PREPROCESSOR_BIN"
        )

    def _run_gpu_preprocessor(
        self,
        input_path: str,
        output_path: str,
        frame_rate: float,
        callback: BackendCallback,
    ) -> Dict[str, Any]:
        sidecar_bin = self._resolve_gpu_preprocessor_bin()
        command = [
            sidecar_bin,
            "--input",
            input_path,
            "--output",
            output_path,
            "--fps",
            str(frame_rate),
        ]
        logger.info("Running GPU preprocessor sidecar: %s", command)

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

        benchmark: Optional[Dict[str, Any]] = None
        diagnostic_lines: List[str] = []

        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                diagnostic_lines.append(line)
                logger.warning("GPU preprocessor emitted non-JSON stdout: %s", line)
                continue

            event_type = event.get("type")
            if event_type == "progress":
                current_frame = int(event.get("currentFrame") or 0)
                total_frames = int(event.get("totalFrames") or 0)
                elapsed_seconds = float(event.get("elapsedSeconds") or 0.0)
                progress = current_frame / total_frames if total_frames > 0 else 0.0
                emitted = callback.update_progress(
                    {
                        "status": "PREPROCESSING",
                        "phase": "GPU 预处理视频中",
                        "progress": round(progress, 4),
                        "currentFrame": current_frame,
                        "totalFrames": total_frames,
                        "preprocessingDuration": int(elapsed_seconds),
                    }
                )
                if not emitted:
                    self._terminate_sidecar(process, "GPU preprocessor", "worker stdout is closed")
                    raise RuntimeError("worker stdout 已关闭，已终止 GPU 预处理 sidecar")
            elif event_type == "result":
                raw_benchmark = event.get("preprocessingBenchmark")
                if isinstance(raw_benchmark, dict):
                    benchmark = self._normalize_preprocessing_benchmark(raw_benchmark)
            elif event_type == "self_check":
                if not event.get("ok", False):
                    logger.error("GPU preprocessor self-check failed: %s", event)
            elif event_type == "error":
                logger.error("GPU preprocessor error event: %s", event)
            else:
                logger.debug("GPU preprocessor event: %s", event)

        return_code = process.wait()
        if diagnostic_lines:
            logger.error("GPU preprocessor diagnostics: %s", "\n".join(diagnostic_lines))

        if return_code != 0:
            detail = f"；diagnostics: {' | '.join(diagnostic_lines[-5:])}" if diagnostic_lines else ""
            raise RuntimeError(f"GPU 预处理失败，sidecar 退出码 {return_code}{detail}")

        if benchmark is None:
            raise RuntimeError("GPU 预处理失败：sidecar 未输出 preprocessingBenchmark")

        if not os.path.exists(output_path):
            raise RuntimeError(f"GPU 预处理失败：输出视频不存在 {output_path}")

        return benchmark

    def _normalize_preprocessing_benchmark(self, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(benchmark)
        total_frames = int(normalized.get("totalFrames") or 0)

        aliases = {
            "decodeDurationSeconds": "gpuDecodeDurationSeconds",
            "frameProcessingDurationSeconds": "gpuProcessingDurationSeconds",
            "encodeDurationSeconds": "gpuEncodeDurationSeconds",
        }
        for target_key, source_key in aliases.items():
            if target_key not in normalized and source_key in normalized:
                normalized[target_key] = float(normalized.get(source_key) or 0.0)
            normalized.pop(source_key, None)

        normalized.setdefault("decodeDurationSeconds", 0.0)
        normalized.setdefault("frameProcessingDurationSeconds", 0.0)
        normalized.setdefault("encodeDurationSeconds", 0.0)
        normalized.setdefault("otherDurationSeconds", 0.0)

        average_fields = {
            "decodeAverageMs": "decodeDurationSeconds",
            "frameProcessingAverageMs": "frameProcessingDurationSeconds",
            "encodeAverageMs": "encodeDurationSeconds",
        }
        for average_key, duration_key in average_fields.items():
            if average_key not in normalized:
                duration_seconds = float(normalized.get(duration_key) or 0.0)
                normalized[average_key] = (duration_seconds * 1000.0 / total_frames) if total_frames > 0 else 0.0

        return normalized

    def _should_use_cpp_video_analyzer(self) -> bool:
        if not getattr(Config, "USE_CPP_VIDEO_ANALYZER", False):
            return False
        return str(self.model_path).lower().endswith(".onnx")

    def _resolve_var_video_analyzer_bin(self) -> str:
        configured = getattr(Config, "VAR_VIDEO_ANALYZER_BIN", "")
        if configured:
            resolved = Config.resolve_path(configured)
            if os.path.exists(resolved):
                return resolved
            if os.path.exists(configured):
                return configured

        exe_name = "var-video-analyzer.exe" if os.name == "nt" else "var-video-analyzer"
        candidates: List[Path] = []

        executable_path = Path(getattr(sys, "executable", "")).resolve()
        if executable_path.name:
            for parent in [executable_path.parent, *executable_path.parents]:
                candidates.append(parent / "tools" / exe_name)
                candidates.append(parent.parent / "tools" / exe_name)

        module_path = Path(__file__).resolve()
        for parent in [module_path.parent, *module_path.parents]:
            candidates.append(parent / "tools" / exe_name)
            candidates.append(parent / "frontend" / "src-tauri" / "resources" / "runtime" / "windows-x64" / "tools" / exe_name)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(
            "未找到 C++ 视频分析 sidecar var-video-analyzer.exe；"
            "请先运行 npm run desktop:build-gpu-sidecars，或设置 VAR_VIDEO_ANALYZER_BIN"
        )

    def _normalize_cpp_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        bbox = detection.get("bbox") or detection.get("box")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"C++ analyzer 输出了无效 bbox: {detection}")

        class_id = detection.get("class_id", detection.get("classId"))
        class_name = detection.get("class_name", detection.get("className"))
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "bbox": [x1, y1, x2, y2],
            "center_x": float(detection.get("center_x", (x1 + x2) / 2.0)),
            "center_y": float(detection.get("center_y", (y1 + y2) / 2.0)),
            "width": float(detection.get("width", x2 - x1)),
            "height": float(detection.get("height", y2 - y1)),
            "confidence": float(detection["confidence"]),
        }

    def _load_cpp_detections(self, output_path: str) -> List[List[Dict[str, Any]]]:
        with open(output_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, list):
            raise ValueError("C++ analyzer 检测结果不是 frame 列表")

        normalized: List[List[Dict[str, Any]]] = []
        for frame_detections in raw_data:
            if not isinstance(frame_detections, list):
                raise ValueError("C++ analyzer 检测结果包含无效 frame 项")
            normalized.append([self._normalize_cpp_detection(item) for item in frame_detections])
        return normalized

    def _run_cpp_video_analyzer(
        self,
        input_path: str,
        output_detections_path: str,
        confidence_threshold: float,
        iou_threshold: float,
        callback: BackendCallback,
        preprocessing_duration: int,
        timeout_threshold: int,
        total_frames: int,
        started_at: float,
    ) -> Dict[str, Any]:
        sidecar_bin = self._resolve_var_video_analyzer_bin()
        model_path = Config.resolve_path(str(self.model_path))
        command = [
            sidecar_bin,
            "--input",
            Config.resolve_path(input_path),
            "--model",
            model_path,
            "--output-detections",
            output_detections_path,
            "--conf",
            str(confidence_threshold),
            "--iou",
            str(iou_threshold),
            "--progress-interval",
            str(max(1, int(Config.PROGRESS_UPDATE_INTERVAL))),
        ]
        logger.info("Running C++ video analyzer sidecar: %s", command)

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

        detection_benchmark: Optional[Dict[str, Any]] = None
        diagnostic_lines: List[str] = []

        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                diagnostic_lines.append(line)
                logger.warning("C++ video analyzer emitted non-JSON stdout: %s", line)
                continue

            event_type = event.get("type")
            if event_type == "progress":
                current_frame = int(event.get("currentFrame") or 0)
                analyzer_elapsed = float(event.get("elapsedSeconds") or (time.time() - started_at))
                total_elapsed = preprocessing_duration + int(analyzer_elapsed)
                progress = current_frame / total_frames if total_frames > 0 else 0.0
                emitted = callback.update_progress(
                    {
                        "status": "ANALYZING",
                        "phase": "C++ GPU 视频分析中",
                        "progress": round(progress, 4),
                        "currentFrame": current_frame,
                        "totalFrames": total_frames,
                        "preprocessingDuration": preprocessing_duration,
                        "analyzingElapsedTime": int(analyzer_elapsed),
                        "isTimeout": total_elapsed > timeout_threshold,
                        "timeoutWarning": total_elapsed > (timeout_threshold * 0.8),
                    }
                )
                if not emitted:
                    self._terminate_sidecar(process, "C++ video analyzer", "worker stdout is closed")
                    raise RuntimeError("worker stdout 已关闭，已终止 C++ 视频分析 sidecar")
            elif event_type == "result":
                raw_benchmark = event.get("detectionBenchmark")
                if isinstance(raw_benchmark, dict):
                    detection_benchmark = raw_benchmark
            elif event_type == "error":
                diagnostic_lines.append(json.dumps(event, ensure_ascii=False))
                logger.error("C++ video analyzer error event: %s", event)
            else:
                logger.debug("C++ video analyzer event: %s", event)

        return_code = process.wait()
        if diagnostic_lines:
            logger.error("C++ video analyzer diagnostics: %s", "\n".join(diagnostic_lines[-20:]))

        if return_code != 0:
            detail = f"；diagnostics: {' | '.join(diagnostic_lines[-5:])}" if diagnostic_lines else ""
            raise RuntimeError(f"C++ GPU 视频分析失败，sidecar 退出码 {return_code}{detail}")

        if detection_benchmark is None:
            raise RuntimeError("C++ GPU 视频分析失败：sidecar 未输出 detectionBenchmark")

        if not os.path.exists(output_detections_path):
            raise RuntimeError(f"C++ GPU 视频分析失败：检测结果不存在 {output_detections_path}")

        return {
            "detections": self._load_cpp_detections(output_detections_path),
            "detectionBenchmark": detection_benchmark,
        }

    def get_timing_summary(self) -> Dict[str, Any]:
        return self.performance_trace.summary(self._last_analyzed_frame_count)

    def _detection_backend(self, use_cpp_video_analyzer: bool = False) -> str:
        if use_cpp_video_analyzer:
            return "onnxruntime-cuda-cpp"

        raise RuntimeError("C++ ONNX GPU analyzer is required")

    def get_info(self) -> Dict[str, Any]:
        if self._should_use_cpp_video_analyzer():
            return {
                "yolo_model": {
                    "backend": "onnxruntime-cuda-cpp",
                    "model_path": str(self.model_path),
                    "model_version": f"onnxruntime-cpp:{Path(self.model_path).name}",
                },
                "device": "cuda",
            }

        raise RuntimeError("C++ ONNX GPU analyzer is required")

    def cleanup(self):
        if self._is_cleaned_up:
            return

        try:
            self._is_cleaned_up = True
        except Exception as e:
            logger.error("Error during cleanup: %s", e)

    def __del__(self):
        if not self._is_cleaned_up:
            try:
                self.cleanup()
            except Exception as e:
                logger.error("Error in destructor cleanup: %s", e)
