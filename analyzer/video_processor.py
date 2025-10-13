"""
视频处理主逻辑模块
整合YOLO检测、事件检测、动态参数计算等功能
"""
import cv2
import time
import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image, ImageDraw, ImageFont
from .yolo_tracker import YOLOTracker
from .trajectory_recorder import TrajectoryRecorder
from .metrics_calculator import MetricsCalculator
from .anomaly_event_generator import AnomalyEventGenerator
from utils.callback import BackendCallback
from utils.video_storage import VideoStorageManager
from utils.atomic_write import atomic_write_json, safe_read_json
from config import Config
from preprocessor import OptimizedVideoPreprocessor

logger = logging.getLogger(__name__)


def cv2_add_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在OpenCV图像上添加中文文本

    Args:
        img: OpenCV图像 (numpy array)
        text: 要添加的文本
        position: 文本位置 (x, y)
        font_size: 字体大小
        color: 文本颜色 (B, G, R)

    Returns:
        添加文本后的图像
    """
    # 将OpenCV图像(BGR)转换为PIL图像(RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 尝试使用系统字体
    font = None
    try:
        # 优先使用的字体路径列表(按优先级排序)
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # Windows 微软雅黑 (优先)
            'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
            'C:/Windows/Fonts/simsun.ttc',  # Windows 宋体
            '/System/Library/Fonts/PingFang.ttc',  # macOS PingFang
            '/System/Library/Fonts/STHeiti Medium.ttc',  # macOS 黑体
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Ubuntu 24.04 Noto CJK (优先)
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Ubuntu Noto CJK 备选路径
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # Ubuntu WQY Zen Hei
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux WQY Micro Hei
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux Droid (旧版)
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    logger.debug(f"成功加载字体: {font_path}")
                    break
                except Exception as font_error:
                    logger.debug(f"尝试加载字体失败 {font_path}: {font_error}")
                    continue

        if font is None:
            # Windows环境下的额外尝试：使用系统默认字体目录
            import platform
            if platform.system() == 'Windows':
                try:
                    # 尝试直接使用字体名称(Windows可以识别)
                    font = ImageFont.truetype("msyh.ttc", font_size)
                    logger.info("成功通过字体名称加载 msyh.ttc")
                except:
                    try:
                        font = ImageFont.truetype("simhei.ttf", font_size)
                        logger.info("成功通过字体名称加载 simhei.ttf")
                    except:
                        logger.warning("无法加载任何中文字体，文本可能无法正常显示")
                        font = ImageFont.load_default()
            else:
                logger.warning("未找到可用的中文字体，使用默认字体")
                font = ImageFont.load_default()
    except Exception as e:
        logger.warning(f"加载字体时发生异常: {e}, 使用默认字体")
        font = ImageFont.load_default()

    # 将BGR颜色转换为RGB
    color_rgb = (color[2], color[1], color[0])

    # 绘制文本
    draw.text(position, text, font=font, fill=color_rgb)

    # 将PIL图像转换回OpenCV图像
    img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_with_text


class VideoAnalyzer:
    """视频分析器"""

    def __init__(self, model_path: str, device: str = ''):
        """
        初始化视频分析器

        Args:
            model_path: YOLO模型路径
            device: 计算设备（空字符串表示自动选择，优先级：CUDA > MPS > CPU）
        """
        # 自动选择设备（CUDA > MPS > CPU）
        self.device = Config.auto_select_device(device)

        # 初始化各个组件
        self.yolo_tracker = YOLOTracker(model_path, self.device)
        self.trajectory_recorder = TrajectoryRecorder()
        self.metrics_calculator = MetricsCalculator()

        # 资源管理标志
        self._is_cleaned_up = False

        # 初始化视频预处理器（仅使用 CPU）
        self.preprocessor = OptimizedVideoPreprocessor()
        
        # 初始化视频存储管理器
        self.storage_manager = VideoStorageManager()

        logger.info(f"VideoAnalyzer initialized with device: {self.device}")

    def analyze_video_task(self, task_id: int, video_path: str,
                          video_duration: int, timeout_threshold: int,
                          confidence_threshold: float = 0.5,
                          iou_threshold: float = 0.45,
                          enable_preprocessing: bool = False,
                          preprocessing_strength: str = 'moderate',
                          preprocessing_enhance_pool: bool = True,
                          enable_tracking_merge: bool = True,
                          tracking_merge_strategy: str = 'auto',
                          callback_url: Optional[str] = None,
                          frame_rate: float = 25.0) -> tuple[str, str]:
        """
        分析视频任务（主处理函数）

        Args:
            task_id: 任务ID
            video_path: 视频路径
            video_duration: 视频时长（秒）
            timeout_threshold: 超时阈值（秒）
            confidence_threshold: 置信度阈值
            iou_threshold: IoU阈值
            enable_preprocessing: 是否启用视频预处理
            preprocessing_strength: 预处理强度（mild/moderate/strong）
            preprocessing_enhance_pool: 是否启用熔池增强
            enable_tracking_merge: 是否启用追踪轨迹合并
            tracking_merge_strategy: 追踪合并策略
            callback_url: 回调URL（可选，从MQ消息中获取）
            frame_rate: 视频帧率（从后端TaskConfig获取，由FFmpeg解析）

        Returns:
            tuple[str, str]: (任务状态, 实际分析的视频路径)
                - 任务状态: 'COMPLETED', 'COMPLETED_TIMEOUT', 'FAILED'
                - 实际分析的视频路径: 如果启用预处理则为预处理后的视频路径，否则为原始视频路径
        """

        callback = BackendCallback(task_id, callback_url)
        preprocessing_start = time.time()
        final_video_path = None  # 记录实际使用的视频路径

        try:
            # 初始化数据结构
            all_metrics = []
            all_detections = []

            # ===== 视频预处理阶段（可选） =====
            preprocessed_video_path = video_path
            if enable_preprocessing:
                logger.info(f"Task {task_id}: Starting video preprocessing (strength={preprocessing_strength}, enhance_pool={preprocessing_enhance_pool})")
                callback.notify_preprocessing(f"正在预处理视频（强度：{preprocessing_strength}）...")

                # 创建预处理视频存储目录
                # 使用推荐的 get_storage_path() 方法获取绝对路径
                preprocessed_dir = Path(Config.get_storage_path(Config.STORAGE_PREPROCESSED_VIDEOS_SUBDIR))
                preprocessed_dir.mkdir(parents=True, exist_ok=True)

                # 生成预处理后的视频文件名（提取基础名称，添加后缀，再添加时间戳）
                from utils.filename_utils import add_or_update_timestamp, extract_base_name
                
                video_stem = Path(video_path).stem
                # 提取原始基础名称（去掉时间戳）
                base_name = extract_base_name(video_stem)
                # 添加 _preprocessed 后缀，然后添加时间戳
                base_filename = f"{base_name}_preprocessed.mp4"
                preprocessed_filename = Path(add_or_update_timestamp(base_filename, update_existing=True)).name
                preprocessed_video_path = str(preprocessed_dir / preprocessed_filename)

                # 进度回调函数
                def preprocessing_progress_callback(current_frame, total_frames, elapsed_time):
                    progress = current_frame / total_frames
                    callback.update_progress({
                        'status': 'PREPROCESSING',
                        'phase': f'预处理视频中（{preprocessing_strength}）',
                        'progress': round(progress, 4),
                        'currentFrame': current_frame,
                        'totalFrames': total_frames,
                        'preprocessingDuration': int(elapsed_time)  # 预处理已耗时（秒）
                    })

                # 执行预处理
                self.preprocessor.process_video(
                    input_path=Config.resolve_path(video_path),
                    output_path=preprocessed_video_path,
                    frame_rate=frame_rate,  # 传递从后端获取的帧率
                    strength=preprocessing_strength,
                    enhance_pool=preprocessing_enhance_pool,
                    progress_callback=preprocessing_progress_callback
                )

                logger.info(f"Task {task_id}: Preprocessing completed, output: {preprocessed_video_path}")

                # 通知后端更新预处理视频路径
                try:
                    import requests
                    # 转换为相对于codes/目录的路径（如 storage/preprocessed_videos/xxx.mp4）
                    relative_path = Config.to_relative_path(os.path.abspath(preprocessed_video_path))
                    # 确保路径以 storage/ 开头（而不是 ../storage/）
                    if relative_path.startswith('../storage/'):
                        relative_path = relative_path[3:]  # 移除 '../'
                    elif relative_path.startswith('storage/'):
                        pass  # 已经是正确格式
                    
                    update_url = f"{Config.BACKEND_BASE_URL}/api/tasks/{task_id}/preprocessed-video"
                    response = requests.put(
                        update_url,
                        json={'preprocessedVideoPath': relative_path},
                        timeout=10
                    )
                    if response.status_code == 200:
                        logger.info(f"Task {task_id}: Preprocessed video path updated: {relative_path}")
                    else:
                        logger.warning(f"Task {task_id}: Failed to update preprocessed video path: {response.status_code}")
                except Exception as e:
                    logger.error(f"Task {task_id}: Failed to notify backend about preprocessed video: {e}")

            # ===== 元数据读取阶段 =====
            logger.info(f"Task {task_id}: Reading video metadata")
            callback.notify_preprocessing("正在读取视频元数据...")

            # 打开视频文件（如果启用预处理，则使用预处理后的视频）
            # preprocessed_video_path 现在要么是原始的 video_path（相对路径），
            # 要么是预处理后的绝对路径，统一通过 resolve_path 处理
            if enable_preprocessing:
                # 预处理后的路径已经是绝对路径
                final_video_path = preprocessed_video_path
            else:
                # 原始路径需要解析
                final_video_path = Config.resolve_path(preprocessed_video_path)
            
            logger.info(f"Task {task_id}: Opening video file: {final_video_path}")
            cap = cv2.VideoCapture(final_video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {final_video_path}")

            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 使用从后端传递的帧率，而不是从视频文件读取
            fps = frame_rate
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Task {task_id}: Video info - {total_frames} frames, {fps} fps (from backend config), {width}x{height}")

            preprocessing_duration = int(time.time() - preprocessing_start)

            # ===== 分析阶段 =====
            logger.info(f"Task {task_id}: Starting analysis")
            callback.notify_analyzing_start(total_frames, preprocessing_duration)

            analyzing_start = time.time()
            frame_count = 0

            # 重置组件
            self.yolo_tracker.reset_tracking()
            self.trajectory_recorder = TrajectoryRecorder()
            self.metrics_calculator.reset()

            # 逐帧处理
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp = frame_count / fps

                # 1. YOLO检测和追踪
                detections = self.yolo_tracker.track_frame(
                    frame,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    persist=True
                )

                # 保存当前帧的检测结果
                all_detections.append(detections)

                # 2. 轨迹记录
                self.trajectory_recorder.process_detections(frame_count, timestamp, detections)

                # 3. 计算动态参数
                metrics = self.metrics_calculator.calculate_metrics(
                    frame_count, timestamp, frame
                )
                all_metrics.append(metrics)

                # 4. 定期更新进度和保存检查点
                if frame_count % Config.PROGRESS_UPDATE_INTERVAL == 0:
                    analyzing_elapsed = int(time.time() - analyzing_start)
                    total_elapsed = preprocessing_duration + analyzing_elapsed
                    is_timeout = total_elapsed > timeout_threshold
                    timeout_warning = total_elapsed > (timeout_threshold * 0.8)

                    progress = frame_count / total_frames
                    estimated_remaining = int((total_frames - frame_count) / fps * analyzing_elapsed / frame_count) if frame_count > 0 else 0

                    callback.update_progress({
                        'status': 'ANALYZING',
                        'phase': '视频分析中',
                        'progress': round(progress, 4),
                        'currentFrame': frame_count,
                        'totalFrames': total_frames,
                        'preprocessingDuration': preprocessing_duration,
                        'analyzingElapsedTime': analyzing_elapsed,
                        'isTimeout': is_timeout,
                        'timeoutWarning': timeout_warning
                    })

                    if frame_count % (Config.PROGRESS_UPDATE_INTERVAL * 10) == 0:
                        logger.info(f"Task {task_id}: Progress {progress:.1%}, Frame {frame_count}/{total_frames}")

            # 释放视频资源
            cap.release()

            # ===== 完成处理 =====
            analyzing_duration = int(time.time() - analyzing_start)
            total_duration = preprocessing_duration + analyzing_duration
            is_timeout = total_duration > timeout_threshold

            logger.info(f"Task {task_id}: Analysis completed - analyzing: {analyzing_duration}s, total: {total_duration}s (timeout: {is_timeout})")

            # 进行全局频率分析
            logger.info(f"Task {task_id}: Performing global frequency analysis")
            global_analysis = self.metrics_calculator.analyze_all(fps)
            logger.info(f"Task {task_id}: Global analysis completed: {global_analysis}")

            # 完成轨迹记录，获取所有追踪物体
            self.trajectory_recorder.finalize_tracking()
            tracking_objects = self.trajectory_recorder.get_tracking_objects()

            # ✨ 应用追踪轨迹合并算法(如果启用)
            logger.info(f"Task {task_id}: Tracking merge enabled: {enable_tracking_merge}, Tracking objects count: {len(tracking_objects)}")
            if enable_tracking_merge and len(tracking_objects) > 0:
                logger.info(f"Task {task_id}: Applying tracking merge algorithm with strategy '{tracking_merge_strategy}'")
                logger.info(f"Task {task_id}: Original tracking objects: {len(tracking_objects)}")
                try:
                    from utils.tracking_utils import smart_merge
                    
                    # 应用智能合并
                    if tracking_merge_strategy == 'auto':
                        unified_objects, merge_report = smart_merge(tracking_objects, auto_scenario=True)
                    elif tracking_merge_strategy == 'adhesion':
                        from utils.tracking_utils import merge_for_adhesion
                        unified_objects, merge_report = merge_for_adhesion(tracking_objects)
                    elif tracking_merge_strategy == 'ingot_crown':
                        from utils.tracking_utils import merge_for_ingot_crown
                        unified_objects, merge_report = merge_for_ingot_crown(tracking_objects)
                    elif tracking_merge_strategy == 'conservative':
                        from utils.tracking_utils import merge_conservative
                        unified_objects, merge_report = merge_conservative(tracking_objects)
                    elif tracking_merge_strategy == 'aggressive':
                        from utils.tracking_utils import merge_aggressive
                        unified_objects, merge_report = merge_aggressive(tracking_objects)
                    else:
                        logger.warning(f"Task {task_id}: Unknown merge strategy '{tracking_merge_strategy}', using auto")
                        unified_objects, merge_report = smart_merge(tracking_objects, auto_scenario=True)
                    
                    # 使用合并后的结果
                    tracking_objects = unified_objects
                    logger.info(f"Task {task_id}: Merge completed - {merge_report['total_original_objects']} → {merge_report['total_unified_objects']} objects ({merge_report['merge_rate']})")
                    logger.info(f"Task {task_id}: Merged {merge_report['merged_groups']} groups")
                    if merge_report['merge_details']:
                        logger.info(f"Task {task_id}: Merge details: {merge_report['merge_details']}")
                except Exception as e:
                    logger.error(f"Task {task_id}: Failed to apply tracking merge: {e}, using original tracking objects", exc_info=True)
            else:
                logger.info(f"Task {task_id}: Tracking merge skipped (enabled={enable_tracking_merge}, objects={len(tracking_objects)})")

            # ✨ 生成基于轨迹的异常事件(在轨迹合并之后)
            logger.info(f"Task {task_id}: Generating anomaly events from tracking objects")
            anomaly_generator = AnomalyEventGenerator(
                fps=fps,
                video_path=final_video_path,
                debug_mode=Config.DEBUG  # 从配置读取调试模式开关
            )
            
            # 从视频路径中提取文件名
            video_filename = os.path.basename(final_video_path if final_video_path else video_path)
            
            # 生成异常事件
            anomaly_events = anomaly_generator.generate_events(
                tracking_objects=tracking_objects,
                video_filename=video_filename,
                total_frames=total_frames
            )
            logger.info(f"Task {task_id}: Generated {len(anomaly_events)} anomaly events")


            # 保存检测结果到文件,供生成结果视频时使用（使用原子写入）
            tracking_results_dir = Path(Config.get_storage_path('tracking_results'))
            tracking_results_dir.mkdir(parents=True, exist_ok=True)
            tracking_file = tracking_results_dir / f"{task_id}_tracking.json"

            try:
                atomic_write_json(
                    str(tracking_file),
                    all_detections,
                    indent=2,
                    ensure_ascii=False,
                    use_lock=True
                )
                logger.info(f"Task {task_id}: Tracking results saved atomically to {tracking_file}")
            except Exception as e:
                logger.error(f"Task {task_id}: Failed to save tracking results: {e}")

            # 提交结果
            result_status = 'COMPLETED_TIMEOUT' if is_timeout else 'COMPLETED'

            result_data = {
                'status': result_status,
                'isTimeout': is_timeout,
                'preprocessingDuration': preprocessing_duration,
                'analyzingDuration': analyzing_duration,
                'totalDuration': total_duration,
                'dynamicMetrics': all_metrics,  # 每帧的动态参数（面积、周长、亮度）
                'globalAnalysis': global_analysis,  # 全局频率分析结果（闪烁频率、趋势等）
                'anomalyEvents': anomaly_events,  # 基于轨迹生成的异常事件
                'trackingObjects': tracking_objects
            }

            callback.submit_result(result_data)

            logger.info(f"Task {task_id}: Result submitted successfully")
            logger.info(f"Task {task_id}: Tracked objects: {len(tracking_objects)}, Metrics: {len(all_metrics)}")
            
            # 返回最终状态和实际使用的视频路径
            return result_status, final_video_path

        except Exception as e:
            logger.error(f"Task {task_id}: Failed with error: {e}", exc_info=True)

            # 提交失败结果
            try:
                callback.submit_result({
                    'status': 'FAILED',
                    'failureReason': str(e)
                })
            except Exception as submit_error:
                logger.error(f"Task {task_id}: Failed to submit error result: {submit_error}")
            
            # 返回失败状态（使用原始视频路径或已处理的路径）
            return "FAILED", final_video_path if final_video_path else Config.resolve_path(video_path)

    def get_info(self) -> Dict[str, Any]:
        """
        获取分析器信息

        Returns:
            分析器信息字典
        """
        return {
            'yolo_model': self.yolo_tracker.get_model_info(),
            'device': str(self.device)
        }

    def cleanup(self):
        """
        清理资源（释放GPU/CPU内存）

        Note:
            - 释放YOLO模型占用的GPU/CPU内存
            - 清理视频预处理器的临时资源
            - 避免内存泄漏，特别是在创建多个分析器实例时
        """
        if self._is_cleaned_up:
            return

        try:
            # 清理YOLO模型资源
            if hasattr(self, 'yolo_tracker') and self.yolo_tracker is not None:
                if hasattr(self.yolo_tracker, 'model') and self.yolo_tracker.model is not None:
                    # 将模型移到CPU并清理GPU缓存
                    try:
                        import torch
                        # 只有当模型不在CPU上时才移动
                        model_device = next(self.yolo_tracker.model.parameters()).device
                        if model_device.type != 'cpu':
                            self.yolo_tracker.model.to('cpu')
                            logger.debug(f"Model moved from {model_device} to CPU")

                        # 清理设备缓存
                        if self.device == 'cuda':
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                logger.debug("CUDA cache cleared")
                        elif self.device == 'mps':
                            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                                torch.mps.empty_cache()
                                logger.debug("MPS cache cleared")
                    except StopIteration:
                        # 模型没有参数，无需清理
                        logger.debug("Model has no parameters, skipping device cleanup")
                    except Exception as e:
                        logger.warning(f"Failed to clear device cache: {e}", exc_info=False)

            # 清理预处理器资源（如果有临时文件）
            if hasattr(self, 'preprocessor') and self.preprocessor is not None:
                # 预处理器当前没有需要清理的资源，但保留扩展点
                pass

            self._is_cleaned_up = True
            logger.debug("VideoAnalyzer resources cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """
        析构函数：确保资源被释放
        """
        if not self._is_cleaned_up:
            logger.warning("VideoAnalyzer was not explicitly cleaned up, cleaning in destructor")
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"Error in destructor cleanup: {e}")

    def export_annotated_video(self, task_id: int, video_path: str,
                               output_path: str,
                               confidence_threshold: float = 0.5,
                               iou_threshold: float = 0.45,
                               callback_url: Optional[str] = None,
                               frame_rate: float = 25.0) -> bool:
        """
        导出带标注的视频（包含bbox、标签和ID）

        Args:
            task_id: 任务ID
            video_path: 输入视频路径
            output_path: 输出视频路径
            confidence_threshold: 置信度阈值（此参数已不使用，保留用于兼容）
            iou_threshold: IoU阈值（此参数已不使用，保留用于兼容）
            callback_url: 回调URL（可选）
            frame_rate: 视频帧率（从后端TaskConfig获取，由FFmpeg解析）

        Returns:
            是否成功导出
        """
        callback = BackendCallback(task_id, callback_url)

        try:
            logger.info(f"Task {task_id}: Starting export annotated video")

            # 读取之前保存的追踪结果
            tracking_file = Path(Config.get_storage_path('tracking_results')) / f"{task_id}_tracking.json"
            if not tracking_file.exists():
                raise FileNotFoundError(f"Tracking results not found: {tracking_file}. Please run analyze_video_task first.")

            logger.info(f"Task {task_id}: Loading tracking results from {tracking_file}")
            all_detections = safe_read_json(
                str(tracking_file),
                use_lock=True,
                lock_timeout=30.0
            )
            logger.info(f"Task {task_id}: Loaded {len(all_detections)} frames of tracking results")

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 使用从后端传递的帧率，而不是从视频文件读取
            fps = frame_rate
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 验证帧数是否匹配
            if len(all_detections) != total_frames:
                logger.warning(f"Task {task_id}: Frame count mismatch - video has {total_frames} frames, but tracking results have {len(all_detections)} frames")

            logger.info(f"Task {task_id}: Exporting video - {total_frames} frames, {fps} fps (from backend config), {width}x{height}")
            
            # 估算输出文件大小
            estimate_size_mb = self.storage_manager.estimate_video_size(
                width, height, total_frames, fps
            )
            
            # 创建视频写入器
            try:
                out, actual_output_path, finalize = self.storage_manager.create_video_writer(
                    output_path, fps, width, height, estimate_size_mb=estimate_size_mb
                )
            except Exception as e:
                cap.release()
                raise

            frame_count = 0
            export_start = time.time()
            success = False  # 标记是否成功完成

            try:
                # 逐帧处理
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 使用保存的检测结果（不再重新运行BotSORT）
                    if frame_count < len(all_detections):
                        detections = all_detections[frame_count]
                    else:
                        logger.warning(f"Task {task_id}: No tracking results for frame {frame_count + 1}")
                        detections = []

                    frame_count += 1

                    # 在帧上绘制检测结果
                    annotated_frame = self._draw_detections(frame, detections)

                    # 写入视频
                    out.write(annotated_frame)

                    # 定期更新进度
                    if frame_count % Config.PROGRESS_UPDATE_INTERVAL == 0:
                        progress = frame_count / total_frames
                        logger.info(f"Task {task_id}: Export progress {progress:.1%}, {frame_count}/{total_frames} frames")
                
                success = True  # 标记成功完成
                
            finally:
                # 释放资源
                cap.release()
                
                # 调用清理函数（会自动处理临时文件的移动或删除）
                finalize(success=success)

            # 验证输出文件
            try:
                validation_result = self.storage_manager.validate_video_file(output_path, check_frames=False)
                export_duration = int(time.time() - export_start)
                logger.info(f"Task {task_id}: Export completed in {export_duration}s, output: {output_path}")
                logger.info(f"Task {task_id}: Output file size: {validation_result['size_mb']:.2f}MB")
            except Exception as e:
                logger.error(f"Task {task_id}: Output file validation failed: {e}")
                raise

            return True

        except Exception as e:
            logger.error(f"Task {task_id}: Failed to export video: {e}", exc_info=True)
            return False

    def _draw_detections(self, frame, detections: List) -> Any:
        """
        在帧上绘制检测结果

        Args:
            frame: 视频帧
            detections: 检测结果列表

        Returns:
            标注后的帧
        """
        annotated_frame = frame.copy()
        
        # 定义颜色（BGR格式）
        colors = {
            '熔池未到边': (0, 100, 0),      # 深绿色
            '粘连物': (0, 0, 255),            # 红色
            '电极粘连物': (0, 0, 255),        # 红色
            '锭冠': (255, 0, 0),              # 蓝色
            '辉光': (255, 255, 0),            # 青色
            '边弧（侧弧）': (128, 0, 128),    # 紫色
            '边弧': (128, 0, 128),            # 紫色
            '侧弧': (128, 0, 128),            # 紫色
            '爬弧': (0, 165, 255)             # 橙色
        }

        for det in detections:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # 获取类别和ID
            category = det.get('class_name', 'Unknown')  # 使用class_name而不是category
            track_id = det.get('track_id', -1)
            confidence = det.get('confidence', 0.0)
            
            # 选择颜色
            color = colors.get(category, (0, 255, 0))  # 默认绿色
            
            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            if track_id >= 0:
                label = f"{category} ID:{track_id} {confidence:.2f}"
            else:
                label = f"{category} {confidence:.2f}"
            
            # 使用PIL绘制中文文本
            # 绘制文本背景 (简化处理，使用固定大小)
            text_bg_height = 25
            text_bg_width = len(label) * 12  # 粗略估计宽度
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_bg_height),
                (x1 + text_bg_width, y1),
                color,
                -1
            )
            
            # 使用PIL绘制中文文本
            annotated_frame = cv2_add_chinese_text(
                annotated_frame,
                label,
                (x1 + 2, y1 - text_bg_height + 2),
                font_size=16,
                color=(255, 255, 255)
            )

        return annotated_frame
