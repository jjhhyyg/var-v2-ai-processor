"""
视频处理主逻辑模块
整合YOLO检测、事件检测、动态参数计算等功能
"""
import cv2
import time
import logging
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image, ImageDraw, ImageFont
from .yolo_tracker import YOLOTracker
from .event_detector import EventDetector
from .metrics_calculator import MetricsCalculator
from .anomaly_event_generator import AnomalyEventGenerator
from utils.callback import BackendCallback
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
    try:
        # macOS字体路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',  # macOS PingFang
            '/System/Library/Fonts/STHeiti Medium.ttc',  # macOS 黑体
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # Linux WQY
            'C:/Windows/Fonts/msyh.ttc',  # Windows 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        
        if font is None:
            # 如果没有找到字体，使用默认字体
            font = ImageFont.load_default()
    except Exception as e:
        logger.warning(f"加载字体失败: {e}, 使用默认字体")
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
        self.event_detector = EventDetector()
        self.metrics_calculator = MetricsCalculator()

        # 初始化视频预处理器
        # CUDA和MPS都支持GPU加速，对于OpenCV CUDA，只有CUDA可用
        use_gpu_opencv = 'cuda' in str(self.device).lower()
        self.preprocessor = OptimizedVideoPreprocessor(use_gpu=use_gpu_opencv)

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
                          callback_url: Optional[str] = None) -> tuple[str, str]:
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
            callback_url: 回调URL（可选，从MQ消息中获取）

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

                # 创建预处理视频存储目录（使用codes/storage/preprocessed_videos）
                # 确保使用绝对路径
                preprocessed_dir = Path(Config.resolve_path(Config.PREPROCESSED_VIDEO_PATH))
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
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Task {task_id}: Video info - {total_frames} frames, {fps} fps, {width}x{height}")

            preprocessing_duration = int(time.time() - preprocessing_start)

            # ===== 分析阶段 =====
            logger.info(f"Task {task_id}: Starting analysis")
            callback.notify_analyzing_start(total_frames, preprocessing_duration)

            analyzing_start = time.time()
            frame_count = 0

            # 重置组件
            self.yolo_tracker.reset_tracking()
            self.event_detector = EventDetector()
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

                # 2. 事件检测
                self.event_detector.process_detections(frame_count, timestamp, detections)

                # 3. 计算动态参数
                metrics = self.metrics_calculator.calculate_metrics(
                    frame_count, timestamp, frame
                )
                all_metrics.append(metrics)

                # 4. 定期更新进度和保存检查点
                if frame_count % Config.PROGRESS_UPDATE_INTERVAL == 0:
                    analyzing_elapsed = int(time.time() - analyzing_start)
                    is_timeout = analyzing_elapsed > timeout_threshold
                    timeout_warning = analyzing_elapsed > (timeout_threshold * 0.8)

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
            is_timeout = analyzing_duration > timeout_threshold

            logger.info(f"Task {task_id}: Analysis completed - {analyzing_duration}s (timeout: {is_timeout})")

            # 进行全局频率分析
            logger.info(f"Task {task_id}: Performing global frequency analysis")
            global_analysis = self.metrics_calculator.analyze_all(fps)
            logger.info(f"Task {task_id}: Global analysis completed: {global_analysis}")

            # 获取追踪物体信息（不再生成异常事件）
            final_events = self.event_detector.finalize_events()  # 返回空列表
            tracking_objects = self.event_detector.get_tracking_objects()

            # ✨ 生成基于轨迹的异常事件
            logger.info(f"Task {task_id}: Generating anomaly events from tracking objects")
            anomaly_generator = AnomalyEventGenerator(fps=fps)
            
            # 从视频路径中提取文件名
            video_filename = os.path.basename(final_video_path if final_video_path else video_path)
            
            # 生成异常事件
            anomaly_events = anomaly_generator.generate_events(
                tracking_objects=tracking_objects,
                video_filename=video_filename,
                total_frames=total_frames
            )
            logger.info(f"Task {task_id}: Generated {len(anomaly_events)} anomaly events")

            # ✨ 应用追踪轨迹合并算法（如果启用）
            if enable_tracking_merge and len(tracking_objects) > 0:
                logger.info(f"Task {task_id}: Applying tracking merge algorithm with strategy '{tracking_merge_strategy}'")
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
                    logger.info(f"Task {task_id}: Merged {merge_report['total_merge_groups']} groups")
                except Exception as e:
                    logger.error(f"Task {task_id}: Failed to apply tracking merge: {e}, using original tracking objects")

            # 保存检测结果到文件,供生成结果视频时使用
            tracking_results_dir = Path('storage/tracking_results')
            tracking_results_dir.mkdir(parents=True, exist_ok=True)
            tracking_file = tracking_results_dir / f"{task_id}_tracking.json"

            try:
                with open(tracking_file, 'w', encoding='utf-8') as f:
                    json.dump(all_detections, f, ensure_ascii=False, indent=2)
                logger.info(f"Task {task_id}: Tracking results saved to {tracking_file}")
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

    def export_annotated_video(self, task_id: int, video_path: str,
                               output_path: str,
                               confidence_threshold: float = 0.5,
                               iou_threshold: float = 0.45,
                               callback_url: Optional[str] = None) -> bool:
        """
        导出带标注的视频（包含bbox、标签和ID）

        Args:
            task_id: 任务ID
            video_path: 输入视频路径
            output_path: 输出视频路径
            confidence_threshold: 置信度阈值（此参数已不使用，保留用于兼容）
            iou_threshold: IoU阈值（此参数已不使用，保留用于兼容）
            callback_url: 回调URL（可选）

        Returns:
            是否成功导出
        """
        callback = BackendCallback(task_id, callback_url)

        try:
            logger.info(f"Task {task_id}: Starting export annotated video")

            # 读取之前保存的追踪结果
            tracking_file = Path('storage/tracking_results') / f"{task_id}_tracking.json"
            if not tracking_file.exists():
                raise FileNotFoundError(f"Tracking results not found: {tracking_file}. Please run analyze_video_task first.")

            logger.info(f"Task {task_id}: Loading tracking results from {tracking_file}")
            with open(tracking_file, 'r', encoding='utf-8') as f:
                all_detections = json.load(f)
            logger.info(f"Task {task_id}: Loaded {len(all_detections)} frames of tracking results")

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 尝试使用浏览器兼容的编码器
            # 优先使用H.264编码（浏览器原生支持），降级到mp4v
            codecs_to_try = [
                ('avc1', 'H.264 (AVC)'),  # H.264编码，浏览器友好
                ('x264', 'x264'),         # x264编码器
                ('H264', 'H.264'),        # H.264别名
                ('mp4v', 'MPEG-4')        # 降级方案（可能不被所有浏览器支持）
            ]

            out = None
            used_codec = None

            for codec, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        used_codec = codec_name
                        logger.info(f"Task {task_id}: Using {codec_name} codec")
                        break
                    else:
                        out.release()
                except Exception as e:
                    logger.debug(f"Task {task_id}: Codec {codec} not available: {e}")
                    continue

            if not out or not out.isOpened():
                raise ValueError(f"Cannot create output video with any available codec")

            if used_codec == 'MPEG-4':
                logger.warning(f"Task {task_id}: Using MPEG-4 codec, video may not be playable in all browsers. Consider installing OpenCV with H.264 support.")

            logger.info(f"Task {task_id}: Exporting video - {total_frames} frames, {fps} fps, {width}x{height}")

            # 验证帧数是否匹配
            if len(all_detections) != total_frames:
                logger.warning(f"Task {task_id}: Frame count mismatch - video has {total_frames} frames, but tracking results have {len(all_detections)} frames")

            frame_count = 0
            export_start = time.time()

            # 逐帧处理
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 使用保存的检测结果（不再重新运行ByteTrack）
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

                    # 生成结果视频时不更新任务状态，避免触发completedAt更新
                    # 结果视频生成不属于任务的生命周期
                    # callback.update_progress({
                    #     'status': 'COMPLETED',
                    #     'phase': '生成结果视频',
                    #     'progress': round(progress, 4),
                    #     'currentFrame': frame_count,
                    #     'totalFrames': total_frames
                    # })

                    logger.info(f"Task {task_id}: Export progress {progress:.1%}, {frame_count}/{total_frames} frames")

            # 释放资源
            cap.release()
            out.release()

            # 验证输出文件是否存在且有效
            if not os.path.exists(output_path):
                logger.error(f"Task {task_id}: Result video file was not created: {output_path}")
                raise ValueError(f"结果视频文件未成功创建: {output_path}")

            output_file_size = os.path.getsize(output_path)
            if output_file_size == 0:
                logger.error(f"Task {task_id}: Result video file size is 0: {output_path}")
                raise ValueError(f"结果视频文件创建失败（文件大小为0）")

            export_duration = int(time.time() - export_start)
            logger.info(f"Task {task_id}: Export completed in {export_duration}s, output: {output_path}")
            logger.info(f"Task {task_id}: Output file size: {output_file_size / 1024 / 1024:.2f} MB")

            # 生成结果视频时不更新任务状态，避免触发completedAt更新
            # 结果视频生成不属于任务的生命周期
            # callback.update_progress({
            #     'status': 'COMPLETED',
            #     'phase': '生成结果视频',
            #     'progress': 1.0,
            #     'currentFrame': total_frames,
            #     'totalFrames': total_frames
            # })
            # logger.info(f"Task {task_id}: Export progress pushed to 100%")

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
