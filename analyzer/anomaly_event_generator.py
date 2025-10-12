"""
异常事件生成模块
基于物体追踪轨迹生成异常事件（包括粘连物脱落位置判断）

功能：
1. 粘连物形成事件：取前3秒作为异常事件
2. 粘连物脱落事件：取后3秒作为异常事件，使用连通域分析判断脱落位置
3. 锭冠脱落事件：取后3秒作为异常事件，固定为落入熔池
4. 其他事件按原有逻辑生成

作者：侯阳洋
日期：2025-10-11（更新）
"""
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)


class AnomalyEventGenerator:
    """异常事件生成器"""

    def __init__(self, fps: float = 30.0, video_path: Optional[str] = None, debug_mode: bool = False):
        """
        初始化异常事件生成器

        Args:
            fps: 视频帧率（用于计算时间窗口）
            video_path: 视频文件路径（用于读取帧进行图像分析）
            debug_mode: 调试模式，启用时会保存中间处理结果到 debug_output 目录
        """
        self.fps = fps
        self.time_window_frames = int(5 * fps)  # 5秒对应的帧数（用于其他事件）
        self.event_window_frames = int(3 * fps)  # 3秒对应的帧数（用于粘连物/锭冠事件）
        self.video_path = video_path
        self.frame_cache = {}  # 缓存读取的帧，避免重复IO
        self.video_cap = None  # VideoCapture对象
        self.debug_mode = debug_mode  # 调试模式开关
        self.debug_output_dir = None  # 调试输出目录

        # 如果启用调试模式，创建调试输出目录
        if self.debug_mode and self.video_path:
            video_name = Path(self.video_path).stem
            self.debug_output_dir = Path(self.video_path).parent / 'debug_output' / video_name
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Debug mode enabled, output dir: {self.debug_output_dir}")

        logger.info(f"AnomalyEventGenerator initialized with fps={fps}, "
                   f"time_window={self.time_window_frames} frames, "
                   f"event_window={self.event_window_frames} frames, "
                   f"video_path={video_path}, debug_mode={debug_mode}")

    def generate_events(self, 
                       tracking_objects: List[Dict[str, Any]], 
                       video_filename: str,
                       total_frames: int) -> List[Dict[str, Any]]:
        """
        基于追踪物体生成异常事件

        Args:
            tracking_objects: 追踪物体列表（来自 TrajectoryRecorder.get_tracking_objects()）
            video_filename: 视频文件名（用于判断左/右视角）
            total_frames: 视频总帧数

        Returns:
            异常事件列表，每个事件包含：
            - eventType: 事件类型（对应后端 EventType 枚举）
            - startFrame: 起始帧号
            - endFrame: 结束帧号
            - objectId: 物体ID（可为None）
            - metadata: 元数据（JSON格式）
        """
        events = []
        
        # 判断视频视角（左/右）
        video_perspective = self._determine_video_perspective(video_filename)
        logger.info(f"Video perspective determined: {video_perspective} (filename: {video_filename})")
        
        # 按类别分组
        objects_by_category = {}
        for obj in tracking_objects:
            category = obj['category']
            if category not in objects_by_category:
                objects_by_category[category] = []
            objects_by_category[category].append(obj)
        
        # 为每个类别生成事件
        for category, objects in objects_by_category.items():
            if not objects:
                continue
            
            # 按起始帧排序
            objects.sort(key=lambda x: x['firstFrame'])
            
            # 特殊处理：粘连物和锭冠
            if category == 'ADHESION':
                # 不再过滤短时粘连物，所有粘连物都需要检查
                # 只要OpenCV判断在熔池内，就生成事件
                logger.info(f"Processing {len(objects)} adhesion objects (including short-duration ones)")
                
                # 为每个粘连物生成事件（如果掉到熔池里）
                for obj in objects:
                    adhesion_events = self._generate_adhesion_events(obj, video_perspective)
                    events.extend(adhesion_events)
                    if len(adhesion_events) > 0:
                        logger.info(f"Generated {len(adhesion_events)} events for adhesion objectId={obj['objectId']}")
            
            elif category == 'CROWN':
                # 过滤掉持续时间太短的锭冠（噪声/误检）
                min_duration_frames = int(0.5 * self.fps)  # 至少0.5秒
                valid_objects = [
                    obj for obj in objects 
                    if (obj['lastFrame'] - obj['firstFrame'] + 1) >= min_duration_frames
                ]
                
                logger.info(f"Filtered crown objects: {len(objects)} → {len(valid_objects)} (min duration: {min_duration_frames} frames)")
                
                # 为每个锭冠生成脱落事件
                for obj in valid_objects:
                    crown_events = self._generate_crown_events(obj)
                    events.extend(crown_events)
                    logger.info(f"Generated {len(crown_events)} events for crown objectId={obj['objectId']}")
            
            else:
                # 其他类别按原有逻辑处理（取最早和最晚物体）
                # 找出最早出现的物体（前5秒无此类物体）
                first_obj = self._find_first_appearance(objects, total_frames)
                # 找出最晚消失的物体（后5秒无此类物体）
                last_obj = self._find_last_appearance(objects, total_frames)
                
                if first_obj and last_obj:
                    # 生成时间段事件
                    event_type = self._get_event_type_for_category(category)
                    
                    if event_type:
                        event = {
                            'eventType': event_type,
                            'startFrame': first_obj['firstFrame'],
                            'endFrame': last_obj['lastFrame'],
                            'objectId': None,  # 按要求留空
                            'metadata': None  # 这些事件不需要metadata
                        }
                        
                        events.append(event)
                        logger.info(f"Generated event for category {category}: {event_type}, "
                                  f"frames {first_obj['firstFrame']}-{last_obj['lastFrame']}, "
                                  f"{len(objects)} objects")
        
        # 清理资源
        self._cleanup()
        
        logger.info(f"Generated {len(events)} anomaly events from {len(tracking_objects)} tracking objects")
        return events

    def _generate_adhesion_events(self, adhesion_obj: Dict[str, Any], 
                                  video_perspective: str) -> List[Dict[str, Any]]:
        """
        为粘连物生成形成和脱落事件
        
        策略：
        1. 先判断脱落位置（是否在熔池中）
        2. 只有在熔池中的才生成事件
        3. 对于持续时间<6秒的，只生成脱落事件
        4. 对于持续时间≥6秒的，生成形成+脱落事件
        
        Args:
            adhesion_obj: 粘连物对象
            video_perspective: 视频视角 ('LEFT' 或 'RIGHT')
        
        Returns:
            事件列表（包含形成事件和脱落事件，或只有脱落事件，或空列表）
        """
        events = []
        first_frame = adhesion_obj['firstFrame']
        last_frame = adhesion_obj['lastFrame']
        duration = last_frame - first_frame + 1
        
        # 判断脱落位置
        drop_location = self._analyze_adhesion_drop(adhesion_obj, video_perspective)
        
        # 只有掉到熔池里的粘连物才生成事件
        if drop_location != 'pool':
            logger.debug(f"Adhesion objectId={adhesion_obj['objectId']}, duration={duration}f, not in pool (location={drop_location}), skipping")
            return events
        
        logger.debug(f"Adhesion objectId={adhesion_obj['objectId']}, duration={duration}f, dropped to pool, generating events")
        
        # 如果持续时间太短（<6秒），只生成脱落事件，避免形成和脱落事件重叠
        if duration < (2 * self.event_window_frames):
            # 只生成脱落事件
            drop_event = {
                'eventType': 'ADHESION_DROPPED',
                'startFrame': first_frame,
                'endFrame': last_frame,
                'objectId': None,
                'metadata': {
                    'dropped_location': drop_location
                }
            }
            events.append(drop_event)
            logger.debug(f"Short adhesion ({duration} frames): only drop event, frames {first_frame}-{last_frame}")
        else:
            # 正常情况：生成形成和脱落两个事件
            # 1. 生成粘连物形成事件（前3秒）
            formation_end_frame = min(first_frame + self.event_window_frames, last_frame)
            formation_event = {
                'eventType': 'ADHESION_FORMED',
                'startFrame': first_frame,
                'endFrame': formation_end_frame,
                'objectId': None,
                'metadata': None
            }
            events.append(formation_event)
            logger.debug(f"Adhesion formation event: frames {first_frame}-{formation_end_frame}")
            
            # 2. 生成粘连物脱落事件（后3秒）
            drop_start_frame = max(last_frame - self.event_window_frames, first_frame)
            
            drop_event = {
                'eventType': 'ADHESION_DROPPED',
                'startFrame': drop_start_frame,
                'endFrame': last_frame,
                'objectId': None,
                'metadata': {
                    'dropped_location': drop_location
                }
            }
            events.append(drop_event)
            logger.debug(f"Adhesion drop event: frames {drop_start_frame}-{last_frame}")
        
        return events

    def _generate_crown_events(self, crown_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        为锭冠生成脱落事件
        
        Args:
            crown_obj: 锭冠对象
        
        Returns:
            事件列表（只包含脱落事件）
        """
        events = []
        first_frame = crown_obj['firstFrame']
        last_frame = crown_obj['lastFrame']
        
        # 生成锭冠脱落事件（后3秒，固定为落入熔池）
        drop_start_frame = max(last_frame - self.event_window_frames, first_frame)
        
        drop_event = {
            'eventType': 'CROWN_DROPPED',
            'startFrame': drop_start_frame,
            'endFrame': last_frame,
            'objectId': None,
            'metadata': {
                'dropped_location': 'pool'  # 锭冠脱落固定为落入熔池
            }
        }
        events.append(drop_event)
        logger.debug(f"Crown drop event: frames {drop_start_frame}-{last_frame}, location=pool")
        
        return events

    def _determine_video_perspective(self, video_filename: str) -> str:
        """
        根据视频文件名判断视角（左/右）
        
        规则：如果文件名包含 'left'、'LEFT'、'L'、'左' 等，判定为左视角；
             如果包含 'right'、'RIGHT'、'R'、'右' 等，判定为右视角；
             否则默认为左视角
        
        Args:
            video_filename: 视频文件名
        
        Returns:
            'LEFT' 或 'RIGHT'
        """
        filename_lower = video_filename.lower()
        
        # 左视角标识
        left_markers = ['left', '_l_', '_l.', 'l_', '左']
        # 右视角标识
        right_markers = ['right', '_r_', '_r.', 'r_', '右']
        
        for marker in right_markers:
            if marker in filename_lower:
                return 'RIGHT'
        
        for marker in left_markers:
            if marker in filename_lower:
                return 'LEFT'
        
        # 默认左视角
        logger.warning(f"Unable to determine video perspective from filename '{video_filename}', "
                      f"defaulting to LEFT")
        return 'LEFT'

    def _find_first_appearance(self, objects: List[Dict[str, Any]], total_frames: int) -> Optional[Dict[str, Any]]:
        """
        找出最早出现的物体（前5秒无此类物体）
        
        Args:
            objects: 同类物体列表（已按起始帧排序）
            total_frames: 视频总帧数
        
        Returns:
            最早出现的物体，如果不满足条件则返回 None
        """
        if not objects:
            return None
        
        # 第一个物体
        first_obj = objects[0]
        
        # 检查前5秒是否有此类物体
        # 如果起始帧在视频开始的5秒内，或者前5秒确实没有此类物体，则满足条件
        if first_obj['firstFrame'] <= self.time_window_frames:
            return first_obj
        
        # 检查前5秒帧范围内是否有其他同类物体
        time_window_start = max(0, first_obj['firstFrame'] - self.time_window_frames)
        has_object_in_window = any(
            obj['lastFrame'] >= time_window_start and obj['firstFrame'] < first_obj['firstFrame']
            for obj in objects
        )
        
        if not has_object_in_window:
            return first_obj
        
        return None

    def _find_last_appearance(self, objects: List[Dict[str, Any]], total_frames: int) -> Optional[Dict[str, Any]]:
        """
        找出最晚消失的物体（后5秒无此类物体）
        
        Args:
            objects: 同类物体列表（已按起始帧排序）
            total_frames: 视频总帧数
        
        Returns:
            最晚消失的物体，如果不满足条件则返回 None
        """
        if not objects:
            return None
        
        # 按结束帧排序，取最后一个
        objects_by_end = sorted(objects, key=lambda x: x['lastFrame'])
        last_obj = objects_by_end[-1]
        
        # 检查后5秒是否有此类物体
        # 如果结束帧在视频最后5秒内，或者后5秒确实没有此类物体，则满足条件
        if last_obj['lastFrame'] >= total_frames - self.time_window_frames:
            return last_obj
        
        # 检查后5秒帧范围内是否有其他同类物体
        time_window_end = min(total_frames, last_obj['lastFrame'] + self.time_window_frames)
        has_object_in_window = any(
            obj['firstFrame'] <= time_window_end and obj['lastFrame'] > last_obj['lastFrame']
            for obj in objects
        )
        
        if not has_object_in_window:
            return last_obj
        
        return None

    def _get_event_type_for_category(self, category: str) -> Optional[str]:
        """
        根据物体类别获取对应的事件类型
        
        Args:
            category: 物体类别（对应后端 ObjectCategory 枚举）
        
        Returns:
            事件类型（对应后端 EventType 枚举），如果不需要生成事件则返回 None
        """
        # 类别到事件类型的映射
        category_to_event = {
            'POOL_NOT_REACHED': 'POOL_NOT_REACHED',
            'ADHESION': 'ADHESION_FORMED',  # 粘连物出现事件
            'CROWN': 'CROWN_DROPPED',       # 锭冠脱落事件
            'GLOW': 'GLOW',
            'SIDE_ARC': 'SIDE_ARC',
            'CREEPING_ARC': 'CREEPING_ARC'
        }
        
        return category_to_event.get(category)

    def _analyze_adhesion_drop(self, adhesion_obj: Dict[str, Any], 
                               video_perspective: str) -> str:
        """
        分析粘连物的脱落位置
        
        判断逻辑优先级：
        1. 使用连通域分析判断最后几帧是否与电极断开（被熔池包围）→ 'pool'
        2. 检查粘连物存在前的最后一帧，向前10帧是否有明显运动且不在熔池中 → 'crystallizer'
        3. 使用原有的方向判断作为fallback
        
        改进：检查最后3-5帧的稳定性，避免单帧误判
        
        Args:
            adhesion_obj: 粘连物对象
            video_perspective: 视频视角 ('LEFT' 或 'RIGHT')
        
        Returns:
            脱落位置：'pool'（熔池）或 'crystallizer'（结晶器）
        """
        trajectory = adhesion_obj.get('trajectory', [])
        
        if len(trajectory) < 2:
            # 轨迹太短，默认为熔池
            logger.debug(f"Trajectory too short, defaulting to pool")
            return 'pool'
        
        last_frame_num = adhesion_obj['lastFrame']
        last_trajectory = trajectory[-1]
        last_bbox = last_trajectory.get('bbox', [])
        
        # 策略1: 检查最后几帧是否被熔池包围（与电极断开连接）
        # 改进：检查最后3-5帧，避免单帧误判
        frames_to_check = min(5, len(trajectory))
        pool_count = 0
        
        for i in range(1, frames_to_check + 1):
            traj_point = trajectory[-i]
            frame_num = traj_point['frame']
            bbox_to_check = traj_point.get('bbox', [])
            
            if bbox_to_check:
                frame = self._read_frame(frame_num)
                if frame is not None:
                    is_in_pool = self._is_surrounded_by_pool(frame, bbox_to_check, frame_num)
                    if is_in_pool:
                        pool_count += 1
        
        # 如果最后几帧中大部分都在熔池中（至少60%），判定为脱落到熔池
        pool_ratio = pool_count / frames_to_check
        if pool_ratio >= 0.6:
            logger.debug(f"Last {frames_to_check} frames: {pool_count} in pool ({pool_ratio:.1%}), dropped to pool")
            return 'pool'
        
        # 策略2: 检查是否被结晶器捕获
        # 粘连物存在前的最后一帧，检查前10帧的运动情况
        if len(trajectory) >= 10:
            # 取倒数第10帧到最后一帧的轨迹
            check_trajectory = trajectory[-10:]
            
            # 计算累计运动距离
            movement_distance = self._calculate_trajectory_movement(check_trajectory)
            
            # 检查这10帧是否都不在熔池中
            all_not_in_pool = True
            for traj_point in check_trajectory:
                frame_num = traj_point['frame']
                bbox = traj_point.get('bbox', [])
                if bbox:
                    frame_img = self._read_frame(frame_num)
                    if frame_img is not None:
                        if self._is_surrounded_by_pool(frame_img, bbox, frame_num):
                            all_not_in_pool = False
                            break
            
            # 如果运动距离小于100像素，且10帧都不在熔池中，判定为被结晶器捕获
            if movement_distance < 100 and all_not_in_pool:
                logger.debug(f"Movement distance {movement_distance:.1f}px < 100px and not in pool, captured by crystallizer")
                return 'crystallizer'
        
        # 策略3: 使用原有的方向判断作为fallback
        drop_location = self._determine_drop_location(trajectory, video_perspective)
        if drop_location:
            logger.debug(f"Using direction-based fallback: {drop_location}")
            return drop_location
        
        # 默认为熔池
        logger.debug(f"No clear indication, defaulting to pool")
        return 'pool'

    def _determine_drop_location(self, trajectory: List[Dict[str, Any]], 
                                 video_perspective: str) -> Optional[str]:
        """
        根据轨迹判断脱落位置
        
        Args:
            trajectory: 轨迹数据，格式为 [{'frame': int, 'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
            video_perspective: 视频视角 ('LEFT' 或 'RIGHT')
        
        Returns:
            脱落位置：'crystallizer'（结晶器）或 'pool'（熔池），如果无法判断则返回 None
        """
        if len(trajectory) < 2:
            return None
        
        # 取最后几帧的轨迹用于判断运动方向
        num_frames_to_check = min(10, len(trajectory))
        recent_trajectory = trajectory[-num_frames_to_check:]
        
        # 计算中心点的水平移动方向
        centers = [self._get_bbox_center(t['bbox']) for t in recent_trajectory]
        
        # 计算平均水平移动速度
        horizontal_movement = 0
        for i in range(1, len(centers)):
            horizontal_movement += centers[i][0] - centers[i-1][0]
        
        # 判断是否突然消失（最后一帧的置信度显著下降）
        sudden_disappearance = False
        if len(trajectory) >= 3:
            last_confidence = trajectory[-1]['confidence']
            avg_confidence = sum(t['confidence'] for t in trajectory[:-1]) / (len(trajectory) - 1)
            if last_confidence < avg_confidence * 0.5:
                sudden_disappearance = True
        
        # 根据视角和运动方向判断脱落位置
        if video_perspective == 'LEFT':
            if horizontal_movement < -5 or sudden_disappearance:
                # 向左飘落或突然消失 → 结晶器
                return 'crystallizer'
            elif horizontal_movement > 5:
                # 向右飘落 → 熔池
                return 'pool'
        else:  # RIGHT
            if horizontal_movement < -5:
                # 向左飘落 → 熔池
                return 'pool'
            elif horizontal_movement > 5 or sudden_disappearance:
                # 向右飘落或突然消失 → 结晶器
                return 'crystallizer'
        
        # 无法明确判断
        logger.debug(f"Unable to determine drop location: horizontal_movement={horizontal_movement}, "
                    f"sudden_disappearance={sudden_disappearance}, perspective={video_perspective}")
        return None

    def _get_bbox_center(self, bbox: List[float]) -> tuple[float, float]:
        """
        计算边界框的中心点
        
        Args:
            bbox: 边界框 [x1, y1, x2, y2]
        
        Returns:
            中心点坐标 (center_x, center_y)
        """
        if not bbox or len(bbox) < 4:
            return (0, 0)
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _is_surrounded_by_pool(self, frame: np.ndarray, bbox: List[float], frame_number: int) -> bool:
        """
        使用改进的连通域分析判断物体是否被熔池包围（即与电极断开连接）
        
        改进算法：
        1. 从bbox中心点开始，找到该点所在的连通域
        2. 计算该连通域的bbox与输入bbox的IoU重合度
        3. 判断该连通域是否与电极连通域相同
        4. 检查粘连物bbox周围的连接情况（多点采样）
        5. 只有当多个条件同时满足时，才判定为在熔池中
        
        Args:
            frame: 原始帧图像
            bbox: 物体边界框 [x1, y1, x2, y2]
            frame_number: 帧号（用于日志）
        
        Returns:
            True: 物体与电极断开，被熔池包围
            False: 物体仍与电极连接
        """
        if frame is None or not bbox or len(bbox) < 4:
            return False
        
        # IoU阈值：提高到0.5，更严格判断
        IOU_THRESHOLD = 0.5
        
        try:
            # 1. 转灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # 2. 高斯模糊去噪
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 3. 二值化：使用Otsu自适应阈值
            # 暗色(0) = 电极及粘连物，亮色(255) = 熔池
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 反转：让暗色区域为255（前景），亮色为0（背景）
            binary_inv = cv2.bitwise_not(binary)

            # 4. 形态学操作：渐进式多尺度闭运算
            # 目的：连接粘连物与电极之间的细微连接（可能只有几个像素宽）
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

            # 三次渐进闭运算：小→中→大
            closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel_small)
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_medium)
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_large)
            
            # 5. 找连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
            
            # 6. 找到中心区域的电极（假设在图像中心附近）
            center_x, center_y = w // 2, h // 2
            electrode_label = labels[center_y, center_x]
            
            if electrode_label == 0:  # 背景，说明中心是亮区
                # 尝试在中心附近搜索暗色区域
                search_radius = 50
                found = False
                for dy in range(-search_radius, search_radius, 10):
                    for dx in range(-search_radius, search_radius, 10):
                        y, x = center_y + dy, center_x + dx
                        if 0 <= y < h and 0 <= x < w:
                            label = labels[y, x]
                            if label > 0:
                                electrode_label = label
                                found = True
                                break
                    if found:
                        break
            
            if electrode_label == 0:
                # 无法找到电极，默认认为未脱离
                logger.debug(f"Frame {frame_number}: Cannot find electrode region")
                return False
            
            # 7. 获取bbox中心点和边界点，多点采样判断连接情况
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2
            
            # 多点采样：中心点 + bbox底部中心点 + bbox四个角的内缩点
            sample_points = [
                (bbox_center_x, bbox_center_y),  # 中心点
                (bbox_center_x, min(y2 - 2, bbox_center_y + (y2 - y1) // 4)),  # 下部中心点（粘连物通常从底部连接电极）
                (x1 + 3, y1 + 3),  # 左上角内缩
                (x2 - 3, y1 + 3),  # 右上角内缩
                (x1 + 3, y2 - 3),  # 左下角内缩
                (x2 - 3, y2 - 3),  # 右下角内缩
            ]
            
            # 确保采样点在图像范围内
            valid_sample_points = [
                (x, y) for x, y in sample_points
                if 0 <= x < w and 0 <= y < h
            ]
            
            # 统计各采样点的连通域标签
            object_labels = [labels[y, x] for x, y in valid_sample_points]
            
            # 如果大部分采样点都在背景上（亮色熔池区域），说明已经脱落到熔池
            background_count = sum(1 for label in object_labels if label == 0)
            if background_count >= len(object_labels) * 0.5:  # 50%以上的点在背景上
                logger.debug(f"Frame {frame_number}: Object has {background_count}/{len(object_labels)} points in background (pool)")
                return True
            
            # 使用中心点的标签作为主要判断
            object_label = labels[bbox_center_y, bbox_center_x]
            
            # 如果中心点在背景上，也判定为在熔池中
            if object_label == 0:
                logger.debug(f"Frame {frame_number}: Object center is in background (pool)")
                return True
            
            # 8. 获取该连通域的bbox
            object_stats = stats[object_label]
            obj_x, obj_y, obj_w, obj_h = object_stats[:4]
            centroid_bbox = [obj_x, obj_y, obj_x + obj_w, obj_y + obj_h]
            
            # 9. 计算连通域bbox与输入bbox的IoU
            iou = self._calculate_bbox_iou(bbox, centroid_bbox)
            
            # 10. 判断是否与电极连通域相同
            is_same_as_electrode = (object_label == electrode_label)
            
            # 检查采样点中是否有多个点与电极连接
            electrode_connected_count = sum(1 for label in object_labels if label == electrode_label and label > 0)
            has_strong_electrode_connection = electrode_connected_count >= 2  # 至少2个点连接到电极
            
            # 11. 综合判断（更严格的条件）
            # 条件1: IoU要足够高，说明连通域准确覆盖了物体
            # 条件2: 物体不能与电极在同一连通域
            # 条件3: 物体不能有多个采样点与电极连接
            is_in_pool = (iou >= IOU_THRESHOLD) and (not is_same_as_electrode) and (not has_strong_electrode_connection)

            if is_in_pool:
                logger.debug(f"Frame {frame_number}: Object is SEPARATED from electrode "
                           f"(IoU={iou:.3f}, is_electrode={is_same_as_electrode}, electrode_connections={electrode_connected_count})")
            else:
                logger.debug(f"Frame {frame_number}: Object is CONNECTED to electrode "
                           f"(IoU={iou:.3f}, is_electrode={is_same_as_electrode}, electrode_connections={electrode_connected_count})")

            # 12. 调试模式：保存中间处理结果
            if self.debug_mode and self.debug_output_dir:
                self._save_debug_images(
                    frame_number=frame_number,
                    original=frame,
                    gray=gray,
                    blurred=blurred,
                    binary=binary,
                    binary_inv=binary_inv,
                    closed=closed,
                    labels=labels,
                    bbox=bbox,
                    electrode_label=electrode_label,
                    object_label=object_label,
                    is_in_pool=is_in_pool,
                    iou=iou,
                    electrode_connections=electrode_connected_count
                )

            return is_in_pool
            
        except Exception as e:
            logger.warning(f"Frame {frame_number}: Error in pool detection: {e}")
            return False
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算两个bbox的IoU（交并比）
        
        Args:
            bbox1: 边界框1 [x1, y1, x2, y2]
            bbox2: 边界框2 [x1, y1, x2, y2]
        
        Returns:
            IoU值（0-1之间）
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集区域
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集区域
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _calculate_trajectory_movement(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        计算轨迹的累计运动距离（绝对距离）
        
        Args:
            trajectory: 轨迹数据列表
        
        Returns:
            累计运动距离（像素）
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        
        for i in range(1, len(trajectory)):
            prev_bbox = trajectory[i-1].get('bbox', [])
            curr_bbox = trajectory[i].get('bbox', [])
            
            if prev_bbox and curr_bbox:
                prev_center = self._get_bbox_center(prev_bbox)
                curr_center = self._get_bbox_center(curr_bbox)
                
                # 计算欧氏距离
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                total_distance += distance
        
        return total_distance

    def _read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        从视频读取指定帧（带缓存）
        
        Args:
            frame_number: 帧号
        
        Returns:
            帧图像（numpy数组），如果读取失败则返回None
        """
        # 检查缓存
        if frame_number in self.frame_cache:
            return self.frame_cache[frame_number]
        
        # 检查是否有视频路径
        if not self.video_path:
            logger.warning(f"No video path provided, cannot read frame {frame_number}")
            return None
        
        try:
            # 延迟初始化VideoCapture
            if self.video_cap is None:
                self.video_cap = cv2.VideoCapture(self.video_path)
                if not self.video_cap.isOpened():
                    logger.error(f"Failed to open video: {self.video_path}")
                    return None
            
            # 设置帧位置
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # 读取帧
            ret, frame = self.video_cap.read()
            
            if ret and frame is not None:
                # 缓存帧（限制缓存大小，避免内存溢出）
                if len(self.frame_cache) < 100:  # 最多缓存100帧
                    self.frame_cache[frame_number] = frame
                return frame
            else:
                logger.warning(f"Failed to read frame {frame_number} from video")
                return None
                
        except Exception as e:
            logger.error(f"Error reading frame {frame_number}: {e}")
            return None

    def _save_debug_images(self, frame_number: int, original: np.ndarray,
                           gray: np.ndarray, blurred: np.ndarray,
                           binary: np.ndarray, binary_inv: np.ndarray,
                           closed: np.ndarray, labels: np.ndarray,
                           bbox: List[float], electrode_label: int,
                           object_label: int, is_in_pool: bool,
                           iou: float, electrode_connections: int):
        """
        保存调试图像，用于可视化连通域分析的中间步骤

        Args:
            frame_number: 帧号
            original: 原始彩色帧
            gray: 灰度图
            blurred: 模糊后的图
            binary: 二值化图
            binary_inv: 反转后的二值化图
            closed: 闭运算后的图
            labels: 连通域标签矩阵
            bbox: 物体边界框
            electrode_label: 电极连通域标签
            object_label: 物体连通域标签
            is_in_pool: 判断结果
            iou: IoU值
            electrode_connections: 与电极连接的采样点数量
        """
        try:
            # 创建帧专用输出目录
            frame_dir = self.debug_output_dir / f"frame_{frame_number:06d}"
            frame_dir.mkdir(exist_ok=True)

            # 1. 保存原图+bbox标注
            original_annotated = original.copy()
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if is_in_pool else (0, 0, 255)  # 绿色=在熔池，红色=连接电极
            cv2.rectangle(original_annotated, (x1, y1), (x2, y2), color, 2)

            # 添加文本标注
            status_text = f"IN_POOL" if is_in_pool else f"CONNECTED"
            info_text = f"IoU={iou:.2f} EC={electrode_connections}"
            cv2.putText(original_annotated, status_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(original_annotated, info_text, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imwrite(str(frame_dir / "01_original_annotated.jpg"), original_annotated)

            # 2. 保存灰度图
            cv2.imwrite(str(frame_dir / "02_gray.jpg"), gray)

            # 3. 保存模糊图
            cv2.imwrite(str(frame_dir / "03_blurred.jpg"), blurred)

            # 4. 保存二值化图
            cv2.imwrite(str(frame_dir / "04_binary.jpg"), binary)

            # 5. 保存反转图
            cv2.imwrite(str(frame_dir / "05_binary_inv.jpg"), binary_inv)

            # 6. 保存闭运算结果
            cv2.imwrite(str(frame_dir / "06_closed.jpg"), closed)

            # 7. 生成连通域可视化图
            # 为每个连通域分配不同的颜色
            label_hue = np.uint8(179 * labels / np.max(labels)) if np.max(labels) > 0 else np.zeros_like(labels, dtype=np.uint8)
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

            # 标记电极连通域（蓝色边框）
            electrode_mask = (labels == electrode_label).astype(np.uint8) * 255
            electrode_contours, _ = cv2.findContours(electrode_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(labeled_img, electrode_contours, -1, (255, 0, 0), 2)  # 蓝色=电极

            # 标记物体连通域（绿色/红色边框）
            if object_label > 0:
                object_mask = (labels == object_label).astype(np.uint8) * 255
                object_contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled_img, object_contours, -1, color, 2)

            # 标记bbox
            cv2.rectangle(labeled_img, (x1, y1), (x2, y2), color, 2)

            cv2.imwrite(str(frame_dir / "07_connected_components.jpg"), labeled_img)

            # 8. 保存判断结果文本
            result_text = f"""Frame {frame_number} Analysis Result
====================================
Status: {"IN POOL (Separated)" if is_in_pool else "CONNECTED to Electrode"}
IoU: {iou:.3f}
Electrode Label: {electrode_label}
Object Label: {object_label}
Electrode Connections: {electrode_connections}
BBox: [{x1}, {y1}, {x2}, {y2}]
"""
            with open(frame_dir / "result.txt", 'w', encoding='utf-8') as f:
                f.write(result_text)

            logger.debug(f"Debug images saved to {frame_dir}")

        except Exception as e:
            logger.warning(f"Failed to save debug images for frame {frame_number}: {e}")

    def _cleanup(self):
        """清理资源（释放视频文件和缓存）"""
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

        self.frame_cache.clear()
        logger.debug("Cleaned up video resources and frame cache")
