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

    def __init__(self, fps: float = 30.0, video_path: Optional[str] = None):
        """
        初始化异常事件生成器
        
        Args:
            fps: 视频帧率（用于计算时间窗口）
            video_path: 视频文件路径（用于读取帧进行图像分析）
        """
        self.fps = fps
        self.time_window_frames = int(5 * fps)  # 5秒对应的帧数（用于其他事件）
        self.event_window_frames = int(3 * fps)  # 3秒对应的帧数（用于粘连物/锭冠事件）
        self.video_path = video_path
        self.frame_cache = {}  # 缓存读取的帧，避免重复IO
        self.video_cap = None  # VideoCapture对象
        
        logger.info(f"AnomalyEventGenerator initialized with fps={fps}, "
                   f"time_window={self.time_window_frames} frames, "
                   f"event_window={self.event_window_frames} frames, "
                   f"video_path={video_path}")

    def generate_events(self, 
                       tracking_objects: List[Dict[str, Any]], 
                       video_filename: str,
                       total_frames: int) -> List[Dict[str, Any]]:
        """
        基于追踪物体生成异常事件
        
        Args:
            tracking_objects: 追踪物体列表（来自 EventDetector.get_tracking_objects()）
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
                # 为每个粘连物生成形成和脱落事件
                for obj in objects:
                    adhesion_events = self._generate_adhesion_events(obj, video_perspective)
                    events.extend(adhesion_events)
                    logger.info(f"Generated {len(adhesion_events)} events for adhesion objectId={obj['objectId']}")
            
            elif category == 'CROWN':
                # 为每个锭冠生成脱落事件
                for obj in objects:
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
        
        Args:
            adhesion_obj: 粘连物对象
            video_perspective: 视频视角 ('LEFT' 或 'RIGHT')
        
        Returns:
            事件列表（包含形成事件和脱落事件）
        """
        events = []
        first_frame = adhesion_obj['firstFrame']
        last_frame = adhesion_obj['lastFrame']
        
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
        
        # 2. 生成粘连物脱落事件（后3秒，带位置判断）
        drop_start_frame = max(last_frame - self.event_window_frames, first_frame)
        
        # 判断脱落位置
        drop_location = self._analyze_adhesion_drop(adhesion_obj, video_perspective)
        
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
        logger.debug(f"Adhesion drop event: frames {drop_start_frame}-{last_frame}, location={drop_location}")
        
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
        1. 使用连通域分析判断最后一帧是否与电极断开（被熔池包围）→ 'pool'
        2. 检查粘连物存在前的最后一帧，向前10帧是否有明显运动且不在熔池中 → 'crystallizer'
        3. 使用原有的方向判断作为fallback
        
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
        
        # 策略1: 检查最后一帧是否被熔池包围（与电极断开连接）
        frame = self._read_frame(last_frame_num)
        if frame is not None and last_bbox:
            is_in_pool = self._is_surrounded_by_pool(frame, last_bbox, last_frame_num)
            if is_in_pool:
                logger.debug(f"Frame {last_frame_num}: Object is surrounded by pool (disconnected from electrode)")
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
        使用连通域分析判断物体是否被熔池包围（即与电极断开连接）
        
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
            
            # 4. 形态学操作：闭运算连接断裂的暗色区域
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel)
            
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
            
            # 7. 获取粘连物bbox区域的标签
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 检查bbox区域是否与电极连通域有交集
            bbox_region = labels[y1:y2, x1:x2]
            has_electrode = np.any(bbox_region == electrode_label)
            
            # 8. 判断结果
            is_separated = not has_electrode
            
            if is_separated:
                logger.debug(f"Frame {frame_number}: Object is SEPARATED from electrode (floating in pool)")
            else:
                logger.debug(f"Frame {frame_number}: Object is CONNECTED to electrode")
            
            return is_separated
            
        except Exception as e:
            logger.warning(f"Frame {frame_number}: Error in pool detection: {e}")
            return False

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

    def _cleanup(self):
        """清理资源（释放视频文件和缓存）"""
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        
        self.frame_cache.clear()
        logger.debug("Cleaned up video resources and frame cache")
