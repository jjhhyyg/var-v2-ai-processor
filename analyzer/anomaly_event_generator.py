"""
异常事件生成模块
基于物体追踪轨迹生成异常事件（包括粘连物脱落位置判断）

功能：
1. 对于同类物体，取最早出现帧（前5s无此类物体）和最晚出现帧（后5s无此类物体）作为时间段
2. 对于粘连物，判断脱落位置（熔池/结晶器）
3. 生成符合后端 AnomalyEvent 实体要求的事件数据

作者：侯阳洋
日期：2025-10-10
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)


class AnomalyEventGenerator:
    """异常事件生成器"""

    def __init__(self, fps: float = 30.0):
        """
        初始化异常事件生成器
        
        Args:
            fps: 视频帧率（用于计算时间窗口）
        """
        self.fps = fps
        self.time_window_frames = int(5 * fps)  # 5秒对应的帧数
        
        logger.info(f"AnomalyEventGenerator initialized with fps={fps}, time_window={self.time_window_frames} frames")

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
            
            # 对于粘连物，额外判断脱落位置
            if category == 'ADHESION':
                for obj in objects:
                    drop_event = self._analyze_adhesion_drop(obj, video_perspective)
                    if drop_event:
                        events.append(drop_event)
                        logger.info(f"Generated adhesion drop event: objectId={obj['objectId']}, "
                                  f"location={drop_event['metadata'].get('dropped_location', 'unknown')}")
        
        logger.info(f"Generated {len(events)} anomaly events from {len(tracking_objects)} tracking objects")
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
                               video_perspective: str) -> Optional[Dict[str, Any]]:
        """
        分析粘连物的脱落情况并判断脱落位置
        
        判断逻辑：
        - 左视角：
          - 向左飘落或突然消失 → 结晶器
          - 向右飘落 → 熔池
        - 右视角：
          - 向左飘落 → 熔池
          - 向右飘落或突然消失 → 结晶器
        
        Args:
            adhesion_obj: 粘连物对象
            video_perspective: 视频视角 ('LEFT' 或 'RIGHT')
        
        Returns:
            脱落事件（如果发生脱落），否则返回 None
        """
        trajectory = adhesion_obj.get('trajectory', [])
        
        if len(trajectory) < 2:
            # 轨迹太短，无法判断运动方向
            return None
        
        # 检查是否发生脱落（简单判断：物体持续时间超过一定帧数）
        duration = adhesion_obj['duration']
        if duration < 10:  # 少于10帧的物体可能是噪声，不认为是脱落
            return None
        
        # 分析运动方向
        drop_location = self._determine_drop_location(trajectory, video_perspective)
        
        if drop_location:
            # 生成脱落事件，仅包含 dropped_location 字段
            event = {
                'eventType': 'ADHESION_DROPPED',
                'startFrame': adhesion_obj['firstFrame'],
                'endFrame': adhesion_obj['lastFrame'],
                'objectId': None,  # 按要求留空
                'metadata': {
                    'dropped_location': drop_location  # 只有这一个字段
                }
            }
            return event
        
        return None

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
