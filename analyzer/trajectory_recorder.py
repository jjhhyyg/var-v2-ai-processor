"""
物体追踪记录模块
基于ByteTrack追踪结果记录各种物体的出现和消失
"""
import logging
from typing import Dict, List, Any
from config import Config

logger = logging.getLogger(__name__)


class EventDetector:
    """物体追踪记录器（简化版本，只记录物体，不生成事件）"""

    def __init__(self):
        """初始化追踪记录器"""
        # 追踪物体状态管理
        self.active_tracks = {}  # {track_id: track_info}
        self.completed_tracks = {}  # {track_id: track_info}

        # 类别名称映射
        self.class_names = Config.CLASS_NAMES
        
        # 未追踪物体计数器（用于生成临时ID）
        self.untracked_counter = 0

    def process_detections(self, frame_number: int, timestamp: float,
                          detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理当前帧的检测结果,更新物体追踪信息

        Args:
            frame_number: 当前帧号
            timestamp: 当前时间戳（秒）
            detections: 检测结果列表,每个包含：
                - track_id: 追踪ID
                - class_id: 类别ID
                - class_name: 类别名称
                - bbox: 边界框 [x1, y1, x2, y2]
                - confidence: 置信度

        Returns:
            空列表（不再生成事件）
        """
        # 获取当前帧所有活跃的track_id
        current_track_ids = set()

        for det in detections:
            track_id = det.get('track_id')
            
            # 处理未追踪的检测（track_id为None或-1）
            if track_id is None or track_id == -1:
                # 为未追踪物体生成唯一的负数临时ID
                self.untracked_counter += 1
                track_id = -(1000000 + self.untracked_counter)  # 使用大负数避免与正常track_id冲突
                logger.debug(f"检测到未追踪物体，分配临时ID: {track_id}, 类别: {det.get('class_name')}, 帧: {frame_number}")

            current_track_ids.add(track_id)

            class_name = det.get('class_name', '')
            bbox = det.get('bbox', [])
            confidence = det.get('confidence', 0.0)

            # 1. 检查是否是新物体（首次出现）或者是已消失物体重新出现
            if track_id not in self.active_tracks:
                # 检查是否在 completed_tracks 中（物体重新出现）
                if track_id in self.completed_tracks:
                    # 物体重新出现，恢复到 active_tracks
                    self.active_tracks[track_id] = self.completed_tracks[track_id]
                    del self.completed_tracks[track_id]
                    
                    # 更新信息
                    self.active_tracks[track_id]['last_frame'] = frame_number
                    self.active_tracks[track_id]['last_time'] = timestamp
                    self.active_tracks[track_id]['trajectory'].append({'frame': frame_number, 'bbox': bbox, 'confidence': confidence})
                    
                    logger.debug(f"物体重新出现: ID={track_id}, 类别={class_name}, 帧={frame_number}")
                else:
                    # 全新物体
                    self.active_tracks[track_id] = {
                        'track_id': track_id,
                        'class_name': class_name,
                        'class_id': det.get('class_id'),
                        'first_frame': frame_number,
                        'first_time': timestamp,
                        'last_frame': frame_number,
                        'last_time': timestamp,
                        'trajectory': [{'frame': frame_number, 'bbox': bbox, 'confidence': confidence}],  # 存储帧号、bbox和置信度
                        'first_position': self._get_center(bbox)
                    }
                    
                    logger.debug(f"新物体追踪开始: ID={track_id}, 类别={class_name}, 帧={frame_number}")

            else:
                # 更新已存在物体的信息
                self.active_tracks[track_id]['last_frame'] = frame_number
                self.active_tracks[track_id]['last_time'] = timestamp
                self.active_tracks[track_id]['trajectory'].append({'frame': frame_number, 'bbox': bbox, 'confidence': confidence})

        # 2. 检测消失的物体，移动到已完成追踪
        disappeared_tracks = set(self.active_tracks.keys()) - current_track_ids

        for track_id in disappeared_tracks:
            track_info = self.active_tracks[track_id]
            class_name = track_info['class_name']
            
            logger.debug(f"物体暂时消失: ID={track_id}, 类别={class_name}, "
                        f"帧={track_info['first_frame']}-{track_info['last_frame']}")

            # 移动到已完成追踪（可能会重新出现）
            self.completed_tracks[track_id] = track_info
            del self.active_tracks[track_id]

        # 不再生成事件，返回空列表
        return []

    def finalize_events(self) -> List[Dict[str, Any]]:
        """
        完成追踪，处理视频结束时仍在活跃的追踪

        Returns:
            空列表（不再生成事件）
        """
        # 将所有仍在活跃的追踪移到已完成追踪
        for track_id, track_info in list(self.active_tracks.items()):
            logger.debug(f"视频结束，物体追踪完成: ID={track_id}, 类别={track_info['class_name']}, "
                        f"帧={track_info['first_frame']}-{track_info['last_frame']}")
            self.completed_tracks[track_id] = track_info

        self.active_tracks.clear()

        # 不再生成事件，返回空列表
        return []

    def get_tracking_objects(self) -> List[Dict[str, Any]]:
        """
        获取所有追踪物体的信息（包含所有类别）

        Returns:
            追踪物体列表，每个物体包含：
            - objectId: 追踪ID
            - category: 物体类别（英文）
            - className: 类别名称（中文）
            - classId: 类别ID
            - firstFrame: 起始帧
            - lastFrame: 结束帧
            - firstTime: 起始时间（秒）
            - lastTime: 结束时间（秒）
            - duration: 持续帧数
        """
        tracking_objects = []

        # 包括已完成和仍活跃的追踪
        all_tracks = {**self.completed_tracks, **self.active_tracks}

        for track_id, track_info in all_tracks.items():
            class_name = track_info['class_name']
            
            # 获取英文类别名称
            category = Config.CATEGORY_MAPPING.get(class_name, class_name)
            
            # 记录所有检测到的物体（不再只记录粘连物和锭冠）
            tracking_obj = {
                'objectId': track_id,
                'category': category,  # 后端需要的英文类别
                'className': class_name,  # 中文类别名称
                'classId': track_info['class_id'],
                'firstFrame': track_info['first_frame'],
                'lastFrame': track_info['last_frame'],
                'firstTime': round(track_info['first_time'], 2),
                'lastTime': round(track_info['last_time'], 2),
                'duration': track_info['last_frame'] - track_info['first_frame'] + 1,
                'trajectory': track_info.get('trajectory', [])  # 添加轨迹数据
            }
            
            tracking_objects.append(tracking_obj)

        # 按起始帧排序
        tracking_objects.sort(key=lambda x: x['firstFrame'])

        logger.info(f"追踪完成，共记录 {len(tracking_objects)} 个物体")

        return tracking_objects

    def _get_center(self, bbox: List[float]) -> Dict[str, float]:
        """获取边界框中心点（bbox格式：[x1, y1, x2, y2]）"""
        if not bbox or len(bbox) < 4:
            return {'x': 0, 'y': 0}
        x1, y1, x2, y2 = bbox
        return {'x': round((x1 + x2) / 2, 1), 'y': round((y1 + y2) / 2, 1)}
