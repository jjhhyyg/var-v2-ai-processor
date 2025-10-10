"""
追踪对象合并与重新关联模块（配合BoT-SORT使用）

背景：
- 系统使用 BoT-SORT (Byte On Track with ReID) 进行多目标追踪
- BoT-SORT 相比 ByteTrack 具有更强的重识别能力（通过外观特征匹配）
- 但在极端形变场景（如粘连物/锭冠脱落）下，ReID 可能仍然失效

问题场景：
1. 粘连物脱落：形状从块状→拉长→碎片，外观特征变化极大
2. 锭冠脱落：从附着状态→分离→下落，位置和形状剧烈变化
3. 电弧遮挡：粘连物被强烈电弧遮挡后重新出现，外观可能改变

解决策略：
本模块在 BoT-SORT 追踪结果的基础上进行后处理，通过以下方式弥补：
1. 空间-时间连续性：即使外观变化，但位置和运动轨迹应该连续
2. 运动预测：基于卡尔曼滤波思想预测物体位置
3. 渐进形变容忍：允许相邻追踪片段之间存在形状渐变
4. 物理约束：考虑重力、惯性等物理规律（如脱落物向下运动）

与 BoT-SORT 的配合：
- BoT-SORT 负责实时追踪，通过 ReID 处理短时遮挡
- 本模块负责离线后处理，通过更长时间窗口和宽松约束连接断裂片段
- track_buffer=100 帧意味着 BoT-SORT 会保留失踪轨迹约 3-4 秒
- 本模块的 max_frame_gap=15 用于连接超出 BoT-SORT buffer 的情况

用途：
1. 解决粘连物/锭冠在脱落过程中形状变化导致的ID断裂问题
2. 基于运动轨迹、位置连续性和时间连续性重新关联追踪对象
3. 为同一物体的多个追踪片段分配统一ID

作者：侯阳洋
日期：2025-10-10
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """边界框"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: 'BBox') -> float:
        """计算IoU"""
        x1_inter = max(self.x1, other.x1)
        y1_inter = max(self.y1, other.y1)
        x2_inter = min(self.x2, other.x2)
        y2_inter = min(self.y2, other.y2)
        
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def distance_to(self, other: 'BBox') -> float:
        """计算中心点距离"""
        c1 = self.center
        c2 = other.center
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


@dataclass
class TrackingFrame:
    """单帧追踪数据"""
    frame: int
    bbox: BBox
    confidence: float
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrackingFrame':
        bbox_data = data['bbox']
        return cls(
            frame=data['frame'],
            bbox=BBox(bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]),
            confidence=data['confidence']
        )


@dataclass
class TrackingObject:
    """追踪对象"""
    tracking_id: str
    object_id: int
    category: str
    first_frame: int
    last_frame: int
    trajectory: List[TrackingFrame]
    merged_from: Optional[List[str]] = None  # 记录合并来源
    
    def __post_init__(self):
        if self.merged_from is None:
            self.merged_from = []
    
    @classmethod
    def from_api_response(cls, data: dict) -> 'TrackingObject':
        trajectory = [TrackingFrame.from_dict(t) for t in data['trajectory']]
        return cls(
            tracking_id=data['trackingId'],
            object_id=data['objectId'],
            category=data['category'],
            first_frame=data['firstFrame'],
            last_frame=data['lastFrame'],
            trajectory=trajectory
        )
    
    def get_velocity(self) -> Tuple[float, float]:
        """计算平均运动速度（像素/帧）"""
        if len(self.trajectory) < 2:
            return (0.0, 0.0)
        
        first = self.trajectory[0].bbox.center
        last = self.trajectory[-1].bbox.center
        frames = self.last_frame - self.first_frame
        
        if frames == 0:
            return (0.0, 0.0)
        
        vx = (last[0] - first[0]) / frames
        vy = (last[1] - first[1]) / frames
        
        return (vx, vy)
    
    def predict_position(self, target_frame: int) -> BBox:
        """
        预测在目标帧的位置
        基于线性运动模型
        """
        if not self.trajectory:
            raise ValueError("No trajectory data")
        
        vx, vy = self.get_velocity()
        last_frame = self.trajectory[-1]
        frame_diff = target_frame - last_frame.frame
        
        # 预测中心点
        cx, cy = last_frame.bbox.center
        pred_cx = cx + vx * frame_diff
        pred_cy = cy + vy * frame_diff
        
        # 使用最后一帧的尺寸（允许±30%变化）
        w = last_frame.bbox.width
        h = last_frame.bbox.height
        
        return BBox(
            pred_cx - w/2,
            pred_cy - h/2,
            pred_cx + w/2,
            pred_cy + h/2
        )
    
    def get_shape_change_rate(self) -> float:
        """计算形状变化率（面积变化）"""
        if len(self.trajectory) < 2:
            return 0.0
        
        areas = [t.bbox.area for t in self.trajectory]
        area_changes = [abs(areas[i] - areas[i-1]) / areas[i-1] 
                       for i in range(1, len(areas)) if areas[i-1] > 0]
        
        return float(np.mean(area_changes)) if area_changes else 0.0


class TrackingMerger:
    """
    追踪对象合并器（配合BoT-SORT后处理）
    
    设计思路：
    1. BoT-SORT 已经处理了大部分追踪场景（包括短时遮挡的ReID恢复）
    2. 本模块专注于 BoT-SORT 无法处理的极端情况：
       - 超长时间失踪（> track_buffer 帧）
       - 极端形变导致ReID失败
       - 物体分裂/合并
    
    参数说明：
        max_frame_gap: 最大帧间隔
            - 通常设置为 track_buffer 的 10-20%
            - 用于连接超出 BoT-SORT 保留时间的断裂
            - 默认 15 帧（约 0.5-1 秒，取决于帧率）
        
        max_distance: 最大中心点距离（像素）
            - 考虑物体可能的最大移动距离
            - 粘连物脱落：可能快速下落，需要较大值
            - 默认 100 像素
        
        min_iou: 最小IoU
            - 放宽要求以适应形状变化
            - 0.1 表示只需 10% 的重叠
            - BoT-SORT 内部使用 match_thresh=0.5，这里更宽松
        
        allow_shape_change: 是否允许形状变化
            - True: 允许面积、长宽比显著变化（适合脱落场景）
            - False: 要求形状相对稳定
    """
    
    def __init__(
        self,
        max_frame_gap: int = 15,  # 最大帧间隔（约为track_buffer的15%）
        max_distance: float = 100.0,  # 最大中心点距离（像素）
        min_iou: float = 0.1,  # 最小IoU（放宽以适应形状变化）
        velocity_tolerance: float = 0.5,  # 速度容差
        allow_shape_change: bool = True,  # 是否允许形状变化
        max_shape_change_rate: float = 0.5,  # 最大形状变化率
    ):
        self.max_frame_gap = max_frame_gap
        self.max_distance = max_distance
        self.min_iou = min_iou
        self.velocity_tolerance = velocity_tolerance
        self.allow_shape_change = allow_shape_change
        self.max_shape_change_rate = max_shape_change_rate
        
        self.objects: List[TrackingObject] = []
        self.merged_groups: List[List[TrackingObject]] = []
    
    def load_objects(self, objects_data: List[dict]) -> None:
        """加载追踪对象数据"""
        self.objects = [TrackingObject.from_api_response(obj) for obj in objects_data]
        logger.info(f"加载了 {len(self.objects)} 个追踪对象")
    
    def calculate_association_score(
        self, 
        obj1: TrackingObject, 
        obj2: TrackingObject
    ) -> Tuple[float, str]:
        """
        计算两个追踪对象的关联得分
        
        返回: (得分, 原因说明)
        得分越高越可能是同一物体
        """
        # 必须同类别
        if obj1.category != obj2.category:
            return (0.0, "类别不匹配")
        
        # 检查时间顺序（obj1应该在obj2之前）
        if obj1.last_frame >= obj2.first_frame:
            # 有时间重叠，不应该合并（除非是短暂重叠）
            overlap = min(obj1.last_frame, obj2.last_frame) - max(obj1.first_frame, obj2.first_frame) + 1
            if overlap > 3:
                return (0.0, "时间重叠过多")
        
        # 计算时间间隔
        frame_gap = obj2.first_frame - obj1.last_frame
        if frame_gap > self.max_frame_gap:
            return (0.0, f"时间间隔过大({frame_gap}帧)")
        
        # 获取边界框
        bbox1_last = obj1.trajectory[-1].bbox
        bbox2_first = obj2.trajectory[0].bbox
        
        # 1. 空间距离得分（0-1，越近越高）
        distance = bbox1_last.distance_to(bbox2_first)
        if distance > self.max_distance:
            return (0.0, f"距离过远({distance:.1f}px)")
        
        distance_score = 1.0 - (distance / self.max_distance)
        
        # 2. 位置预测得分（基于运动模型）
        prediction_distance = 0.0
        try:
            predicted_bbox = obj1.predict_position(obj2.first_frame)
            prediction_distance = predicted_bbox.distance_to(bbox2_first)
            prediction_score = 1.0 - min(prediction_distance / self.max_distance, 1.0)
        except:
            prediction_score = 0.0
            prediction_distance = distance  # 使用实际距离
        
        # 3. IoU得分（放宽要求）
        iou = bbox1_last.iou(bbox2_first)
        iou_score = iou / self.min_iou if iou > 0 else 0.0
        
        # 4. 速度一致性得分
        v1 = obj1.get_velocity()
        v2 = obj2.get_velocity()
        if len(obj1.trajectory) >= 2 and len(obj2.trajectory) >= 2:
            v1_mag = np.sqrt(v1[0]**2 + v1[1]**2)
            v2_mag = np.sqrt(v2[0]**2 + v2[1]**2)
            if v1_mag > 0 and v2_mag > 0:
                # 计算速度方向的余弦相似度
                cos_sim = (v1[0]*v2[0] + v1[1]*v2[1]) / (v1_mag * v2_mag)
                velocity_score = (cos_sim + 1) / 2  # 归一化到0-1
            else:
                velocity_score = 0.5  # 静止物体
        else:
            velocity_score = 0.5
        
        # 5. 时间连续性得分
        time_score = 1.0 - (frame_gap / self.max_frame_gap)
        
        # 6. 形状变化容忍度
        if self.allow_shape_change:
            # 对于粘连物和锭冠，允许形状逐渐变化
            shape_change1 = obj1.get_shape_change_rate()
            shape_change2 = obj2.get_shape_change_rate()
            
            # 如果前一个物体正在发生形状变化（可能在脱落），给予额外分数
            if shape_change1 > 0.1:  # 正在变化
                shape_bonus = 0.2
            else:
                shape_bonus = 0.0
        else:
            shape_bonus = 0.0
        
        # 综合得分（加权平均）
        weights = {
            'distance': 0.25,
            'prediction': 0.20,
            'iou': 0.15,
            'velocity': 0.15,
            'time': 0.20,
            'shape_bonus': 0.05
        }
        
        total_score = (
            distance_score * weights['distance'] +
            prediction_score * weights['prediction'] +
            min(iou_score, 1.0) * weights['iou'] +
            velocity_score * weights['velocity'] +
            time_score * weights['time'] +
            shape_bonus * weights['shape_bonus']
        )
        
        reason = (
            f"距离:{distance:.1f}px(得分{distance_score:.2f}), "
            f"预测:{prediction_distance:.1f}px(得分{prediction_score:.2f}), "
            f"IoU:{iou:.3f}(得分{iou_score:.2f}), "
            f"时间:{frame_gap}帧(得分{time_score:.2f}), "
            f"总分:{total_score:.3f}"
        )
        
        return (total_score, reason)
    
    def merge_objects(self, threshold: float = 0.5) -> List[List[TrackingObject]]:
        """
        合并追踪对象
        
        参数:
            threshold: 关联得分阈值
        
        返回: 合并后的对象组列表
        """
        # 按类别分组
        category_groups = defaultdict(list)
        for obj in self.objects:
            category_groups[obj.category].append(obj)
        
        all_merged_groups = []
        
        for category, objs in category_groups.items():
            logger.info(f"处理类别: {category}, 对象数: {len(objs)}")
            
            # 按首次出现帧号排序
            objs.sort(key=lambda x: x.first_frame)
            
            # 构建关联图
            graph = defaultdict(set)
            association_scores = {}
            
            for i, obj1 in enumerate(objs):
                for j, obj2 in enumerate(objs[i+1:], start=i+1):
                    # 只考虑时间上在后面的对象
                    if obj1.last_frame < obj2.first_frame:
                        score, reason = self.calculate_association_score(obj1, obj2)
                        
                        if score >= threshold:
                            graph[i].add(j)
                            association_scores[(i, j)] = (score, reason)
                            logger.debug(f"关联 {obj1.object_id}→{obj2.object_id}: {reason}")
            
            # 使用贪心算法构建追踪链
            # 优先连接得分最高的对
            visited = set()
            merged_groups = []
            
            # 按得分排序关联对
            sorted_pairs = sorted(
                association_scores.items(),
                key=lambda x: x[1][0],
                reverse=True
            )
            
            # 构建追踪链
            chains = {}  # {起始索引: [索引列表]}
            
            for (i, j), (score, reason) in sorted_pairs:
                # 查找i所在的链
                chain_start = None
                for start, chain in chains.items():
                    if i in chain:
                        chain_start = start
                        break
                
                if chain_start is not None:
                    # i已在某条链中
                    chain = chains[chain_start]
                    if j not in chain:
                        # 确保j在i之后且不在链中
                        if chain[-1] == i:
                            chain.append(j)
                            logger.debug(f"扩展链 {chain_start}: 添加 {j}")
                else:
                    # 创建新链
                    chains[i] = [i, j]
                    logger.debug(f"创建新链 {i}: [{i}, {j}]")
            
            # 将链转换为对象组
            for start_idx, chain_indices in chains.items():
                group = [objs[idx] for idx in chain_indices]
                merged_groups.append(group)
                logger.info(f"合并组: {[obj.object_id for obj in group]}")
            
            # 添加未被合并的单独对象
            all_indices = set(range(len(objs)))
            merged_indices = set()
            for chain in chains.values():
                merged_indices.update(chain)
            
            for idx in all_indices - merged_indices:
                merged_groups.append([objs[idx]])
            
            all_merged_groups.extend(merged_groups)
        
        self.merged_groups = all_merged_groups
        return all_merged_groups
    
    def create_unified_objects(self) -> List[Dict]:
        """
        创建统一ID的追踪对象
        
        返回: 合并后的追踪对象列表（API格式）
        """
        unified_objects = []
        
        for group_idx, group in enumerate(self.merged_groups):
            if len(group) == 1:
                # 单个对象，直接使用
                obj = group[0]
                unified_objects.append({
                    'trackingId': obj.tracking_id,
                    'objectId': obj.object_id,
                    'category': obj.category,
                    'firstFrame': obj.first_frame,
                    'lastFrame': obj.last_frame,
                    'trajectory': [
                        {
                            'frame': t.frame,
                            'bbox': [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2],
                            'confidence': t.confidence
                        }
                        for t in obj.trajectory
                    ]
                })
            else:
                # 合并多个对象
                # 使用第一个对象的ID作为统一ID
                unified_id = group[0].object_id
                category = group[0].category
                
                # 合并轨迹
                all_trajectory = []
                for obj in group:
                    all_trajectory.extend(obj.trajectory)
                
                # 按帧号排序并去重
                trajectory_dict = {}
                for t in all_trajectory:
                    if t.frame not in trajectory_dict:
                        trajectory_dict[t.frame] = t
                    else:
                        # 如果同一帧有多个检测，选择置信度高的
                        if t.confidence > trajectory_dict[t.frame].confidence:
                            trajectory_dict[t.frame] = t
                
                sorted_trajectory = sorted(trajectory_dict.values(), key=lambda x: x.frame)
                
                unified_objects.append({
                    'trackingId': group[0].tracking_id,  # 使用第一个的tracking_id
                    'objectId': unified_id,
                    'category': category,
                    'firstFrame': sorted_trajectory[0].frame,
                    'lastFrame': sorted_trajectory[-1].frame,
                    'trajectory': [
                        {
                            'frame': t.frame,
                            'bbox': [t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2],
                            'confidence': t.confidence
                        }
                        for t in sorted_trajectory
                    ],
                    'mergedFrom': [obj.tracking_id for obj in group],  # 记录合并来源
                    'mergedObjectIds': [obj.object_id for obj in group]  # 记录原始ID
                })
                
                logger.info(
                    f"合并对象: ID {unified_id}, "
                    f"帧范围 {sorted_trajectory[0].frame}-{sorted_trajectory[-1].frame}, "
                    f"来源: {[obj.object_id for obj in group]}"
                )
        
        return unified_objects
    
    def generate_merge_report(self) -> Dict:
        """生成合并报告"""
        total_objects = len(self.objects)
        total_groups = len(self.merged_groups)
        merged_count = sum(1 for g in self.merged_groups if len(g) > 1)
        single_count = sum(1 for g in self.merged_groups if len(g) == 1)
        
        # 统计每个合并组的详情
        merge_details = []
        for idx, group in enumerate(self.merged_groups):
            if len(group) > 1:
                merge_details.append({
                    'group_id': idx,
                    'unified_id': group[0].object_id,
                    'category': group[0].category,
                    'object_count': len(group),
                    'original_ids': [obj.object_id for obj in group],
                    'frame_range': [group[0].first_frame, group[-1].last_frame],
                    'total_frames': group[-1].last_frame - group[0].first_frame + 1
                })
        
        return {
            'total_original_objects': total_objects,
            'total_unified_objects': total_groups,
            'merged_groups': merged_count,
            'single_objects': single_count,
            'merge_rate': f"{merged_count / total_objects * 100:.1f}%",
            'merge_details': merge_details
        }


def process_tracking_objects(
    tracking_objects: List[dict],
    max_frame_gap: int = 15,
    max_distance: float = 100.0,
    association_threshold: float = 0.5
) -> Tuple[List[dict], Dict]:
    """
    处理追踪对象，合并可能是同一物体的片段
    
    参数:
        tracking_objects: 原始追踪对象列表（API格式）
        max_frame_gap: 最大允许的帧间隔
        max_distance: 最大允许的空间距离
        association_threshold: 关联得分阈值
    
    返回:
        (统一后的追踪对象列表, 合并报告)
    """
    merger = TrackingMerger(
        max_frame_gap=max_frame_gap,
        max_distance=max_distance,
        allow_shape_change=True  # 允许形状变化以适应粘连物脱落
    )
    
    merger.load_objects(tracking_objects)
    merger.merge_objects(threshold=association_threshold)
    
    unified_objects = merger.create_unified_objects()
    report = merger.generate_merge_report()
    
    return unified_objects, report
