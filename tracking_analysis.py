"""
追踪对象数据分析脚本

用途：
1. 从后端API获取已完成任务的追踪对象数据
2. 分析可能为同一物体的追踪ID（基于位置和时间重叠）
3. 识别可合并的追踪对象
4. 为后处理异常事件检测提供数据支持

作者：侯阳洋
日期：2025-10-10
"""

import requests
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class BBox:
    """边界框"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """计算中心点"""
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
        """计算与另一个边界框的IoU（交并比）"""
        # 计算交集区域
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
        """计算与另一个边界框中心点的欧氏距离"""
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
        """从字典创建"""
        bbox_data = data['bbox']
        return cls(
            frame=data['frame'],
            bbox=BBox(bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]),
            confidence=data['confidence']
        )


@dataclass
class TrackingObject:
    """追踪物体"""
    tracking_id: str
    object_id: int
    category: str
    first_frame: int
    last_frame: int
    trajectory: List[TrackingFrame]
    
    @classmethod
    def from_api_response(cls, data: dict) -> 'TrackingObject':
        """从API响应创建"""
        trajectory = [TrackingFrame.from_dict(t) for t in data['trajectory']]
        return cls(
            tracking_id=data['trackingId'],
            object_id=data['objectId'],
            category=data['category'],
            first_frame=data['firstFrame'],
            last_frame=data['lastFrame'],
            trajectory=trajectory
        )
    
    @property
    def duration(self) -> int:
        """持续时间（帧数）"""
        return self.last_frame - self.first_frame + 1
    
    @property
    def actual_frames(self) -> int:
        """实际出现的帧数"""
        return len(self.trajectory)
    
    @property
    def avg_confidence(self) -> float:
        """平均置信度"""
        if not self.trajectory:
            return 0.0
        return np.mean([t.confidence for t in self.trajectory])
    
    def get_bbox_at_frame(self, frame: int) -> Optional[BBox]:
        """获取指定帧的边界框"""
        for t in self.trajectory:
            if t.frame == frame:
                return t.bbox
        return None
    
    def get_average_bbox(self) -> BBox:
        """获取平均边界框"""
        if not self.trajectory:
            raise ValueError("No trajectory data")
        
        x1_avg = np.mean([t.bbox.x1 for t in self.trajectory])
        y1_avg = np.mean([t.bbox.y1 for t in self.trajectory])
        x2_avg = np.mean([t.bbox.x2 for t in self.trajectory])
        y2_avg = np.mean([t.bbox.y2 for t in self.trajectory])
        
        return BBox(x1_avg, y1_avg, x2_avg, y2_avg)


class TrackingAnalyzer:
    """追踪对象分析器"""
    
    def __init__(self, backend_url: str = "http://localhost:8080"):
        self.backend_url = backend_url
        self.objects: List[TrackingObject] = []
        self.task_info: Optional[Dict] = None
    
    def fetch_tracking_data(self, task_id: str) -> None:
        """从后端获取追踪数据"""
        print(f"正在获取任务 {task_id} 的追踪数据...")
        
        url = f"{self.backend_url}/api/tasks/{task_id}/result"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if data['code'] != 200:
            raise Exception(f"API返回错误: {data['message']}")
        
        result_data = data['data']
        self.task_info = {
            'taskId': result_data['taskId'],
            'name': result_data['name'],
            'status': result_data['status']
        }
        
        tracking_objects = result_data.get('trackingObjects', [])
        self.objects = [TrackingObject.from_api_response(obj) for obj in tracking_objects]
        
        print(f"成功获取 {len(self.objects)} 个追踪对象")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.objects:
            return {}
        
        # 按类别统计
        category_stats = defaultdict(int)
        for obj in self.objects:
            category_stats[obj.category] += 1
        
        # 持续时间统计
        durations = [obj.duration for obj in self.objects]
        actual_frames = [obj.actual_frames for obj in self.objects]
        confidences = [obj.avg_confidence for obj in self.objects]
        
        # 负ID对象（未被长期追踪的对象）
        negative_id_objects = [obj for obj in self.objects if obj.object_id < 0]
        positive_id_objects = [obj for obj in self.objects if obj.object_id >= 0]
        
        return {
            'total_objects': len(self.objects),
            'category_distribution': dict(category_stats),
            'negative_id_count': len(negative_id_objects),
            'positive_id_count': len(positive_id_objects),
            'duration_stats': {
                'min': int(np.min(durations)),
                'max': int(np.max(durations)),
                'mean': float(np.mean(durations)),
                'median': float(np.median(durations))
            },
            'actual_frames_stats': {
                'min': int(np.min(actual_frames)),
                'max': int(np.max(actual_frames)),
                'mean': float(np.mean(actual_frames)),
                'median': float(np.median(actual_frames))
            },
            'confidence_stats': {
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences))
            }
        }
    
    def find_temporal_overlaps(self, min_overlap_frames: int = 1) -> List[Tuple[TrackingObject, TrackingObject, int]]:
        """
        查找时间上重叠的追踪对象
        
        返回: [(obj1, obj2, 重叠帧数), ...]
        """
        overlaps = []
        
        for i, obj1 in enumerate(self.objects):
            for obj2 in self.objects[i+1:]:
                # 计算时间重叠
                overlap_start = max(obj1.first_frame, obj2.first_frame)
                overlap_end = min(obj1.last_frame, obj2.last_frame)
                overlap_frames = overlap_end - overlap_start + 1
                
                if overlap_frames >= min_overlap_frames:
                    overlaps.append((obj1, obj2, overlap_frames))
        
        return overlaps
    
    def find_spatial_temporal_similar_objects(
        self,
        iou_threshold: float = 0.3,
        max_frame_gap: int = 5,
        same_category: bool = True
    ) -> List[Tuple[TrackingObject, TrackingObject, float, str]]:
        """
        查找空间-时间上相似的追踪对象（可能是同一物体）
        
        参数:
            iou_threshold: IoU阈值
            max_frame_gap: 最大帧间隔
            same_category: 是否要求同类别
        
        返回: [(obj1, obj2, 相似度得分, 原因), ...]
        """
        similar_pairs = []
        
        for i, obj1 in enumerate(self.objects):
            for obj2 in self.objects[i+1:]:
                # 检查类别
                if same_category and obj1.category != obj2.category:
                    continue
                
                # 检查时间关系
                frame_gap = abs(obj1.last_frame - obj2.first_frame)
                if obj1.last_frame < obj2.first_frame:
                    # obj1 在 obj2 之前
                    gap = obj2.first_frame - obj1.last_frame
                elif obj2.last_frame < obj1.first_frame:
                    # obj2 在 obj1 之前
                    gap = obj1.first_frame - obj2.last_frame
                else:
                    # 有时间重叠
                    gap = 0
                
                if gap > max_frame_gap:
                    continue
                
                # 计算空间相似度
                if gap == 0:
                    # 有重叠，计算重叠区域的平均IoU
                    overlap_start = max(obj1.first_frame, obj2.first_frame)
                    overlap_end = min(obj1.last_frame, obj2.last_frame)
                    
                    ious = []
                    for frame in range(overlap_start, overlap_end + 1):
                        bbox1 = obj1.get_bbox_at_frame(frame)
                        bbox2 = obj2.get_bbox_at_frame(frame)
                        if bbox1 and bbox2:
                            ious.append(bbox1.iou(bbox2))
                    
                    if ious:
                        avg_iou = np.mean(ious)
                        if avg_iou >= iou_threshold:
                            similar_pairs.append((
                                obj1, obj2, avg_iou,
                                f"时间重叠{len(ious)}帧，平均IoU={avg_iou:.3f}"
                            ))
                else:
                    # 无重叠，比较最接近的帧
                    if obj1.last_frame < obj2.first_frame:
                        bbox1 = obj1.trajectory[-1].bbox
                        bbox2 = obj2.trajectory[0].bbox
                    else:
                        bbox1 = obj1.trajectory[0].bbox
                        bbox2 = obj2.trajectory[-1].bbox
                    
                    iou = bbox1.iou(bbox2)
                    distance = bbox1.distance_to(bbox2)
                    
                    # 使用综合评分：IoU + 距离
                    score = iou - distance / 1000.0  # 简单的评分公式
                    
                    if iou >= iou_threshold or (score > 0.2 and distance < 50):
                        similar_pairs.append((
                            obj1, obj2, score,
                            f"帧间隔{gap}帧，IoU={iou:.3f}，距离={distance:.1f}px"
                        ))
        
        return similar_pairs
    
    def find_mergeable_objects(self) -> List[List[TrackingObject]]:
        """
        查找可以合并的追踪对象组
        
        返回: [[obj1, obj2, obj3], [obj4, obj5], ...]
        """
        # 获取相似对象对
        similar_pairs = self.find_spatial_temporal_similar_objects(
            iou_threshold=0.2,
            max_frame_gap=10,
            same_category=True
        )
        
        # 构建图结构
        from collections import defaultdict
        graph = defaultdict(set)
        
        for obj1, obj2, score, reason in similar_pairs:
            graph[obj1.tracking_id].add(obj2.tracking_id)
            graph[obj2.tracking_id].add(obj1.tracking_id)
        
        # 查找连通分量
        visited = set()
        groups = []
        
        def dfs(obj_id, group):
            if obj_id in visited:
                return
            visited.add(obj_id)
            group.append(obj_id)
            for neighbor in graph[obj_id]:
                dfs(neighbor, group)
        
        for obj_id in graph:
            if obj_id not in visited:
                group = []
                dfs(obj_id, group)
                if len(group) > 1:
                    groups.append(group)
        
        # 转换为对象列表
        id_to_obj = {obj.tracking_id: obj for obj in self.objects}
        result = []
        for group_ids in groups:
            group_objects = [id_to_obj[obj_id] for obj_id in group_ids]
            # 按首次出现帧号排序
            group_objects.sort(key=lambda x: x.first_frame)
            result.append(group_objects)
        
        return result
    
    def analyze_single_frame_objects(self) -> List[TrackingObject]:
        """分析只在单帧出现的对象（可能是误检）"""
        return [obj for obj in self.objects if obj.duration == 1]
    
    def analyze_short_lived_objects(self, max_frames: int = 5) -> List[TrackingObject]:
        """分析短暂出现的对象"""
        return [obj for obj in self.objects if obj.duration <= max_frames]
    
    def print_analysis_report(self):
        """打印分析报告"""
        print("\n" + "="*80)
        print(f"追踪对象分析报告 - 任务: {self.task_info['name']}")
        print("="*80)
        
        # 基础统计
        stats = self.get_statistics()
        print("\n【基础统计】")
        print(f"  总对象数: {stats['total_objects']}")
        print(f"  正ID对象数（持续追踪）: {stats['positive_id_count']}")
        print(f"  负ID对象数（短暂检测）: {stats['negative_id_count']}")
        print(f"\n  类别分布:")
        for category, count in stats['category_distribution'].items():
            print(f"    {category}: {count}")
        
        print(f"\n  持续时间统计（帧）:")
        print(f"    最小: {stats['duration_stats']['min']}")
        print(f"    最大: {stats['duration_stats']['max']}")
        print(f"    平均: {stats['duration_stats']['mean']:.2f}")
        print(f"    中位数: {stats['duration_stats']['median']:.2f}")
        
        print(f"\n  平均置信度: {stats['confidence_stats']['mean']:.3f}")
        
        # 单帧对象
        single_frame_objs = self.analyze_single_frame_objects()
        print(f"\n【单帧对象分析】")
        print(f"  单帧对象数: {len(single_frame_objs)} ({len(single_frame_objs)/len(self.objects)*100:.1f}%)")
        if single_frame_objs:
            print(f"  示例（前5个）:")
            for obj in single_frame_objs[:5]:
                print(f"    - ID:{obj.object_id}, 类别:{obj.category}, 帧:{obj.first_frame}, 置信度:{obj.avg_confidence:.3f}")
        
        # 短暂对象
        short_objs = self.analyze_short_lived_objects(max_frames=5)
        print(f"\n【短暂对象分析】（≤5帧）")
        print(f"  短暂对象数: {len(short_objs)} ({len(short_objs)/len(self.objects)*100:.1f}%)")
        
        # 可能的同一物体
        similar_pairs = self.find_spatial_temporal_similar_objects()
        print(f"\n【可能为同一物体的追踪对】")
        print(f"  发现 {len(similar_pairs)} 对相似对象")
        if similar_pairs:
            print(f"  示例（前10个）:")
            for obj1, obj2, score, reason in similar_pairs[:10]:
                print(f"    - ID:{obj1.object_id}[帧{obj1.first_frame}-{obj1.last_frame}] ↔ "
                      f"ID:{obj2.object_id}[帧{obj2.first_frame}-{obj2.last_frame}]")
                print(f"      {reason}")
        
        # 可合并组
        mergeable_groups = self.find_mergeable_objects()
        print(f"\n【可合并的追踪对象组】")
        print(f"  发现 {len(mergeable_groups)} 个可合并组")
        if mergeable_groups:
            for i, group in enumerate(mergeable_groups[:5], 1):
                print(f"\n  组 {i} ({len(group)} 个对象):")
                for obj in group:
                    print(f"    - ID:{obj.object_id}, 类别:{obj.category}, "
                          f"帧:{obj.first_frame}-{obj.last_frame}, "
                          f"持续:{obj.duration}帧, 置信度:{obj.avg_confidence:.3f}")
        
        print("\n" + "="*80)
    
    def visualize_timeline(self, max_objects: int = 50, save_path: Optional[str] = None):
        """
        可视化追踪对象时间线
        
        参数:
            max_objects: 最多显示的对象数
            save_path: 保存路径（可选）
        """
        import matplotlib.pyplot as plt
        
        # 选择要显示的对象（优先显示持续时间长的）
        objects_to_show = sorted(self.objects, key=lambda x: -x.duration)[:max_objects]
        
        fig, ax = plt.subplots(figsize=(15, max(8, len(objects_to_show) * 0.3)))
        
        # 按类别分配颜色
        categories = list(set(obj.category for obj in objects_to_show))
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        category_colors = dict(zip(categories, colors))
        
        for i, obj in enumerate(objects_to_show):
            color = category_colors[obj.category]
            
            # 绘制时间线
            ax.barh(i, obj.duration, left=obj.first_frame, 
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # 标注ID
            label = f"ID:{obj.object_id}" if obj.object_id >= 0 else f"ID:{obj.object_id}"
            ax.text(obj.first_frame + obj.duration/2, i, label, 
                   va='center', ha='center', fontsize=6)
        
        ax.set_yticks(range(len(objects_to_show)))
        ax.set_yticklabels([f"{obj.category[:10]}" for obj in objects_to_show], fontsize=8)
        ax.set_xlabel('帧号', fontsize=12)
        ax.set_ylabel('追踪对象', fontsize=12)
        ax.set_title(f'追踪对象时间线 - {self.task_info["name"]}', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加图例
        legend_elements = [patches.Patch(facecolor=color, label=cat) 
                          for cat, color in category_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"时间线图已保存到: {save_path}")
        
        plt.show()
    
    def export_mergeable_suggestions(self, output_file: str):
        """
        导出可合并对象建议（JSON格式）
        
        用于后处理时参考
        """
        mergeable_groups = self.find_mergeable_objects()
        
        suggestions = []
        for group in mergeable_groups:
            group_data = {
                'group_id': f"group_{len(suggestions)+1}",
                'object_count': len(group),
                'category': group[0].category,
                'frame_range': [group[0].first_frame, group[-1].last_frame],
                'objects': [
                    {
                        'tracking_id': obj.tracking_id,
                        'object_id': obj.object_id,
                        'first_frame': obj.first_frame,
                        'last_frame': obj.last_frame,
                        'duration': obj.duration,
                        'avg_confidence': obj.avg_confidence
                    }
                    for obj in group
                ]
            }
            suggestions.append(group_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'task_info': self.task_info,
                'total_groups': len(suggestions),
                'mergeable_groups': suggestions
            }, f, indent=2, ensure_ascii=False)
        
        print(f"合并建议已导出到: {output_file}")


def main():
    """主函数"""
    # 创建分析器
    analyzer = TrackingAnalyzer()
    
    # 获取已完成的任务列表
    print("正在获取已完成的任务列表...")
    response = requests.get("http://localhost:8080/api/tasks?status=COMPLETED&size=10")
    tasks_data = response.json()
    
    if tasks_data['code'] != 200:
        print(f"获取任务列表失败: {tasks_data['message']}")
        return
    
    tasks = tasks_data['data']['items']
    
    if not tasks:
        print("没有找到已完成的任务")
        return
    
    # 显示任务列表
    print(f"\n找到 {len(tasks)} 个已完成的任务:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. [{task['taskId']}] {task['name']} "
              f"(时长: {task['videoDuration']}s, "
              f"完成于: {task['completedAt']})")
    
    # 选择第一个任务进行分析
    selected_task = tasks[0]
    task_id = selected_task['taskId']
    
    print(f"\n分析任务: {selected_task['name']} (ID: {task_id})")
    
    # 获取追踪数据
    analyzer.fetch_tracking_data(task_id)
    
    # 打印分析报告
    analyzer.print_analysis_report()
    
    # 导出合并建议
    output_file = f"mergeable_suggestions_{task_id}.json"
    analyzer.export_mergeable_suggestions(output_file)
    
    # 可视化时间线
    print("\n正在生成时间线可视化...")
    try:
        analyzer.visualize_timeline(
            max_objects=30,
            save_path=f"tracking_timeline_{task_id}.png"
        )
    except Exception as e:
        print(f"可视化失败（可能是显示环境问题）: {e}")
    
    # 额外分析：为异常事件检测提供建议
    print("\n【异常事件检测建议】")
    
    # 1. 粘连物检测
    adhesion_objects = [obj for obj in analyzer.objects if obj.category == "ADHESION"]
    if adhesion_objects:
        print(f"\n  粘连物对象: {len(adhesion_objects)} 个")
        # 持续时间较长的粘连物可能是真实异常
        long_adhesions = [obj for obj in adhesion_objects if obj.duration >= 10]
        print(f"  持续≥10帧的粘连物: {len(long_adhesions)} 个 → 建议记录为异常事件")
        
        # 合并后的粘连物组
        mergeable = analyzer.find_mergeable_objects()
        adhesion_groups = [g for g in mergeable if g[0].category == "ADHESION"]
        print(f"  可合并的粘连物组: {len(adhesion_groups)} 组 → 建议合并后记录为单个异常事件")
    
    # 2. 爬弧检测
    arc_objects = [obj for obj in analyzer.objects if obj.category == "CREEPING_ARC"]
    if arc_objects:
        print(f"\n  爬弧对象: {len(arc_objects)} 个")
        long_arcs = [obj for obj in arc_objects if obj.duration >= 5]
        print(f"  持续≥5帧的爬弧: {len(long_arcs)} 个 → 建议记录为异常事件")
    
    # 3. 频繁出现的短暂对象
    short_objs = analyzer.analyze_short_lived_objects(max_frames=3)
    if len(short_objs) > len(analyzer.objects) * 0.5:
        print(f"\n  检测到大量短暂对象({len(short_objs)}个)，可能表示:")
        print(f"    - 环境不稳定（频繁的飞溅、火花等）")
        print(f"    - 建议记录为'不稳定期'异常事件")


if __name__ == "__main__":
    main()
