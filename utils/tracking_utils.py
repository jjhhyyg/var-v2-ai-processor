"""
追踪对象合并工具 - 简化接口

提供简单易用的API来合并BoT-SORT的追踪结果

作者：侯阳洋
日期：2025-10-10
"""

from typing import List, Dict, Tuple
from analyzer.tracking_merger import process_tracking_objects
import logging

logger = logging.getLogger(__name__)


def merge_tracking_objects(
    tracking_objects: List[Dict],
    scenario: str = "default"
) -> Tuple[List[Dict], Dict]:
    """
    合并追踪对象（简化接口）
    
    参数:
        tracking_objects: BoT-SORT的追踪结果列表
        scenario: 场景类型，预定义的参数配置
            - "default": 通用场景（默认）
            - "adhesion_falling": 粘连物脱落场景
            - "ingot_crown_falling": 锭冠脱落场景
            - "fast_motion": 快速运动场景
            - "conservative": 保守合并（避免误匹配）
            - "aggressive": 激进合并（最大化连接断裂片段）
    
    返回:
        (统一后的追踪对象列表, 合并报告)
    """
    
    # 预定义场景参数
    scenario_params = {
        "default": {
            "max_frame_gap": 15,
            "max_distance": 100.0,
            "association_threshold": 0.5
        },
        "adhesion_falling": {
            "max_frame_gap": 20,  # 粘连物可能有较长时间的形变
            "max_distance": 120.0,  # 可能快速下落
            "association_threshold": 0.45  # 稍微宽松以适应形状变化
        },
        "ingot_crown_falling": {
            "max_frame_gap": 15,
            "max_distance": 150.0,  # 锭冠下落速度快
            "association_threshold": 0.5
        },
        "fast_motion": {
            "max_frame_gap": 10,  # 快速运动，间隔小
            "max_distance": 200.0,  # 但移动距离大
            "association_threshold": 0.4
        },
        "conservative": {
            "max_frame_gap": 10,
            "max_distance": 80.0,
            "association_threshold": 0.6  # 更高的阈值，避免误匹配
        },
        "aggressive": {
            "max_frame_gap": 30,
            "max_distance": 150.0,
            "association_threshold": 0.3  # 更低的阈值，激进合并
        }
    }
    
    params = scenario_params.get(scenario, scenario_params["default"])
    
    logger.info(f"=== 轨迹合并开始 ===")
    logger.info(f"使用场景: {scenario}, 参数: {params}")
    logger.info(f"输入对象数量: {len(tracking_objects)}")
    
    # 打印输入对象的简要信息
    if tracking_objects:
        logger.info(f"输入对象示例（前3个）:")
        for i, obj in enumerate(tracking_objects[:3]):
            logger.info(f"  对象{i+1}: objectId={obj.get('objectId')}, category={obj.get('category')}, "
                       f"帧范围={obj.get('firstFrame')}-{obj.get('lastFrame')}, "
                       f"轨迹点数={len(obj.get('trajectory', []))}")
    
    unified_objects, report = process_tracking_objects(
        tracking_objects,
        **params
    )
    
    logger.info(
        f"合并完成: 原始{len(tracking_objects)}个 -> "
        f"统一{len(unified_objects)}个 (合并率: {report['merge_rate']})"
    )
    logger.info(f"合并组数: {report['merged_groups']}, 单独对象: {report['single_objects']}")
    
    # 打印合并详情
    if report['merge_details']:
        logger.info(f"合并详情:")
        for detail in report['merge_details']:
            logger.info(f"  合并组{detail['group_id']}: {detail['original_ids']} -> ID {detail['unified_id']}, "
                       f"类别={detail['category']}, 帧范围={detail['frame_range']}")
    
    logger.info(f"=== 轨迹合并结束 ===")
    
    return unified_objects, report


def auto_select_scenario(tracking_objects: List[Dict]) -> str:
    """
    根据追踪对象特征自动选择合并场景
    
    参数:
        tracking_objects: 追踪对象列表
    
    返回:
        推荐的场景名称
    """
    if not tracking_objects:
        return "default"
    
    # 统计特征
    total = len(tracking_objects)
    categories = [obj.get('category', '') for obj in tracking_objects]
    durations = [obj['lastFrame'] - obj['firstFrame'] + 1 for obj in tracking_objects]
    
    # 短暂对象比例
    short_ratio = sum(1 for d in durations if d <= 5) / total
    
    # 类别统计
    adhesion_count = sum(1 for c in categories if c == 'ADHESION')
    ingot_crown_count = sum(1 for c in categories if c == 'INGOT_CROWN')
    
    # 决策逻辑
    if adhesion_count / total > 0.6:
        # 大部分是粘连物
        if short_ratio > 0.5:
            # 很多短暂片段，可能是脱落场景
            return "adhesion_falling"
        else:
            return "default"
    
    elif ingot_crown_count / total > 0.3:
        # 有较多锭冠
        return "ingot_crown_falling"
    
    elif short_ratio > 0.7:
        # 大量短暂对象，可能是快速运动
        return "fast_motion"
    
    else:
        return "default"


def smart_merge(
    tracking_objects: List[Dict],
    auto_scenario: bool = True,
    scenario: str = "default"
) -> Tuple[List[Dict], Dict]:
    """
    智能合并（自动选择场景或使用指定场景）
    
    参数:
        tracking_objects: 追踪对象列表
        auto_scenario: 是否自动选择场景
        scenario: 手动指定的场景（当auto_scenario=False时使用）
    
    返回:
        (统一后的追踪对象列表, 合并报告)
    """
    if auto_scenario:
        selected_scenario = auto_select_scenario(tracking_objects)
        logger.info(f"自动选择场景: {selected_scenario}")
    else:
        selected_scenario = scenario
    
    return merge_tracking_objects(tracking_objects, selected_scenario)


# 便捷函数：直接用于不同场景
def merge_for_adhesion(tracking_objects: List[Dict]) -> Tuple[List[Dict], Dict]:
    """粘连物场景专用合并"""
    return merge_tracking_objects(tracking_objects, "adhesion_falling")


def merge_for_ingot_crown(tracking_objects: List[Dict]) -> Tuple[List[Dict], Dict]:
    """锭冠场景专用合并"""
    return merge_tracking_objects(tracking_objects, "ingot_crown_falling")


def merge_conservative(tracking_objects: List[Dict]) -> Tuple[List[Dict], Dict]:
    """保守合并（避免误匹配）"""
    return merge_tracking_objects(tracking_objects, "conservative")


def merge_aggressive(tracking_objects: List[Dict]) -> Tuple[List[Dict], Dict]:
    """激进合并（最大化连接断裂片段）"""
    return merge_tracking_objects(tracking_objects, "aggressive")
