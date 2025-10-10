"""
追踪对象合并 - 快速使用示例

展示如何在video_processor.py中集成追踪合并功能

作者：侯阳洋
日期：2025-10-10
"""

# ============================================================================
# 示例 1: 在 VideoProcessor 中集成（推荐方式）
# ============================================================================

"""
在 analyzer/video_processor.py 的 process_video() 方法中：

def process_video(self, ...) -> ProcessResult:
    # ... 现有的预处理代码 ...
    
    # 步骤1: YOLO追踪（现有代码）
    tracking_results = self.yolo_tracker.track_video(
        video_path=preprocessed_path,
        output_path=output_video_path,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # ========== 新增：追踪对象合并 ==========
    # 步骤2: 应用追踪合并（解决ID断裂问题）
    from utils.tracking_utils import smart_merge
    
    logger.info("应用追踪对象合并算法...")
    unified_tracking_results, merge_report = smart_merge(
        tracking_results,
        auto_scenario=True  # 自动选择合并策略
    )
    
    # 记录合并效果
    logger.info(
        f"追踪合并完成: {merge_report['total_original_objects']} -> "
        f"{merge_report['total_unified_objects']} 对象 "
        f"(合并了 {merge_report['merged_groups']} 组)"
    )
    # =====================================
    
    # 步骤3: 使用合并后的结果进行后续处理
    tracking_objects_data = self._convert_tracking_to_data(
        unified_tracking_results  # 使用合并后的结果
    )
    
    # ... 其余代码保持不变 ...
"""

# ============================================================================
# 示例 2: 手动指定场景
# ============================================================================

"""
如果你知道当前处理的是特定场景，可以手动指定：

from utils.tracking_utils import merge_for_adhesion, merge_for_ingot_crown

# 针对粘连物视频
if '粘连' in video_name:
    unified_results, report = merge_for_adhesion(tracking_results)

# 针对锭冠视频
elif '锭冠' in video_name:
    unified_results, report = merge_for_ingot_crown(tracking_results)

# 其他场景使用默认
else:
    unified_results, report = smart_merge(tracking_results)
"""

# ============================================================================
# 示例 3: 保守/激进模式
# ============================================================================

"""
根据需求选择合并强度：

from utils.tracking_utils import merge_conservative, merge_aggressive

# 保守模式：避免误匹配，适合高精度要求场景
unified_results, report = merge_conservative(tracking_results)

# 激进模式：最大化连接断裂片段，适合ID一致性要求高的场景
unified_results, report = merge_aggressive(tracking_results)
"""

# ============================================================================
# 示例 4: 完整的集成代码（复制粘贴可用）
# ============================================================================

def example_integration():
    """
    完整的集成示例代码
    可以直接复制到 video_processor.py 中使用
    """
    
    # 假设这是从 YOLO 追踪获得的结果
    tracking_results = [
        {
            'trackingId': '123',
            'objectId': 1,
            'category': 'ADHESION',
            'firstFrame': 1,
            'lastFrame': 50,
            'trajectory': [...]
        },
        {
            'trackingId': '124',
            'objectId': 2,
            'category': 'ADHESION',
            'firstFrame': 55,  # 注意：有5帧间隔
            'lastFrame': 100,
            'trajectory': [...]
        },
        # ... 更多追踪对象
    ]
    
    # 方法1: 智能合并（推荐）
    from utils.tracking_utils import smart_merge
    
    unified_results, report = smart_merge(
        tracking_results,
        auto_scenario=True
    )
    
    # 打印合并报告
    print(f"原始对象数: {report['total_original_objects']}")
    print(f"合并后对象数: {report['total_unified_objects']}")
    print(f"合并了 {report['merged_groups']} 组")
    
    # 查看合并详情
    for detail in report['merge_details']:
        print(f"\n统一ID {detail['unified_id']}:")
        print(f"  合并了 {detail['object_count']} 个追踪片段")
        print(f"  原始ID: {detail['original_ids']}")
        print(f"  帧范围: {detail['frame_range']}")
    
    return unified_results


# ============================================================================
# 示例 5: 在 callback 中集成（后端接收前处理）
# ============================================================================

"""
在 utils/callback.py 的 submit_result() 方法中：

def submit_result(self, task_id: int, result_data: dict):
    # 获取原始追踪结果
    tracking_objects = result_data.get('trackingObjects', [])
    
    # 应用合并
    from utils.tracking_utils import smart_merge
    
    if tracking_objects:
        unified_objects, merge_report = smart_merge(tracking_objects)
        
        # 替换为合并后的结果
        result_data['trackingObjects'] = unified_objects
        
        # 添加合并报告到元数据
        result_data['trackingMergeReport'] = merge_report
    
    # 提交到后端
    response = requests.post(
        f"{self.base_url}/api/tasks/{task_id}/result",
        json=result_data
    )
"""

# ============================================================================
# 示例 6: 测试和验证
# ============================================================================

def test_merge_effectiveness():
    """
    测试合并效果的代码示例
    """
    import requests
    from utils.tracking_utils import smart_merge
    
    # 从后端获取一个任务的追踪结果
    task_id = "102104311357505536"
    response = requests.get(f"http://localhost:8080/api/tasks/{task_id}/result")
    data = response.json()['data']
    
    original_objects = data['trackingObjects']
    
    print("="*80)
    print("合并前:")
    print(f"  总对象数: {len(original_objects)}")
    
    # 统计正负ID
    positive_ids = [obj for obj in original_objects if obj['objectId'] >= 0]
    negative_ids = [obj for obj in original_objects if obj['objectId'] < 0]
    
    print(f"  正ID对象: {len(positive_ids)}")
    print(f"  负ID对象: {len(negative_ids)}")
    
    # 应用合并
    unified_objects, report = smart_merge(original_objects)
    
    print("\n合并后:")
    print(f"  总对象数: {len(unified_objects)}")
    print(f"  合并了 {report['merged_groups']} 组")
    print(f"  合并率: {report['merge_rate']}")
    
    # 计算改善效果
    reduction_rate = (len(original_objects) - len(unified_objects)) / len(original_objects) * 100
    print(f"\n对象数减少: {reduction_rate:.1f}%")
    
    # 找出最显著的合并案例
    if report['merge_details']:
        max_merge = max(report['merge_details'], key=lambda x: x['object_count'])
        print(f"\n最显著的合并案例:")
        print(f"  统一ID: {max_merge['unified_id']}")
        print(f"  合并了 {max_merge['object_count']} 个片段")
        print(f"  原始ID: {max_merge['original_ids']}")
        print(f"  帧范围: {max_merge['frame_range']}")


if __name__ == "__main__":
    # 运行测试
    test_merge_effectiveness()
