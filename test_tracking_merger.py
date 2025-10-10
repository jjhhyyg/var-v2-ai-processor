"""
追踪对象合并测试脚本

演示如何使用tracking_merger模块解决粘连物/锭冠脱落过程中ID断裂的问题

作者：侯阳洋
日期：2025-10-10
"""

import sys
import requests
import json
from analyzer.tracking_merger import process_tracking_objects, TrackingMerger
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_tracking_merger(task_id: str, backend_url: str = "http://localhost:8080"):
    """
    测试追踪对象合并功能
    
    参数:
        task_id: 任务ID
        backend_url: 后端URL
    """
    print("="*80)
    print(f"追踪对象合并测试 - 任务ID: {task_id}")
    print("="*80)
    
    # 1. 获取原始追踪数据
    print("\n步骤1: 获取原始追踪数据...")
    url = f"{backend_url}/api/tasks/{task_id}/result"
    response = requests.get(url)
    response.raise_for_status()
    
    data = response.json()
    if data['code'] != 200:
        raise Exception(f"API返回错误: {data['message']}")
    
    result_data = data['data']
    tracking_objects = result_data.get('trackingObjects', [])
    
    print(f"  原始追踪对象数: {len(tracking_objects)}")
    
    # 统计原始对象
    positive_ids = [obj for obj in tracking_objects if obj['objectId'] >= 0]
    negative_ids = [obj for obj in tracking_objects if obj['objectId'] < 0]
    
    print(f"  正ID对象数（持续追踪）: {len(positive_ids)}")
    print(f"  负ID对象数（短暂检测）: {len(negative_ids)}")
    
    # 2. 应用合并算法
    print("\n步骤2: 应用追踪对象合并算法...")
    print("  配置参数:")
    print("    - 最大帧间隔: 15帧")
    print("    - 最大空间距离: 100像素")
    print("    - 关联得分阈值: 0.5")
    print("    - 允许形状变化: True (适应粘连物脱落)")
    
    unified_objects, report = process_tracking_objects(
        tracking_objects,
        max_frame_gap=15,
        max_distance=100.0,
        association_threshold=0.5
    )
    
    # 3. 显示合并报告
    print("\n步骤3: 合并结果报告")
    print("-"*80)
    print(f"  原始对象数: {report['total_original_objects']}")
    print(f"  合并后对象数: {report['total_unified_objects']}")
    print(f"  合并组数: {report['merged_groups']}")
    print(f"  单独对象数: {report['single_objects']}")
    print(f"  合并率: {report['merge_rate']}")
    
    # 显示合并详情
    if report['merge_details']:
        print(f"\n  合并详情（共{len(report['merge_details'])}组）:")
        for detail in report['merge_details'][:10]:  # 只显示前10个
            print(f"\n    组 {detail['group_id']}:")
            print(f"      统一ID: {detail['unified_id']}")
            print(f"      类别: {detail['category']}")
            print(f"      合并对象数: {detail['object_count']}")
            print(f"      原始ID: {detail['original_ids']}")
            print(f"      帧范围: {detail['frame_range'][0]}-{detail['frame_range'][1]} (共{detail['total_frames']}帧)")
    
    # 4. 对比分析
    print("\n步骤4: 对比分析")
    print("-"*80)
    
    # 找出合并效果最显著的例子
    if report['merge_details']:
        max_merge = max(report['merge_details'], key=lambda x: x['object_count'])
        print(f"\n  最显著的合并案例:")
        print(f"    统一ID: {max_merge['unified_id']}")
        print(f"    类别: {max_merge['category']}")
        print(f"    合并了 {max_merge['object_count']} 个原始追踪片段")
        print(f"    原始ID列表: {max_merge['original_ids']}")
        print(f"    完整追踪范围: 第{max_merge['frame_range'][0]}帧 到 第{max_merge['frame_range'][1]}帧")
        print(f"    总持续时间: {max_merge['total_frames']}帧")
        print(f"\n  ✅ 解释: 这个物体原本被分割成{max_merge['object_count']}个不同的ID，")
        print(f"          现在通过合并算法识别为同一个物体，维持统一ID {max_merge['unified_id']}")
    
    # 5. 保存结果
    print("\n步骤5: 保存合并结果...")
    
    # 保存统一后的追踪对象
    output_file = f"unified_tracking_{task_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'task_id': task_id,
            'task_name': result_data['name'],
            'original_count': len(tracking_objects),
            'unified_count': len(unified_objects),
            'merge_report': report,
            'unified_objects': unified_objects
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 统一后的追踪对象已保存到: {output_file}")
    
    # 保存合并报告
    report_file = f"merge_report_{task_id}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 合并报告已保存到: {report_file}")
    
    # 6. 使用建议
    print("\n步骤6: 集成建议")
    print("-"*80)
    print("\n  📋 如何在后处理中使用:")
    print("     1. 在AI模块完成追踪后，调用此合并算法")
    print("     2. 使用统一后的追踪对象替代原始追踪数据")
    print("     3. 基于统一ID检测异常事件（如粘连物持续时间、脱落事件等）")
    print("\n  📋 参数调优建议:")
    print("     - max_frame_gap: 根据视频帧率调整，帧率越高可以设置越大")
    print("     - max_distance: 根据物体运动速度调整，移动快的物体需要更大值")
    print("     - association_threshold: 0.4-0.6之间，越低合并越激进")
    print("\n  📋 特殊场景处理:")
    print("     - 粘连物脱落: 算法已考虑形状变化，可以追踪脱落全过程")
    print("     - 锭冠脱落: 通过运动预测和位置连续性，即使形变大也能保持ID")
    print("     - 遮挡恢复: 短暂消失后重新出现的物体会被正确关联")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    
    return unified_objects, report


def compare_before_after(task_id: str, backend_url: str = "http://localhost:8080"):
    """对比合并前后的差异"""
    
    # 获取原始数据
    url = f"{backend_url}/api/tasks/{task_id}/result"
    response = requests.get(url)
    data = response.json()['data']
    tracking_objects = data.get('trackingObjects', [])
    
    # 合并处理
    unified_objects, report = process_tracking_objects(tracking_objects)
    
    print("\n对比分析：合并前 vs 合并后")
    print("="*80)
    
    # 找一个被合并的例子
    for detail in report['merge_details']:
        if detail['object_count'] >= 3:  # 至少合并了3个
            print(f"\n案例：统一ID {detail['unified_id']} ({detail['category']})")
            print("-"*80)
            
            print(f"\n【合并前】{detail['object_count']} 个独立的追踪片段:")
            for orig_id in detail['original_ids']:
                orig_obj = next(obj for obj in tracking_objects if obj['objectId'] == orig_id)
                print(f"  ID {orig_id:>6}: 帧 {orig_obj['firstFrame']:>4} - {orig_obj['lastFrame']:>4} "
                      f"(持续 {orig_obj['lastFrame']-orig_obj['firstFrame']+1:>3}帧)")
            
            print(f"\n【合并后】1 个连续的追踪对象:")
            unified_obj = next(obj for obj in unified_objects if obj['objectId'] == detail['unified_id'])
            print(f"  ID {detail['unified_id']:>6}: 帧 {unified_obj['firstFrame']:>4} - {unified_obj['lastFrame']:>4} "
                  f"(持续 {unified_obj['lastFrame']-unified_obj['firstFrame']+1:>3}帧)")
            
            print(f"\n✅ 效果: 将断裂的追踪片段重新连接，维持了物体的ID一致性")
            break


def main():
    """主函数"""
    # 获取已完成的任务
    print("正在获取已完成的任务列表...")
    response = requests.get("http://localhost:8080/api/tasks?status=COMPLETED&size=5")
    tasks_data = response.json()
    
    if tasks_data['code'] != 200:
        print(f"获取任务列表失败: {tasks_data['message']}")
        return
    
    tasks = tasks_data['data']['items']
    
    if not tasks:
        print("没有找到已完成的任务")
        return
    
    print(f"\n找到 {len(tasks)} 个已完成的任务:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. [{task['taskId']}] {task['name']}")
    
    # 选择第一个任务进行测试
    selected_task = tasks[0]
    task_id = selected_task['taskId']
    
    print(f"\n选择任务: {selected_task['name']} (ID: {task_id})")
    
    # 运行测试
    unified_objects, report = test_tracking_merger(task_id)
    
    # 对比分析
    compare_before_after(task_id)


if __name__ == "__main__":
    main()
