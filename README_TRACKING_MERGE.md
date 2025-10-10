# 追踪对象合并模块 - 使用指南

## 📋 概述

本模块用于解决粘连物和锭冠在脱落过程中因形状剧变导致的 **ID 断裂问题**。

### 问题背景

你的系统使用 **BoT-SORT (Byte On Track with ReID)** 进行多目标追踪，虽然 BoT-SORT 具有重识别能力，但在以下极端场景下仍可能失效：

- 🔹 **粘连物脱落**：从块状 → 拉长 → 断裂，外观特征剧变
- 🔹 **锭冠脱落**：从固定 → 分离 → 快速下落，位置和形状同时变化
- 🔹 **电弧遮挡**：被强光遮挡后重新出现，外观可能改变

导致同一个物体被识别为多个不同的ID。

### 解决方案

**两层防护机制：**

1. **第一层：优化 BoT-SORT 配置** (`botsort.yaml`)
   - 放宽匹配阈值
   - 延长轨迹保留时间
   - 调整 ReID 参数

2. **第二层：后处理合并算法** (本模块)
   - 基于空间-时间连续性
   - 运动预测和轨迹关联
   - 渐进形变容忍

---

## 🚀 快速开始

### 1. 最简单的用法

```python
from utils.tracking_utils import smart_merge

# 获取 BoT-SORT 的追踪结果
tracking_results = [...]  # 从 YOLO tracker 获得

# 一行代码解决问题
unified_results, report = smart_merge(tracking_results, auto_scenario=True)

# unified_results 就是合并后的结果，可以直接使用
```

### 2. 集成到处理流程

在 `analyzer/video_processor.py` 中添加：

```python
def process_video(self, ...) -> ProcessResult:
    # 现有代码：YOLO 追踪
    tracking_results = self.yolo_tracker.track_video(...)
    
    # ✨ 新增：追踪合并
    from utils.tracking_utils import smart_merge
    unified_results, report = smart_merge(tracking_results, auto_scenario=True)
    
    # 使用合并后的结果
    tracking_objects_data = self._convert_tracking_to_data(unified_results)
    
    # 其余代码保持不变...
```

### 3. 查看效果

```python
print(f"合并前: {report['total_original_objects']} 个对象")
print(f"合并后: {report['total_unified_objects']} 个对象")
print(f"合并率: {report['merge_rate']}")
```

---

## 📚 详细文档

### 核心文件

| 文件 | 说明 |
|------|------|
| `analyzer/tracking_merger.py` | 核心合并算法（无需直接调用） |
| `utils/tracking_utils.py` | **简化接口（推荐使用）** |
| `TRACKING_ID_MAINTENANCE.md` | 完整的问题分析和解决方案 |
| `MERGE_USAGE_EXAMPLES.py` | 各种使用示例代码 |

### 预定义场景

```python
from utils.tracking_utils import (
    merge_for_adhesion,      # 粘连物场景
    merge_for_ingot_crown,   # 锭冠场景
    merge_conservative,      # 保守合并（避免误匹配）
    merge_aggressive,        # 激进合并（最大化连接）
)

# 根据场景选择
if '粘连' in video_name:
    unified, report = merge_for_adhesion(tracking_results)
elif '锭冠' in video_name:
    unified, report = merge_for_ingot_crown(tracking_results)
```

---

## ⚙️ 参数配置

### 自动场景选择（推荐）

```python
unified, report = smart_merge(tracking_results, auto_scenario=True)
```

系统会根据追踪对象的特征自动选择最佳场景。

### 手动调参

```python
from analyzer.tracking_merger import process_tracking_objects

unified, report = process_tracking_objects(
    tracking_results,
    max_frame_gap=20,        # 最大帧间隔
    max_distance=120.0,      # 最大空间距离（像素）
    association_threshold=0.45  # 关联得分阈值
)
```

**参数说明：**

| 参数 | 默认值 | 说明 | 调整建议 |
|-----|--------|------|---------|
| `max_frame_gap` | 15 | 最大允许的帧间隔 | 脱落过程慢：增大到 20-30<br>脱落过程快：保持 10-15 |
| `max_distance` | 100 | 最大空间距离（px） | 快速下落：150-200<br>慢速移动：50-100 |
| `association_threshold` | 0.5 | 关联得分阈值 | 严格合并：0.6-0.7<br>宽松合并：0.4-0.5<br>激进合并：0.3-0.4 |

---

## 🔧 BoT-SORT 配置优化

在 `botsort.yaml` 中，针对脱落场景优化：

```yaml
# 推荐配置
track_buffer: 120           # 延长轨迹保留（~4-5秒）
match_thresh: 0.3           # 降低IoU阈值，适应形状变化
proximity_thresh: 0.3       # 降低ReID空间邻近度要求
appearance_thresh: 0.2      # 降低外观相似度要求
```

**⚠️ 注意：** 过于宽松的阈值可能导致误匹配，建议先测试后调整。

---

## 📊 效果验证

### 定量指标

```python
# ID切换率（越低越好）
id_switches = count_id_switches(tracking_results)
id_switch_rate = id_switches / total_frames
# 目标：< 0.001

# 碎片化率（越高说明合并效果越好）
fragmentation_rate = (original_count - unified_count) / original_count
# 目标：> 40%

# 平均追踪持续时间（应显著增加）
avg_duration_before = mean([obj['lastFrame'] - obj['firstFrame'] for obj in original])
avg_duration_after = mean([obj['lastFrame'] - obj['firstFrame'] for obj in unified])
# 目标：avg_duration_after > 2 * avg_duration_before
```

### 定性验证

在结果视频中检查：

- ✅ 粘连物从出现到脱落保持同一ID
- ✅ 没有明显的误匹配
- ✅ 边界框连续跟踪

---

## 🐛 故障排查

### 问题 1: 仍然有大量 ID 断裂

**解决方案：**

1. 先调整 BoT-SORT 配置（降低 `match_thresh`，提高 `track_buffer`）
2. 放宽合并参数：

   ```python
   unified, report = merge_aggressive(tracking_results)
   ```

### 问题 2: 出现误匹配（不同物体被合并）

**解决方案：**

1. 提高关联阈值：

   ```python
   unified, report = merge_conservative(tracking_results)
   ```

2. 或手动调整：

   ```python
   unified, report = process_tracking_objects(
       tracking_results,
       association_threshold=0.6  # 从0.5提高到0.6
   )
   ```

### 问题 3: 合并算法耗时过长

**解决方案：**

1. 预过滤短暂对象：

   ```python
   long_objects = [obj for obj in tracking_results 
                   if obj['lastFrame'] - obj['firstFrame'] > 3]
   unified, report = smart_merge(long_objects)
   ```

2. 提高 BoT-SORT 的 `new_track_thresh`，减少误检

---

## 📈 最佳实践

### ✅ DO（推荐做法）

1. **优先调整 BoT-SORT 配置**
   - BoT-SORT 在线处理效果优于离线后处理
   - 先尝试降低 `match_thresh` 和提高 `track_buffer`

2. **使用自动场景选择**
   - `smart_merge(auto_scenario=True)` 可以自动适配

3. **持续监控和验证**
   - 记录每次合并的详细日志
   - 定期检查结果视频
   - 建立测试集评估指标

### ❌ DON'T（避免做法）

1. **不要盲目激进合并**
   - 误匹配比ID断裂更难修复
   - 从保守参数开始，逐步调整

2. **不要忽略 BoT-SORT 优化**
   - 后处理不是万能的
   - 根本解决在于追踪器本身

3. **不要一刀切**
   - 不同场景可能需要不同参数
   - 粘连物 vs 锭冠 vs 爬弧应分别配置

---

## 🎯 典型案例

### 案例：粘连物逐渐拉长脱落

**合并前：**

```
ID 1:  帧 1-50   (附着阶段)
ID 2:  帧 55-80  (拉长变形)
ID 3:  帧 85-100 (继续拉长)
ID 4:  帧 105-120 (断裂脱落)
```

**合并后：**

```
统一ID 1:  帧 1-120  (完整追踪)
合并来源: [1, 2, 3, 4]
```

**效果：**

- 对象数减少：4 → 1
- 完整追踪了粘连物的整个生命周期
- 便于异常事件检测（基于统一ID）

---

## 🔗 相关资源

- BoT-SORT 论文: [arxiv.org/abs/2206.14651](https://arxiv.org/abs/2206.14651)
- Ultralytics 追踪文档: [docs.ultralytics.com/modes/track](https://docs.ultralytics.com/modes/track/)
- 项目完整文档: `TRACKING_ID_MAINTENANCE.md`

---

## 📞 支持

如有问题，请参考：

1. `MERGE_USAGE_EXAMPLES.py` - 各种使用示例
2. `TRACKING_ID_MAINTENANCE.md` - 详细的问题分析和解决方案
3. 或联系开发团队

---

**🎉 现在，你的追踪系统已经具备了强大的ID维持能力！**
