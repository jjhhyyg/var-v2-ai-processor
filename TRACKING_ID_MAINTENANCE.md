# 粘连物/锭冠脱落过程中的ID维持策略

## 问题描述

在 VAR 熔池视频分析中，粘连物和锭冠在脱落过程中会经历剧烈的形状和位置变化：

1. **粘连物脱落**：
   - 初始：附着在电极或锭冠上，形状规则
   - 中期：被拉长变形，形状从块状变为细长条
   - 后期：断裂成碎片或完全脱落

2. **锭冠脱落**：
   - 初始：固定在锭子顶部
   - 中期：开始分离，倾斜或翻转
   - 后期：自由下落，可能旋转

这些过程中：

- **外观特征变化大**：ReID 可能失效
- **运动速度快**：匹配阈值可能无法覆盖
- **形状剧变**：IoU 急剧下降

导致 BoT-SORT 无法维持同一个 ID，将其识别为多个不同的物体。

---

## 解决方案：BoT-SORT + 后处理合并

### 1. BoT-SORT 配置优化

#### 当前配置 (`botsort.yaml`)

```yaml
tracker_type: botsort
track_high_thresh: 0.5      # 高置信度阈值
track_low_thresh: 0.15      # 低置信度阈值
track_buffer: 100           # 保留失踪轨迹 100 帧（~3-4秒）
match_thresh: 0.5           # IoU 匹配阈值
with_reid: true             # 启用 ReID
gmc_method: None            # 固定摄像头，无需运动补偿
```

#### 针对脱落场景的优化建议

**方案 A：保守优化（推荐，先尝试这个）**

```yaml
# 放宽匹配约束，允许更大的位置和形状变化
track_buffer: 120           # 延长轨迹保留时间（~4-5秒）
match_thresh: 0.3           # 降低 IoU 阈值，适应形状变化
proximity_thresh: 0.3       # 降低 ReID 空间邻近度要求
appearance_thresh: 0.2      # 降低外观相似度要求
```

**方案 B：激进优化（如果方案A效果不佳）**

```yaml
track_buffer: 150           # 进一步延长（~5-6秒）
match_thresh: 0.2           # 更宽松的 IoU
track_low_thresh: 0.1       # 允许更低置信度参与关联
```

**注意事项：**

- 过于宽松的阈值可能导致误匹配（将不同物体关联到一起）
- 需要在实际视频上测试并调整
- 建议先用方案A，观察效果后再决定是否采用方案B

---

### 2. 后处理合并算法

当 BoT-SORT 仍然产生ID断裂时，使用后处理合并模块：

#### 使用示例

```python
from analyzer.tracking_merger import process_tracking_objects

# 获取 BoT-SORT 的追踪结果
tracking_objects = result_data['trackingObjects']

# 应用合并算法
unified_objects, report = process_tracking_objects(
    tracking_objects,
    max_frame_gap=15,        # 最大允许帧间隔
    max_distance=100.0,      # 最大空间距离（像素）
    association_threshold=0.5  # 关联得分阈值
)

# unified_objects 即为合并后的结果
```

#### 参数调优指南

| 参数 | 默认值 | 说明 | 调整建议 |
|-----|--------|------|---------|
| `max_frame_gap` | 15 | 最大帧间隔 | - 25fps视频：10-20帧<br>- 如果脱落过程较慢，可增大到30帧 |
| `max_distance` | 100.0 | 最大空间距离（px） | - 快速下落：150-200<br>- 慢速移动：50-100 |
| `association_threshold` | 0.5 | 关联得分阈值 | - 严格合并：0.6-0.7<br>- 宽松合并：0.4-0.5<br>- 激进合并：0.3-0.4 |

---

### 3. 集成到处理流程

#### 方案 A：在 AI 模块中集成（推荐）

在 `video_processor.py` 中，追踪完成后立即应用合并：

```python
# 在 VideoProcessor.process_video() 中
# 1. 完成 YOLO 追踪
tracking_results = self.yolo_tracker.track_video(...)

# 2. 应用合并算法
from analyzer.tracking_merger import process_tracking_objects

unified_tracking, merge_report = process_tracking_objects(
    tracking_results,
    max_frame_gap=15,
    max_distance=100.0,
    association_threshold=0.5
)

# 3. 使用 unified_tracking 替代原始 tracking_results
# 提交到后端时使用合并后的数据
callback.submit_result(task_id, {
    'trackingObjects': unified_tracking,
    ...
})
```

#### 方案 B：在后端中集成

在后端接收追踪结果后进行后处理：

```java
// AnalysisTaskServiceImpl.java
@Override
public void submitResult(Long taskId, ResultSubmitRequest request) {
    // 接收原始追踪对象
    List<TrackingObjectData> trackingObjects = request.getTrackingObjects();
    
    // 调用 Python 合并脚本（通过 ProcessBuilder）
    List<TrackingObjectData> unifiedObjects = mergeTrackingObjects(trackingObjects);
    
    // 使用合并后的对象保存
    saveTrackingObjects(taskId, unifiedObjects);
}
```

---

### 4. 效果验证

#### 定量指标

1. **ID 切换率**（ID Switch Rate）

   ```
   ID切换次数 / 总帧数
   ```

   - 目标：< 0.001（每1000帧少于1次切换）

2. **轨迹碎片化率**

   ```
   (原始追踪对象数 - 合并后对象数) / 原始追踪对象数
   ```

   - 目标：> 40%（说明成功合并了40%以上的断裂）

3. **平均追踪持续时间**

   ```
   合并前：平均每个对象持续X帧
   合并后：平均每个对象持续Y帧
   ```

   - 目标：Y > 2X

#### 定性验证

在结果视频中检查：

- 粘连物从出现到完全脱落是否保持同一个ID
- 是否有明显的误匹配（不同物体被合并）
- 脱落过程的边界框是否连续

---

## 典型场景分析

### 场景 1：粘连物逐渐拉长脱落

```
帧1-50:   粘连物附着，正常追踪，ID=1
帧51-70:  开始拉长，形状变化，BoT-SORT可能丢失
          → 创建新ID=2
帧71-100: 继续拉长，再次丢失
          → 创建新ID=3
帧101-120: 断裂脱落
          → 创建新ID=4
```

**后处理合并结果：**

```
统一ID=1: 帧1-120（完整追踪）
合并来源：[1, 2, 3, 4]
```

### 场景 2：锭冠快速脱落下落

```
帧1-30:   锭冠附着，ID=5
帧31-35:  开始分离（遮挡），BoT-SORT丢失
帧36-50:  快速下落，重新检测，ID=6
帧51-60:  落出画面
```

**后处理合并结果：**

```
统一ID=5: 帧1-60
合并来源：[5, 6]
关联依据：位置连续（虽然有5帧间隔）+ 向下运动一致
```

---

## 故障排查

### 问题：合并过于激进，将不同物体合并

**原因：**

- `association_threshold` 设置过低
- `max_distance` 设置过大

**解决：**

```python
# 提高阈值
association_threshold=0.6  # 从0.5提高到0.6
max_distance=80.0          # 从100降低到80
```

### 问题：仍然有大量ID断裂

**原因：**

- 脱落速度太快，超出`max_distance`
- BoT-SORT 配置仍然过于保守

**解决：**

```python
# 放宽合并约束
max_frame_gap=30           # 从15增加到30
max_distance=150.0         # 从100增加到150
association_threshold=0.4  # 从0.5降低到0.4
```

同时调整 BoT-SORT：

```yaml
track_buffer: 150
match_thresh: 0.25
```

### 问题：合并算法耗时过长

**原因：**

- 追踪对象数量过多（如有大量短暂检测）

**解决：**

1. 预过滤短暂对象：

   ```python
   # 只合并持续 > 3 帧的对象
   long_objects = [obj for obj in tracking_objects 
                   if obj['lastFrame'] - obj['firstFrame'] > 3]
   ```

2. 优化 BoT-SORT 配置，减少误检：

   ```yaml
   new_track_thresh: 0.65  # 提高新轨迹门槛
   ```

---

## 最佳实践总结

1. **优先调整 BoT-SORT 配置**
   - 先尝试降低 `match_thresh` 和提高 `track_buffer`
   - BoT-SORT 在线处理效果好于离线后处理

2. **谨慎使用后处理合并**
   - 只在 BoT-SORT 确实无法处理时才使用
   - 从保守参数开始，逐步调整

3. **分场景配置**
   - 粘连物场景：允许较大形状变化
   - 锭冠场景：关注快速下落运动
   - 可以为不同类别使用不同参数

4. **持续监控和验证**
   - 记录每次合并的详细日志
   - 定期检查结果视频
   - 建立测试集评估指标

---

## 实施计划

### 阶段 1：BoT-SORT 优化（1-2天）

- [ ] 调整 `botsort.yaml` 配置
- [ ] 在测试视频上验证效果
- [ ] 记录ID断裂情况

### 阶段 2：后处理集成（2-3天）

- [ ] 集成 `tracking_merger` 模块到处理流程
- [ ] 编写参数配置接口
- [ ] 测试合并效果

### 阶段 3：调优和验证（3-5天）

- [ ] 在多个视频上测试
- [ ] 调整参数获得最佳效果
- [ ] 建立评估基准

### 阶段 4：生产部署（1-2天）

- [ ] 性能优化
- [ ] 添加监控和日志
- [ ] 文档和使用手册
