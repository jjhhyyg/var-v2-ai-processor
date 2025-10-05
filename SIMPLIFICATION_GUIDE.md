# 物体追踪模块简化说明

## 更新日期

2025-10-04

## 变更概述

将 `EventDetector` 模块从"异常事件检测器"简化为"物体追踪记录器"，**不再生成和判断异常事件**，仅记录检测到的各种物体及其起始帧和结束帧。

---

## 主要变更

### 1. 模块重命名和定位

- **文件名**：保持 `event_detector.py` 不变（避免影响导入）
- **模块描述**：从"异常事件检测模块" → "物体追踪记录模块"
- **类描述**：从"异常事件检测器" → "物体追踪记录器（简化版本，只记录物体，不生成事件）"

### 2. 移除的功能

- ❌ 事件生成逻辑（粘连物形成、粘连物脱落、锭冠脱落等）
- ❌ 持续性事件检测（辉光、边弧、爬弧、熔池未到边）
- ❌ 机位识别 (`_determine_camera_position()`)
- ❌ 脱落位置判断 (`_determine_drop_location()`)
- ❌ 锭冠脱落判断 (`_is_crown_dropped()`)
- ❌ 事件创建方法 (`_create_event()`)
- ❌ 边界框转字典方法 (`_bbox_to_dict()`)
- ❌ 事件列表存储 (`self.events`)
- ❌ 事件ID计数器 (`self.event_id_counter`)
- ❌ 机位缓存 (`self.camera_position`)

### 3. 保留的功能

- ✅ 追踪物体状态管理 (`active_tracks`, `completed_tracks`)
- ✅ 物体首次出现记录
- ✅ 物体消失检测
- ✅ 轨迹记录
- ✅ 中心点计算 (`_get_center()`)

### 4. 修改的方法

#### `__init__()`

```python
# 移除：
- self.events = []
- self.event_id_counter = 1
- self.camera_position = None

# 保留：
- self.active_tracks = {}
- self.completed_tracks = {}
- self.class_names = Config.CLASS_NAMES
```

#### `process_detections()`

```python
# 返回值：从返回事件列表 → 返回空列表 []
# 逻辑：
- 记录新物体的出现（first_frame, first_time）
- 更新物体的最后出现（last_frame, last_time）
- 将消失的物体移到 completed_tracks
- 不再生成任何事件
```

#### `finalize_events()`

```python
# 返回值：从返回事件列表 → 返回空列表 []
# 逻辑：
- 将所有活跃追踪移到 completed_tracks
- 清空 active_tracks
- 不再生成任何事件
```

#### `get_tracking_objects()`

**重大改进**：现在返回**所有类别**的物体信息

之前只返回：粘连物、锭冠  
现在返回：所有检测到的物体（辉光、边弧、爬弧、熔池未到边、粘连物、锭冠）

**返回格式**：

```python
[
    {
        'objectId': 1,           # 追踪ID
        'className': '粘连物',    # 类别名称
        'classId': 1,            # 类别ID
        'firstFrame': 100,       # 起始帧
        'lastFrame': 250,        # 结束帧
        'firstTime': 3.33,       # 起始时间（秒）
        'lastTime': 8.33,        # 结束时间（秒）
        'duration': 151          # 持续帧数
    },
    ...
]
```

**排序**：按 `firstFrame` 升序排序

---

## 代码文件变更

### 修改的文件

1. **`analyzer/event_detector.py`**
   - 完全重写，大幅简化
   - 从 ~440 行减少到 ~163 行
   - 移除所有事件生成和判断逻辑

2. **`analyzer/video_processor.py`**
   - 更新注释，说明不再生成事件
   - 更新日志输出，不再报告事件数量
   - API保持兼容（`anomalyEvents` 字段仍存在，只是为空列表）

### 保持兼容性

为了避免影响前端和后端：

- `anomalyEvents` 字段仍然存在于返回结果中，只是值为空列表 `[]`
- `trackingObjects` 字段仍然存在，但现在包含所有类别的物体

---

## API 返回格式变化

### 之前的返回结果

```python
{
    'status': 'COMPLETED',
    'anomalyEvents': [
        {'eventType': 'ADHESION_FORMED', ...},
        {'eventType': 'ADHESION_DROPPED', ...},
        {'eventType': 'GLOW', ...},
        ...
    ],
    'trackingObjects': [
        # 只包含粘连物和锭冠
    ]
}
```

### 现在的返回结果

```python
{
    'status': 'COMPLETED',
    'anomalyEvents': [],  # 始终为空
    'trackingObjects': [
        # 包含所有类别：辉光、边弧、爬弧、熔池未到边、粘连物、锭冠
        {
            'objectId': 1,
            'className': '辉光',
            'classId': 3,
            'firstFrame': 50,
            'lastFrame': 120,
            'firstTime': 1.67,
            'lastTime': 4.0,
            'duration': 71
        },
        ...
    ]
}
```

---

## 前端需要的调整

前端需要根据 `trackingObjects` 来显示检测到的物体信息，而不是依赖 `anomalyEvents`。

### 建议的前端展示方式

1. **按类别分组显示**

   ```
   辉光: 3次
   边弧（侧弧）: 2次
   爬弧: 1次
   粘连物: 5次
   锭冠: 2次
   熔池未到边: 4次
   ```

2. **时间线展示**
   - 在视频播放器下方显示物体出现的时间段
   - 不同类别用不同颜色表示

3. **详细列表**

   ```
   ID  类别      起始帧  结束帧  持续时间
   1   粘连物    100    250     5.00s
   2   辉光      50     120     2.33s
   ...
   ```

---

## 日志输出变化

### 新增的调试日志

```
DEBUG: 新物体追踪开始: ID=1, 类别=粘连物, 帧=100
DEBUG: 物体追踪结束: ID=1, 类别=粘连物, 帧=100-250
DEBUG: 视频结束，物体追踪完成: ID=2, 类别=辉光, 帧=50-120
INFO: 追踪完成，共记录 10 个物体
```

### 修改的信息日志

```
# 之前：
INFO: Task 123: Metrics: 1000, Events: 15, Tracks: 5

# 现在：
INFO: Task 123: Tracked objects: 10, Metrics: 1000
```

---

## 优势

1. **逻辑简化**：不需要判断复杂的事件规则
2. **代码更少**：减少约 60% 的代码量
3. **维护性好**：更容易理解和修改
4. **灵活性高**：前端可以根据物体信息自定义展示和分析
5. **扩展性强**：未来如需添加事件判断，可以在前端或单独的分析模块中实现

---

## 后续建议

1. **前端自定义事件规则**
   - 可以在前端根据 `trackingObjects` 数据自定义事件规则
   - 例如：粘连物持续时间 > 5秒视为异常

2. **独立的事件分析服务**
   - 如需复杂的事件判断，可以创建独立的分析服务
   - 接收 `trackingObjects` 数据，返回分析结果

3. **统计分析**
   - 可以基于追踪数据生成统计报告
   - 例如：各类异常的出现频率、持续时间分布等
