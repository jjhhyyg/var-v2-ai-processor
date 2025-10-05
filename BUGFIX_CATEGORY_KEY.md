# Bug修复：KeyError 'category'

## 问题描述

在导出标注视频时遇到以下错误：

```
KeyError: 'category'
File "/Users/erikssonhou/Projects/VAR熔池挑战/codes/ai-processor/analyzer/video_processor.py", line 323, in _draw_detections
    category = det['category']
               ~~~^^^^^^^^^^^^
```

## 根本原因

数据字段命名不一致：

1. **`yolo_tracker.track_frame()`** 返回的检测结果使用 **`class_name`** 键
2. **`_draw_detections()`** 方法错误地尝试访问 **`category`** 键

### 数据流分析

```python
# yolo_tracker.py 返回的数据结构
detection = {
    'track_id': int,
    'class_id': int,
    'class_name': str,        # ← 这里使用 class_name
    'bbox': [x1, y1, x2, y2],
    'center_x': float,
    'center_y': float,
    'width': float,
    'height': float,
    'confidence': float
}

# video_processor.py 错误地尝试访问
category = det['category']    # ✗ KeyError!
```

## 修复方案

### 修改文件

`ai-processor/analyzer/video_processor.py`

### 修改位置

第 323 行

### 修改内容

**修改前：**

```python
# 获取类别和ID
category = det['category']
track_id = det.get('track_id', -1)
confidence = det.get('confidence', 0.0)
```

**修改后：**

```python
# 获取类别和ID
category = det.get('class_name', 'Unknown')  # 使用class_name而不是category
track_id = det.get('track_id', -1)
confidence = det.get('confidence', 0.0)
```

### 改进说明

1. ✅ 使用 `det.get('class_name', 'Unknown')` 而不是 `det['category']`
2. ✅ 使用 `.get()` 方法提供默认值，避免 KeyError
3. ✅ 与 `yolo_tracker.py` 返回的数据结构保持一致

## 验证

### 检查检测结果结构

```python
# 在 yolo_tracker.py 的 track_frame() 方法中
for i in range(len(boxes_xyxy)):
    detection = {
        'track_id': int(track_ids[i]),
        'class_id': int(class_ids[i]),
        'class_name': class_names[i],     # ✓ 使用 class_name
        'bbox': [float(x1), float(y1), float(x2), float(y2)],
        'center_x': float(center_x),
        'center_y': float(center_y),
        'width': float(width),
        'height': float(height),
        'confidence': float(confidences[i])
    }
```

### 关于 'category' 字段的说明

`category` 字段**仅**在 `event_detector.py` 中使用，用于后端API：

```python
# event_detector.py
def get_tracking_objects(self):
    for track_id, track_info in all_tracks.items():
        class_name = track_info['class_name']
        
        # 将中文类别名映射为英文（用于后端）
        category = Config.CATEGORY_MAPPING.get(class_name, class_name)
        
        tracking_obj = {
            'objectId': track_id,
            'category': category,      # 英文类别（POOL_NOT_REACHED, ADHESION等）
            'className': class_name,   # 中文类别（熔池未到边、粘连物等）
            ...
        }
```

**两个不同的数据流：**

1. **检测流**：`yolo_tracker` → `video_processor` (使用 `class_name`)
2. **API流**：`event_detector` → 后端 (使用 `category` 映射)

## 测试步骤

1. **重启AI处理器**

   ```bash
   cd ai-processor
   python app.py
   ```

2. **上传测试视频**
   通过前端上传视频并启动分析

3. **验证结果视频生成**
   - 检查日志没有 `KeyError: 'category'` 错误
   - 确认结果视频成功生成
   - 验证视频中的标注（边界框、标签、ID）正确显示

4. **检查日志**

   ```bash
   tail -f ai-processor/logs/app.log
   ```

   应该看到：

   ```
   Task {taskId}: Result video exported successfully
   ```

## 相关文件

- `ai-processor/analyzer/yolo_tracker.py` - 定义检测结果数据结构
- `ai-processor/analyzer/video_processor.py` - 使用检测结果绘制标注
- `ai-processor/analyzer/event_detector.py` - 类别映射（与此bug无关）
- `ai-processor/config.py` - CATEGORY_MAPPING 配置

## 修复时间

2025-10-04 14:30

## 修复人员

AI Assistant (Claude)

---

**状态：已修复 ✅**
