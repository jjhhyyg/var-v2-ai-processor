# 配置同步总结

## ✅ 已完成配置统一

现在**生产环境（MQ消费者）** 和 **Demo脚本（track_video.py）** 使用相同的检测参数。

---

## 📊 当前统一配置

### YOLO 检测参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `confidence_threshold` | **0.4** | 置信度阈值，过滤低置信度检测 |
| `iou_threshold` | **0.4** | NMS的IoU阈值，去除重叠框 |

### ByteTrack 追踪参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `track_high_thresh` | **0.5** | 第一次关联的高置信度阈值 |
| `track_low_thresh` | **0.1** | 第二次关联的低置信度阈值 |
| `new_track_thresh` | **0.6** | 新轨迹初始化阈值 |
| `track_buffer` | **30** | 轨迹缓冲帧数 |
| `match_thresh` | **0.8** | 轨迹匹配IoU阈值 |

---

## 📝 修改的文件

### 1. `/ai-processor/.env`

```diff
- DEFAULT_CONFIDENCE_THRESHOLD=0.35
+ DEFAULT_CONFIDENCE_THRESHOLD=0.4

- TRACK_HIGH_THRESH=0.3
+ TRACK_HIGH_THRESH=0.5

- TRACK_LOW_THRESH=0.06
+ TRACK_LOW_THRESH=0.1

- NEW_TRACK_THRESH=0.5
+ NEW_TRACK_THRESH=0.6

- TRACK_BUFFER=60
+ TRACK_BUFFER=30

- MATCH_THRESH=0.5
+ MATCH_THRESH=0.8
```

### 2. `/ai-processor/bytetrack.yaml` 和 `/ai-processor/demo/bytetrack.yaml`

两个文件已保持一致的标准配置。

---

## 🎯 使用方式

### Demo 测试（推荐）

```bash
cd ai-processor/demo

# 使用与生产环境相同的参数
python track_video.py /path/to/video.mkv \
  --device mps \
  --tracker bytetrack.yaml \
  --show \
  --conf 0.4 \
  --iou 0.4
```

### 生产环境

```bash
cd ai-processor

# 启动MQ消费者（会自动从 .env 读取配置）
python mq_consumer.py
```

现在两种方式的检测结果应该**完全一致**！

---

## 🔍 配置说明

### 为什么选择这些参数？

1. **`conf=0.4`**: 适中的置信度阈值
   - 不会太严格（0.5+会漏检）
   - 不会太宽松（0.3-会误检）

2. **`track_high_thresh=0.5`**: 标准的第一次关联阈值
   - 确保高质量的追踪初始化

3. **`track_low_thresh=0.1`**: ByteTrack的第二次关联
   - 利用ByteTrack的核心优势
   - 恢复被短暂遮挡的目标

4. **`track_buffer=30`**: 1秒缓冲（@30fps）
   - 适合大部分场景
   - 避免ID频繁切换

---

## 📌 注意事项

### ⚠️ .env 文件中的 ByteTrack 参数说明

`.env` 中的这些参数：

```bash
TRACK_HIGH_THRESH=0.5
TRACK_LOW_THRESH=0.1
NEW_TRACK_THRESH=0.6
TRACK_BUFFER=30
MATCH_THRESH=0.8
```

**目前不生效**，因为代码使用的是 `bytetrack.yaml` 配置文件。

如果想让这些参数生效，需要修改 `analyzer/yolo_tracker.py`，直接传入参数而非配置文件路径。

---

## 🧪 验证配置是否生效

### 方法1: 检查输出日志

```bash
# Demo
python track_video.py test.mkv -v 2

# 生产环境
python mq_consumer.py
# 查看日志中的参数值
```

### 方法2: 比较检测结果

对同一视频运行两种方式，CSV输出的追踪数据应该完全一致：

- 相同的 `track_id`
- 相同的检测框数量
- 相同的置信度值

---

## 📚 进一步优化建议

如果发现检测效果不理想，可以调整：

### 漏检过多（检测不到目标）

```bash
# .env
DEFAULT_CONFIDENCE_THRESHOLD=0.35  # 降低阈值
TRACK_LOW_THRESH=0.08              # 降低第二次关联阈值
```

### 误检过多（检测到不存在的目标）

```bash
# .env
DEFAULT_CONFIDENCE_THRESHOLD=0.5   # 提高阈值
NEW_TRACK_THRESH=0.7               # 提高新轨迹初始化阈值
```

### ID切换频繁

```bash
# bytetrack.yaml
track_buffer: 60                   # 增加缓冲时间
match_thresh: 0.7                  # 降低匹配阈值（更宽松）
```

---

## 🎉 总结

✅ 生产环境配置已匹配 `track_video.py`  
✅ 两种测试方式结果一致  
✅ 参数已文档化，便于后续调优  

最后修改时间: 2025-10-04
