# AI处理模块使用指南

## 快速开始

### 1. 准备工作

```bash
# 进入ai-processor目录
cd ai-processor

# 复制环境变量配置文件
cp .env.example .env

# 编辑配置文件，根据实际情况修改
vim .env
```

### 2. 准备模型文件

将训练好的YOLO模型文件放置在 `weights/best.pt`，或在 `.env` 中配置模型路径：

```bash
# 创建weights目录
mkdir -p weights

# 将模型文件复制到weights目录
cp /path/to/your/model.pt weights/best.pt
```

### 3. 配置ByteTrack追踪器

编辑 `bytetrack.yaml` 文件，调整追踪参数：

```yaml
# 针对熔池场景的推荐配置
track_high_thresh: 0.5      # 高置信度阈值
track_low_thresh: 0.1       # 低置信度阈值
new_track_thresh: 0.6       # 新轨迹初始化阈值
track_buffer: 30            # 轨迹缓冲帧数
match_thresh: 0.8           # 轨迹匹配阈值
fuse_score: false           # 是否融合置信度分数
gmc_method: sparseOptFlow   # 全局运动补偿方法
```

### 4. 启动服务

**方式一：使用启动脚本（推荐）**

```bash
./start.sh
```

**方式二：手动启动**

```bash
# 激活conda环境
conda activate pytorch

# 安装依赖（首次运行）
pip install -r requirements.txt

# 启动服务
python app.py
```

### 5. 验证服务

```bash
# 健康检查
curl http://localhost:5000/health

# 预期输出
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "yolo11n",
  "gpu_available": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

## 配置说明

### 环境变量配置

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `YOLO_MODEL_PATH` | YOLO模型路径 | `weights/best.pt` | `weights/best.pt` |
| `YOLO_DEVICE` | 计算设备 | 自动选择 | `cuda`, `cpu`, `mps`, `0` |
| `TRACKER_CONFIG` | ByteTrack配置文件 | `bytetrack.yaml` | `bytetrack.yaml` |
| `DEFAULT_CONFIDENCE_THRESHOLD` | 默认置信度阈值 | `0.5` | `0.6` |
| `DEFAULT_IOU_THRESHOLD` | 默认IoU阈值 | `0.45` | `0.5` |
| `PROGRESS_UPDATE_INTERVAL` | 进度更新频率（帧） | `30` | `60` |

### ByteTrack参数说明

参考demo代码中的配置，关键参数：

1. **track_high_thresh** (0.0-1.0)
   - 第一次关联的置信度阈值
   - 熔池场景推荐：0.5-0.6

2. **track_low_thresh** (0.0-1.0)
   - 第二次关联的置信度阈值
   - 通常为 track_high_thresh 的 1/3 到 1/5
   - 熔池场景推荐：0.1-0.15

3. **track_buffer** (整数)
   - 目标丢失后保留的帧数
   - 30fps视频推荐：30-60（1-2秒）
   - 熔池场景（有遮挡）推荐：60-90

4. **match_thresh** (0.0-1.0)
   - IoU匹配阈值
   - 快速运动目标：0.6-0.7
   - 慢速运动：0.8-0.9

5. **gmc_method**
   - 全局运动补偿方法
   - 固定摄像头：`None`
   - 轻微抖动：`sparseOptFlow`
   - 移动摄像头：`orb` 或 `ecc`

## 接口使用

### 1. 健康检查

```bash
GET /health
```

**响应示例：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "yolo11n",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "device": "cuda",
  "version": "1.0.0"
}
```

### 2. 视频分析

```bash
POST /api/analyze
Content-Type: application/json
```

**请求体：**
```json
{
  "taskId": 123,
  "videoPath": "/path/to/video.mp4",
  "videoDuration": 1800,
  "timeoutThreshold": 7200,
  "config": {
    "confidenceThreshold": 0.5,
    "iouThreshold": 0.45
  }
}
```

**响应示例：**
```json
{
  "status": "accepted",
  "taskId": 123,
  "message": "任务已接受，开始处理"
}
```

## 输出格式

### 检测结果格式

参照demo代码的CSV输出格式：

```python
{
    'frame': 100,              # 帧号
    'track_id': 12,            # 追踪ID
    'class_id': 1,             # 类别ID
    'class_name': '粘连物',     # 类别名称
    'confidence': 0.95,        # 置信度
    'bbox': [x1, y1, x2, y2],  # 边界框（xyxy格式）
    'center_x': 320.5,         # 中心点x坐标
    'center_y': 240.3,         # 中心点y坐标
    'width': 40.0,             # 宽度
    'height': 50.0             # 高度
}
```

### 事件类型

系统会自动推断以下事件：

| 事件类型 | 说明 | 推断逻辑 |
|---------|------|---------|
| `ADHESION_FORMED` | 电极形成粘连物 | 粘连物首次被检测到 |
| `ADHESION_DROPPED` | 粘连物脱落 | 粘连物轨迹消失 |
| `CROWN_DROPPED` | 锭冠脱落 | 锭冠从边缘运动至熔池 |
| `GLOW` | 辉光 | 持续事件 |
| `SIDE_ARC` | 边弧（侧弧） | 持续事件 |
| `CLIMBING_ARC` | 爬弧 | 持续事件 |
| `POOL_NOT_EDGE` | 熔池未到边 | 持续事件 |

## 调试技巧

### 1. 启用详细输出

```bash
# 在.env中设置
YOLO_VERBOSE=True
```

### 2. 调整进度更新频率

```bash
# 更频繁的进度更新（每10帧）
PROGRESS_UPDATE_INTERVAL=10
```

### 3. 测试不同设备

```bash
# 使用CPU
YOLO_DEVICE=cpu

# 使用第一块GPU
YOLO_DEVICE=0

# 使用Apple Silicon加速
YOLO_DEVICE=mps
```

### 4. 调整追踪参数

编辑 `bytetrack.yaml`，实时生效（需重启服务）。

## 常见问题

### Q1: 模型加载失败

**A:** 检查模型文件路径和权限：
```bash
ls -lh weights/best.pt
```

### Q2: GPU不可用

**A:** 检查CUDA安装：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Q3: 追踪ID频繁变化

**A:** 调整ByteTrack参数：
- 降低 `track_high_thresh`（如从0.6降到0.5）
- 增加 `track_buffer`（如从30增到60）
- 降低 `match_thresh`（如从0.8降到0.7）

### Q4: 漏检或误检较多

**A:** 调整置信度阈值：
- 漏检多：降低 `DEFAULT_CONFIDENCE_THRESHOLD`
- 误检多：提高 `DEFAULT_CONFIDENCE_THRESHOLD`

### Q5: 处理速度慢

**A:** 优化建议：
- 使用GPU加速（`YOLO_DEVICE=cuda`）
- 降低进度更新频率（`PROGRESS_UPDATE_INTERVAL=60`）
- 使用更小的模型（如yolo11n.pt）
- 关闭详细输出（`YOLO_VERBOSE=False`）

## 性能监控

### 查看GPU使用情况

```bash
# NVIDIA GPU
nvidia-smi

# Apple Silicon
sudo powermetrics --samplers gpu_power
```

### 查看日志

```bash
# 实时查看日志
tail -f logs/ai-processor.log
```

## 与后端集成

AI处理模块会通过HTTP回调方式与后端通信：

1. **进度更新**: `POST {AI_CALLBACK_URL}/{taskId}/progress`
2. **结果提交**: `POST {AI_CALLBACK_URL}/{taskId}/result`

确保后端接口已实现。

## 参考资料

- [Ultralytics官方文档](https://docs.ultralytics.com/)
- [ByteTrack论文](https://arxiv.org/abs/2110.06864)
- [接口设计文档](../接口设计文档.md)
- [系统设计文档](../系统设计文档.md)
