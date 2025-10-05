# VAR熔池视频分析系统 - AI处理模块

基于Flask + PyTorch + Ultralytics YOLO的视频分析服务。

## 功能特性

- ✅ YOLOv11目标检测和BotSort多目标追踪
- ✅ 异常事件自动检测（粘连物、锭冠、辉光、边弧、爬弧等）
- ✅ 动态参数计算（熔池闪烁频率、面积、周长）
- ✅ 实时进度回调
- ✅ 超时检测和预警
- ✅ 健康检查接口

## 系统要求

### Python环境

- Python 3.9+
- Conda环境（推荐）

### 硬件要求

- CPU: 多核处理器
- GPU: NVIDIA GPU（推荐，需要CUDA支持）或Apple Silicon（支持MPS）
- 内存: 8GB+（推荐16GB+）

## 安装步骤

### 1. 创建并激活Conda环境

```bash
# 使用已配置好的pytorch环境
conda activate pytorch
```

### 2. 安装依赖包

```bash
cd ai-processor
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑.env文件，修改相应配置
vim .env
```

### 4. 下载YOLO模型

```bash
# 模型会在首次运行时自动下载
# 或手动下载并放置在指定路径
# 下载地址: https://github.com/ultralytics/assets/releases
```

## 使用方法

### 启动服务

```bash
conda activate pytorch
python app.py
```

服务将在 `http://localhost:5000` 启动。

### 健康检查

```bash
curl http://localhost:5000/health
```

### API接口

#### 1. 健康检查

```
GET /health
```

响应示例：

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

#### 2. 视频分析

```
POST /api/analyze
```

请求体：

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

响应示例：

```json
{
  "status": "accepted",
  "taskId": 123,
  "message": "任务已接受，开始处理"
}
```

## 项目结构

```
ai-processor/
├── analyzer/              # 核心分析模块
│   ├── __init__.py
│   ├── video_processor.py    # 视频处理主逻辑
│   ├── yolo_tracker.py       # YOLO检测和追踪
│   ├── event_detector.py     # 事件检测
│   └── metrics_calculator.py # 动态参数计算
├── utils/                 # 工具模块
│   ├── __init__.py
│   └── callback.py           # 后端回调工具
├── app.py                 # Flask主应用
├── config.py              # 配置文件
├── requirements.txt       # Python依赖
├── .env.example           # 环境变量示例
└── README.md              # 本文档
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| AI_PROCESSOR_HOST | 服务监听地址 | 0.0.0.0 |
| AI_PROCESSOR_PORT | 服务端口 | 5000 |
| AI_CALLBACK_URL | 后端回调URL | <http://localhost:8080/api/tasks> |
| YOLO_MODEL_PATH | YOLO模型路径 | yolo11n.pt |
| YOLO_DEVICE | 计算设备 | （自动选择） |
| DEFAULT_CONFIDENCE_THRESHOLD | 默认置信度阈值 | 0.5 |
| DEFAULT_IOU_THRESHOLD | 默认IoU阈值 | 0.45 |

### 类别定义

系统检测以下6类物体/现象：

| ID | 类别名称 | 说明 |
|----|---------|------|
| 0 | 熔池未到边 | 熔池边缘尚未达到结晶器边缘 |
| 1 | 粘连物 | 电极表面形成的黑色不规则粘连物 |
| 2 | 锭冠 | 结晶器边缘的锭冠 |
| 3 | 辉光 | 电极环缝区域气体异常放电现象 |
| 4 | 边弧（侧弧） | 电弧持续出现在电极边缘 |
| 5 | 爬弧 | 电极表面出现的电弧迹线 |

## 事件推断逻辑

### 1. 粘连物相关事件

- **电极形成粘连物**：粘连物首次被检测到
- **电极粘连物脱落**：粘连物轨迹消失，根据消失位置判断落入熔池还是被结晶器捕获

### 2. 锭冠相关事件

- **锭冠脱落**：锭冠从结晶器边缘运动至熔池

### 3. 电弧异常事件

- **辉光、边弧、爬弧**：持续事件，记录起始和结束帧

## 开发说明

### 动态参数计算

当前版本的动态参数计算使用假数据模拟。如需实现真实算法，请参考 `analyzer/metrics_calculator.py` 中的 `RealMetricsCalculator` 类。

实现思路：

1. **闪烁频率**：使用FFT分析亮度时域信号
2. **熔池面积**：图像分割 + 像素计数
3. **熔池周长**：边缘检测 + 周长计算

### 模型训练

如需训练自定义YOLO模型：

```bash
# 使用Ultralytics训练
yolo train data=var_dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```

## 故障排查

### GPU不可用

检查CUDA安装：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 内存不足

降低batch size或使用更小的模型（如yolo11n.pt）。

### 模型加载失败

确保模型文件存在且路径正确：

```bash
ls -lh yolo11n.pt
```

## 性能优化建议

1. **使用GPU加速**：设置 `YOLO_DEVICE=cuda` 或 `YOLO_DEVICE=0`
2. **降低进度更新频率**：修改 `PROGRESS_UPDATE_INTERVAL`
3. **使用更小的模型**：如 `yolo11n.pt` 替代 `yolo11x.pt`
4. **降低视频分辨率**：预处理时缩放视频

## 许可证

Apache 2.0

## 联系方式

- 开发者：侯阳洋
- 项目：VAR熔池视频分析系统
