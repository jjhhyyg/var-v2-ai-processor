# VAR 熔池视频分析系统 - AI 处理模块

简体中文 | [English](README.md)

> 负责视频预处理、YOLO 检测、追踪和结果导出的 Flask AI 服务。

## 模块职责

AI 模块负责：

- 从 RabbitMQ 消费分析任务
- 可选地执行视频预处理
- 执行 YOLO11 检测与 BoT-SORT 追踪
- 生成异常事件和动态指标
- 导出带标注的结果视频
- 向后端回调进度、结果、模型版本和生成文件路径

## 当前运行方式

本模块本地通常采用两层能力同时运行：

1. 通过 `app.py` 启动 Flask 服务和健康检查接口
2. 在同一运行时内启动 RabbitMQ 消费线程，异步处理分析任务

所以真正的本地联调入口应以 `app.py` 为准。

## 当前真实配置口径

配置以 `config.py` 为准。

必须记住这几个事实：

- `config.py` 会加载主仓库根目录 `.env`，不是只看 `ai-processor/.env`
- 回调后端使用的是 `BACKEND_BASE_URL`
- 默认模型路径是 `weights/best.pt`
- 默认队列名是 `video_analysis_queue`

核心变量包括：

- `AI_PROCESSOR_HOST`
- `AI_PROCESSOR_PORT`
- `AI_PROCESSOR_DEBUG`
- `BACKEND_BASE_URL`
- `YOLO_MODEL_PATH`
- `YOLO_DEVICE`
- `DEFAULT_CONFIDENCE_THRESHOLD`
- `DEFAULT_IOU_THRESHOLD`
- `TRACKER_CONFIG`
- `PROGRESS_UPDATE_INTERVAL`
- `RABBITMQ_*`
- `STORAGE_*`

环境文件建议从主仓库根目录生成：

```bash
./scripts/use-env.sh dev
```

## 本地开发

安装依赖：

```bash
cd ai-processor
pip install -r requirements.txt
```

如果你明确只想装 CPU 依赖，可使用：

```bash
pip install -r requirements-cpu.txt
```

启动服务：

```bash
python app.py
```

默认本地地址：

- `http://localhost:5000`

健康检查：

```bash
curl http://localhost:5000/health
```

## 模型与设备

本地测试或部署前，必须确认：

- `ai-processor/weights/best.pt` 存在
- `YOLO_DEVICE` 与实际环境匹配，或者明确留空走自动选择

常见设备选项：

- `cuda`：NVIDIA GPU
- `mps`：Apple Silicon
- `cpu`：只使用 CPU
- 空字符串：自动选择

## 关键文件

- `app.py`：Flask 入口与健康检查
- `mq_consumer.py`：RabbitMQ 消费与任务分发
- `config.py`：配置来源
- `analyzer/video_processor.py`：预处理、分析、导出主流程
- `preprocessor/video_preprocessor.py`：基于 CPU 的视频预处理

## 本地测试要求

部署前至少完成这些检查：

- `GET /health` 返回 healthy
- 模型成功加载
- 设备识别正确
- RabbitMQ 连接成功
- 至少一条真实分析任务能够被消费并完成

需要说清楚：当前模块缺少完善的自动化测试体系，因此手工联调测试是硬性要求，不是可选项。

## Docker 说明

- GPU 生产部署走主仓库 `docker-compose.prod.yml`
- CPU 生产部署走 `docker-compose.prod.cpu.yml`
- CPU 生产镜像使用 `Dockerfile.cpu`
- 生产环境仍然依赖 `weights/best.pt`

## 下一步阅读

- 主仓库地址：
  `https://github.com/jjhhyyg/VAR-melting-defect-detection-source-code.git`
- 主仓库中的交接文档：
  `docs/项目接手、开发测试与部署指南.md`
