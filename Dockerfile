# AI处理模块Dockerfile
FROM python:3.12-slim

LABEL maintainer="侯阳洋"
LABEL description="VAR熔池视频分析系统 - AI处理模块"

WORKDIR /app

# 安装系统依赖
# libglib2.0-0, libsm6, libxext6, libxrender-dev, libgomp1: OpenCV依赖
# libgl1: OpenGL库,OpenCV的cv2模块需要
# curl: 健康检查
# 注意:不需要安装 ffmpeg,OpenCV 自带视频编解码功能
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /var/var-analysis/storage/videos \
    /var/var-analysis/storage/result_videos \
    /var/var-analysis/storage/preprocessed_videos \
    /var/var-analysis/storage/temp \
    logs

# 设置文件权限
RUN chown -R appuser:appgroup /app /var/var-analysis

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 启动应用（同时启动Flask API和RabbitMQ消费者）
CMD ["python", "mq_consumer.py"]
