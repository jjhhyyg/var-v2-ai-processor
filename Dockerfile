# ========================================
# 多阶段构建 - AI处理模块Dockerfile
# 阶段1: 代码混淆
# 阶段2: 运行环境
# ========================================

# ========================================
# 阶段1: 代码混淆（使用 PyArmor）
# ========================================
FROM python:3.12-slim AS obfuscator

LABEL stage="obfuscation"

WORKDIR /build

# 安装 PyArmor
RUN pip install --no-cache-dir pyarmor==8.5.9

# 复制源代码（只复制需要的文件）
COPY requirements.txt .
COPY *.py ./
COPY analyzer/ ./analyzer/
COPY preprocessor/ ./preprocessor/
COPY utils/ ./utils/
COPY botsort.yaml ./

# 使用 PyArmor 混淆代码（试用版基础混淆）
# 注意：试用版不支持高级选项如 --restrict, --enable-jit, --obf-code 2
# 基础混淆仍然提供字节码加密和运行时保护
# 如需更强保护，请购买 PyArmor 许可证后使用高级选项
RUN pyarmor gen \
    --output /obfuscated \
    --recursive \
    .

# ========================================
# 阶段2: 运行环境（生产镜像）
# ========================================
FROM python:3.12-slim

LABEL maintainer="侯阳洋"
LABEL description="VAR熔池视频分析系统 - AI处理模块（源码已混淆保护）"
LABEL security="code-obfuscated"

WORKDIR /app/ai-processor

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

# 从混淆阶段复制混淆后的代码（替代 COPY . .）
COPY --from=obfuscator /obfuscated/ .

# 复制配置文件（不需要混淆）
COPY botsort.yaml .

# 创建必要的目录并设置权限
RUN mkdir -p ../storage/videos \
    ../storage/result_videos \
    ../storage/preprocessed_videos \
    ../storage/temp \
    ../storage/tracking_results \
    logs && \
    chown -R appuser:appgroup /app

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
