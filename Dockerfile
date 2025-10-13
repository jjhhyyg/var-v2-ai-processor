# ========================================
# 多阶段构建 - AI处理模块Dockerfile
# 阶段1: 代码混淆 (已优化缓存)
# 阶段2: 运行环境 (已优化缓存)
# ========================================

# ========================================
# 阶段1: 代码混淆(使用 PyArmor)
# ========================================
FROM python:3.12-slim AS obfuscator
LABEL stage="obfuscation"
WORKDIR /build

# 1. 先只复制 requirements.txt
COPY requirements.txt .

# 2. 再安装依赖 (PyArmor)
# 这样只有在 requirements.txt 变化时，这一层才会重新执行
RUN pip install --no-cache-dir pyarmor==8.5.9

# 3. 最后复制源代码和配置文件
# 这样修改代码不会导致上面的 pip install 缓存失效
COPY *.py ./
COPY analyzer/ ./analyzer/
COPY preprocessor/ ./preprocessor/
COPY utils/ ./utils/
COPY botsort.yaml ./

# 使用 PyArmor 混淆代码(试用版基础混淆)
RUN pyarmor gen \
    --output /obfuscated \
    --recursive \
    --exclude requirements.txt \
    --exclude botsort.yaml \
    .

# ========================================
# 阶段2: 运行环境(生产镜像)
# ========================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

LABEL maintainer="侯阳洋"
LABEL description="VAR熔池视频分析系统 - AI处理模块(源码已混淆保护)"
LABEL security="code-obfuscated"

WORKDIR /app/ai-processor

# 安装系统依赖 (此部分不常变动，放在前面)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    ffmpeg \
    fonts-noto-cjk \
    fonts-wqy-zenhei \
    fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 创建非root用户
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# --- 缓存优化核心部分 ---
# 1. 只复制依赖文件
# 这一层只有在 requirements.txt 内容变化时才会缓存失效
COPY requirements.txt .

# 2. 安装所有Python依赖
# --mount 使用BuildKit的缓存，即使上一层失效，也能从持久化缓存中快速安装
# 注意：请确保你的 requirements.txt 顶部包含了 PyTorch 的 --index-url
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r requirements.txt

# 3. 最后再复制你的代码和配置文件
# 这样，日常开发中只修改代码时，上面耗时的 pip install 会直接使用缓存
COPY --from=obfuscator /obfuscated/ .
COPY botsort.yaml .
# --- 缓存优化结束 ---

# 创建必要的目录并设置权限
RUN mkdir -p \
    ../storage/videos \
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

# 启动应用
CMD ["python3", "app.py"]
