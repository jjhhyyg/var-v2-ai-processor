#!/bin/bash
# VAR熔池视频分析系统 - AI处理模块启动脚本

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VAR熔池视频分析系统 - AI处理模块${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: conda未安装或未添加到PATH${NC}"
    exit 1
fi

# 检查pytorch环境
if ! conda env list | grep -q "^pytorch "; then
    echo -e "${RED}错误: pytorch conda环境不存在${NC}"
    echo -e "${YELLOW}请先创建pytorch环境：conda create -n pytorch python=3.9${NC}"
    exit 1
fi

# 检查.env文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}警告: .env文件不存在，正在从.env.example创建...${NC}"
    cp .env.example .env
    echo -e "${GREEN}.env文件已创建，请根据需要修改配置${NC}"
fi

# 检查模型文件
MODEL_PATH=$(grep "YOLO_MODEL_PATH=" .env | cut -d '=' -f2)
MODEL_PATH=${MODEL_PATH:-weights/best.pt}

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}警告: 模型文件 ${MODEL_PATH} 不存在${NC}"
    echo -e "${YELLOW}首次运行时会自动下载预训练模型${NC}"
    echo -e "${YELLOW}如果要使用自定义模型，请将模型文件放置在：${MODEL_PATH}${NC}"
fi

# 检查bytetrack.yaml配置文件
if [ ! -f "bytetrack.yaml" ]; then
    echo -e "${YELLOW}警告: bytetrack.yaml配置文件不存在${NC}"
    echo -e "${YELLOW}将使用Ultralytics默认配置${NC}"
fi

# 激活conda环境并启动服务
echo -e "${GREEN}正在激活pytorch环境...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

echo -e "${GREEN}检查依赖包...${NC}"
python -c "import flask, torch, ultralytics, cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}部分依赖包缺失，正在安装...${NC}"
    pip install -r requirements.txt
fi

echo -e "${GREEN}启动AI处理模块...${NC}"
python app.py

# 捕获退出信号
trap 'echo -e "${YELLOW}正在停止AI处理模块...${NC}"; exit 0' SIGINT SIGTERM
