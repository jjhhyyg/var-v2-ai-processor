# VAR 熔池分析桌面 worker

简体中文 | [English](README.md)

> 当前 AI 模块服务于桌面端本地分析链路，不再提供旧的 Flask/RabbitMQ 服务入口，也不再提供 Docker 镜像入口。

## 模块职责

- 接收 Tauri/Rust 写入的 job JSON
- 执行视频预处理、YOLO 检测、BoT-SORT 追踪和结果视频导出
- 通过 stdout NDJSON 事件向桌面端上报状态、进度、模型版本和结果文件路径
- 为 macOS 桌面端打包提供最小依赖清单和 worker 入口

## 当前入口

```bash
python desktop_worker.py /path/to/job.json
python desktop_worker.py --self-check
```

桌面端实际由 `frontend/src-tauri/src/lib.rs` 创建 job 文件并启动 worker。开发者通常不需要手写 job，除非在定位 worker 本身的问题。

## 依赖

```bash
pip install -r requirements-desktop-macos.txt
pip install -r requirements.txt
```

## 模型权重

模型权重不提交到 Git。默认构建脚本会读取 `weights/best.pt`，请在本地放置该文件，或通过 `YOLO_MODEL_PATH` 指向其他本地模型路径。

## 关键文件

- `desktop_worker.py`：桌面 worker 入口
- `analyzer/video_processor.py`：预处理、检测、追踪和结果导出主流程
- `preprocessor/video_preprocessor.py`：视频预处理
- `utils/callback.py`：stdout/HTTP 回调抽象，桌面端使用 `stdout://`
- `requirements-desktop-macos.txt`：macOS worker 最小依赖

## 测试要求

- `python3 -m py_compile desktop_worker.py utils/callback.py analyzer/video_processor.py`
- `python desktop_worker.py --self-check`
- 从 macOS 桌面端导入真实视频并完成一次分析

## 说明

`BackendCallback` 仍保留 HTTP 回调分支，目的是降低算法主流程改造风险；当前桌面端路径使用 `stdout://`，不会依赖旧后端服务。
