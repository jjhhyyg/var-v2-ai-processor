# VAR 熔池分析桌面 worker

简体中文 | [English](README.md)

> 当前 AI 模块服务于桌面端本地分析链路，桌面入口是 `desktop_worker.py`。

## 模块职责

- 接收 Tauri/Rust 写入的 job JSON
- 调用 `var-gpu-preprocessor` 生成预处理视频
- 调用 `var-video-analyzer` 使用 `best.onnx` 和 ONNX Runtime CUDA EP 输出逐帧 detections
- 在 Python 侧生成异常事件、动态指标和结果汇总
- 通过 stdout NDJSON 事件向桌面端上报状态、进度、模型版本和结果文件路径

## 当前入口

```bash
python desktop_worker.py /path/to/job.json
python desktop_worker.py --self-check
```

桌面端实际由 `frontend/src-tauri/src/lib.rs` 创建 job 文件并启动 worker。开发者通常不需要手写 job，除非在定位 worker 本身的问题。

## 依赖

```bash
pip install -r requirements-desktop-windows-cuda.txt
```

Windows worker 打包由 `frontend/scripts/build-desktop-worker.mjs` 自动创建并维护 `frontend/.desktop-worker-venv`，该环境只安装 worker 所需的轻量依赖。ONNX 导出仍使用 `TAURI_ONNX_EXPORT_PYTHON` / `TAURI_WORKER_PYTHON` / conda `var-env` 中的 `torch + ultralytics`。

## 模型权重

模型权重不提交到 Git。源模型为 `weights/best.pt`，Windows runtime 构建第一步会执行 `npm run desktop:export-onnx` 导出 `weights/best.onnx`。运行时只使用 `best.onnx`，不再使用 Python PT/ONNX fallback。

## 关键文件

- `desktop_worker.py`：桌面 worker 入口
- `analyzer/video_processor.py`：GPU 预处理 sidecar、C++ ONNX analyzer、事件生成和结果汇总主流程
- `utils/callback.py`：桌面 worker 的 stdout 事件输出
- `requirements-desktop-windows-cuda.txt`：Windows worker 最小依赖

## 测试要求

- `python3 -m py_compile desktop_worker.py utils/callback.py analyzer/video_processor.py`
- `python desktop_worker.py --self-check`
- `desktop_worker.py --self-check --backend cpp-onnx --model weights/best.onnx`
- 从 Windows 桌面端导入 runtime zip 后，导入真实视频并完成一次分析

## 说明

`BackendCallback` 通过 `stdout://` 输出桌面 NDJSON 事件，运行时不依赖旧后端服务。默认不再生成 `result.mp4`；桌面端播放原始/预处理视频，并读取 `output/detections.json` 做前端 bbox overlay。
