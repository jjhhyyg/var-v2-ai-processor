# 模型文件目录

模型文件不提交到 Git。本目录用于放置本地训练权重和导出的 Windows runtime 模型。

## 当前文件约定

```text
weights/
├── best.pt      # 源模型，来自 Ultralytics 训练结果
└── best.onnx    # Windows runtime 使用的 ONNX 模型，由 npm run desktop:export-onnx 生成
```

## 导出 ONNX

从 `frontend/` 运行：

```powershell
npm run desktop:export-onnx
```

导出要求：

- `weights/best.pt` 必须存在。
- 导出环境需要 `torch + ultralytics`。
- 必须有可用 NVIDIA GPU，导出脚本会检查 `torch.cuda.is_available()`。

导出参数由 `frontend/scripts/export-onnx-model.mjs` 固定，当前 ONNX 输入为 `[1,3,640,1024]`，输出为 `[1,10,13440]`。

## 运行时使用

Windows 正式分析链路只使用 `best.onnx`：

- `var-video-analyzer.exe --self-check-onnx --model best.onnx`
- `var-video-analyzer.exe --input ... --model best.onnx --output-detections ...`

Python PT/ONNX fallback 已删除，运行时不再自动下载或加载预训练 YOLO 模型。

## 训练

如需训练自定义模型，参考 Ultralytics 文档：

```bash
yolo train data=var_dataset.yaml model=yolo11n.pt epochs=100 imgsz=1024
```

训练完成后，将 `runs/detect/train/weights/best.pt` 复制到本目录，再运行 `npm run desktop:export-onnx`。

## 注意事项

不要将 `best.pt` 或 `best.onnx` 提交到 Git。模型文件通常很大，已在 `.gitignore` 中排除。
