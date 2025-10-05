# YOLO模型文件目录

## 模型文件放置

将训练好的YOLO模型文件放置在此目录下。

### 推荐命名

```
weights/
├── best.pt          # 最佳模型（推荐）
├── last.pt          # 最后一个epoch的模型
└── yolo11n.pt       # 预训练模型（可选）
```

### 配置模型路径

在 `ai-processor/.env` 中配置：

```bash
# 使用best.pt
YOLO_MODEL_PATH=weights/best.pt

# 或使用绝对路径
YOLO_MODEL_PATH=/path/to/your/model.pt
```

## 模型训练

如需训练自定义模型，参考Ultralytics文档：

```bash
# 使用YOLOv11训练
yolo train data=var_dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```

训练完成后，将 `runs/detect/train/weights/best.pt` 复制到此目录。

## 模型要求

- 格式：PyTorch (.pt)
- 框架：Ultralytics YOLO
- 类别：6类（熔池未到边、粘连物、锭冠、辉光、边弧、爬弧）
- 输入尺寸：推荐640x640或更大

## 预训练模型

如果没有训练好的模型，首次运行时会自动下载YOLOv11预训练模型：

- `yolo11n.pt` - Nano（最快）
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - XLarge（最准确）

## 注意事项

⚠️ **不要将模型文件提交到Git仓库**

模型文件通常很大（几MB到几百MB），已在 `.gitignore` 中排除。

如需分享模型，请使用以下方式：
- 云存储（Google Drive、百度网盘等）
- 模型版本管理平台（HuggingFace、ModelScope等）
- 内部文件服务器
