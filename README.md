# VAR Molten Pool Analysis System - AI Processing Module

[简体中文](README.zh.md) | English

> Video analysis service based on Flask + PyTorch + Ultralytics YOLO

## Features

- ✅ YOLOv11 object detection and BoT-SORT multi-object tracking
- ✅ Automatic detection of anomalous events (adhesion, ingot crown, glow, side arc, creeping arc, etc.)
- ✅ Dynamic parameter calculation (pool flicker frequency, area, perimeter)
- ✅ Real-time progress callbacks
- ✅ Timeout detection and alerts
- ✅ Health check endpoint

## System Requirements

### Python Environment

- Python 3.9+
- Conda environment (recommended)

### Hardware Requirements

- CPU: Multi-core processor
- GPU: NVIDIA GPU (recommended, requires CUDA support) or Apple Silicon (supports MPS)
- Memory: 8GB+ (16GB+ recommended)

## Installation

### 1. Create and Activate Conda Environment

```bash
# Use the pre-configured pytorch environment
conda activate pytorch
```

### 2. Install Dependencies

```bash
cd ai-processor
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy the environment variable example file
cp .env.example .env

# Edit the .env file and modify the configuration
vim .env
```

### 4. Download YOLO Model

```bash
# The model will be automatically downloaded on first run
# Or manually download and place it in the specified path
# Download URL: https://github.com/ultralytics/assets/releases
```

## Usage

### Start the Service

```bash
conda activate pytorch
python app.py
```

The service will start at `http://localhost:5000`.

### Health Check

```bash
curl http://localhost:5000/health
```

### API Endpoints

#### 1. Health Check

```text
GET /health
```

Response example:

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

#### 2. Video Analysis

```text
POST /api/analyze
```

Request body:

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

Response example:

```json
{
  "status": "accepted",
  "taskId": 123,
  "message": "Task accepted, processing started"
}
```

## Project Structure

```text
ai-processor/
├── analyzer/              # Core analysis module
│   ├── __init__.py
│   ├── video_processor.py    # Main video processing logic
│   ├── yolo_tracker.py       # YOLO detection and tracking
│   ├── event_detector.py     # Event detection
│   └── metrics_calculator.py # Dynamic parameter calculation
├── utils/                 # Utility modules
│   ├── __init__.py
│   └── callback.py           # Backend callback utilities
├── app.py                 # Flask main application
├── config.py              # Configuration file
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable example
└── README.md              # This document
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| AI_PROCESSOR_HOST | Service listening address | 0.0.0.0 |
| AI_PROCESSOR_PORT | Service port | 5000 |
| AI_CALLBACK_URL | Backend callback URL | http://localhost:8080/api/tasks |
| YOLO_MODEL_PATH | YOLO model path | yolo11n.pt |
| YOLO_DEVICE | Computing device | (auto-select) |
| DEFAULT_CONFIDENCE_THRESHOLD | Default confidence threshold | 0.5 |
| DEFAULT_IOU_THRESHOLD | Default IoU threshold | 0.45 |

### Class Definitions

The system detects the following 6 classes of objects/phenomena:

| ID | Class Name | Description |
|----|-----------|-------------|
| 0 | Pool Not to Edge | Pool edge has not reached crystallizer edge |
| 1 | Adhesion | Black irregular adhesion on electrode surface |
| 2 | Ingot Crown | Ingot crown at crystallizer edge |
| 3 | Glow | Abnormal gas discharge in electrode ring gap area |
| 4 | Side Arc | Arc continuously appears at electrode edge |
| 5 | Creeping Arc | Arc traces on electrode surface |

## Event Inference Logic

### 1. Adhesion-Related Events

- **Electrode Adhesion Formation**: First detection of adhesion
- **Electrode Adhesion Detachment**: Adhesion trajectory disappears, judged to fall into pool or be captured by crystallizer based on disappearance position

### 2. Ingot Crown-Related Events

- **Ingot Crown Detachment**: Ingot crown moves from crystallizer edge to pool

### 3. Arc Anomaly Events

- **Glow, Side Arc, Creeping Arc**: Continuous events, recording start and end frames

## Development Notes

### Dynamic Parameter Calculation

The current version uses simulated data for dynamic parameter calculation. To implement real algorithms, refer to the `RealMetricsCalculator` class in `analyzer/metrics_calculator.py`.

Implementation approach:

1. **Flicker Frequency**: Use FFT to analyze brightness time-domain signals
2. **Pool Area**: Image segmentation + pixel counting
3. **Pool Perimeter**: Edge detection + perimeter calculation

### Model Training

To train a custom YOLO model:

```bash
# Train using Ultralytics
yolo train data=var_dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```

## Troubleshooting

### GPU Not Available

Check CUDA installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

Reduce batch size or use a smaller model (e.g., yolo11n.pt).

### Model Loading Failed

Ensure the model file exists and the path is correct:

```bash
ls -lh yolo11n.pt
```

## Performance Optimization Tips

1. **Use GPU Acceleration**: Set `YOLO_DEVICE=cuda` or `YOLO_DEVICE=0`
2. **Reduce Progress Update Frequency**: Modify `PROGRESS_UPDATE_INTERVAL`
3. **Use Smaller Models**: e.g., `yolo11n.pt` instead of `yolo11x.pt`
4. **Reduce Video Resolution**: Scale videos during preprocessing

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

**Important:** Any modified version of this software used over a network must make the source code available to users.
