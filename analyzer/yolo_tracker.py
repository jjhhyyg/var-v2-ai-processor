"""
YOLO目标检测和BotSort追踪模块
使用Ultralytics YOLO进行目标检测和多目标追踪
"""
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from config import Config

logger = logging.getLogger(__name__)


class YOLOTracker:
    """YOLO目标检测和追踪器"""

    def __init__(self, model_path: str, device: str = ''):
        """
        初始化YOLO追踪器

        Args:
            model_path: YOLO模型路径
            device: 设备类型（'cpu', 'cuda', 'mps', '0', '1'等），空字符串表示自动选择
                   优先级：CUDA > MPS > CPU
        """
        self.model_path = model_path
        self.device = Config.auto_select_device(device)
        self.class_names = Config.CLASS_NAMES

        logger.info(f"Loading YOLO model from {model_path}")

        try:
            # 加载YOLO模型
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info("YOLO model loaded successfully")
            
            # 动态获取模型版本
            self.model_version = self._get_model_version(model_path)
            logger.info(f"Model version: {self.model_version}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _get_model_version(self, model_path: str) -> str:
        """
        从模型检查点文件中获取模型版本
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            模型版本字符串 (去掉.pt后缀)
        """
        try:
            # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含自定义类的检查点
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'train_args' in ckpt and 'model' in ckpt['train_args']:
                model_version = ckpt['train_args']['model']
                # 去掉.pt后缀
                if model_version.endswith('.pt'):
                    model_version = model_version[:-3]
                return model_version
            else:
                logger.warning(f"Model version not found in checkpoint, using default 'yolo11n'")
                return 'yolo11n'
        except Exception as e:
            logger.warning(f"Failed to read model version from checkpoint: {e}, using default 'yolo11n'")
            return 'yolo11n'

    def track_frame(self, frame: np.ndarray, conf: float = 0.4,
                   iou: float = 0.4, persist: bool = True) -> List[Dict[str, Any]]:
        """
        对单帧进行目标检测和追踪

        Args:
            frame: 输入视频帧
            conf: 置信度阈值
            iou: IoU阈值
            persist: 是否保持追踪（跨帧持续）

        Returns:
            检测结果列表，每个包含：
                - track_id: 追踪ID
                - class_id: 类别ID
                - class_name: 类别名称
                - bbox: 边界框 [x1, y1, x2, y2] (xyxy格式)
                - center_x: 中心点x坐标
                - center_y: 中心点y坐标
                - width: 边界框宽度
                - height: 边界框高度
                - confidence: 置信度
        """
        try:
            # 运行YOLO追踪
            results = self.model.track(
                source=frame,
                conf=conf,
                iou=iou,
                persist=persist,
                tracker=Config.TRACKER_CONFIG,
                verbose=Config.VERBOSE
            )

            # 解析结果
            detections = []

            if results and len(results) > 0:
                result = results[0]

                # 检查是否有检测结果
                if result.boxes is not None and len(result.boxes) > 0:
                    # 安全地转换tensor到numpy，参照demo代码
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else np.array(result.boxes.xyxy)
                    confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'cpu') else np.array(result.boxes.conf)
                    class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'cpu') else np.array(result.boxes.cls)
                    
                    # 处理追踪ID：可能为None（没有追踪）或有值
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy() if hasattr(result.boxes.id, 'cpu') else np.array(result.boxes.id)
                    else:
                        # 没有追踪ID时，使用-1标记（表示只检测到但未追踪）
                        track_ids = np.full(len(boxes_xyxy), -1, dtype=np.int32)
                        logger.warning(f"No track IDs available for {len(boxes_xyxy)} detections, using -1 as placeholder")

                    # 获取类别名称
                    class_names = [result.names.get(int(cls_id), f'Unknown_{cls_id}') for cls_id in class_ids]

                    # 遍历每个检测框
                    for i in range(len(boxes_xyxy)):
                        x1, y1, x2, y2 = boxes_xyxy[i]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1

                        detection = {
                            'track_id': int(track_ids[i]),
                            'class_id': int(class_ids[i]),
                            'class_name': class_names[i],
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'center_x': float(center_x),
                            'center_y': float(center_y),
                            'width': float(width),
                            'height': float(height),
                            'confidence': float(confidences[i])
                        }

                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during tracking: {e}", exc_info=True)
            return []

    def reset_tracking(self):
        """
        重置追踪状态
        用于处理新视频时清除之前的追踪信息
        """
        try:
            # Ultralytics的persist参数会自动管理追踪状态
            # 如果需要手动重置，可以重新初始化追踪器
            # 这里暂时不做任何操作，因为每个新视频处理时会自动重置
            pass
        except Exception as e:
            logger.error(f"Error resetting tracking: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            'model_path': self.model_path,
            'model_version': self.model_version,
            'device': str(self.device),
            'class_names': self.class_names,
            'num_classes': len(self.class_names)
        }
