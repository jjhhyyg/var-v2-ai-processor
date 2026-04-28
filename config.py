"""
AI处理模块配置文件

环境变量配置说明：
所有配置项都可以通过环境变量覆盖，环境变量在仓库根目录 .env 文件中定义。

核心配置：
- AI_LOG_LEVEL: 日志级别（默认：INFO）支持：DEBUG, INFO, WARNING, ERROR, CRITICAL
- AI_DEBUG: 是否启用算法调试输出（默认：False）

ONNX 模型配置：
- YOLO_MODEL_PATH: 模型文件路径（默认：weights/best.onnx）
- DEFAULT_CONFIDENCE_THRESHOLD: 默认置信度阈值（默认：0.5）
- DEFAULT_IOU_THRESHOLD: 默认IOU阈值（默认：0.45）

存储路径配置：
- STORAGE_BASE_PATH: 基础存储目录（默认：storage，相对于codes/目录）
- STORAGE_PREPROCESSED_VIDEOS_SUBDIR: 预处理输出子目录（默认：output）
- STORAGE_DETECTION_RESULTS_SUBDIR: 检测结果输出子目录（默认：output）
- DETECTION_RESULTS_FILENAME_TEMPLATE: 检测结果文件名（默认：detections.json）
"""
import logging
import os
from dotenv import load_dotenv

# 加载环境变量（从项目根目录的 .env 文件）
# codes/ai-processor/config.py -> codes/.env
_root_dir = os.path.join(os.path.dirname(__file__), '..')
load_dotenv(os.path.join(_root_dir, '.env'))

logger = logging.getLogger(__name__)


class Config:
    """
    AI处理模块配置类

    所有配置项都可以通过环境变量覆盖。
    推荐使用提供的类方法来获取路径，而不是直接访问路径常量。
    """

    # ========================================
    # 日志配置
    # ========================================
    # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL = os.getenv('AI_LOG_LEVEL', 'INFO').upper()
    # 算法调试开关：用于控制异常事件生成等模块的调试输出
    DEBUG = os.getenv('AI_DEBUG', os.getenv('DEBUG', 'False')).lower() in ('1', 'true', 'yes', 'on')

    # ========================================
    # YOLO模型配置
    # ========================================
    MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'weights/best.onnx')  # 模型文件路径

    # ========================================
    # 默认检测参数
    # ========================================
    DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv('DEFAULT_CONFIDENCE_THRESHOLD', '0.5'))  # 置信度阈值
    DEFAULT_IOU_THRESHOLD = float(os.getenv('DEFAULT_IOU_THRESHOLD', '0.45'))  # IoU阈值（NMS）

    # ========================================
    # 类别和事件定义
    # ========================================
    # 类别定义（严格对应YOLO模型权重中的定义）
    # 必须与YOLO模型训练时的类别名称完全一致
    CLASS_NAMES = {
        0: '熔池未到边',
        1: '电极粘连物',  # YOLO模型权重中的实际名称
        2: '锭冠',
        3: '辉光',
        4: '边弧（侧弧）',
        5: '爬弧'
    }

    # 事件类型映射（对应桌面端持久化字段）
    # Phase 0 中事件直接由 YOLO 检测类别产生，不再基于追踪生命周期推导。
    EVENT_TYPE_MAPPING = {
        '熔池未到边': 'POOL_NOT_REACHED',
        '电极粘连物': 'ADHESION',
        '锭冠': 'CROWN',
        '辉光': 'GLOW',
        '边弧（侧弧）': 'SIDE_ARC',
        '爬弧': 'CREEPING_ARC',

        # 兼容性映射（支持可能的别名）
        '粘连物': 'ADHESION',
        '边弧': 'SIDE_ARC',
        '侧弧': 'SIDE_ARC'
    }

    # 物体类别映射（用于结果事件的 category 字段）
    # 主映射严格对应YOLO模型权重中的类别名称，同时提供兼容性别名映射
    CATEGORY_MAPPING = {
        # 主映射（对应YOLO模型权重中的实际类别名称）
        '熔池未到边': 'POOL_NOT_REACHED',
        '电极粘连物': 'ADHESION',
        '锭冠': 'CROWN',
        '辉光': 'GLOW',
        '边弧（侧弧）': 'SIDE_ARC',
        '爬弧': 'CREEPING_ARC',
        
        # 兼容性映射（支持可能的别名）
        '粘连物': 'ADHESION',              # 简化别名
        '边弧': 'SIDE_ARC',                # 简化别名
        '侧弧': 'SIDE_ARC'                 # 别名
    }

    # ========================================
    # 分析任务配置
    # ========================================
    # 进度更新频率（每处理多少帧更新一次回调；桌面 stdout 链路另有时间节流）
    PROGRESS_UPDATE_INTERVAL = int(os.getenv('PROGRESS_UPDATE_INTERVAL', '1'))

    # ========================================
    # 存储路径配置（相对于 codes/ 目录）
    # ========================================

    # 基础存储目录
    STORAGE_BASE_PATH = os.getenv('STORAGE_BASE_PATH', 'storage')

    # 子目录配置（推荐使用 get_storage_path(subdir) 方法获取完整路径）
    STORAGE_PREPROCESSED_VIDEOS_SUBDIR = os.getenv('STORAGE_PREPROCESSED_VIDEOS_SUBDIR', 'output')
    STORAGE_DETECTION_RESULTS_SUBDIR = os.getenv('STORAGE_DETECTION_RESULTS_SUBDIR', 'output')
    DETECTION_RESULTS_FILENAME_TEMPLATE = os.getenv('DETECTION_RESULTS_FILENAME_TEMPLATE', 'detections.json')

    # ========================================
    # 外部二进制配置
    # ========================================
    FFMPEG_BIN = os.getenv('FFMPEG_BIN', 'ffmpeg')
    FFPROBE_BIN = os.getenv('FFPROBE_BIN', 'ffprobe')
    GPU_PREPROCESSOR_BIN = os.getenv('GPU_PREPROCESSOR_BIN', '')
    VAR_VIDEO_ANALYZER_BIN = os.getenv('VAR_VIDEO_ANALYZER_BIN', '')
    USE_CPP_VIDEO_ANALYZER = os.getenv('USE_CPP_VIDEO_ANALYZER', '1').lower() not in ('0', 'false', 'no', 'off')

    # ========================================
    # 内部路径常量（不建议直接使用）
    # ========================================
    # codes/ 目录的绝对路径 (ai-processor的父目录)
    # 推荐使用 resolve_path() 和 get_storage_path() 方法来处理路径
    CODES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # ========================================
    # 工具方法（推荐使用）
    # ========================================

    @classmethod
    def get_storage_path(cls, subdir: str = '') -> str:
        """
        获取存储路径（绝对路径）- 推荐使用此方法代替直接使用路径常量

        Args:
            subdir: 子目录名称，支持：
                - 'output': 预处理视频和检测结果
                - 或任意自定义子目录名称

        Returns:
            绝对路径（规范化后）

        Examples:
            >>> Config.get_storage_path('output')
            '/path/to/codes/storage/output'

            >>> Config.get_storage_path()
            '/path/to/codes/storage'
        """
        if subdir:
            path = os.path.join(cls.STORAGE_BASE_PATH, subdir)
        else:
            path = cls.STORAGE_BASE_PATH
        return cls.resolve_path(path)
    
    @classmethod
    def resolve_path(cls, relative_path: str) -> str:
        """
        将相对于codes/目录的路径转换为绝对路径
        
        Args:
            relative_path: 相对于codes/目录的路径，例如 'storage/videos/xxx.mp4'
            
        Returns:
            绝对路径（规范化后）
        """
        if os.path.isabs(relative_path):
            # 如果已经是绝对路径，规范化后返回
            return os.path.normpath(relative_path)
        # 相对路径：拼接后规范化（消除 .. 等）
        return os.path.normpath(os.path.join(cls.CODES_DIR, relative_path))
    
    @classmethod
    def to_relative_path(cls, absolute_path: str) -> str:
        """
        将绝对路径转换为相对于codes/目录的路径

        Args:
            absolute_path: 绝对路径

        Returns:
            相对于codes/目录的路径
        """
        if not os.path.isabs(absolute_path):
            return absolute_path
        return os.path.relpath(absolute_path, cls.CODES_DIR)

    @classmethod
    def get_detection_results_path(cls, task_id: int) -> str:
        filename = cls.DETECTION_RESULTS_FILENAME_TEMPLATE.format(task_id=task_id)
        return os.path.join(cls.get_storage_path(cls.STORAGE_DETECTION_RESULTS_SUBDIR), filename)
