"""
AI处理模块配置文件

环境变量配置说明：
所有配置项都可以通过环境变量覆盖，环境变量在 codes/.env 文件中定义。

核心配置：
- AI_LOG_LEVEL: 日志级别（默认：INFO）支持：DEBUG, INFO, WARNING, ERROR, CRITICAL
- AI_DEBUG: 是否启用算法调试输出（默认：False）

YOLO模型配置：
- YOLO_MODEL_PATH: 模型文件路径（默认：weights/best.pt）
- YOLO_DEVICE: 计算设备（默认：''，自动选择，优先级：CUDA > MPS > CPU）
- DEFAULT_CONFIDENCE_THRESHOLD: 默认置信度阈值（默认：0.5）
- DEFAULT_IOU_THRESHOLD: 默认IOU阈值（默认：0.45）

存储路径配置：
- STORAGE_BASE_PATH: 基础存储目录（默认：storage，相对于codes/目录）
- STORAGE_*_SUBDIR: 各子目录名称（推荐使用 get_storage_path() 方法获取完整路径）

HTTP 回调兼容配置：
- BACKEND_BASE_URL: 兼容 HTTP 回调路径的基础 URL（默认：http://localhost:8080）；桌面端 stdout:// 路径不会依赖它
"""
import logging
import os
import torch
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
    推荐使用提供的类方法来获取路径和URL，而不是直接访问路径常量。
    """

    # ========================================
    # 日志配置
    # ========================================
    # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_LEVEL = os.getenv('AI_LOG_LEVEL', 'INFO').upper()
    # 算法调试开关：用于控制异常事件生成等模块的调试输出
    DEBUG = os.getenv('AI_DEBUG', os.getenv('DEBUG', 'False')).lower() in ('1', 'true', 'yes', 'on')

    # ========================================
    # HTTP 回调兼容配置
    # ========================================
    # HTTP 回调兼容路径。桌面端使用 stdout://，不会依赖后端服务。
    BACKEND_BASE_URL = os.getenv('BACKEND_BASE_URL', 'http://localhost:8080')

    # ========================================
    # YOLO模型配置
    # ========================================
    MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'weights/best.pt')  # 模型文件路径
    # MODEL_VERSION 已弃用,现在从模型检查点文件动态读取
    # MODEL_VERSION = os.getenv('YOLO_MODEL_VERSION', 'yolo11n')
    DEVICE = os.getenv('YOLO_DEVICE', '')  # 计算设备（''=自动选择，优先级：CUDA > MPS > CPU）

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
    # 注意：事件不是YOLO直接输出，而是基于追踪物体的出现/消失等行为分析得出
    # 
    # 异常事件类型：
    #   - POOL_NOT_REACHED: 熔池未到边（状态检测）
    #   - ADHESION_FORMED: 电极形成粘连物（检测到粘连物出现）
    #   - ADHESION_DROPPED: 电极粘连物脱落（检测到粘连物消失）
    #   - CROWN_DROPPED: 锭冠脱落（检测到锭冠消失）
    #
    # 电弧异常事件类型：
    #   - GLOW: 辉光
    #   - SIDE_ARC: 边弧（侧弧）
    #   - CREEPING_ARC: 爬弧
    EVENT_TYPE_MAPPING = {
        '熔池未到边': 'POOL_NOT_REACHED',
        '电极形成粘连物': 'ADHESION_FORMED',
        '电极粘连物脱落': 'ADHESION_DROPPED',
        '锭冠脱落': 'CROWN_DROPPED',
        '辉光': 'GLOW',
        '边弧（侧弧）': 'SIDE_ARC',
        '爬弧': 'CREEPING_ARC',
        
        # 兼容性映射（支持可能的别名）
        '形成粘连物': 'ADHESION_FORMED',
        '粘连物脱落': 'ADHESION_DROPPED',
        '边弧': 'SIDE_ARC',
        '侧弧': 'SIDE_ARC'
    }

    # 物体类别映射（用于结果事件和追踪对象的 category 字段）
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
    # BotSort追踪器配置
    # ========================================
    # 追踪器配置文件路径（YOLO内置配置）
    TRACKER_CONFIG = os.getenv('TRACKER_CONFIG', 'botsort.yaml')

    # BotSort详细参数（当使用自定义配置时）
    TRACKER_PARAMS = {
        'tracker_type': 'botsort',
        'track_high_thresh': float(os.getenv('TRACK_HIGH_THRESH', '0.5')),
        'track_low_thresh': float(os.getenv('TRACK_LOW_THRESH', '0.1')),
        'new_track_thresh': float(os.getenv('NEW_TRACK_THRESH', '0.6')),
        'track_buffer': int(os.getenv('TRACK_BUFFER', '30')),
        'match_thresh': float(os.getenv('MATCH_THRESH', '0.8')),
        'fuse_score': os.getenv('FUSE_SCORE', 'True').lower() == 'true',
        'gmc_method': os.getenv('GMC_METHOD', 'None'),
        # BotSort特有参数
        'with_reid': os.getenv('WITH_REID', 'False').lower() == 'true',
        'proximity_thresh': float(os.getenv('PROXIMITY_THRESH', '0.5')),
        'appearance_thresh': float(os.getenv('APPEARANCE_THRESH', '0.25')),
    }

    # ========================================
    # 分析任务配置
    # ========================================
    # 进度更新频率（每处理多少帧更新一次回调；桌面 stdout 链路另有时间节流）
    PROGRESS_UPDATE_INTERVAL = int(os.getenv('PROGRESS_UPDATE_INTERVAL', '1'))

    # 是否显示YOLO详细输出（调试时启用）
    VERBOSE = os.getenv('YOLO_VERBOSE', 'False').lower() == 'true'

    # ========================================
    # 存储路径配置（相对于 codes/ 目录）
    # ========================================

    # 基础存储目录
    STORAGE_BASE_PATH = os.getenv('STORAGE_BASE_PATH', 'storage')

    # 子目录配置（推荐使用 get_storage_path(subdir) 方法获取完整路径）
    STORAGE_VIDEOS_SUBDIR = os.getenv('STORAGE_VIDEOS_SUBDIR', 'videos')
    STORAGE_RESULT_VIDEOS_SUBDIR = os.getenv('STORAGE_RESULT_VIDEOS_SUBDIR', 'result_videos')
    STORAGE_PREPROCESSED_VIDEOS_SUBDIR = os.getenv('STORAGE_PREPROCESSED_VIDEOS_SUBDIR', 'preprocessed_videos')
    STORAGE_TRACKING_RESULTS_SUBDIR = os.getenv('STORAGE_TRACKING_RESULTS_SUBDIR', 'tracking_results')
    TRACKING_RESULTS_FILENAME_TEMPLATE = os.getenv('TRACKING_RESULTS_FILENAME_TEMPLATE', '{task_id}_tracking.json')

    # 完整路径（⚠️ 已废弃，保留用于向后兼容，请使用 get_storage_path() 方法）
    RESULT_VIDEO_PATH = os.getenv('RESULT_VIDEO_PATH', './storage/result_videos')
    PREPROCESSED_VIDEO_PATH = os.getenv('PREPROCESSED_VIDEO_PATH', './storage/preprocessed_videos')

    # ========================================
    # 外部二进制配置
    # ========================================
    FFMPEG_BIN = os.getenv('FFMPEG_BIN', 'ffmpeg')
    FFPROBE_BIN = os.getenv('FFPROBE_BIN', 'ffprobe')

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
    def get_callback_url(cls, task_id, endpoint='progress'):
        """
        获取 HTTP 回调 URL。桌面端 stdout:// 路径不会调用该方法。
        
        Args:
            task_id: 任务ID
            endpoint: 端点类型，可选 'progress' 或 'result'
            
        Returns:
            完整的回调URL
            
        Example:
            >>> Config.get_callback_url(123, 'progress')
            'http://localhost:8080/api/tasks/123/progress'
        """
        if endpoint == 'progress':
            return f"{cls.BACKEND_BASE_URL}/api/tasks/{task_id}/progress"
        elif endpoint == 'result':
            return f"{cls.BACKEND_BASE_URL}/api/tasks/{task_id}/result"
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
    
    @classmethod
    def get_storage_path(cls, subdir: str = '') -> str:
        """
        获取存储路径（绝对路径）- 推荐使用此方法代替直接使用路径常量

        Args:
            subdir: 子目录名称，支持：
                - 'videos': 原始上传视频
                - 'result_videos': 带标注的结果视频
                - 'preprocessed_videos': 预处理后的视频
                - 'tracking_results': 追踪结果JSON文件
                - 或任意自定义子目录名称

        Returns:
            绝对路径（规范化后）

        Examples:
            >>> Config.get_storage_path('videos')
            '/path/to/codes/storage/videos'

            >>> Config.get_storage_path('tracking_results')
            '/path/to/codes/storage/tracking_results'

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

    @staticmethod
    def auto_select_device(preferred_device: str = '') -> str:
        """
        自动选择PyTorch设备，优先级：CUDA > MPS > CPU

        Args:
            preferred_device: 用户指定的设备（可选）
                - 如果指定了具体设备（如'cuda', 'mps', 'cpu'），则使用指定设备
                - 如果为空字符串，则按优先级自动选择

        Returns:
            设备字符串 ('cuda', 'mps', 或 'cpu')
        """
        # 如果用户指定了设备，直接使用
        if preferred_device:
            return preferred_device

        # 按优先级自动选择
        if torch.cuda.is_available():
            device = 'cuda'
            device_name = torch.cuda.get_device_name(0)
            logger.info("使用 CUDA 设备: %s", device_name)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            logger.info("使用 Apple MPS (Metal Performance Shaders) 设备")
        else:
            device = 'cpu'
            logger.info("使用 CPU 设备")

        return device

    @classmethod
    def get_tracking_results_path(cls, task_id: int) -> str:
        filename = cls.TRACKING_RESULTS_FILENAME_TEMPLATE.format(task_id=task_id)
        return os.path.join(cls.get_storage_path(cls.STORAGE_TRACKING_RESULTS_SUBDIR), filename)
