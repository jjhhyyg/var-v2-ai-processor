"""
视频存储管理工具模块
统一处理视频文件的创建、写入、验证等操作
解决中文路径、编码器兼容性、临时文件清理等问题
"""
import os
import sys
import cv2
import uuid
import atexit
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, Callable

logger = logging.getLogger(__name__)


class VideoStorageManager:
    """
    视频存储管理器
    
    功能：
    1. 统一处理视频写入器创建
    2. 自动处理 Windows 中文路径问题
    3. 尝试多种编码器确保兼容性
    4. 磁盘空间检查
    5. 临时文件自动清理
    6. 视频文件完整性验证
    """
    
    # 类级别的临时文件跟踪列表
    _temp_files = []
    _cleanup_registered = False
    
    # 默认编码器列表（优先级从高到低）
    DEFAULT_CODECS = [
        ('avc1', 'H.264 (AVC)'),   # H.264编码，浏览器友好
        ('x264', 'x264'),          # x264编码器
        ('H264', 'H.264'),         # H.264别名
        ('mp4v', 'MPEG-4')         # 降级方案
    ]
    
    def __init__(self):
        """初始化视频存储管理器"""
        # 注册清理函数（只注册一次）
        if not VideoStorageManager._cleanup_registered:
            atexit.register(self._cleanup_temp_files)
            VideoStorageManager._cleanup_registered = True
            logger.debug("临时文件清理函数已注册")
    
    @classmethod
    def _cleanup_temp_files(cls):
        """清理所有临时文件"""
        if not cls._temp_files:
            return
        
        logger.info(f"开始清理 {len(cls._temp_files)} 个临时文件...")
        cleaned = 0
        failed = 0
        
        for temp_file in cls._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    cleaned += 1
                    logger.debug(f"已清理临时文件: {temp_file}")
            except Exception as e:
                failed += 1
                logger.warning(f"清理临时文件失败 {temp_file}: {e}")
        
        logger.info(f"临时文件清理完成: 成功 {cleaned}, 失败 {failed}")
        cls._temp_files.clear()
    
    @staticmethod
    def check_disk_space(path: str, required_mb: float = 100) -> dict:
        """
        检查磁盘空间是否足够
        
        Args:
            path: 文件路径（用于确定磁盘位置）
            required_mb: 需要的空间（MB）
        
        Returns:
            dict: {'available_mb': float, 'sufficient': bool}
        
        Raises:
            IOError: 磁盘空间不足时抛出
        """
        try:
            # 确保目录存在
            dir_path = os.path.dirname(path) or '.'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            # 获取磁盘使用情况
            stat = shutil.disk_usage(dir_path)
            available_mb = stat.free / (1024 * 1024)
            sufficient = available_mb >= required_mb
            
            result = {
                'available_mb': available_mb,
                'required_mb': required_mb,
                'sufficient': sufficient
            }
            
            if not sufficient:
                raise IOError(
                    f"磁盘空间不足: 需要 {required_mb:.0f}MB, "
                    f"可用 {available_mb:.0f}MB (缺少 {required_mb - available_mb:.0f}MB)"
                )
            
            logger.debug(f"磁盘空间检查通过: {available_mb:.0f}MB 可用")
            return result
            
        except IOError:
            raise
        except Exception as e:
            logger.warning(f"磁盘空间检查失败: {e}")
            # 检查失败不阻断流程，返回假设充足
            return {'available_mb': -1, 'required_mb': required_mb, 'sufficient': True}
    
    @staticmethod
    def validate_video_file(path: str, check_frames: bool = True) -> dict:
        """
        验证视频文件完整性
        
        Args:
            path: 视频文件路径
            check_frames: 是否检查帧数（较慢但更准确）
        
        Returns:
            dict: {'valid': bool, 'size': int, 'frames': int}
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件无效
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"视频文件不存在: {path}")
        
        # 检查文件大小
        size = os.path.getsize(path)
        if size == 0:
            raise ValueError(f"视频文件大小为0: {path}")
        
        # 尝试打开视频
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {path}")
        
        frames = 0
        if check_frames:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frames == 0:
                cap.release()
                raise ValueError(f"视频帧数为0: {path}")
        
        cap.release()
        
        result = {
            'valid': True,
            'size': size,
            'size_mb': size / (1024 * 1024),
            'frames': frames
        }
        
        logger.debug(f"视频文件验证通过: {path} ({result['size_mb']:.2f}MB, {frames} 帧)")
        return result
    
    def create_video_writer(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codecs: Optional[list] = None,
        estimate_size_mb: Optional[float] = None
    ) -> Tuple[cv2.VideoWriter, str, Callable]:
        """
        创建视频写入器（统一处理中文路径、编码器选择等）
        
        Args:
            output_path: 输出路径（可能包含中文）
            fps: 帧率
            width: 视频宽度
            height: 视频高度
            codecs: 编码器列表（可选，默认使用 DEFAULT_CODECS）
            estimate_size_mb: 预估文件大小（MB），用于磁盘空间检查
        
        Returns:
            tuple: (VideoWriter对象, 实际写入路径, 清理函数)
        
        Raises:
            ValueError: 无法创建写入器
            IOError: 磁盘空间不足
        """
        # 检查磁盘空间
        if estimate_size_mb:
            self.check_disk_space(output_path, estimate_size_mb)
        
        # 处理 Windows 下中文路径问题
        use_temp_output = False
        temp_output_path = None
        actual_output_path = output_path
        
        if sys.platform == 'win32':
            try:
                output_path.encode('ascii')
            except UnicodeEncodeError:
                # 包含非ASCII字符，使用临时文件
                use_temp_output = True
                output_dir = os.path.dirname(output_path)
                
                # 确保目录存在
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                # 在同一目录下创建临时文件
                temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
                temp_output_path = os.path.join(output_dir, temp_filename)
                actual_output_path = temp_output_path
                
                # 添加到清理列表
                VideoStorageManager._temp_files.append(temp_output_path)
                
                logger.info(f"Windows环境检测到非ASCII路径，使用临时文件: {temp_output_path}")
        
        # 尝试多种编码器
        codecs_to_try = codecs or self.DEFAULT_CODECS
        out = None
        used_codec = None
        
        for codec, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(actual_output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    used_codec = codec_name
                    logger.info(f"使用视频编码器: {codec_name}")
                    break
                else:
                    out.release()
            except Exception as e:
                logger.debug(f"编码器 {codec} 不可用: {e}")
                continue
        
        # 验证是否成功创建
        if not out or not out.isOpened():
            # 清理临时文件
            if use_temp_output and temp_output_path:
                self._remove_temp_file(temp_output_path)
            raise ValueError(f"无法创建输出视频文件: {output_path}，所有编码器均不可用")
        
        if used_codec == 'MPEG-4':
            logger.warning("使用 MPEG-4 编码器，视频可能无法在所有浏览器中播放。建议安装支持H.264的OpenCV版本。")
        
        # 创建清理函数
        def finalize(success: bool = True):
            """
            完成视频写入的清理函数
            
            Args:
                success: 是否成功完成（成功则移动临时文件，失败则删除）
            """
            out.release()
            
            if use_temp_output and temp_output_path:
                if success:
                    try:
                        # 确保目标目录存在
                        output_dir = os.path.dirname(output_path)
                        if output_dir and not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        
                        # 如果目标文件已存在，先删除
                        if os.path.exists(output_path):
                            os.remove(output_path)
                        
                        # 验证临时文件
                        if not os.path.exists(temp_output_path):
                            raise ValueError(f"临时文件不存在: {temp_output_path}")
                        
                        temp_size = os.path.getsize(temp_output_path)
                        if temp_size == 0:
                            raise ValueError(f"临时文件大小为0: {temp_output_path}")
                        
                        logger.info(f"开始移动临时文件 ({temp_size / 1024 / 1024:.2f}MB)...")
                        
                        # 使用 shutil.move（更高效）
                        try:
                            shutil.move(temp_output_path, output_path)
                            logger.info(f"临时文件已移动为: {output_path}")
                        except Exception as move_error:
                            # 如果 move 失败（可能跨分区），降级到 copy+remove
                            logger.warning(f"shutil.move 失败，使用 copy+remove: {move_error}")
                            shutil.copyfile(temp_output_path, output_path)
                            os.remove(temp_output_path)
                            logger.info(f"临时文件已复制并删除")
                        
                        # 验证目标文件
                        final_size = os.path.getsize(output_path)
                        logger.info(f"最终文件大小: {final_size / 1024 / 1024:.2f}MB")
                        
                        # 从清理列表中移除
                        self._remove_from_cleanup_list(temp_output_path)
                        
                    except Exception as e:
                        logger.error(f"移动临时文件失败: {e}", exc_info=True)
                        self._remove_temp_file(temp_output_path)
                        raise ValueError(f"无法保存视频到: {output_path}, 错误: {e}")
                else:
                    # 失败时删除临时文件
                    logger.info(f"写入失败，删除临时文件: {temp_output_path}")
                    self._remove_temp_file(temp_output_path)
        
        return out, actual_output_path, finalize
    
    @classmethod
    def _remove_temp_file(cls, temp_path: str):
        """删除临时文件并从清理列表中移除"""
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"已删除临时文件: {temp_path}")
        except Exception as e:
            logger.warning(f"删除临时文件失败 {temp_path}: {e}")
        
        cls._remove_from_cleanup_list(temp_path)
    
    @classmethod
    def _remove_from_cleanup_list(cls, temp_path: str):
        """从清理列表中移除"""
        if temp_path in cls._temp_files:
            cls._temp_files.remove(temp_path)
            logger.debug(f"已从清理列表移除: {temp_path}")
    
    @staticmethod
    def estimate_video_size(
        width: int,
        height: int,
        total_frames: int,
        fps: float,
        compression_ratio: float = 0.05
    ) -> float:
        """
        估算视频文件大小（MB）
        
        Args:
            width: 视频宽度
            height: 视频高度
            total_frames: 总帧数
            fps: 帧率
            compression_ratio: 压缩比（默认0.05，即压缩到原始大小的5%）
        
        Returns:
            float: 预估大小（MB）
        """
        # 原始大小 = 宽 × 高 × 帧数 × 3字节(BGR)
        raw_size_mb = width * height * total_frames * 3 / (1024 * 1024)
        
        # 考虑压缩比
        estimated_size_mb = raw_size_mb * compression_ratio
        
        # 添加10%的缓冲
        estimated_size_mb *= 1.1
        
        logger.debug(
            f"视频大小估算: {width}x{height}, {total_frames}帧, "
            f"原始{raw_size_mb:.0f}MB → 压缩{estimated_size_mb:.0f}MB"
        )
        
        return estimated_size_mb
