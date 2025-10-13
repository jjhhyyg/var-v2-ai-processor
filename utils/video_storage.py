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
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Callable
from utils.file_lock import FileLock

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

    @staticmethod
    def convert_to_h264_with_faststart(video_path: str) -> bool:
        """
        使用 FFmpeg 将视频转换为 H.264 编码并应用 faststart（支持浏览器流式播放）

        如果视频已经是 H.264 编码，则只应用 faststart（快速）
        如果是其他编码（如 MPEG-4），则重新编码为 H.264（较慢）

        Args:
            video_path: 视频文件路径

        Returns:
            bool: 是否成功

        Raises:
            RuntimeError: FFmpeg 处理失败
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 创建临时输出文件
        temp_output = video_path + ".h264.tmp.mp4"

        try:
            # 先检测视频编码格式
            logger.info(f"检测视频编码格式: {video_path}")
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]

            probe_result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )

            codec_name = probe_result.stdout.strip().lower() if probe_result.returncode == 0 else 'unknown'
            logger.info(f"当前视频编码: {codec_name}")

            # 根据编码格式选择处理方式
            if codec_name in ['h264', 'avc']:
                # 已经是 H.264，只需要应用 faststart
                logger.info(f"视频已是 H.264 编码，应用 faststart: {video_path}")
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-c', 'copy',  # 不重新编码
                    '-movflags', 'faststart',
                    '-y',
                    temp_output
                ]
            else:
                # 需要重新编码为 H.264
                logger.info(f"视频编码为 {codec_name}，重新编码为 H.264: {video_path}")
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-c:v', 'libx264',  # 使用 H.264 编码器
                    '-preset', 'medium',  # 编码速度/质量平衡
                    '-crf', '23',  # 恒定质量因子（18-28，23为默认，越小质量越好）
                    '-c:a', 'copy',  # 音频流直接复制（如果有）
                    '-movflags', 'faststart',  # 同时应用 faststart
                    '-y',
                    temp_output
                ]

            # 执行 FFmpeg 命令
            logger.info(f"开始处理视频...")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600  # 10分钟超时（重新编码需要更长时间）
            )

            if result.returncode != 0:
                error_msg = result.stderr
                logger.error(f"FFmpeg 处理失败: {error_msg}")
                raise RuntimeError(f"FFmpeg 处理失败: {error_msg}")

            # 验证输出文件
            if not os.path.exists(temp_output):
                raise RuntimeError("FFmpeg 未生成输出文件")

            output_size = os.path.getsize(temp_output)
            if output_size == 0:
                raise RuntimeError("FFmpeg 输出文件大小为0")

            # 替换原文件
            original_size = os.path.getsize(video_path)
            os.remove(video_path)
            shutil.move(temp_output, video_path)

            action = "Faststart 应用" if codec_name in ['h264', 'avc'] else f"重新编码 ({codec_name} -> H.264)"
            logger.info(
                f"视频处理完成 ({action}): {video_path} "
                f"(原始: {original_size / 1024 / 1024:.2f}MB, "
                f"处理后: {output_size / 1024 / 1024:.2f}MB)"
            )

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg 处理超时: {video_path}")
            # 清理临时文件
            if os.path.exists(temp_output):
                os.remove(temp_output)
            raise RuntimeError("FFmpeg 处理超时")

        except Exception as e:
            logger.error(f"视频处理失败: {e}", exc_info=True)
            # 清理临时文件
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except:
                    pass
            raise

    @staticmethod
    def apply_faststart(video_path: str) -> bool:
        """
        使用 FFmpeg 将视频的 moov atom 移到文件开头，使其支持流式播放

        已弃用：请使用 convert_to_h264_with_faststart() 以确保浏览器兼容性

        Args:
            video_path: 视频文件路径

        Returns:
            bool: 是否成功

        Raises:
            RuntimeError: FFmpeg 处理失败
        """
        logger.warning("apply_faststart() 已弃用，将调用 convert_to_h264_with_faststart()")
        return VideoStorageManager.convert_to_h264_with_faststart(video_path)

    def create_video_writer(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        codecs: Optional[list] = None,
        estimate_size_mb: Optional[float] = None,
        use_file_lock: bool = True,
        lock_timeout: float = 300.0
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
            use_file_lock: 是否使用文件锁（防止并发写入）
            lock_timeout: 文件锁超时时间（秒）

        Returns:
            tuple: (VideoWriter对象, 实际写入路径, 清理函数)

        Raises:
            ValueError: 无法创建写入器
            IOError: 磁盘空间不足
            TimeoutError: 无法获取文件锁
        """
        # 检查磁盘空间
        if estimate_size_mb:
            self.check_disk_space(output_path, estimate_size_mb)

        # 获取文件锁（如果启用）
        file_lock = None
        if use_file_lock:
            try:
                file_lock = FileLock(output_path, exclusive=True, timeout=lock_timeout)
                file_lock.acquire()
                logger.info(f"File lock acquired for {output_path}")
            except TimeoutError as e:
                logger.error(f"Failed to acquire file lock for {output_path}: {e}")
                raise
            except Exception as e:
                logger.warning(f"Failed to create file lock for {output_path}: {e}, continuing without lock")
                file_lock = None

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
            # 释放文件锁
            if file_lock:
                try:
                    file_lock.release()
                except Exception as e:
                    logger.warning(f"Failed to release file lock: {e}")
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
            try:
                out.release()
            except Exception as e:
                logger.warning(f"Failed to release VideoWriter: {e}")

            # 释放文件锁
            if file_lock:
                try:
                    file_lock.release()
                    logger.debug(f"File lock released for {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to release file lock: {e}")

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

            # 成功完成后，转换为 H.264 编码并应用 faststart（确保浏览器兼容性）
            # 注意：如果视频已经是 H.264，只会应用 faststart（快速）
            # 如果是 MPEG-4 等其他编码，会重新编码为 H.264（较慢但必要）
            if success:
                try:
                    VideoStorageManager.convert_to_h264_with_faststart(output_path)
                except Exception as e:
                    logger.error(f"视频编码转换失败，浏览器可能无法播放: {e}")
                    # 不抛出异常，因为文件本身是有效的，只是可能无法在浏览器中播放
        
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
