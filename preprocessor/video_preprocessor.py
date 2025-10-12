# -*- coding: utf-8 -*-
"""
本模块负责对VAR熔池视频进行预处理。
主要目标是提升视频质量，包括去噪、增强对比度等，为后续的参数识别和缺陷检测做准备。
核心特性：
- 使用 CPU 进行所有 OpenCV 视频处理操作（保证最大兼容性）
- 提供不同强度的预设处理方案 ('mild', 'moderate', 'strong')
- 包含针对熔池高亮区域的特定增强算法
- 在处理长视频时，提供详细的进度日志，包括处理速度和预计剩余时间
"""

# 导入必要的库
import os  # 用于文件系统操作
import cv2  # 用于图像和视频处理
import numpy as np  # 用于数值计算
import logging  # 用于日志记录
import time  # 用于计算执行时间
from typing import Optional, Callable
from utils.video_storage import VideoStorageManager

# --- 全局配置：设置日志记录器 ---
logger = logging.getLogger(__name__)


class OptimizedVideoPreprocessor:
    """
    VAR熔池视频预处理类 (VAR Pool Video Preprocessor)

    本类封装了视频预处理的完整流程，旨在通过一系列图像处理技术
    消除视频中的噪声、增强熔池区域的细节，同时保持较高的处理效率。
    
    注意：视频预处理使用 CPU 进行，不使用 GPU 加速（OpenCV 的 CUDA 模块兼容性问题）
    """

    def __init__(self):
        """
        构造函数，用于初始化预处理器实例。
        
        视频预处理统一使用 CPU 模式，确保最大兼容性。
        """
        logger.info("视频预处理器初始化：使用 CPU 模式")
        
        # 初始化视频存储管理器
        self.storage_manager = VideoStorageManager()

    def process_video(self, input_path: str, output_path: str,
                     frame_rate: float = 25.0,
                     strength: str = 'moderate', enhance_pool: bool = True,
                     progress_callback: Optional[Callable[[int, int, float], None]] = None):
        """
        处理整个视频文件的主函数。

        :param input_path: str, 输入视频文件的路径。
        :param output_path: str, 处理后输出视频文件的路径。
        :param frame_rate: float, 视频帧率（从后端TaskConfig获取，由FFmpeg解析）
        :param strength: str, 预处理强度。可选值为 'mild'(轻度), 'moderate'(中度), 'strong'(强度)。
                         不同强度对应不同的去噪和增强参数。
        :param enhance_pool: bool, 是否启用针对熔池的特定增强算法。
        :param progress_callback: 进度回调函数,接收(当前帧数, 总帧数, 已耗时秒数)
        """
        # --- 步骤 1: 打开视频文件 ---
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {input_path}")
            raise ValueError(f"无法打开视频文件: {input_path}")

        # --- 步骤 2: 获取视频基本属性 ---
        # 使用传入的 frame_rate 而不是从视频文件读取
        fps = int(frame_rate)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- 步骤 3: 估算视频大小并创建写入器 ---
        logger.info(f"开始预处理视频: {input_path}")
        logger.info(f"分辨率: {width}x{height}, 帧率: {fps}, 总帧数: {total_frames}")
        logger.info(f"预处理强度: {strength}, 熔池增强: {'启用' if enhance_pool else '禁用'}")
        
        # 估算输出文件大小
        estimate_size_mb = self.storage_manager.estimate_video_size(
            width, height, total_frames, fps
        )
        logger.info(f"预估输出文件大小: {estimate_size_mb:.0f}MB")
        
        # 创建视频写入器
        try:
            out, actual_output_path, finalize = self.storage_manager.create_video_writer(
                output_path, fps, width, height, estimate_size_mb=estimate_size_mb
            )
        except Exception as e:
            cap.release()
            raise

        frame_count = 0
        start_time = time.time()
        
        success = False  # 标记是否成功完成

        try:
            # --- 步骤 4: 逐帧读取、处理和写入 ---
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # 视频读取结束

                # 使用 CPU 处理帧
                processed_frame = self.process_frame_cpu(frame, strength, enhance_pool)

                # 将处理好的帧写入输出文件
                out.write(processed_frame)
                frame_count += 1

                # --- 步骤 5: 每帧都通过回调更新进度和耗时 ---
                elapsed_time = time.time() - start_time
                
                # 每帧都调用进度回调，传递当前帧数、总帧数和已耗时
                if progress_callback:
                    progress_callback(frame_count, total_frames, elapsed_time)
                
                # 每处理100帧记录一次日志，避免过于频繁的日志输出
                if frame_count % 100 == 0:
                    fps_processed = frame_count / elapsed_time  # 计算平均处理速度
                    eta_seconds = (total_frames - frame_count) / fps_processed  # 计算预计剩余秒数
                    eta_minutes = eta_seconds / 60  # 转换为分钟
                    logger.info(f"预处理进度: {frame_count}/{total_frames} 帧 | "
                               f"速度: {fps_processed:.2f} FPS | "
                               f"已耗时: {elapsed_time:.1f} 秒 | "
                               f"预计剩余: {eta_minutes:.1f} 分钟")
            
            success = True  # 标记成功完成
            
        finally:
            # --- 步骤 6: 释放资源 ---
            end_time = time.time()
            cap.release()
            
            # 调用清理函数（会自动处理临时文件的移动或删除）
            finalize(success=success)
            cv2.destroyAllWindows()

        # --- 步骤 7: 验证输出文件 ---
        try:
            validation_result = self.storage_manager.validate_video_file(output_path, check_frames=False)
            logger.info(f"视频预处理完成！总共处理 {frame_count} 帧，耗时 {(end_time - start_time) / 60:.2f} 分钟。")
            logger.info(f"输出文件: {output_path}, 大小: {validation_result['size_mb']:.2f} MB")
        except Exception as e:
            logger.error(f"输出文件验证失败: {e}")
            raise

    def process_frame_cpu(self, frame, strength='moderate', enhance_pool=True):
        """ 在CPU上对单帧图像应用预处理流水线 """
        # --- 步骤 1: 图像去噪 ---
        # 根据强度选择不同的去噪算法和参数
        if strength == 'strong':
            # 双边滤波：能有效去除噪声，同时较好地保留边缘信息
            denoised_frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        elif strength == 'moderate':
            denoised_frame = cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)
        else:  # 'mild'
            # 中值滤波：对椒盐噪声效果好，计算速度快
            denoised_frame = cv2.medianBlur(frame, 3)

        result = denoised_frame

        # --- 步骤 2: 图像增强 (可选) ---
        # 对中度和强度模式应用Gamma校正，以调整图像的整体亮度和对比度
        if strength in ['moderate', 'strong']:
            result = self.gamma_correction_cpu(result)

        # --- 步骤 3: 熔池特定增强 (可选) ---
        if enhance_pool:
            result = self.enhance_melting_pool_cpu(result)

        # --- 步骤 4: 结果混合 ---
        # 将处理后的图像与原始图像进行加权融合，可以使效果更自然，避免过度处理
        blend_ratios = {'mild': 0.4, 'moderate': 0.6, 'strong': 0.8}
        ratio = blend_ratios.get(strength, 0.6)  # 处理后图像的权重
        result = cv2.addWeighted(frame, 1 - ratio, result, ratio, 0)

        return result

    def gamma_correction_cpu(self, frame):
        """(CPU)对亮度通道应用自适应Gamma校正，以增强暗部或亮部细节。"""
        # 将图像从BGR颜色空间转换到YCrCb，Y是亮度通道，Cr和Cb是色度通道
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 根据平均亮度自适应地选择gamma值
        mean_luminance = np.mean(y)
        if mean_luminance < 60:  # 图像偏暗，需要提亮
            gamma = 0.8
        elif mean_luminance > 180:  # 图像偏亮，需要压暗
            gamma = 1.2
        else:  # 亮度适中，不处理
            return frame

        # 构建Gamma查找表，避免对每个像素进行幂运算，大大提高效率
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # 应用查找表
        y_enhanced = cv2.LUT(y, table)

        # 合并通道并转换回BGR空间
        ycrcb_enhanced = cv2.merge([y_enhanced, cr, cb])
        return cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2BGR)

    def enhance_melting_pool_cpu(self, frame):
        """(CPU)保守的熔池增强：仅增强熔池中特定颜色区域的亮度。"""
        # 转换到HSV颜色空间，H(色相)通道更容易分离颜色
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 创建掩码(mask)，只选中绿色、且有一定饱和度的区域
        mask_green = cv2.inRange(h, 50, 80)
        mask_sat = cv2.inRange(s, 30, 255)
        mask = cv2.bitwise_and(mask_green, mask_sat)

        # 使用形态学开运算去除掩码中的噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 对掩码进行高斯模糊，使其边缘平滑过渡
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (11, 11), 3)

        # 对V(亮度)通道进行线性增强
        v_enhanced = cv2.convertScaleAbs(v, alpha=1.1, beta=5)

        # 使用平滑后的掩码将原亮度图和增强后的亮度图进行融合
        v_final = (v * (1 - mask_float) + v_enhanced * mask_float).astype(np.uint8)

        hsv_enhanced = cv2.merge([h, s, v_final])
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
