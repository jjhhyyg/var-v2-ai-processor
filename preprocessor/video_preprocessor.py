# -*- coding: utf-8 -*-
"""
本模块负责对VAR熔池视频进行预处理。
主要目标是提升视频质量，包括去噪、增强对比度等，为后续的参数识别和缺陷检测做准备。
核心特性：
- 支持GPU (CUDA) 和 CPU 两种处理模式，优先使用GPU以提高效率。
- 提供不同强度的预设处理方案 ('mild', 'moderate', 'strong')。
- 包含针对熔池高亮区域的特定增强算法。
- 在处理长视频时，提供详细的进度日志，包括处理速度和预计剩余时间。
"""

# 导入必要的库
import os  # 用于文件系统操作
import cv2  # 用于图像和视频处理
import numpy as np  # 用于数值计算
import logging  # 用于日志记录
import time  # 用于计算执行时间
from typing import Optional, Callable

# --- 全局配置：设置日志记录器 ---
logger = logging.getLogger(__name__)


class OptimizedVideoPreprocessor:
    """
    优化后的VAR熔池视频预处理类 (Optimized VAR Pool Video Preprocessor)

    本类封装了视频预处理的完整流程，旨在通过一系列图像处理技术
    消除视频中的噪声、增强熔池区域的细节，同时保持较高的处理效率。
    """

    def __init__(self, use_gpu=True):
        """
        构造函数，用于初始化预处理器实例。

        :param use_gpu: bool, 是否尝试使用GPU进行加速。
        """
        # --- 检查CUDA环境，决定最终的运行模式（GPU或CPU）---
        # 只有在用户请求使用GPU且系统中存在可用的CUDA设备时，才会真正启用GPU模式
        self.use_gpu = use_gpu and (cv2.cuda.getCudaEnabledDeviceCount() > 0)

        if use_gpu and not self.use_gpu:
            logger.warning("请求使用GPU，但未检测到启用了CUDA的设备。将回退到CPU模式。")
        elif self.use_gpu:
            logger.info(f"成功启用GPU加速模式，检测到 {cv2.cuda.getCudaEnabledDeviceCount()} 个CUDA设备。")
        else:
            logger.info("当前运行在CPU模式。")

    def process_video(self, input_path: str, output_path: str,
                     strength: str = 'moderate', enhance_pool: bool = True,
                     progress_callback: Optional[Callable[[int, int, float], None]] = None):
        """
        处理整个视频文件的主函数。

        :param input_path: str, 输入视频文件的路径。
        :param output_path: str, 处理后输出视频文件的路径。
        :param strength: str, 预处理强度。可选值为 'mild'(轻度), 'moderate'(中度), 'strong'(强度)。
                         不同强度对应不同的去噪和增强参数。
        :param enhance_pool: bool, 是否启用针对熔池的特定增强算法。
        :param progress_callback: 进度回调函数,接收(当前帧数, 总帧数, 已耗时秒数)
        """
        import sys
        import tempfile
        import shutil
        
        # 处理Windows下中文路径乱码问题
        # 在Windows下使用临时英文路径，处理完成后再重命名
        use_temp_output = False
        temp_output_path = None
        if sys.platform == 'win32':
            try:
                # 检测路径中是否包含非ASCII字符
                output_path.encode('ascii')
            except UnicodeEncodeError:
                # 包含非ASCII字符（如中文），使用临时文件
                use_temp_output = True
                # 创建临时文件（自动清理）
                temp_fd, temp_output_path = tempfile.mkstemp(suffix='.mp4', prefix='temp_preprocessed_')
                os.close(temp_fd)  # 关闭文件描述符
                logger.info(f"Windows环境检测到非ASCII路径，使用临时文件: {temp_output_path}")
        
        # 实际写入的路径（可能是临时路径）
        actual_output_path = temp_output_path if use_temp_output else output_path
        
        # --- 步骤 1: 打开视频文件 ---
        # Windows下读取视频也可能有中文路径问题，但这里输入路径通常是从数据库来的相对路径
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {input_path}")
            if use_temp_output and temp_output_path:
                os.remove(temp_output_path)
            raise ValueError(f"无法打开视频文件: {input_path}")

        # --- 步骤 2: 获取视频基本属性 ---
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- 步骤 3: 设置输出视频的写入器 ---
        # 尝试使用浏览器兼容的编码器（优先H.264，浏览器友好）
        codecs_to_try = [
            ('avc1', 'H.264 (AVC)'),  # H.264编码，浏览器友好
            ('x264', 'x264'),         # x264编码器
            ('H264', 'H.264'),        # H.264别名
            ('mp4v', 'MPEG-4')        # 降级方案（可能不被所有浏览器支持）
        ]

        out = None
        used_codec = None

        for codec, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    used_codec = codec_name
                    logger.info(f"使用视频编码器: {codec_name}")
                    break
                else:
                    out.release()
            except Exception as e:
                logger.debug(f"编码器 {codec} 不可用: {e}")
                continue

        # 验证视频写入器是否成功创建
        if not out or not out.isOpened():
            cap.release()
            logger.error(f"无法创建输出视频文件: {output_path}")
            raise ValueError(f"无法创建输出视频文件: {output_path}，所有编码器均不可用")

        if used_codec == 'MPEG-4':
            logger.warning("使用 MPEG-4 编码器，视频可能无法在所有浏览器中播放。建议安装支持H.264的OpenCV版本。")

        # 打印处理任务的摘要信息
        logger.info(f"开始预处理视频: {input_path}")
        logger.info(f"分辨率: {width}x{height}, 帧率: {fps}, 总帧数: {total_frames}")
        logger.info(f"预处理强度: {strength}, 熔池增强: {'启用' if enhance_pool else '禁用'}")

        frame_count = 0
        start_time = time.time()

        # --- 步骤 4: 逐帧读取、处理和写入 ---
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取结束

            # 根据初始化时确定的模式（GPU/CPU）选择不同的处理函数
            if self.use_gpu:
                # GPU处理流程
                gpu_frame = cv2.cuda_GpuMat()  # 创建一个GPU矩阵对象
                gpu_frame.upload(frame)  # 将CPU内存中的帧上传到GPU显存
                processed_gpu_frame = self.process_frame_gpu(gpu_frame, strength, enhance_pool)
                processed_frame = processed_gpu_frame.download()  # 将处理后的结果从GPU下载回CPU内存
            else:
                # CPU处理流程
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

        # --- 步骤 6: 释放资源并验证输出 ---
        end_time = time.time()

        cap.release()  # 释放视频读取器
        out.release()  # 释放视频写入器
        cv2.destroyAllWindows()

        # Windows环境：如果使用了临时文件，现在将其重命名为目标文件名
        if use_temp_output and temp_output_path:
            try:
                # 确保目标目录存在
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                # 如果目标文件已存在，先删除
                if os.path.exists(output_path):
                    os.remove(output_path)
                
                # 移动临时文件到目标位置
                shutil.move(temp_output_path, output_path)
                logger.info(f"临时文件已重命名为: {output_path}")
            except Exception as e:
                logger.error(f"重命名临时文件失败: {e}")
                # 清理临时文件
                if os.path.exists(temp_output_path):
                    try:
                        os.remove(temp_output_path)
                    except:
                        pass
                raise ValueError(f"无法保存预处理视频到: {output_path}, 错误: {e}")

        # 验证输出文件是否存在且有效
        if not os.path.exists(output_path):
            logger.error(f"预处理视频文件未成功创建: {output_path}")
            raise ValueError(f"预处理视频文件未成功创建: {output_path}")

        output_file_size = os.path.getsize(output_path)
        if output_file_size == 0:
            logger.error(f"预处理视频文件大小为0: {output_path}")
            raise ValueError(f"预处理视频文件创建失败（文件大小为0）")

        logger.info(f"视频预处理完成！总共处理 {frame_count} 帧，耗时 {(end_time - start_time) / 60:.2f} 分钟。")
        logger.info(f"输出文件: {output_path}, 大小: {output_file_size / 1024 / 1024:.2f} MB")

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

    def process_frame_gpu(self, gpu_frame, strength='moderate', enhance_pool=True):
        """ 在GPU上对单帧图像应用预处理流水线 (使用 cv2.cuda 模块) """
        # --- 步骤 1: GPU去噪 ---
        if strength == 'strong':
            # 非局部均值去噪：效果非常好，但计算量巨大，在GPU上执行才具有可行性
            denoised_gpu_frame = cv2.cuda.fastNlMeansDenoisingColored(gpu_frame, h=10)
        elif strength == 'moderate':
            denoised_gpu_frame = cv2.cuda.fastNlMeansDenoisingColored(gpu_frame, h=7)
        else:  # 'mild'
            # 高斯滤波：一种简单快速的线性平滑滤波
            gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 1.5)
            denoised_gpu_frame = gaussian_filter.apply(gpu_frame)

        # --- 步骤 2: GPU图像增强 (可选) ---
        if strength in ['moderate', 'strong']:
            enhanced_gpu_frame = self.gamma_correction_gpu(denoised_gpu_frame)
        else:
            enhanced_gpu_frame = denoised_gpu_frame

        # --- 步骤 3: GPU熔池特定增强 (可选) ---
        if enhance_pool:
            enhanced_gpu_frame = self.enhance_melting_pool_gpu(enhanced_gpu_frame)

        # --- 步骤 4: GPU结果混合 ---
        blend_ratios = {'mild': 0.4, 'moderate': 0.6, 'strong': 0.8}
        ratio = blend_ratios.get(strength, 0.6)
        result_gpu_frame = cv2.cuda.addWeighted(gpu_frame, 1 - ratio, enhanced_gpu_frame, ratio, 0)

        return result_gpu_frame

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

    def gamma_correction_gpu(self, gpu_frame):
        """(GPU)对亮度通道应用Gamma校正。"""
        gamma = 0.85  # 在GPU上使用固定的gamma值以简化计算
        gpu_ycrcb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.cuda.split(gpu_ycrcb)

        # GPU上的幂运算需要浮点数格式
        y_float = cv2.cuda_GpuMat()
        y.convertTo(y_float, cv2.CV_32F, 1.0 / 255.0)  # 归一化到 [0, 1]

        inv_gamma = 1.0 / gamma
        cv2.cuda.pow(y_float, inv_gamma, y_float)  # 执行幂运算

        y_float.convertTo(y, cv2.CV_8U, 255.0)  # 转换回8位整数格式 [0, 255]

        cv2.cuda.merge([y, cr, cb], gpu_ycrcb)
        return cv2.cuda.cvtColor(gpu_ycrcb, cv2.COLOR_YCrCb2BGR)

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

    def enhance_melting_pool_gpu(self, gpu_frame):
        """(GPU)保守的熔池增强。"""
        try:
            # 与CPU版本逻辑相同，但所有操作都使用cv2.cuda模块的函数
            gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.cuda.split(gpu_hsv)

            # 创建绿色和饱和度掩码
            mask_green = cv2.cuda.inRange(h, (50,), (80,))
            mask_sat = cv2.cuda.inRange(s, (30,), (255,))

            # 合并掩码
            gpu_mask = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC1)
            cv2.cuda.bitwise_and(mask_green, mask_sat, gpu_mask)

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morph_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, kernel)
            morph_filter.apply(gpu_mask, gpu_mask)

            # 平滑掩码
            gpu_mask_float = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
            gpu_mask.convertTo(gpu_mask_float, cv2.CV_32F, 1.0 / 255.0)
            blur_filter = cv2.cuda.createGaussianFilter(cv2.CV_32FC1, cv2.CV_32FC1, (11, 11), 3)
            blur_filter.apply(gpu_mask_float, gpu_mask_float)

            # 增强V通道
            v_enhanced = cv2.cuda_GpuMat()
            v.convertTo(v_enhanced, cv2.CV_8UC1, 1.1, 5)

            # 混合V通道 - 使用CPU计算以提高兼容性
            v_cpu = v.download()
            v_enhanced_cpu = v_enhanced.download()
            mask_cpu = gpu_mask_float.download()

            v_final_cpu = (v_cpu * (1 - mask_cpu) + v_enhanced_cpu * mask_cpu).astype(np.uint8)

            v_final = cv2.cuda_GpuMat()
            v_final.upload(v_final_cpu)

            # 合并并转换回BGR
            cv2.cuda.merge([h, s, v_final], gpu_hsv)
            return cv2.cuda.cvtColor(gpu_hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            logger.warning(f"GPU熔池增强失败，回退到CPU模式: {e}")
            # 回退到CPU处理
            frame_cpu = gpu_frame.download()
            processed_cpu = self.enhance_melting_pool_cpu(frame_cpu)
            result_gpu = cv2.cuda_GpuMat()
            result_gpu.upload(processed_cpu)
            return result_gpu
