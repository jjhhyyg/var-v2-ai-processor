"""
动态参数计算模块
计算熔池的闪烁频率、面积、周长等动态参数
基于图像处理和频域分析实现真实参数提取
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """动态参数计算器 - 真实算法实现"""

    def __init__(self, method='adaptive', roi_rect=None):
        """
        初始化计算器

        Args:
            method: 熔池区域分割方法，'adaptive'（自适应阈值）或 'hsv'（颜色空间阈值）
            roi_rect: 感兴趣区域 (x, y, width, height)，为None则处理整帧
        """
        self.method = method
        self.roi_rect = roi_rect

        # 用于存储时间序列数据（用于后续频率分析）
        self.brightness_series = []
        self.area_series = []
        self.perimeter_series = []
        self.timestamps = []

        logger.info(f"MetricsCalculator initialized: method={method}, roi={'set' if roi_rect else 'full frame'}")

    def extract_pool_properties(self, frame: np.ndarray) -> tuple:
        """
        从单帧图像中提取熔池的关键参数

        Args:
            frame: 输入的单帧图像 (BGR格式)

        Returns:
            (净面积, 外周长, 亮度) 的元组
        """
        if frame is None:
            return 0, 0, 0

        # 如果设置了ROI，先对图像进行裁剪
        if self.roi_rect:
            x, y, w, h = self.roi_rect
            frame = frame[y:y + h, x:x + w]

        # 步骤 1: 图像分割，将熔池区域与背景分离
        if self.method == 'adaptive':
            # 自适应阈值法：对光照不均的场景效果较好
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=151, C=-10
            )
        elif self.method == 'hsv':
            # HSV颜色空间法：适用于熔池颜色特征明显的场景
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            binary = cv2.inRange(hsv, lower_green, upper_green)
        else:
            raise ValueError("方法(method)必须是 'adaptive' 或 'hsv'")

        # 步骤 2: 形态学操作，优化二值图像
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 闭运算：填充轮廓内部的小黑点
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        # 开运算：去除轮廓外部的噪点
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

        # 步骤 3: 轮廓发现与分析
        contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        net_area = 0
        perimeter = 0

        if contours and hierarchy is not None:
            hierarchy = hierarchy[0]

            # 找到所有最外层的轮廓 (父轮廓ID为-1)
            parent_contours_info = [
                (i, cv2.contourArea(contours[i]))
                for i, h in enumerate(hierarchy) if h[3] == -1
            ]

            if parent_contours_info:
                # 按面积从大到小排序，找到最大的最外层轮廓（主熔池）
                parent_contours_info.sort(key=lambda x: x[1], reverse=True)
                main_contour_idx = parent_contours_info[0][0]
                main_contour = contours[main_contour_idx]

                # 计算父轮廓的面积和周长
                parent_area = parent_contours_info[0][1]
                perimeter = cv2.arcLength(main_contour, True)

                # 找到所有直接属于该父轮廓的子轮廓（内部孔洞）
                holes_area = sum(
                    cv2.contourArea(contours[i])
                    for i, h in enumerate(hierarchy) if h[3] == main_contour_idx
                )

                # 核心计算：净面积 = 父轮廓面积 - 所有内部孔洞的总面积
                net_area = parent_area - holes_area

        # 步骤 4: 亮度计算
        # 计算灰度值最高的10%像素的平均值（关注熔池区域的亮度）
        gray_for_brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray_for_brightness, 90)
        bright_pixels = gray_for_brightness[gray_for_brightness > threshold]
        brightness = np.mean(bright_pixels) if bright_pixels.size > 0 else 0

        return net_area, perimeter, brightness

    def calculate_metrics(self, frame_number: int, timestamp: float,
                          frame: np.ndarray = None) -> Dict[str, Any]:
        """
        计算当前帧的动态参数

        Args:
            frame_number: 帧号
            timestamp: 时间戳（秒）
            frame: 视频帧

        Returns:
            包含动态参数的字典
        """
        if frame is None:
            # 如果没有提供帧数据，返回空值
            area, perimeter, brightness = 0, 0, 0
        else:
            # 提取真实参数
            area, perimeter, brightness = self.extract_pool_properties(frame)

        # 存储到时间序列（用于后续频率分析）
        self.area_series.append(area)
        self.perimeter_series.append(perimeter)
        self.brightness_series.append(brightness)
        self.timestamps.append(timestamp)

        return {
            'frameNumber': frame_number,
            'timestamp': round(timestamp, 2),
            'brightness': round(brightness, 1),  # 返回亮度值而非频率
            'poolArea': int(max(0, area)),
            'poolPerimeter': round(max(0, perimeter), 1)
        }

    def calculate_frequency(self, data_series: List[float], fps: float,
                           param_name: str = "") -> Optional[Dict[str, Any]]:
        """
        对时间序列数据进行傅里叶变换，计算主导频率和变化趋势

        Args:
            data_series: 时间序列数据
            fps: 视频帧率
            param_name: 参数名称（用于调试）

        Returns:
            包含分析结果的字典，如果数据太短则返回None
        """
        if len(data_series) < 10:
            logger.warning(f"数据序列太短 ({len(data_series)} 帧)，无法进行频率分析")
            return None

        data = np.array(data_series)

        # 步骤1: 去趋势 (Detrending)
        # 使用线性拟合移除数据的长期上升或下降趋势
        x = np.arange(len(data))
        z = np.polyfit(x, data, 1)
        p = np.poly1d(z)
        detrended = data - p(x)

        # 步骤2: 加窗 (Windowing)
        # 使用汉宁窗减少频谱泄漏
        window = np.hanning(len(detrended))
        windowed = detrended * window

        # 步骤3: 快速傅里叶变换 (FFT)
        yf = fft(windowed)
        xf = fftfreq(len(windowed), 1 / fps)[:len(windowed) // 2]
        power = 2.0 / len(windowed) * np.abs(yf[:len(windowed) // 2])

        if len(power) > 1:
            # 找到功率最大的频率点（忽略直流分量 power[0]）
            main_idx = np.argmax(power[1:]) + 1
            main_freq = xf[main_idx]

            # 判断长期趋势
            trend = "上升" if z[0] > 0 else "下降" if z[0] < 0 else "平稳"

            return {
                'frequency': float(main_freq),
                'period': float(1 / main_freq) if main_freq > 0 else None,
                'trend_slope': float(z[0]),
                'trend': trend,
                'mean': float(np.mean(data)),
                'std': float(np.std(data))
            }

        return None

    def analyze_all(self, fps: float) -> Dict[str, Any]:
        """
        对所有提取出的核心参数进行批量分析

        Args:
            fps: 视频帧率

        Returns:
            包含所有参数分析结果的字典
        """
        results = {}

        logger.info(f"开始分析计算结果，共 {len(self.timestamps)} 帧")

        # 分别对亮度、面积和周长序列进行频率和趋势分析
        brightness_result = self.calculate_frequency(self.brightness_series, fps, "闪烁频率")
        if brightness_result:
            results['闪烁'] = brightness_result
            logger.info(f"闪烁分析完成: 主导频率={brightness_result['frequency']:.3f} Hz")

        area_result = self.calculate_frequency(self.area_series, fps, "面积")
        if area_result:
            results['面积'] = area_result
            logger.info(f"面积分析完成: 主导频率={area_result['frequency']:.3f} Hz, 趋势={area_result['trend']}")

        perimeter_result = self.calculate_frequency(self.perimeter_series, fps, "周长")
        if perimeter_result:
            results['周长'] = perimeter_result
            logger.info(f"周长分析完成: 主导频率={perimeter_result['frequency']:.3f} Hz, 趋势={perimeter_result['trend']}")

        # 计算平均圆度
        if '面积' in results and '周长' in results:
            mean_circularity = (
                4 * np.pi * results['面积']['mean'] / (results['周长']['mean'] ** 2)
                if results['周长']['mean'] > 0 else 0
            )
            results['圆度'] = {'mean': float(mean_circularity)}
            logger.info(f"平均圆度: {mean_circularity:.3f}")

        return results

    def reset(self):
        """重置时间序列数据"""
        self.brightness_series = []
        self.area_series = []
        self.perimeter_series = []
        self.timestamps = []
        logger.info("MetricsCalculator reset")
