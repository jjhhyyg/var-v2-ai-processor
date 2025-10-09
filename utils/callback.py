"""
后端回调工具模块
负责向后端发送进度更新和结果提交
"""
import requests
import logging
from typing import Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)


class BackendCallback:
    """后端回调工具类"""

    def __init__(self, task_id: int, callback_url: Optional[str] = None):
        self.task_id = task_id
        self.callback_url = callback_url or Config.BACKEND_BASE_URL
        self.headers = {
            'Content-Type': 'application/json'
        }

    def update_progress(self, progress_data: Dict[str, Any]) -> bool:
        """
        更新任务进度

        Args:
            progress_data: 进度数据，包含以下字段：
                - status: 任务状态 (PREPROCESSING/ANALYZING)
                - progress: 进度 (0.0-1.0)
                - currentFrame: 当前帧号
                - totalFrames: 总帧数
                - phase: 阶段描述
                - preprocessingDuration: 预处理耗时（秒）
                - analyzingElapsedTime: 分析已用时间（秒）
                - isTimeout: 是否超时
                - timeoutWarning: 是否接近超时

        Returns:
            是否更新成功
        """
        url = f"{self.callback_url}/api/tasks/{self.task_id}/progress"

        try:
            response = requests.post(
                url,
                json=progress_data,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Progress updated for task {self.task_id}: {progress_data.get('progress', 0):.2%}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update progress for task {self.task_id}: {e}")
            return False

    def submit_result(self, result_data: Dict[str, Any]) -> bool:
        """
        提交分析结果

        Args:
            result_data: 结果数据，包含以下字段：
                - status: 最终状态 (COMPLETED/COMPLETED_TIMEOUT/FAILED)
                - isTimeout: 是否超时
                - preprocessingDuration: 预处理耗时（秒）
                - analyzingDuration: 分析耗时（秒）
                - totalDuration: 总耗时（秒）
                - dynamicMetrics: 动态参数数据列表
                - anomalyEvents: 异常事件列表
                - trackingObjects: 追踪物体列表
                - failureReason: 失败原因（失败时）

        Returns:
            是否提交成功
        """
        url = f"{self.callback_url}/api/tasks/{self.task_id}/result"

        try:
            response = requests.post(
                url,
                json=result_data,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Result submitted for task {self.task_id}: {result_data.get('status')}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to submit result for task {self.task_id}: {e}")
            return False

    def notify_preprocessing(self, message: str = "正在预处理视频...") -> bool:
        """通知预处理阶段"""
        return self.update_progress({
            'status': 'PREPROCESSING',
            'phase': '视频预处理中',
            'progress': 0.0,
            'message': message
        })

    def notify_analyzing_start(self, total_frames: int, preprocessing_duration: int) -> bool:
        """通知开始分析"""
        return self.update_progress({
            'status': 'ANALYZING',
            'phase': '视频分析中',
            'progress': 0.0,
            'currentFrame': 0,
            'totalFrames': total_frames,
            'preprocessingDuration': preprocessing_duration,
            'analyzingElapsedTime': 0,
            'isTimeout': False,
            'timeoutWarning': False
        })
