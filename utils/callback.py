"""
桌面 worker stdout 事件工具模块。
"""
import json
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BackendCallback:
    """通过 stdout 输出桌面 NDJSON 事件。"""

    STDOUT_CALLBACK_URL = 'stdout://'
    STDOUT_PROGRESS_INTERVAL_SECONDS = 0.5

    def __init__(self, task_id: int, callback_url: Optional[str] = None):
        self.task_id = task_id
        self.callback_url = callback_url or self.STDOUT_CALLBACK_URL
        self._last_stdout_progress_emit_at: Optional[float] = None
        self._last_stdout_progress_signature: Optional[tuple[Optional[str], Optional[str], Optional[bool], Optional[bool]]] = None
        self._stdout_failed = False

    def is_stdout_mode(self) -> bool:
        return self.callback_url == self.STDOUT_CALLBACK_URL

    def stdout_available(self) -> bool:
        return self.is_stdout_mode() and not self._stdout_failed

    def emit_event(self, event_type: str, payload: Dict[str, Any]) -> bool:
        if not self.stdout_available():
            return False

        try:
            # stdout is reserved for the desktop NDJSON protocol. Do not log human-readable text here.
            print(
                json.dumps({
                    'type': event_type,
                    'payload': payload
                }, ensure_ascii=True),
                flush=True
            )
            return True
        except Exception as e:
            self._stdout_failed = True
            logger.error(f"Failed to emit stdout event for task {self.task_id}: {e}")
            return False

    def _stdout_progress_signature(
        self,
        progress_data: Dict[str, Any]
    ) -> tuple[Optional[str], Optional[str], Optional[bool], Optional[bool]]:
        return (
            progress_data.get('status'),
            progress_data.get('phase'),
            progress_data.get('isTimeout'),
            progress_data.get('timeoutWarning')
        )

    def _should_emit_stdout_progress(self, progress_data: Dict[str, Any]) -> bool:
        now = time.monotonic()
        signature = self._stdout_progress_signature(progress_data)

        if self._last_stdout_progress_emit_at is None:
            return True

        if self._last_stdout_progress_signature != signature:
            return True

        return (now - self._last_stdout_progress_emit_at) >= self.STDOUT_PROGRESS_INTERVAL_SECONDS

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
        if not self.is_stdout_mode():
            logger.error("Unsupported callback URL %s; desktop worker requires stdout://", self.callback_url)
            return False

        if not self._should_emit_stdout_progress(progress_data):
            return True

        emitted = self.emit_event('progress', progress_data)
        if emitted:
            self._last_stdout_progress_emit_at = time.monotonic()
            self._last_stdout_progress_signature = self._stdout_progress_signature(progress_data)
        return emitted

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
                - videoInfo: 视频信息
                - performance: 性能信息
                - anomalyEvents: YOLO-only异常事件列表
                - failureReason: 失败原因（失败时）

        Returns:
            是否提交成功
        """
        if not self.is_stdout_mode():
            logger.error("Unsupported callback URL %s; desktop worker requires stdout://", self.callback_url)
            return False
        return self.emit_event('result', result_data)

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
