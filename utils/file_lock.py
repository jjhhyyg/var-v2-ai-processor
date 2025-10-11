"""
跨平台文件锁工具模块
提供进程级和线程级的文件锁机制，防止并发写入冲突
"""
import os
import sys
import time
import logging
import threading
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# 根据平台导入相应的锁模块
if sys.platform == 'win32':
    import msvcrt
    PLATFORM = 'windows'
else:
    import fcntl
    PLATFORM = 'unix'


class FileLock:
    """
    跨平台文件锁实现

    特性:
    1. 支持 Windows (msvcrt) 和 Linux/Mac (fcntl)
    2. 支持上下文管理器 (with语句)
    3. 支持独占锁和共享锁
    4. 支持超时和重试机制
    5. 线程安全

    使用示例:
        # 独占锁
        with FileLock('/path/to/file.mp4'):
            # 写入文件
            pass

        # 共享锁（允许多个读取者）
        with FileLock('/path/to/file.mp4', exclusive=False):
            # 读取文件
            pass

        # 带超时的锁
        with FileLock('/path/to/file.mp4', timeout=10):
            # 写入文件
            pass
    """

    def __init__(
        self,
        file_path: str,
        exclusive: bool = True,
        timeout: float = 60.0,
        retry_interval: float = 0.1
    ):
        """
        初始化文件锁

        Args:
            file_path: 文件路径
            exclusive: 是否为独占锁（True=独占写锁，False=共享读锁）
            timeout: 获取锁的超时时间（秒），None表示无限等待
            retry_interval: 重试间隔（秒）
        """
        self.file_path = str(file_path)
        self.exclusive = exclusive
        self.timeout = timeout
        self.retry_interval = retry_interval

        # 锁文件路径（在原文件旁边创建.lock文件）
        self.lock_file_path = f"{self.file_path}.lock"

        # 文件句柄
        self.lock_file_handle: Optional[int] = None

        # 线程锁（确保同一进程内的线程安全）
        self._thread_lock = threading.Lock()

        # 是否已获取锁
        self._locked = False

        logger.debug(f"FileLock initialized for {self.file_path} (exclusive={exclusive})")

    def acquire(self) -> bool:
        """
        获取文件锁

        Returns:
            bool: 是否成功获取锁

        Raises:
            TimeoutError: 超时未能获取锁
            IOError: 文件操作失败
        """
        # 先获取线程锁
        if not self._thread_lock.acquire(blocking=False):
            logger.warning(f"Thread lock already held for {self.file_path}")
            # 如果当前线程已持有锁，等待
            self._thread_lock.acquire()

        try:
            start_time = time.time()

            while True:
                try:
                    # 确保锁文件所在目录存在
                    lock_dir = os.path.dirname(self.lock_file_path)
                    if lock_dir and not os.path.exists(lock_dir):
                        os.makedirs(lock_dir, exist_ok=True)

                    # 打开或创建锁文件
                    self.lock_file_handle = os.open(
                        self.lock_file_path,
                        os.O_RDWR | os.O_CREAT | os.O_TRUNC
                    )

                    # 尝试获取文件锁
                    if PLATFORM == 'windows':
                        self._lock_windows()
                    else:
                        self._lock_unix()

                    self._locked = True
                    logger.debug(f"Lock acquired for {self.file_path}")
                    return True

                except (IOError, OSError) as e:
                    # 锁被占用，检查是否超时
                    if self.lock_file_handle is not None:
                        os.close(self.lock_file_handle)
                        self.lock_file_handle = None

                    elapsed = time.time() - start_time
                    if self.timeout is not None and elapsed >= self.timeout:
                        self._thread_lock.release()
                        raise TimeoutError(
                            f"Failed to acquire lock for {self.file_path} "
                            f"after {elapsed:.1f} seconds"
                        )

                    # 等待后重试
                    logger.debug(f"Lock busy for {self.file_path}, retrying...")
                    time.sleep(self.retry_interval)

        except Exception:
            # 发生异常，释放线程锁
            if self._thread_lock.locked():
                self._thread_lock.release()
            raise

    def _lock_windows(self):
        """Windows平台的锁实现（使用msvcrt）"""
        if self.exclusive:
            # 独占锁
            msvcrt.locking(self.lock_file_handle, msvcrt.LK_NBLCK, 1)
        else:
            # Windows的msvcrt不直接支持共享锁，这里简化处理
            # 对于读锁也使用独占锁，但超时时间较短
            msvcrt.locking(self.lock_file_handle, msvcrt.LK_NBLCK, 1)

    def _lock_unix(self):
        """Unix/Linux/Mac平台的锁实现（使用fcntl）"""
        if self.exclusive:
            # 独占锁
            fcntl.flock(self.lock_file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            # 共享锁
            fcntl.flock(self.lock_file_handle, fcntl.LOCK_SH | fcntl.LOCK_NB)

    def release(self):
        """
        释放文件锁
        """
        if not self._locked:
            logger.warning(f"Attempting to release unlocked file: {self.file_path}")
            return

        try:
            # 释放文件锁
            if self.lock_file_handle is not None:
                if PLATFORM == 'windows':
                    msvcrt.locking(self.lock_file_handle, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self.lock_file_handle, fcntl.LOCK_UN)

                os.close(self.lock_file_handle)
                self.lock_file_handle = None

            # 尝试删除锁文件（可能失败，不影响功能）
            try:
                if os.path.exists(self.lock_file_path):
                    os.remove(self.lock_file_path)
            except Exception as e:
                logger.debug(f"Failed to remove lock file {self.lock_file_path}: {e}")

            self._locked = False
            logger.debug(f"Lock released for {self.file_path}")

        finally:
            # 释放线程锁
            if self._thread_lock.locked():
                self._thread_lock.release()

    def __enter__(self):
        """上下文管理器入口"""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.release()
        return False  # 不抑制异常

    def __del__(self):
        """析构函数：确保锁被释放"""
        if self._locked:
            logger.warning(f"FileLock for {self.file_path} was not explicitly released")
            try:
                self.release()
            except Exception as e:
                logger.error(f"Error releasing lock in destructor: {e}")


class FileLockManager:
    """
    文件锁管理器（全局单例）

    提供全局的文件锁管理，避免同一进程内重复创建锁
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化锁管理器"""
        if self._initialized:
            return

        # 存储当前活跃的锁（文件路径 -> FileLock对象）
        self._active_locks = {}
        self._locks_lock = threading.Lock()

        self._initialized = True
        logger.debug("FileLockManager initialized")

    def get_lock(
        self,
        file_path: str,
        exclusive: bool = True,
        timeout: float = 60.0
    ) -> FileLock:
        """
        获取文件锁（如果已存在则返回现有锁）

        Args:
            file_path: 文件路径
            exclusive: 是否为独占锁
            timeout: 超时时间

        Returns:
            FileLock: 文件锁对象
        """
        file_path = str(Path(file_path).resolve())

        with self._locks_lock:
            if file_path not in self._active_locks:
                self._active_locks[file_path] = FileLock(
                    file_path,
                    exclusive=exclusive,
                    timeout=timeout
                )
            return self._active_locks[file_path]

    def remove_lock(self, file_path: str):
        """
        移除文件锁记录

        Args:
            file_path: 文件路径
        """
        file_path = str(Path(file_path).resolve())

        with self._locks_lock:
            if file_path in self._active_locks:
                del self._active_locks[file_path]
                logger.debug(f"Removed lock record for {file_path}")
