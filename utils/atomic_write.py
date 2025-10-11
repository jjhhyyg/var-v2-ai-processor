"""
原子性文件写入工具模块
提供原子性的文件写入操作，防止写入过程中的数据损坏
"""
import os
import json
import uuid
import shutil
import logging
from pathlib import Path
from typing import Any, Optional
from utils.file_lock import FileLock

logger = logging.getLogger(__name__)


def atomic_write_json(
    file_path: str,
    data: Any,
    indent: Optional[int] = 2,
    ensure_ascii: bool = False,
    use_lock: bool = True,
    lock_timeout: float = 60.0
) -> bool:
    """
    原子性地写入JSON文件

    实现原理:
    1. 先写入临时文件
    2. 验证临时文件完整性
    3. 使用文件锁保护
    4. 原子性地替换目标文件

    Args:
        file_path: 目标文件路径
        data: 要写入的数据（可序列化为JSON）
        indent: JSON缩进空格数（None表示压缩）
        ensure_ascii: 是否确保ASCII编码（False支持Unicode）
        use_lock: 是否使用文件锁
        lock_timeout: 文件锁超时时间（秒）

    Returns:
        bool: 是否成功写入

    Raises:
        IOError: 文件写入失败
        ValueError: 数据无法序列化为JSON
        TimeoutError: 无法获取文件锁
    """
    file_path = str(file_path)

    # 确保目标目录存在
    target_dir = os.path.dirname(file_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # 生成临时文件路径
    temp_path = f"{file_path}.tmp.{uuid.uuid4().hex}"

    # 获取文件锁（如果启用）
    file_lock = None
    if use_lock:
        try:
            file_lock = FileLock(file_path, exclusive=True, timeout=lock_timeout)
            file_lock.acquire()
            logger.debug(f"File lock acquired for {file_path}")
        except TimeoutError:
            logger.error(f"Failed to acquire file lock for {file_path}")
            raise
        except Exception as e:
            logger.warning(f"Failed to create file lock: {e}, continuing without lock")
            file_lock = None

    try:
        # 步骤1: 写入临时文件
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
            logger.debug(f"Data written to temporary file: {temp_path}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize data to JSON: {e}")
        except IOError as e:
            raise IOError(f"Failed to write to temporary file {temp_path}: {e}")

        # 步骤2: 验证临时文件
        if not os.path.exists(temp_path):
            raise IOError(f"Temporary file not created: {temp_path}")

        temp_size = os.path.getsize(temp_path)
        if temp_size == 0:
            raise IOError(f"Temporary file is empty: {temp_path}")

        # 尝试读取验证JSON格式
        try:
            with open(temp_path, 'r', encoding='utf-8') as f:
                json.load(f)
            logger.debug(f"Temporary file validated: {temp_path} ({temp_size} bytes)")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in temporary file: {e}")

        # 步骤3: 原子性替换
        # 如果目标文件已存在，先备份
        backup_path = None
        if os.path.exists(file_path):
            backup_path = f"{file_path}.backup.{uuid.uuid4().hex}"
            try:
                shutil.copy2(file_path, backup_path)
                logger.debug(f"Backup created: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
                backup_path = None

        try:
            # 使用 os.replace() 实现原子性替换（POSIX标准保证原子性）
            # Windows上也支持，但需要Python 3.3+
            os.replace(temp_path, file_path)
            logger.info(f"JSON file written atomically: {file_path} ({temp_size} bytes)")

            # 删除备份
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                    logger.debug(f"Backup removed: {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove backup: {e}")

            return True

        except Exception as e:
            # 替换失败，尝试恢复备份
            logger.error(f"Failed to replace target file: {e}")
            if backup_path and os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, file_path)
                    logger.info(f"Backup restored: {file_path}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {restore_error}")
            raise IOError(f"Failed to write to {file_path}: {e}")

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

        # 释放文件锁
        if file_lock:
            try:
                file_lock.release()
                logger.debug(f"File lock released for {file_path}")
            except Exception as e:
                logger.warning(f"Failed to release file lock: {e}")


def atomic_write_text(
    file_path: str,
    content: str,
    encoding: str = 'utf-8',
    use_lock: bool = True,
    lock_timeout: float = 60.0
) -> bool:
    """
    原子性地写入文本文件

    Args:
        file_path: 目标文件路径
        content: 要写入的文本内容
        encoding: 文件编码
        use_lock: 是否使用文件锁
        lock_timeout: 文件锁超时时间（秒）

    Returns:
        bool: 是否成功写入

    Raises:
        IOError: 文件写入失败
        TimeoutError: 无法获取文件锁
    """
    file_path = str(file_path)

    # 确保目标目录存在
    target_dir = os.path.dirname(file_path)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # 生成临时文件路径
    temp_path = f"{file_path}.tmp.{uuid.uuid4().hex}"

    # 获取文件锁（如果启用）
    file_lock = None
    if use_lock:
        try:
            file_lock = FileLock(file_path, exclusive=True, timeout=lock_timeout)
            file_lock.acquire()
        except TimeoutError:
            logger.error(f"Failed to acquire file lock for {file_path}")
            raise
        except Exception as e:
            logger.warning(f"Failed to create file lock: {e}, continuing without lock")
            file_lock = None

    try:
        # 写入临时文件
        try:
            with open(temp_path, 'w', encoding=encoding) as f:
                f.write(content)
        except IOError as e:
            raise IOError(f"Failed to write to temporary file {temp_path}: {e}")

        # 验证临时文件
        if not os.path.exists(temp_path):
            raise IOError(f"Temporary file not created: {temp_path}")

        temp_size = os.path.getsize(temp_path)
        if temp_size == 0 and len(content) > 0:
            raise IOError(f"Temporary file is empty but content is not: {temp_path}")

        # 原子性替换
        try:
            os.replace(temp_path, file_path)
            logger.info(f"Text file written atomically: {file_path} ({temp_size} bytes)")
            return True
        except Exception as e:
            raise IOError(f"Failed to write to {file_path}: {e}")

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_path}: {e}")

        # 释放文件锁
        if file_lock:
            try:
                file_lock.release()
            except Exception as e:
                logger.warning(f"Failed to release file lock: {e}")


def safe_read_json(
    file_path: str,
    use_lock: bool = True,
    lock_timeout: float = 30.0,
    default: Any = None
) -> Any:
    """
    安全地读取JSON文件（带文件锁）

    Args:
        file_path: 文件路径
        use_lock: 是否使用文件锁
        lock_timeout: 文件锁超时时间（秒）
        default: 读取失败时的默认返回值

    Returns:
        解析后的JSON数据，失败时返回default

    Raises:
        FileNotFoundError: 文件不存在且default=None
        ValueError: JSON格式错误且default=None
        TimeoutError: 无法获取文件锁
    """
    file_path = str(file_path)

    if not os.path.exists(file_path):
        if default is None:
            raise FileNotFoundError(f"File not found: {file_path}")
        return default

    # 获取共享锁（允许多个读取者）
    file_lock = None
    if use_lock:
        try:
            file_lock = FileLock(file_path, exclusive=False, timeout=lock_timeout)
            file_lock.acquire()
        except TimeoutError:
            logger.error(f"Failed to acquire file lock for {file_path}")
            raise
        except Exception as e:
            logger.warning(f"Failed to create file lock: {e}, continuing without lock")
            file_lock = None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"JSON file read successfully: {file_path}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        if default is None:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        return default

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        if default is None:
            raise
        return default

    finally:
        # 释放文件锁
        if file_lock:
            try:
                file_lock.release()
            except Exception as e:
                logger.warning(f"Failed to release file lock: {e}")
