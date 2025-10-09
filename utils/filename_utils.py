"""
文件名工具模块
提供带时间戳的文件名生成和更新功能
"""
import os
import re
from datetime import datetime
from pathlib import Path


def generate_timestamp():
    """生成当前时间戳字符串（格式：yyyyMMdd_HHmmss）"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_timestamp_from_filename(filename: str) -> tuple[str, str | None]:
    """
    从文件名中提取时间戳和基础名称
    
    Args:
        filename: 文件名（可以包含或不包含扩展名）
    
    Returns:
        tuple[str, str | None]: (基础名称（不含时间戳和扩展名）, 时间戳（如果存在）)
    
    示例:
        "video_20240101_120000.mp4" -> ("video", "20240101_120000")
        "video.mp4" -> ("video", None)
        "video_preprocessed_20240101_120000.mp4" -> ("video_preprocessed", "20240101_120000")
    """
    # 移除扩展名
    name_without_ext = Path(filename).stem
    
    # 匹配时间戳模式：_yyyyMMdd_HHmmss（可能在末尾或者后面还有其他内容）
    timestamp_pattern = r'_(\d{8}_\d{6})(?:_|$)'
    match = re.search(timestamp_pattern, name_without_ext)
    
    if match:
        timestamp = match.group(1)
        base_name = name_without_ext[:match.start()]
        # 如果时间戳后面还有内容，也加到base_name中
        suffix_start = match.end()
        if suffix_start < len(name_without_ext):
            suffix = name_without_ext[suffix_start:]
            base_name = base_name + '_' + suffix
        return base_name, timestamp
    else:
        return name_without_ext, None


def extract_base_name(filename: str, remove_suffixes: list[str] | None = None) -> str:
    """
    提取文件的原始基础名称（去掉时间戳和指定的后缀）
    
    Args:
        filename: 文件名（可以包含或不包含扩展名）
        remove_suffixes: 要移除的后缀列表（如 ['_preprocessed', '_result']）
    
    Returns:
        str: 原始基础名称
    
    示例:
        "video_20240101_120000.mp4" -> "video"
        "video_20240101_120000_preprocessed.mp4" -> "video" (如果 remove_suffixes=['_preprocessed'])
        "video_preprocessed_20240101_120000.mp4" -> "video" (如果 remove_suffixes=['_preprocessed'])
    """
    # 移除扩展名
    name_without_ext = Path(filename).stem
    
    # 移除时间戳
    base_name, _ = extract_timestamp_from_filename(name_without_ext)
    
    # 移除指定的后缀
    if remove_suffixes:
        for suffix in remove_suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
    
    return base_name


def generate_filename_with_timestamp(base_name: str, extension: str = ".mp4", 
                                    update_existing: bool = True) -> str:
    """
    生成带时间戳的文件名
    
    Args:
        base_name: 基础文件名（可能已包含时间戳）
        extension: 文件扩展名（包括点，如 .mp4）
        update_existing: 如果文件名已包含时间戳，是否更新为新时间戳
    
    Returns:
        str: 带时间戳的完整文件名
    
    示例:
        generate_filename_with_timestamp("video") -> "video_20240101_120000.mp4"
        generate_filename_with_timestamp("video_20240101_120000", update_existing=True) 
            -> "video_20240102_130000.mp4"
        generate_filename_with_timestamp("video_20240101_120000", update_existing=False) 
            -> "video_20240101_120000.mp4"
    """
    # 确保扩展名以点开头
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    # 提取基础名称和现有时间戳
    base, existing_timestamp = extract_timestamp_from_filename(base_name)
    
    # 决定是否需要添加/更新时间戳
    if existing_timestamp is None or update_existing:
        # 没有时间戳或需要更新时间戳
        new_timestamp = generate_timestamp()
        return f"{base}_{new_timestamp}{extension}"
    else:
        # 保留现有时间戳
        return f"{base}_{existing_timestamp}{extension}"


def add_or_update_timestamp(filepath: str, update_existing: bool = True) -> str:
    """
    为文件路径添加或更新时间戳
    
    Args:
        filepath: 完整的文件路径
        update_existing: 如果文件名已包含时间戳，是否更新为新时间戳
    
    Returns:
        str: 带时间戳的完整文件路径
    
    示例:
        add_or_update_timestamp("/path/to/video.mp4") 
            -> "/path/to/video_20240101_120000.mp4"
        add_or_update_timestamp("/path/to/video_20240101_120000.mp4", update_existing=True) 
            -> "/path/to/video_20240102_130000.mp4"
    """
    path = Path(filepath)
    directory = path.parent
    filename = path.name
    
    # 提取扩展名和基础名称
    extension = path.suffix
    base_name = path.stem
    
    # 生成新文件名
    new_filename = generate_filename_with_timestamp(base_name, extension, update_existing)
    
    # 返回完整路径
    return str(directory / new_filename)


if __name__ == "__main__":
    # 测试代码
    print("测试文件名时间戳工具:")
    print("=" * 50)
    
    # 测试提取时间戳
    print("\n1. 提取时间戳:")
    test_cases = [
        "video.mp4",
        "video_20240101_120000.mp4",
        "video_preprocessed.mp4",
        "video_preprocessed_20240101_120000.mp4"
    ]
    for case in test_cases:
        base, ts = extract_timestamp_from_filename(case)
        print(f"  {case} -> 基础名: '{base}', 时间戳: {ts}")
    
    # 测试生成文件名
    print("\n2. 生成带时间戳的文件名:")
    print(f"  video -> {generate_filename_with_timestamp('video')}")
    print(f"  video_preprocessed -> {generate_filename_with_timestamp('video_preprocessed')}")
    
    # 测试更新时间戳
    print("\n3. 更新现有时间戳:")
    old_name = "video_20240101_120000"
    print(f"  {old_name} (update=True) -> {generate_filename_with_timestamp(old_name, '.mp4', True)}")
    print(f"  {old_name} (update=False) -> {generate_filename_with_timestamp(old_name, '.mp4', False)}")
    
    # 测试完整路径
    print("\n4. 处理完整路径:")
    test_paths = [
        "/path/to/video.mp4",
        "/path/to/video_20240101_120000.mp4"
    ]
    for path in test_paths:
        print(f"  {path}")
        print(f"    -> (add) {add_or_update_timestamp(path, update_existing=False)}")
        print(f"    -> (update) {add_or_update_timestamp(path, update_existing=True)}")
