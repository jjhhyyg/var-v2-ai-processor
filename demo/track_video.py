#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO多目标追踪脚本
使用训练好的模型对视频进行目标检测和追踪
"""

import argparse
import csv
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np

def print_verbose(message: str, level: int, verbose: int) -> None:
    """
    根据verbose级别控制输出
    
    参数:
        message: 要输出的消息
        level: 消息级别 (0: 总是显示, 1: 基本信息, 2: 详细信息)
        verbose: 当前verbose设置
    """
    if verbose >= level:
        print(message)

def track_video(
    input_video: str,
    output_video: str = "",
    output_csv: str = "",
    model_path: str = "weights/best.pt",
    conf: float = 0.5,
    iou: float = 0.45,
    tracker: str = "bytetrack.yaml",
    device: str = "",
    show: bool = False,
    save: bool = True,
    save_csv: bool = True,
    verbose: int = 1,
):
    """
    对视频执行YOLO多目标追踪

    参数:
        input_video: 输入视频路径
        output_video: 输出视频路径（默认为输入文件名_tracked.mp4）
        output_csv: 输出CSV文件路径（默认为输入文件名_tracks.csv）
        model_path: YOLO模型权重路径
        conf: 置信度阈值 (默认: 0.5)
        iou: NMS的IoU阈值 (默认: 0.45)
        tracker: 追踪器配置文件 (默认: bytetrack.yaml)
        device: 运行设备 (cpu/cuda/cuda:0/mps等，默认自动选择)
        show: 是否实时显示追踪结果
        save: 是否保存追踪结果视频
        save_csv: 是否保存追踪结果为CSV文件
        verbose: 输出详细程度 (0: 静默, 1: 基本信息, 2: 详细信息)
    """

    # 检查输入视频是否存在
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"输入视频不存在: {input_video}")

    # 检查模型文件是否存在
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 设置输出路径
    if output_video.strip() == "":
        output_video = str(input_path.parent / f"{input_path.stem}_tracked{input_path.suffix}")

    if output_csv.strip() == "":
        output_csv = str(input_path.parent / f"{input_path.stem}_tracks.csv")

    print_verbose(f"正在加载模型: {model_path}", 1, verbose)
    model = YOLO(model_path)

    # 设置设备
    if device.strip():
        print_verbose(f"使用设备: {device}", 1, verbose)
    else:
        print_verbose(f"使用设备: 自动选择", 1, verbose)

    print_verbose(f"开始追踪视频: {input_video}", 1, verbose)
    print_verbose(f"追踪器: {tracker}", 2, verbose)
    print_verbose(f"置信度阈值: {conf}", 2, verbose)
    print_verbose(f"IoU阈值: {iou}", 2, verbose)
    print_verbose(f"输出视频: {output_video}", 2, verbose)
    print_verbose(f"输出CSV: {output_csv}", 2, verbose)

    # 执行追踪
    track_params = {
        "source": input_video,
        "conf": conf,
        "iou": iou,
        "tracker": tracker,
        "show": show,
        "save": save,
        "project": str(Path(output_video).parent),
        "name": Path(output_video).stem,
        "stream": True,  # 使用流模式以便逐帧处理
        "exist_ok": True,  # 允许覆盖已存在的文件
        "verbose": verbose >= 2,  # 详细模式下显示更多信息
    }

    # 如果指定了device，添加到参数中
    if device.strip():
        track_params["device"] = device

    results = model.track(**track_params)

    tracking_data = []
    # 保存追踪结果到CSV
    if save_csv:
        print_verbose(f"正在保存追踪结果到CSV...", 1, verbose)

        for frame_idx, result in enumerate(results):
            # 检查是否有追踪框
            if result.boxes is not None and result.boxes.id is not None:
                boxes: np.ndarray = result.boxes.xyxy.numpy() if type(result.boxes.xyxy) is torch.Tensor else np.ndarray(result.boxes.xyxy)  # 边界框坐标
                track_ids: np.ndarray = result.boxes.id.numpy() if type(result.boxes.id) is torch.Tensor else np.ndarray(result.boxes.id)  # 追踪ID
                confidences: np.ndarray = result.boxes.conf.numpy() if type(result.boxes.conf) is torch.Tensor else np.ndarray(result.boxes.conf)  # 置信度
                class_ids: np.ndarray = result.boxes.cls.numpy() if type(result.boxes.cls) is torch.Tensor else np.ndarray(result.boxes.cls)  # 类别ID

                # 获取类别名称
                class_names = [result.names[cls_id] for cls_id in class_ids]

                # 遍历每个检测框
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1

                    tracking_data.append({
                        'frame': frame_idx,
                        'track_id': track_ids[i],
                        'class_id': class_ids[i],
                        'class_name': class_names[i],
                        'confidence': confidences[i],
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                    })
                
                # 在详细模式下显示每帧的处理进度
                if verbose >= 2 and len(boxes) > 0:
                    print_verbose(f"帧 {frame_idx}: 检测到 {len(boxes)} 个目标", 2, verbose)

        # 写入CSV文件
        if tracking_data:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['frame', 'track_id', 'class_id', 'class_name', 'confidence',
                             'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'width', 'height']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(tracking_data)

            print_verbose(f"CSV文件已保存至: {output_csv}", 1, verbose)
            unique_tracks = len(set(d['track_id'] for d in tracking_data))
            print_verbose(f"共检测到 {len(tracking_data)} 个追踪框，{unique_tracks} 个不同的追踪ID", 1, verbose)
        else:
            print_verbose(f"警告: 未检测到任何目标，未生成CSV文件", 0, verbose)

    print_verbose(f"追踪完成！", 0, verbose)
    if save:
        print_verbose(f"视频已保存至: {output_video}", 1, verbose)

    return tracking_data if save_csv else None


def main():
    parser = argparse.ArgumentParser(
        description="YOLO多目标追踪脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python track_video.py input.mp4
  python track_video.py input.mp4 -o output.mp4 -c tracks.csv
  python track_video.py input.mp4 --conf 0.6 --iou 0.5
  python track_video.py input.mp4 --tracker bytetrack.yaml --show
  python track_video.py input.mp4 --device cuda:0  # 使用第一块GPU
  python track_video.py input.mp4 --device mps     # 使用Apple Silicon加速
  python track_video.py input.mp4 --device cpu     # 使用CPU
  python track_video.py input.mp4 --no-csv         # 不保存CSV文件
  python track_video.py input.mp4 -v 0             # 静默模式
  python track_video.py input.mp4 -v 1             # 基本信息（默认）
  python track_video.py input.mp4 -v 2             # 详细信息
        """
    )

    parser.add_argument(
        "input",
        type=str,
        help="输入视频路径"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="",
        help="输出视频路径 (默认: 输入文件名_tracked.mp4)"
    )

    parser.add_argument(
        "-c", "--csv",
        type=str,
        default="",
        help="输出CSV文件路径 (默认: 输入文件名_tracks.csv)"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="weights/best.pt",
        help="YOLO模型权重路径 (默认: weights/best.pt)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="置信度阈值 (默认: 0.5)"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS的IoU阈值 (默认: 0.45)"
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="追踪器配置 (默认: bytetrack.yaml)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="运行设备 (cpu/cuda/cuda:0/cuda:1/mps等，默认自动选择)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="实时显示追踪结果"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存追踪结果视频"
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="不保存追踪结果CSV文件"
    )

    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="输出详细程度 (0: 静默模式，仅显示错误; 1: 基本信息; 2: 详细信息)"
    )

    args = parser.parse_args()

    try:
        track_video(
            input_video=args.input,
            output_video=args.output,
            output_csv=args.csv,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
            device=args.device,
            show=args.show,
            save=not args.no_save,
            save_csv=not args.no_csv,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
