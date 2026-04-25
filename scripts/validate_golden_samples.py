import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2


AI_PROCESSOR_ROOT = Path(__file__).resolve().parents[1]
if str(AI_PROCESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_PROCESSOR_ROOT))

from analyzer.anomaly_event_generator import AnomalyEventGenerator  # noqa: E402
from analyzer.onnx_detector import ONNXDetector  # noqa: E402
from analyzer.yolo_tracker import YOLOTracker  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO-only output against golden samples.")
    parser.add_argument("--samples", required=True, help="Directory containing sample_*.mp4 and sample_*.json")
    parser.add_argument("--model", required=True, help="Path to best.pt or best.onnx")
    parser.add_argument("--backend", choices=["pt", "onnx"], default="pt")
    parser.add_argument("--device", default="", help="Torch device override")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def load_annotation(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    fps = data.get("sourceVideoFps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"{path} 缺少合法 sourceVideoFps")
    if not isinstance(data.get("expectedEvents"), list):
        raise ValueError(f"{path} 缺少 expectedEvents")
    return data


def create_detector(args):
    if args.backend == "onnx":
        return ONNXDetector(args.model, device=args.device)
    return YOLOTracker(args.model, device=args.device)


def run_detection(detector, video_path: Path, fps: float, confidence: float, iou: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_detections: List[List[Dict[str, Any]]] = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_detections.append(detector.detect_frame(frame, conf=confidence, iou=iou))

    cap.release()
    generator = AnomalyEventGenerator(fps=fps, total_frames=total_frames)
    return generator.generate_events(frame_detections)


def group_by_type(events: List[Dict[str, Any]]):
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["eventType"]].append(event)
    for event_type in grouped:
        grouped[event_type].sort(key=lambda event: (event["startFrame"], event["endFrame"]))
    return grouped


def compare_events(sample_name: str, expected: List[Dict[str, Any]], actual: List[Dict[str, Any]], tolerance: int):
    failures = []
    expected_by_type = group_by_type(expected)
    actual_by_type = group_by_type(actual)
    event_types = sorted(set(expected_by_type) | set(actual_by_type))

    for event_type in event_types:
        expected_events = expected_by_type.get(event_type, [])
        actual_events = actual_by_type.get(event_type, [])
        if len(expected_events) != len(actual_events):
            failures.append(
                f"{sample_name} {event_type}: expected {len(expected_events)} events, got {len(actual_events)}"
            )
            continue

        for index, (expected_event, actual_event) in enumerate(zip(expected_events, actual_events), start=1):
            start_delta = abs(int(expected_event["startFrame"]) - int(actual_event["startFrame"]))
            end_delta = abs(int(expected_event["endFrame"]) - int(actual_event["endFrame"]))
            if start_delta > tolerance or end_delta > tolerance:
                failures.append(
                    f"{sample_name} {event_type} #{index}: "
                    f"expected frames {expected_event['startFrame']}-{expected_event['endFrame']}, "
                    f"got {actual_event['startFrame']}-{actual_event['endFrame']}, "
                    f"delta start/end={start_delta}/{end_delta}, tolerance={tolerance}"
                )

    return failures


def main() -> int:
    args = parse_args()
    samples_dir = Path(args.samples)
    detector = create_detector(args)
    all_failures = []

    for annotation_path in sorted(samples_dir.glob("*.json")):
        sample_name = annotation_path.stem
        video_path = annotation_path.with_suffix(".mp4")
        if not video_path.exists():
            all_failures.append(f"{sample_name}: missing video {video_path}")
            continue

        annotation = load_annotation(annotation_path)
        fps = float(annotation["sourceVideoFps"])
        tolerance = int(round(fps * 1.0))
        print(f"{sample_name}: sourceVideoFps={fps:g}, toleranceFrames={tolerance}")
        actual_events = run_detection(detector, video_path, fps, args.confidence, args.iou)
        failures = compare_events(sample_name, annotation["expectedEvents"], actual_events, tolerance)
        if failures:
            all_failures.extend(failures)
        else:
            print(f"{sample_name}: OK ({len(actual_events)} events)")

    if all_failures:
        print("\nGolden sample validation failed:", file=sys.stderr)
        for failure in all_failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nGolden sample validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
