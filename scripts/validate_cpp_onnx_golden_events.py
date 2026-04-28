import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


AI_PROCESSOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_PROCESSOR_ROOT.parent
if str(AI_PROCESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_PROCESSOR_ROOT))

from analyzer.anomaly_event_generator import AnomalyEventGenerator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Validate C++ ONNX analyzer detections against golden sample events.")
    parser.add_argument("--samples", required=True, help="Directory containing sample_*.mp4 and sample_*.json")
    parser.add_argument("--onnx-model", required=True, help="Path to best.onnx")
    parser.add_argument(
        "--cpp-analyzer",
        default=str(REPO_ROOT / "frontend" / "src-tauri" / "resources" / "runtime" / "windows-x64" / "tools" / "var-video-analyzer.exe"),
        help="Path to var-video-analyzer.exe",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos for quick checks")
    parser.add_argument("--keep-temp", action="store_true")
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


def group_by_type(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["eventType"]].append(event)
    for event_type in grouped:
        grouped[event_type].sort(key=lambda event: (event["startFrame"], event["endFrame"]))
    return grouped


def compare_events(
    sample_name: str,
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
    tolerance: int,
) -> List[str]:
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


def normalize_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    bbox = detection.get("bbox") or detection.get("box")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid detection bbox: {detection}")

    class_id = detection.get("class_id", detection.get("classId"))
    class_name = detection.get("class_name", detection.get("className"))
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return {
        "class_id": int(class_id),
        "class_name": str(class_name),
        "bbox": [x1, y1, x2, y2],
        "center_x": float(detection.get("center_x", (x1 + x2) / 2.0)),
        "center_y": float(detection.get("center_y", (y1 + y2) / 2.0)),
        "width": float(detection.get("width", x2 - x1)),
        "height": float(detection.get("height", y2 - y1)),
        "confidence": float(detection["confidence"]),
    }


def run_cpp_detections(
    video_path: Path,
    model_path: Path,
    analyzer_path: Path,
    confidence: float,
    iou: float,
    output_path: Path,
) -> List[List[Dict[str, Any]]]:
    command = [
        str(analyzer_path),
        "--input",
        str(video_path),
        "--model",
        str(model_path),
        "--output-detections",
        str(output_path),
        "--conf",
        str(confidence),
        "--iou",
        str(iou),
        "--progress-interval",
        "1000000",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ analyzer failed for {video_path.name}: exit={result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return [[normalize_detection(item) for item in frame] for frame in data]


def main() -> int:
    args = parse_args()
    samples_dir = Path(args.samples)
    model_path = Path(args.onnx_model)
    analyzer_path = Path(args.cpp_analyzer)

    if not analyzer_path.exists():
        raise FileNotFoundError(f"Missing C++ analyzer: {analyzer_path}")

    sample_paths = sorted(samples_dir.glob("*.json"))
    if args.limit > 0:
        sample_paths = sample_paths[: args.limit]

    failures: List[str] = []
    with tempfile.TemporaryDirectory(prefix="var-cpp-onnx-events-") as temp_dir:
        temp_root = Path(temp_dir)
        for annotation_path in sample_paths:
            sample_name = annotation_path.stem
            video_path = annotation_path.with_suffix(".mp4")
            if not video_path.exists():
                failures.append(f"{sample_name}: missing video {video_path}")
                continue

            annotation = load_annotation(annotation_path)
            fps = float(annotation["sourceVideoFps"])
            tolerance = int(round(fps * 1.0))
            cpp_output_path = temp_root / f"{sample_name}.cpp.detections.json"

            print(f"{sample_name}: running C++ var-video-analyzer")
            frame_detections = run_cpp_detections(
                video_path,
                model_path,
                analyzer_path,
                args.confidence,
                args.iou,
                cpp_output_path,
            )

            generator = AnomalyEventGenerator(fps=fps, total_frames=len(frame_detections))
            actual_events = generator.generate_events(frame_detections)
            sample_failures = compare_events(sample_name, annotation["expectedEvents"], actual_events, tolerance)
            if sample_failures:
                failures.extend(sample_failures)
            else:
                print(
                    f"{sample_name}: OK "
                    f"(frames={len(frame_detections)}, detections={sum(len(frame) for frame in frame_detections)}, "
                    f"events={len(actual_events)}, toleranceFrames={tolerance})"
                )

            if args.keep_temp:
                kept_path = Path.cwd() / f"{sample_name}.cpp.detections.json"
                kept_path.write_text(cpp_output_path.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"{sample_name}: kept {kept_path}")

    if failures:
        print("\nC++ ONNX golden event validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nC++ ONNX golden event validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
