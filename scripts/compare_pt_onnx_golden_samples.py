import argparse
import sys
from pathlib import Path


AI_PROCESSOR_ROOT = Path(__file__).resolve().parents[1]
if str(AI_PROCESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_PROCESSOR_ROOT))

from analyzer.onnx_detector import ONNXDetector  # noqa: E402
from analyzer.yolo_tracker import YOLOTracker  # noqa: E402
from validate_golden_samples import compare_events, load_annotation, run_detection  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PT and ONNX YOLO-only events on golden samples.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--pt-model", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples_dir = Path(args.samples)
    pt_detector = YOLOTracker(args.pt_model, device=args.device)
    onnx_detector = ONNXDetector(args.onnx_model, device=args.device)
    failures = []

    for annotation_path in sorted(samples_dir.glob("*.json")):
        sample_name = annotation_path.stem
        video_path = annotation_path.with_suffix(".mp4")
        if not video_path.exists():
            failures.append(f"{sample_name}: missing video {video_path}")
            continue

        annotation = load_annotation(annotation_path)
        fps = float(annotation["sourceVideoFps"])
        tolerance = int(round(fps * 1.0))
        print(f"{sample_name}: sourceVideoFps={fps:g}, toleranceFrames={tolerance}")

        pt_events = run_detection(pt_detector, video_path, fps, args.confidence, args.iou)
        onnx_events = run_detection(onnx_detector, video_path, fps, args.confidence, args.iou)
        sample_failures = compare_events(sample_name, pt_events, onnx_events, tolerance)
        if sample_failures:
            failures.extend(sample_failures)
        else:
            print(f"{sample_name}: OK (pt={len(pt_events)} events, onnx={len(onnx_events)} events)")

    if failures:
        print("\nPT/ONNX golden comparison failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nPT/ONNX golden comparison passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
