import sys
import unittest
from pathlib import Path


AI_PROCESSOR_ROOT = Path(__file__).resolve().parents[1]
if str(AI_PROCESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_PROCESSOR_ROOT))

from analyzer.anomaly_event_generator import AnomalyEventGenerator  # noqa: E402


def detection(class_id: int, confidence: float = 0.8):
    return {
        "class_id": class_id,
        "class_name": f"class_{class_id}",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "confidence": confidence,
    }


class YoloOnlyEventGeneratorTest(unittest.TestCase):
    def test_clamps_event_window_to_video_bounds(self):
        generator = AnomalyEventGenerator(fps=25.0, total_frames=100)
        frame_detections = [[] for _ in range(100)]
        frame_detections[0] = [detection(1, 0.91)]

        events = generator.generate_events(frame_detections)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["eventType"], "ADHESION")
        self.assertEqual(events[0]["startFrame"], 0)
        self.assertEqual(events[0]["endFrame"], 25)
        self.assertEqual(events[0]["startTime"], 0)
        self.assertEqual(events[0]["endTime"], 1)

    def test_merges_only_same_class_overlapping_or_adjacent_intervals(self):
        generator = AnomalyEventGenerator(fps=10.0, total_frames=100)
        frame_detections = [[] for _ in range(100)]
        frame_detections[20] = [detection(1, 0.5)]
        frame_detections[41] = [detection(1, 0.8)]
        frame_detections[20].append(detection(2, 0.7))

        events = generator.generate_events(frame_detections)

        adhesion = [event for event in events if event["eventType"] == "ADHESION"]
        crown = [event for event in events if event["eventType"] == "CROWN"]
        self.assertEqual(len(adhesion), 1)
        self.assertEqual((adhesion[0]["startFrame"], adhesion[0]["endFrame"]), (10, 51))
        self.assertEqual(adhesion[0]["metadata"]["evidenceFrames"], [20, 41])
        self.assertEqual(adhesion[0]["metadata"]["maxConfidence"], 0.8)
        self.assertEqual(len(crown), 1)
        self.assertEqual((crown[0]["startFrame"], crown[0]["endFrame"]), (10, 30))

    def test_keeps_separate_same_class_intervals_when_gap_exists(self):
        generator = AnomalyEventGenerator(fps=10.0, total_frames=100)
        frame_detections = [[] for _ in range(100)]
        frame_detections[10] = [detection(5, 0.6)]
        frame_detections[32] = [detection(5, 0.7)]

        events = generator.generate_events(frame_detections)

        self.assertEqual(len(events), 2)
        self.assertEqual([event["eventType"] for event in events], ["CREEPING_ARC", "CREEPING_ARC"])
        self.assertEqual((events[0]["startFrame"], events[0]["endFrame"]), (0, 20))
        self.assertEqual((events[1]["startFrame"], events[1]["endFrame"]), (22, 42))

    def test_uses_max_confidence_once_per_frame_per_class(self):
        generator = AnomalyEventGenerator(fps=5.0, total_frames=20)
        frame_detections = [[] for _ in range(20)]
        frame_detections[8] = [detection(3, 0.4), detection(3, 0.93)]

        events = generator.generate_events(frame_detections)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["eventType"], "GLOW")
        self.assertEqual(events[0]["metadata"]["maxConfidence"], 0.93)
        self.assertEqual(events[0]["metadata"]["evidenceFrames"], [8])

    def test_ignores_unknown_classes(self):
        generator = AnomalyEventGenerator(fps=25.0, total_frames=10)
        frame_detections = [[detection(99, 0.99)] for _ in range(10)]

        self.assertEqual(generator.generate_events(frame_detections), [])


if __name__ == "__main__":
    unittest.main()
