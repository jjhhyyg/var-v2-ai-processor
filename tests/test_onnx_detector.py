import sys
import unittest
from pathlib import Path

import numpy as np


AI_PROCESSOR_ROOT = Path(__file__).resolve().parents[1]
if str(AI_PROCESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_PROCESSOR_ROOT))

from analyzer.onnx_detector import ONNXDetector  # noqa: E402


class ONNXDetectorPostprocessTest(unittest.TestCase):
    def make_detector(self):
        detector = ONNXDetector.__new__(ONNXDetector)
        detector.class_names = {
            0: "熔池未到边",
            1: "电极粘连物",
            2: "锭冠",
            3: "辉光",
            4: "边弧（侧弧）",
            5: "爬弧",
        }
        return detector

    def test_letterbox_matches_rect_960x576_to_640x1024(self):
        image = np.zeros((576, 960, 3), dtype=np.uint8)

        padded, scale, pad = ONNXDetector.letterbox(image, (640, 1024))

        self.assertEqual(padded.shape, (640, 1024, 3))
        self.assertAlmostEqual(scale, 1024 / 960)
        self.assertEqual(pad, (0, 13))

    def test_postprocess_maps_boxes_back_to_original_frame(self):
        detector = self.make_detector()
        output = np.zeros((1, 10, 1), dtype=np.float32)
        output[0, 0:4, 0] = [512, 333, 96, 96]
        output[0, 4 + 1, 0] = 0.9

        detections = detector._postprocess(
            output,
            original_shape=(576, 960),
            scale=1024 / 960,
            pad=(0, 13),
            conf_threshold=0.5,
            iou_threshold=0.45,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_id"], 1)
        self.assertEqual(detections[0]["class_name"], "电极粘连物")
        self.assertEqual(detections[0]["confidence"], 0.8999999761581421)
        self.assertAlmostEqual(detections[0]["bbox"][0], 435.0, places=3)
        self.assertAlmostEqual(detections[0]["bbox"][1], 255.0, places=3)
        self.assertAlmostEqual(detections[0]["bbox"][2], 525.0, places=3)
        self.assertAlmostEqual(detections[0]["bbox"][3], 345.0, places=3)

    def test_filters_by_confidence_and_clips_to_bounds(self):
        detector = self.make_detector()
        output = np.zeros((1, 10, 2), dtype=np.float32)
        output[0, 0:4, 0] = [5, 5, 30, 30]
        output[0, 4 + 2, 0] = 0.8
        output[0, 0:4, 1] = [500, 300, 40, 40]
        output[0, 4 + 2, 1] = 0.49

        detections = detector._postprocess(
            output,
            original_shape=(100, 100),
            scale=1.0,
            pad=(0, 0),
            conf_threshold=0.5,
            iou_threshold=0.45,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], [0.0, 0.0, 20.0, 20.0])

    def test_nms_is_class_aware(self):
        detector = self.make_detector()
        output = np.zeros((1, 10, 3), dtype=np.float32)
        output[0, 0:4, 0] = [50, 50, 40, 40]
        output[0, 4 + 1, 0] = 0.9
        output[0, 0:4, 1] = [52, 52, 40, 40]
        output[0, 4 + 1, 1] = 0.8
        output[0, 0:4, 2] = [52, 52, 40, 40]
        output[0, 4 + 2, 2] = 0.85

        detections = detector._postprocess(
            output,
            original_shape=(100, 100),
            scale=1.0,
            pad=(0, 0),
            conf_threshold=0.5,
            iou_threshold=0.45,
        )

        self.assertEqual(len(detections), 2)
        self.assertEqual([item["class_id"] for item in detections], [1, 2])


if __name__ == "__main__":
    unittest.main()
