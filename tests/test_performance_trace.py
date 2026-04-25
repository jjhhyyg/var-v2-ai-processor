import sys
import unittest
from pathlib import Path


AI_PROCESSOR_ROOT = Path(__file__).resolve().parents[1]
if str(AI_PROCESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_PROCESSOR_ROOT))

from utils.performance_trace import PerformanceTrace  # noqa: E402


class PerformanceTraceTest(unittest.TestCase):
    def test_summary_aggregates_stage_samples(self):
        trace = PerformanceTrace()
        trace.record_elapsed_ns("videoRead", 1_000_000)
        trace.record_elapsed_ns("videoRead", 2_000_000)
        trace.record_elapsed_ns("videoRead", 3_000_000)
        trace.record_elapsed_ns("detectorTotal", 10_000_000)

        summary = trace.summary(total_measured_frames=3)
        stages = {stage["key"]: stage for stage in summary["stages"]}

        self.assertEqual(summary["schemaVersion"], 1)
        self.assertEqual(summary["totalMeasuredFrames"], 3)
        self.assertEqual(stages["videoRead"]["samples"], 3)
        self.assertEqual(stages["videoRead"]["totalMs"], 6.0)
        self.assertEqual(stages["videoRead"]["avgMs"], 2.0)
        self.assertEqual(stages["videoRead"]["p50Ms"], 2.0)
        self.assertEqual(stages["videoRead"]["p95Ms"], 3.0)
        self.assertEqual(stages["videoRead"]["maxMs"], 3.0)
        self.assertAlmostEqual(stages["videoRead"]["percentOfMeasuredMs"], 37.5)

    def test_summary_omits_unrecorded_stages(self):
        trace = PerformanceTrace()

        summary = trace.summary(total_measured_frames=0)

        self.assertEqual(summary["stages"], [])


if __name__ == "__main__":
    unittest.main()
