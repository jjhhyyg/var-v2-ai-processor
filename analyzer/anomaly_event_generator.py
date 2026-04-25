"""
YOLO-only anomaly event generation.

Phase 0 event semantics are detection driven: every frame-level YOLO hit in
one of the six defect classes becomes evidence for that defect class. Evidence
frames expand to a +/- 1 second closed interval, and only intervals of the same
class are merged.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from config import Config

logger = logging.getLogger(__name__)


DEFECT_CLASS_IDS = frozenset(Config.CLASS_NAMES.keys())


@dataclass
class _EventInterval:
    event_type: str
    start_frame: int
    end_frame: int
    max_confidence: float = 0.0
    evidence_frames: set[int] = field(default_factory=set)


class AnomalyEventGenerator:
    """Generate Phase 0 YOLO-only anomaly events from per-frame detections."""

    def __init__(self, fps: float = 30.0, total_frames: int = 0):
        if fps <= 0:
            raise ValueError("fps must be greater than 0")
        if total_frames < 0:
            raise ValueError("total_frames must not be negative")
        self.fps = float(fps)
        self.total_frames = int(total_frames)
        self.radius_frames = int(round(self.fps * 1.0))
        logger.info(
            "YOLO-only event generator initialized: fps=%s, total_frames=%s, radius=%s",
            self.fps,
            self.total_frames,
            self.radius_frames,
        )

    def generate_events(
        self,
        frame_detections: Iterable[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Generate events from a 0-based frame-indexed detection stream."""
        if self.total_frames <= 0:
            return []

        intervals_by_type: Dict[str, List[_EventInterval]] = {}

        for frame_index, detections in enumerate(frame_detections):
            if frame_index >= self.total_frames:
                break

            hits_by_type: Dict[str, float] = {}
            for detection in detections or []:
                class_id = self._detection_class_id(detection)
                if class_id not in DEFECT_CLASS_IDS:
                    continue

                event_type = self._event_type_for_class_id(class_id)
                confidence = float(detection.get("confidence", 0.0) or 0.0)
                hits_by_type[event_type] = max(hits_by_type.get(event_type, 0.0), confidence)

            for event_type, confidence in hits_by_type.items():
                start_frame = max(0, frame_index - self.radius_frames)
                end_frame = min(self.total_frames - 1, frame_index + self.radius_frames)
                intervals_by_type.setdefault(event_type, []).append(
                    _EventInterval(
                        event_type=event_type,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        max_confidence=confidence,
                        evidence_frames={frame_index},
                    )
                )

        events: List[Dict[str, Any]] = []
        for event_type in sorted(intervals_by_type):
            merged = self._merge_intervals(intervals_by_type[event_type])
            events.extend(self._interval_to_event(interval) for interval in merged)

        events.sort(key=lambda event: (event["startFrame"], event["eventType"], event["endFrame"]))
        logger.info("Generated %s YOLO-only anomaly events", len(events))
        return events

    def _merge_intervals(self, intervals: List[_EventInterval]) -> List[_EventInterval]:
        if not intervals:
            return []

        intervals.sort(key=lambda interval: (interval.start_frame, interval.end_frame))
        merged: List[_EventInterval] = [intervals[0]]

        for next_interval in intervals[1:]:
            current = merged[-1]
            if next_interval.start_frame <= current.end_frame + 1:
                current.end_frame = max(current.end_frame, next_interval.end_frame)
                current.max_confidence = max(current.max_confidence, next_interval.max_confidence)
                current.evidence_frames.update(next_interval.evidence_frames)
            else:
                merged.append(next_interval)

        return merged

    def _interval_to_event(self, interval: _EventInterval) -> Dict[str, Any]:
        evidence_frames = sorted(interval.evidence_frames)
        return {
            "eventType": interval.event_type,
            "startFrame": interval.start_frame,
            "endFrame": interval.end_frame,
            "startTime": interval.start_frame / self.fps,
            "endTime": interval.end_frame / self.fps,
            "metadata": {
                "defectClass": interval.event_type,
                "maxConfidence": round(interval.max_confidence, 6),
                "evidenceFrames": evidence_frames,
            },
        }

    @staticmethod
    def _detection_class_id(detection: Dict[str, Any]) -> int | None:
        raw_class_id = detection.get("class_id", detection.get("classId"))
        if raw_class_id is None:
            return None
        try:
            return int(raw_class_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _event_type_for_class_id(class_id: int) -> str:
        class_name = Config.CLASS_NAMES[class_id]
        return Config.CATEGORY_MAPPING[class_name]
