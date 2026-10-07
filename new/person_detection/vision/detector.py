"""YOLO person detection with tracking."""

from dataclasses import dataclass

from ultralytics import YOLO

from ..core.geometry import Box, Frame

PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    """One tracked person in a single frame."""

    box: Box
    track_id: int  # -1 until the tracker assigns one
    confidence: float

    @property
    def is_tracked(self) -> bool:
        return self.track_id >= 0


class PersonDetector:
    """Runs YOLO tracking on a frame and returns the people it found."""

    def __init__(self, model_path: str, conf: float):
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(f"Error loading YOLO model '{model_path}'") from exc
        self.conf = conf

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        coordinates = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [-1] * len(coordinates)
        )
        return [
            Detection(tuple(box), track_id, confidence)
            for box, track_id, confidence in zip(coordinates, track_ids, confidences)
        ]
