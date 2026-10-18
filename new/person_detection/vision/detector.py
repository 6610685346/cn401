"""YOLO person detection with tracking."""

from dataclasses import dataclass

from ultralytics import YOLO

from ..core.geometry import Box, Frame

PERSON_CLASS_ID = 0


# COCO keypoint indices used by the pose model
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
KEYPOINT_MIN_CONF = 0.5


@dataclass(frozen=True)
class Detection:
    """One tracked person in a single frame."""

    box: Box
    track_id: int  # -1 until the tracker assigns one
    confidence: float
    # Where the person touches the floor, in frame pixels: between the ankles
    # when the pose model sees them, else the bottom-centre of the box.
    foot: tuple[float, float] | None = None

    @property
    def foot_point(self) -> tuple[float, float]:
        if self.foot is not None:
            return self.foot
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, float(y2))

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
        feet = self._feet(results[0], len(coordinates))
        # The sole is a little below the ankle joint: nudge down, not past the box.
        feet = [None if f is None else (f[0], min(float(b[3]), f[1] + 0.03 * (b[3] - b[1])))
                for f, b in zip(feet, coordinates)]
        return [
            Detection(tuple(box), track_id, confidence, foot)
            for box, track_id, confidence, foot in zip(coordinates, track_ids, confidences, feet)
        ]

    @staticmethod
    def _feet(result, count: int) -> list:
        """Ankle midpoint per person (pose models only), else None."""
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.data is None or len(keypoints.data) != count:
            return [None] * count
        feet = []
        for person in keypoints.data.cpu().numpy():  # (17, 3): x, y, confidence
            if person.shape[0] <= RIGHT_ANKLE or person.shape[1] < 3:
                feet.append(None)
                continue
            ankles = [person[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if person[i][2] >= KEYPOINT_MIN_CONF]
            if ankles:
                feet.append((float(sum(a[0] for a in ankles) / len(ankles)),
                             float(max(a[1] for a in ankles))))
            else:
                feet.append(None)
        return feet
