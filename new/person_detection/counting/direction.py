"""Report which way a person crosses two zones (off unless ENABLE_DIRECTION).

Zones are stored as 0..1 fractions of the frame (x / width, y / height), so
they stay on the same spot of the picture whatever the camera resolution.
Hover the mouse over the window (PRINT_MOUSE_POSITION) to read the fractions
for new zones.
"""

import cv2
import numpy as np

from ..vision.detector import Detection
from ..core.geometry import Frame, to_pixels

# The original pixel zones were drawn on an 889x500 frame (a 16:9 camera shown
# in the old 1020x500 window); dividing by that size gives these fractions.
DEFAULT_AREA = ((0.4736, 0.76), (0.5006, 0.798), (0.7447, 0.636), (0.7233, 0.598))
DEFAULT_AREA1 = ((0.4353, 0.70), (0.4634, 0.744), (0.7199, 0.586), (0.6952, 0.548))


class DirectionTracker:
    """Reports a person's direction after they cross both zones in order."""

    def __init__(self, area=DEFAULT_AREA, area1=DEFAULT_AREA1):
        self.area = area
        self.area1 = area1
        self.entered_area = set()  # track IDs seen inside AREA
        self.entered_area1 = set()  # track IDs seen inside AREA1
        self.reported = set()  # (track_id, direction) already reported

    @staticmethod
    def _inside(polygon: np.ndarray, point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def update(self, detection: Detection, width: int, height: int) -> str | None:
        """Return "Up"/"Down" the first time a person completes a crossing."""
        _, _, x2, y2 = detection.box
        anchor = (float(x2), float(y2))
        track_id = detection.track_id

        in_area = self._inside(to_pixels(self.area, width, height), anchor)
        in_area1 = self._inside(to_pixels(self.area1, width, height), anchor)

        direction = None
        if in_area1 and track_id in self.entered_area:
            direction = "Up"
        elif in_area and track_id in self.entered_area1:
            direction = "Down"

        if in_area:
            self.entered_area.add(track_id)
        if in_area1:
            self.entered_area1.add(track_id)

        if direction is None or (track_id, direction) in self.reported:
            return None
        self.reported.add((track_id, direction))
        return direction

    def forget(self, track_id: int) -> None:
        self.entered_area.discard(track_id)
        self.entered_area1.discard(track_id)
        self.reported = {r for r in self.reported if r[0] != track_id}

    def draw_zones(self, frame: Frame) -> None:
        height, width = frame.shape[:2]
        cv2.polylines(frame, [to_pixels(self.area, width, height)], True, (255, 0, 0), 2)
        cv2.polylines(frame, [to_pixels(self.area1, width, height)], True, (0, 255, 0), 2)
