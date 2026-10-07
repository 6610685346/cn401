"""Report which way a person crosses two zones (off unless ENABLE_DIRECTION).

Hover the mouse over the window (PRINT_MOUSE_POSITION) to re-fit the polygons
to the camera's angle. Coordinates are in the resized display frame.
"""

import cv2
import numpy as np

from ..vision.detector import Detection
from ..core.geometry import Frame


class DirectionTracker:
    """Reports a person's direction after they cross both zones in order."""

    AREA = np.array([(421, 380), (445, 399), (662, 318), (643, 299)], np.int32)
    AREA1 = np.array([(387, 350), (412, 372), (640, 293), (618, 274)], np.int32)

    def __init__(self):
        self.entered_area = set()  # track IDs seen inside AREA
        self.entered_area1 = set()  # track IDs seen inside AREA1
        self.reported = set()  # (track_id, direction) already reported

    @staticmethod
    def _inside(polygon: np.ndarray, point: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def update(self, detection: Detection) -> str | None:
        """Return "Up"/"Down" the first time a person completes a crossing."""
        _, _, x2, y2 = detection.box
        anchor = (x2, y2)
        track_id = detection.track_id

        in_area = self._inside(self.AREA, anchor)
        in_area1 = self._inside(self.AREA1, anchor)

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

    def draw_zones(self, frame: Frame) -> None:
        cv2.polylines(frame, [self.AREA], True, (255, 0, 0), 2)
        cv2.polylines(frame, [self.AREA1], True, (0, 255, 0), 2)
