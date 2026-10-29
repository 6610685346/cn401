"""Count people crossing a door line ("in" / "out").

The line is two points drawn on the camera picture (0..1 fractions of the
processed frame, so any resolution works) and which side of it is *inside*.
A person's floor point (between the ankles, or the bottom of the box) is
followed frame by frame; when it moves from one side of the line to the
other it is one crossing:

    outside -> inside  = "in"        inside -> outside = "out"

Two things keep the count honest:

* a dead band around the line (DEADBAND of the frame's short side): a foot
  wobbling on the line does not count as many crossings;
* only crossings within the length of the door count (plus a small margin),
  so someone walking past far beyond the end of the line is ignored.

A person who walks in and later walks out again on the same track gives one
"in" and one "out".
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..vision.detector import Detection

DEADBAND = 0.025  # fraction of the frame's short side
END_MARGIN = 0.15  # fraction of the line length beyond each end that still counts


class DoorLine:
    def __init__(self, a, b, inside: str = "left"):
        self.a = (float(a[0]), float(a[1]))
        self.b = (float(b[0]), float(b[1]))
        if inside not in ("left", "right"):
            raise ValueError("inside must be 'left' or 'right'")
        self.inside = inside
        if math.dist(self.a, self.b) < 0.02:
            raise ValueError("เส้นประตูสั้นเกินไป")

    @classmethod
    def from_dict(cls, data) -> "DoorLine | None":
        if not data:
            return None
        try:
            return cls(data["a"], data["b"], data.get("inside", "left"))
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    def to_dict(self) -> dict:
        return {"a": [round(v, 4) for v in self.a], "b": [round(v, 4) for v in self.b], "inside": self.inside}

    def measure(self, point, width: int, height: int) -> tuple[float, float]:
        """(signed distance in pixels, position along the line 0..1) of a pixel point.

        Positive distance = the "left" side when looking from a to b in the
        picture (image y grows downwards).
        """
        ax, ay = self.a[0] * width, self.a[1] * height
        bx, by = self.b[0] * width, self.b[1] * height
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy) or 1.0
        px, py = point[0] - ax, point[1] - ay
        signed = (dx * py - dy * px) / length  # cross product
        along = (px * dx + py * dy) / (length * length)
        return -signed, along


class LineCrossCounter:
    """Same interface as DirectionTracker: update() returns "in" / "out" or None."""

    def __init__(self, line: DoorLine, deadband: float = DEADBAND):
        self.line = line
        self.deadband = deadband
        self.side: dict[int, int] = {}  # track_id -> last confirmed side (+1 inside / -1 outside)

    def _side_of(self, detection: Detection, width: int, height: int) -> int | None:
        signed, along = self.line.measure(detection.foot_point, width, height)
        if not -END_MARGIN <= along <= 1 + END_MARGIN:
            return None  # beyond the ends of the door
        if abs(signed) < self.deadband * min(width, height):
            return None  # on the line: wait until clearly on one side
        left = signed > 0
        return 1 if left == (self.line.inside == "left") else -1

    def update(self, detection: Detection, width: int, height: int) -> str | None:
        if not detection.is_tracked:
            return None
        side = self._side_of(detection, width, height)
        if side is None:
            return None
        previous = self.side.get(detection.track_id)
        self.side[detection.track_id] = side
        if previous is None or previous == side:
            return None
        return "in" if side == 1 else "out"

    def forget(self, track_id: int) -> None:
        self.side.pop(track_id, None)

    def draw_zones(self, frame) -> None:
        """Draw the door line and an arrow pointing inside."""
        height, width = frame.shape[:2]
        a = (int(self.line.a[0] * width), int(self.line.a[1] * height))
        b = (int(self.line.b[0] * width), int(self.line.b[1] * height))
        thick = max(2, min(width, height) // 240)
        cv2.line(frame, a, b, (0, 200, 255), thick + 1, cv2.LINE_AA)
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1
        nx, ny = dy / length, -dx / length  # "left" normal in image coordinates
        if self.line.inside == "right":
            nx, ny = -nx, -ny
        arrow = 0.08 * min(width, height)
        tip = (int(mx + nx * arrow), int(my + ny * arrow))
        cv2.arrowedLine(frame, (int(mx), int(my)), tip, (60, 220, 60), thick, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(frame, "IN", (tip[0] + 4, tip[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, min(width, height) / 900), (60, 220, 60), thick, cv2.LINE_AA)


def as_points(line: DoorLine, width: int, height: int) -> np.ndarray:
    return np.array([[line.a[0] * width, line.a[1] * height], [line.b[0] * width, line.b[1] * height]], np.int32)
