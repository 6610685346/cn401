"""Shared types and helpers for boxes and frames."""

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame coordinates
Frame = np.ndarray  # a BGR image

# Clockwise rotations accepted by the ROTATE setting.
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def fit_frame(frame: Frame, max_width: int, max_height: int) -> Frame:
    """Scale a frame to fit the given box, keeping its aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    return cv2.resize(frame, (round(width * scale), round(height * scale)))


def clip_box(frame: Frame, box: Box) -> Box:
    """Clamp a box to the frame bounds."""
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def crop_box(frame: Frame, box: Box) -> Frame:
    """Cut a box out of a frame, clamped to the frame bounds."""
    x1, y1, x2, y2 = clip_box(frame, box)
    return frame[y1:y2, x1:x2]


def crop_band(frame: Frame, box: Box, band: tuple[float, float]) -> Frame:
    """Cut a horizontal slice out of a box, given as fractions of its height."""
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    top = y1 + int(box_height * band[0])
    bottom = y1 + int(box_height * band[1])
    return crop_box(frame, (x1, top, x2, bottom))


class FramePreprocessor:
    """Straightens and resizes raw frames before detection."""

    def __init__(self, rotate: int, max_width: int, max_height: int):
        self.rotation = ROTATIONS.get(rotate)  # None when rotate == 0
        self.max_size = (max_width, max_height)

    def __call__(self, frame: Frame) -> Frame:
        if self.rotation is not None:
            frame = cv2.rotate(frame, self.rotation)
        return fit_frame(frame, *self.max_size)
