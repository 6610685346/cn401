"""Name the colours a person is wearing."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.geometry import Box, Frame, crop_band

COLOR_NAMES = (
    "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
    "brown", "black", "gray", "white",
)  # fmt: skip
COLOR_INDEX = {name: i for i, name in enumerate(COLOR_NAMES)}

# Hue ranges on OpenCV's 0-179 scale; red wraps around both ends.
HUE_RANGES = (
    (0, 10, "red"),
    (11, 20, "orange"),
    (21, 33, "yellow"),
    (34, 85, "green"),
    (86, 100, "cyan"),
    (101, 130, "blue"),
    (131, 150, "purple"),
    (151, 169, "pink"),
    (170, 179, "red"),
)


class ColorClassifier:
    """Names the colour covering most of an image region.

    Pixels are labelled individually and the most common label wins, so a
    patterned shirt keeps its main colour instead of averaging to mud.
    """

    BLACK_MAX_VALUE = 55  # below this, hue is meaningless
    GRAY_MAX_SATURATION = 45  # below this, hue is meaningless
    WHITE_MIN_VALUE = 190
    BROWN_MAX_VALUE = 150  # a dark orange is really brown
    SAMPLE_SIZE = (32, 32)  # regions are shrunk to this before being labelled

    def __init__(self):
        self._hue_lut = self._build_hue_lut()

    @staticmethod
    def _build_hue_lut() -> np.ndarray:
        """Lookup table mapping each hue value to a colour index."""
        lut = np.zeros(180, np.uint8)
        for low, high, name in HUE_RANGES:
            lut[low: high + 1] = COLOR_INDEX[name]
        return lut

    def dominant(self, region: Frame) -> str | None:
        """Name the region's main colour, or None if it is empty."""
        if region.size == 0:
            return None

        sample = cv2.resize(region, self.SAMPLE_SIZE,
                            interpolation=cv2.INTER_AREA)
        hue, saturation, value = cv2.split(
            cv2.cvtColor(sample, cv2.COLOR_BGR2HSV))

        labels = self._hue_lut[hue]
        dark_orange = (labels == COLOR_INDEX["orange"]) & (
            value < self.BROWN_MAX_VALUE)
        labels[dark_orange] = COLOR_INDEX["brown"]

        # Achromatic pixels override whatever hue they happen to carry.
        achromatic = saturation < self.GRAY_MAX_SATURATION
        labels[achromatic] = COLOR_INDEX["gray"]
        labels[achromatic & (value >= self.WHITE_MIN_VALUE)
               ] = COLOR_INDEX["white"]
        labels[value < self.BLACK_MAX_VALUE] = COLOR_INDEX["black"]

        counts = np.bincount(labels.ravel(), minlength=len(COLOR_NAMES))
        return COLOR_NAMES[counts.argmax()]


@dataclass(frozen=True)
class Appearance:
    """What a person appears to be wearing."""

    shirt_color: str
    pants_color: str

    def __str__(self) -> str:
        return f"{self.shirt_color}/{self.pants_color}"


class AppearanceReader:
    """Reads a person's shirt and trouser colours from their box."""

    # Slices of a person's box to sample, as a fraction of its height (head skipped).
    SHIRT_BAND = (0.20, 0.55)
    PANTS_BAND = (0.58, 0.90)

    MIN_BOX_HEIGHT_PX = 60  # smaller than this and the colours are unreadable
    MIN_BOX_WIDTH_PX = 10

    def __init__(self, classifier: ColorClassifier | None = None):
        self.classifier = classifier or ColorClassifier()

    def read(self, frame: Frame, box: Box) -> Appearance | None:
        """Return the person's appearance, or None if the box is unreadable."""
        x1, y1, x2, y2 = box
        if y2 - y1 < self.MIN_BOX_HEIGHT_PX or x2 - x1 < self.MIN_BOX_WIDTH_PX:
            return None

        shirt = self.classifier.dominant(
            crop_band(frame, box, self.SHIRT_BAND))
        pants = self.classifier.dominant(
            crop_band(frame, box, self.PANTS_BAND))
        if shirt is None or pants is None:
            return None
        return Appearance(shirt, pants)
