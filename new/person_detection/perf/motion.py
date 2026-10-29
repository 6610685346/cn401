"""Skip the person detector while nothing happens in front of the camera.

A door camera looks at an empty corridor most of the day. Running YOLO on
every one of those frames costs the same as on a busy frame. MotionGate
compares a tiny grey copy of each frame (160 px wide, ~1 ms) with a slowly
updated background; DetectionScheduler then decides whether YOLO runs:

* people were found on the last processed frame  -> always run (tracking needs it)
* something moved                               -> run, and keep running for HOLD_SEC
* nothing moved, nobody there                   -> run only every IDLE_CHECK_SEC,
                                                   so a person standing perfectly
                                                   still is still found
"""

from __future__ import annotations

import time
from collections.abc import Callable

import cv2
import numpy as np


class MotionGate:
    def __init__(self, width: int = 160, threshold: int = 18, min_area: float = 0.004,
                 learn_rate: float = 0.05, quiet_learn_rate: float = 0.3):
        self.width = width
        self.threshold = threshold  # grey-level change that counts as "moved"
        self.min_area = min_area  # fraction of the small picture that must change
        self.learn_rate = learn_rate  # how fast the background follows light changes
        # ...and when YOLO has just seen nobody: forget the "ghost" of people who
        # left (or a chair that was moved) within a few frames, not 2-3 seconds
        self.quiet_learn_rate = quiet_learn_rate
        self._background: np.ndarray | None = None
        self.last_fraction = 0.0

    def _small(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        size = (self.width, max(1, round(height * self.width / width)))
        grey = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(grey, (5, 5), 0)

    def update(self, frame: np.ndarray, quiet: bool = False) -> bool:
        """True when this frame differs enough from the background.

        quiet: the person detector found nobody last time it ran.
        """
        small = self._small(frame)
        if self._background is None or self._background.shape != small.shape:
            self._background = small.astype(np.float32)
            self.last_fraction = 1.0
            return True
        diff = cv2.absdiff(small, cv2.convertScaleAbs(self._background))
        self.last_fraction = float(np.count_nonzero(diff > self.threshold)) / diff.size
        cv2.accumulateWeighted(small, self._background, self.quiet_learn_rate if quiet else self.learn_rate)
        return self.last_fraction >= self.min_area


class DetectionScheduler:
    def __init__(self, gate: MotionGate | None, idle_check_sec: float = 2.0, hold_sec: float = 1.5,
                 clock: Callable[[], float] = time.monotonic):
        self.gate = gate
        self.idle_check_sec = idle_check_sec
        self.hold_sec = hold_sec
        self.clock = clock
        self._active_until = float("-inf")
        self._last_detect = float("-inf")

    def should_detect(self, frame: np.ndarray, had_people: bool) -> bool:
        if self.gate is None:
            return True
        now = self.clock()
        moved = self.gate.update(frame, quiet=not had_people)
        if had_people or moved:
            self._active_until = now + self.hold_sec
        run = now <= self._active_until or now - self._last_detect >= self.idle_check_sec
        if run:
            self._last_detect = now
        return run
