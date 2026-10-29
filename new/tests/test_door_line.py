"""Door line counting (counting/line.py) and the photo saved with each row.

Run from the `new` folder:   python -m unittest tests.test_door_line -v
"""

import csv
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

NEW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEW_DIR))

from person_detection.counting.direction import DirectionTracker  # noqa: E402
from person_detection.counting.line import DoorLine, LineCrossCounter  # noqa: E402
from person_detection.storage.recorder import CropSaver, CsvLog, PersonRecorder, TrackGate  # noqa: E402
from person_detection.vision.colors import Appearance  # noqa: E402
from person_detection.vision.detector import Detection  # noqa: E402

W, H = 1000, 500
# horizontal door across the middle; "left" of a->b (pointing right) is the upper half
DOOR = DoorLine((0.3, 0.5), (0.7, 0.5), inside="left")


def at(track, x, y):
    """A person whose feet are at (x, y) in pixels."""
    return Detection((x - 40, y - 200, x + 40, y), track, 0.9, foot=(x, y))


def walk(counter, track, points):
    return [r for r in (counter.update(at(track, x, y), W, H) for x, y in points) if r]


class LineTest(unittest.TestCase):
    def test_in_then_out(self):
        c = LineCrossCounter(DOOR)
        self.assertEqual(walk(c, 1, [(500, 400), (500, 300), (500, 200)]), ["in"])  # bottom -> top = inside
        self.assertEqual(walk(c, 1, [(500, 300), (500, 420)]), ["out"])

    def test_other_side_is_inside_when_flipped(self):
        c = LineCrossCounter(DoorLine((0.3, 0.5), (0.7, 0.5), inside="right"))
        self.assertEqual(walk(c, 1, [(500, 400), (500, 200)]), ["out"])

    def test_wobbling_on_the_line_counts_once(self):
        c = LineCrossCounter(DOOR)
        # dead band = 2.5% of 500 px = 12.5 px: 245..255 is "on the line"
        self.assertEqual(walk(c, 1, [(500, 200), (500, 246), (500, 254), (500, 248), (500, 200)]), [])  # stays inside
        c = LineCrossCounter(DOOR)
        # outside (300) -> jitters around the line -> clearly inside (200): exactly one "in"
        self.assertEqual(walk(c, 2, [(500, 300), (500, 256), (500, 244), (500, 255), (500, 246), (500, 200)]), ["in"])

    def test_beyond_the_door_is_ignored(self):
        c = LineCrossCounter(DOOR)
        self.assertEqual(walk(c, 1, [(950, 400), (950, 100)]), [])  # x=950 is far past b (700)
        self.assertEqual(walk(c, 2, [(250, 400), (250, 100)]), ["in"])  # inside the end margin

    def test_each_track_is_separate_and_forgotten(self):
        c = LineCrossCounter(DOOR)
        self.assertEqual(walk(c, 1, [(500, 400)]), [])
        self.assertEqual(walk(c, 2, [(500, 100)]), [])  # first sighting: no previous side
        c.forget(1)
        self.assertEqual(walk(c, 1, [(500, 100)]), [])  # after forget, starts fresh

    def test_validation_and_round_trip(self):
        with self.assertRaises(ValueError):
            DoorLine((0.5, 0.5), (0.505, 0.5))
        with self.assertRaises(ValueError):
            DoorLine((0, 0), (1, 1), inside="up")
        self.assertIsNone(DoorLine.from_dict({"a": [0, 0]}))
        self.assertEqual(DoorLine.from_dict(DOOR.to_dict()).to_dict(), DOOR.to_dict())

    def test_drawing_does_not_fail(self):
        frame = np.zeros((H, W, 3), np.uint8)
        LineCrossCounter(DOOR).draw_zones(frame)
        self.assertGreater(frame.sum(), 0)

    def test_factory_uses_the_line(self):
        from person_detection.pipeline import CameraOptions, PipelineFactory
        tracker = PipelineFactory._direction_tracker(CameraOptions(enable_direction=True, door_line=DOOR.to_dict()))
        self.assertIsInstance(tracker, LineCrossCounter)
        self.assertIsInstance(PipelineFactory._direction_tracker(CameraOptions(enable_direction=True)), DirectionTracker)
        self.assertIsNone(PipelineFactory._direction_tracker(CameraOptions(enable_direction=False, door_line=DOOR.to_dict())))


class Reader:
    def read(self, frame, box):
        return Appearance("red", "blue")


class PhotoPerRowTest(unittest.TestCase):
    def test_every_row_has_a_photo_in_a_day_folder(self):
        tmp = Path(tempfile.mkdtemp())
        log = CsvLog(tmp / "log.csv")
        recorder = PersonRecorder(log, CropSaver(tmp / "crops"), TrackGate(1), reader=Reader(), camera="Gate")
        frame = np.full((480, 640, 3), 120, np.uint8)
        person = Detection((100, 50, 300, 470), 7, 0.9)
        recorder.observe(frame, person, [])
        recorder.record_direction(frame, person, "in")
        rows = list(csv.DictReader(open(tmp / "log.csv", encoding="utf-8-sig")))
        self.assertEqual([r["direction"] for r in rows], ["", "in"])
        day = time.strftime("%Y-%m-%d")
        for r in rows:
            self.assertTrue(r["snapshot"].startswith(day + "/"), r["snapshot"])
            self.assertTrue((tmp / "crops" / r["snapshot"]).is_file())
        self.assertTrue(rows[1]["snapshot"].endswith("_in.jpg"))
        self.assertNotEqual(rows[0]["snapshot"], rows[1]["snapshot"])


if __name__ == "__main__":
    unittest.main()
