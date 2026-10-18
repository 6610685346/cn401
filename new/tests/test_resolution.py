"""Checks that detection works the same at every camera resolution.

Run from the `new` folder:   python -m unittest discover -s tests -v

Uses the two sample photos that ship with ultralytics (zidane.jpg is 16:9,
bus.jpg is portrait), scaled to typical camera sizes from 640x480 to 4K.
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from person_detection import config  # noqa: E402
from person_detection.core.geometry import FramePreprocessor, scale_to_long_side  # noqa: E402
from person_detection.pipeline import CameraOptions, PipelineFactory  # noqa: E402
from person_detection.storage.recorder import DailyCsvLog, CSV_COLUMNS  # noqa: E402
from person_detection.sources.stream import LatestFrameStream, is_live, parse_source  # noqa: E402

import ultralytics  # noqa: E402

ASSETS = Path(ultralytics.__file__).parent / "assets"
LANDSCAPE = cv2.imread(str(ASSETS / "zidane.jpg"))  # 1280x720, 2 people
PORTRAIT = cv2.imread(str(ASSETS / "bus.jpg"))  # 810x1080, 4 people + bus

MODELS_PRESENT = Path(config.MODEL_PATH).exists() and Path(config.FACE_MODEL_PATH).exists()


def resize(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA
                      if width < image.shape[1] else cv2.INTER_CUBIC)


def letterbox(image, width, height):
    """Put an image on a grey canvas of a different aspect ratio (4:3 camera)."""
    canvas = np.full((height, width, 3), 114, np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    small = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    y = (height - small.shape[0]) // 2
    x = (width - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


class PreprocessTest(unittest.TestCase):
    SIZES = [(320, 240), (640, 480), (1280, 720), (1920, 1080), (2560, 1440),
             (3840, 2160), (1080, 1920), (720, 1280), (1000, 1000)]

    def test_long_side_is_always_standard(self):
        pre = FramePreprocessor(0, 960)
        for w, h in self.SIZES:
            out = pre(np.zeros((h, w, 3), np.uint8))
            self.assertEqual(max(out.shape[:2]), 960, (w, h))
            # aspect ratio kept (within one pixel of rounding)
            self.assertAlmostEqual(out.shape[1] / out.shape[0], w / h, delta=0.01)

    def test_rotation_swaps_sides(self):
        out = FramePreprocessor(90, 960)(np.zeros((1080, 1920, 3), np.uint8))
        self.assertEqual(out.shape[:2], (960, 540))

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            FramePreprocessor(45, 960)
        with self.assertRaises(ValueError):
            FramePreprocessor(0, 960)(np.zeros((10, 10), np.uint8))

    def test_identity_when_already_standard(self):
        frame = np.zeros((540, 960, 3), np.uint8)
        self.assertIs(scale_to_long_side(frame, 960), frame)


@unittest.skipUnless(MODELS_PRESENT, "models/ not downloaded")
class ResolutionIndependenceTest(unittest.TestCase):
    """Same scene at different resolutions -> same people, same places."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls._saved = (config.OUTPUT_DIR, config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH)
        config.OUTPUT_DIR = Path(cls.tmp)
        config.PEOPLE_DB_PATH = Path(cls.tmp) / "people.json"
        config.WARDROBE_LOG_PATH = Path(cls.tmp) / "wardrobe.json"
        cls.factory = PipelineFactory()

    @classmethod
    def tearDownClass(cls):
        config.OUTPUT_DIR, config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH = cls._saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_once(self, image, tag):
        # Fresh processor per resolution: as if a different camera were plugged in.
        processor = self.factory.create(CameraOptions(enable_vlm_par=False, csv_tag=tag))
        result = None
        for _ in range(6):  # a few frames so tracking + logging kick in
            result = processor.process(image.copy())
        return result

    def assert_people_match(self, base, other, label):
        self.assertEqual(len(other.people), len(base.people), label)
        a = sorted(base.people, key=lambda p: p.box[0])
        b = sorted(other.people, key=lambda p: p.box[0])
        for pa, pb in zip(a, b):
            # boxes agree to within 3% of the frame
            np.testing.assert_allclose(pa.box, pb.box, atol=0.03, err_msg=label)
            self.assertEqual(pa.clothing, pb.clothing, label)

    def test_landscape_16_9_resolutions(self):
        base = self.run_once(LANDSCAPE, "base")
        self.assertGreaterEqual(len(base.people), 2)
        for w, h in [(854, 480), (1920, 1080), (2560, 1440), (3840, 2160)]:
            with self.subTest(size=f"{w}x{h}"):
                other = self.run_once(resize(LANDSCAPE, w, h), f"{w}x{h}")
                self.assertEqual((other.source_width, other.source_height), (w, h))
                self.assertEqual((other.width, other.height), (960, 540))
                self.assert_people_match(base, other, f"{w}x{h}")

    def test_portrait_resolutions(self):
        base = self.run_once(PORTRAIT, "pbase")
        self.assertGreaterEqual(len(base.people), 3)
        for w, h in [(540, 720), (1620, 2160), (2430, 3240)]:
            with self.subTest(size=f"{w}x{h}"):
                other = self.run_once(resize(PORTRAIT, w, h), f"p{w}x{h}")
                self.assertEqual(max(other.width, other.height), 960)
                self.assert_people_match(base, other, f"{w}x{h}")

    def test_4_3_camera(self):
        for w, h in [(640, 480), (1600, 1200)]:
            with self.subTest(size=f"{w}x{h}"):
                result = self.run_once(letterbox(LANDSCAPE, w, h), f"43_{w}")
                self.assertGreaterEqual(len(result.people), 2)
                for person in result.people:
                    self.assertTrue(all(0 <= v <= 1 for v in person.box))

    def test_csv_written_per_camera_tag(self):
        self.run_once(LANDSCAPE, "Gate A")
        date = time.strftime("%Y-%m-%d")
        path = Path(self.tmp) / f"person_log_{date}_Gate-A.csv"
        self.assertTrue(path.exists())
        rows = path.read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(rows[0].split(","), CSV_COLUMNS)
        self.assertGreaterEqual(len(rows), 3)  # header + 2 people

    def test_annotate_off_leaves_picture_clean(self):
        processor = self.factory.create(CameraOptions(enable_vlm_par=False, save_csv=False))
        image = resize(LANDSCAPE, 960, 540)
        result = processor.process(image.copy(), annotate=False)
        np.testing.assert_array_equal(result.frame, image)
        self.assertTrue(result.people)


class StorageAndStreamTest(unittest.TestCase):
    def test_daily_csv_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = DailyCsvLog(tmp, "ประตู 1/หน้า")
            when = time.strptime("2026-10-13", "%Y-%m-%d")
            self.assertEqual(log.path_for(when).name, "person_log_2026-10-13_ประตู-1-หน้า.csv")
            self.assertEqual(DailyCsvLog(tmp).path_for(when).name, "person_log_2026-10-13.csv")

    def test_source_parsing(self):
        self.assertEqual(parse_source("0"), 0)
        self.assertTrue(is_live("rtsp://a:b@1.2.3.4/x"))
        self.assertTrue(is_live("http://192.168.1.5:8080/video"))
        self.assertFalse(is_live("clip.mp4"))

    def test_latest_frame_stream_ends_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
            for i in range(20):
                writer.write(np.full((240, 320, 3), i * 10, np.uint8))
            writer.release()
            with LatestFrameStream(path, max_failures=1, reconnect_delay=0) as stream:
                frames = list(stream.frames(timeout=5))
            self.assertGreaterEqual(len(frames), 1)  # newest frames only; may skip
            self.assertEqual(frames[-1].shape, (240, 320, 3))


if __name__ == "__main__":
    unittest.main()
