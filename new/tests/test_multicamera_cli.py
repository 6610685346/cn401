"""Features merged from main (Oct 16): several cameras in connect.py, camera
password from the environment, YOLO device/fp16, background VLM-PAR, and the
CSV "camera" column.

Run from the `new` folder:   python -m unittest discover -s tests -v
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

NEW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEW_DIR))

from person_detection import config  # noqa: E402
from person_detection.core.geometry import FramePreprocessor  # noqa: E402
from person_detection.pipeline import CameraOptions, FrameProcessor, PipelineFactory  # noqa: E402
from person_detection.storage.recorder import CropSaver, CsvLog  # noqa: E402
from person_detection.vision.colors import Appearance  # noqa: E402
from person_detection.vision.detector import Detection  # noqa: E402

import ultralytics  # noqa: E402

BUS = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
MODELS_PRESENT = Path(config.MODEL_PATH).exists() and Path(config.FACE_MODEL_PATH).exists()


class TempOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._saved = (config.OUTPUT_DIR, config.CROPS_DIR, config.SAVE_CROPS, config.ENABLE_VLM_PAR,
                       config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH)
        config.OUTPUT_DIR = self.tmp
        config.CROPS_DIR = self.tmp / "crops"
        config.PEOPLE_DB_PATH = self.tmp / "people.json"
        config.WARDROBE_LOG_PATH = self.tmp / "wardrobe.json"
        config.ENABLE_VLM_PAR = False

    def tearDown(self):
        (config.OUTPUT_DIR, config.CROPS_DIR, config.SAVE_CROPS, config.ENABLE_VLM_PAR,
         config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH) = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)


class CameraPasswordTest(unittest.TestCase):
    def test_password_comes_from_the_environment(self):
        code = "from person_detection import config; print(config.CAMERAS['cam1']); print(config.SOURCE)"
        env = {**os.environ, "CAM_USER": "admin", "CAM_PASSWORD": "#Oung 25/48"}
        out = subprocess.run([sys.executable, "-c", code], cwd=NEW_DIR, env=env,
                             capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertTrue(out[0].startswith("rtsp://admin:%23Oung%2025%2F48@"), out[0])  # quoted
        self.assertEqual(out[0], out[1])  # SOURCE = cam1 (capture_faces.py uses it)

    def test_no_password_in_the_code(self):
        text = (NEW_DIR / "person_detection" / "config.py").read_text(encoding="utf-8")
        self.assertNotIn("Oung2548", text)


class CsvCameraColumnTest(TempOutput):
    def test_camera_column_and_crop_name(self):
        detection = Detection((10, 10, 110, 310), 7, 0.9)
        log = CsvLog(self.tmp / "log.csv")
        log.write(time.localtime(), detection, Appearance("red", "blue"), "Arm", 3, camera="cam2")
        with open(self.tmp / "log.csv", newline="", encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f))
        self.assertEqual((row["camera"], row["track_id"], row["name"], row["global_id"]), ("cam2", "7", "Arm", "G3"))
        CropSaver(self.tmp / "crops").save(BUS, detection, time.localtime(), camera="Gate HD")
        self.assertTrue(next((self.tmp / "crops").glob("Gate-HD_person_7_*.jpg"), None))

    def test_old_file_keeps_its_columns(self):
        old = self.tmp / "old.csv"
        old.write_text("timestamp,track_id,name,shirt_color,pants_color,confidence\n", encoding="utf-8-sig")
        CsvLog(old).write(time.localtime(), Detection((0, 0, 9, 9), 1, 0.5), Appearance("red", "blue"),
                          None, camera="cam1")
        rows = list(csv.reader(open(old, encoding="utf-8-sig")))
        self.assertEqual(len(rows[1]), len(rows[0]))  # still lines up with the old header


class SlowReader:
    """Stands in for VLM-PAR: answers after a delay."""

    MIN_BOX_HEIGHT_FRAC = 0.096

    def __init__(self, delay):
        self.delay = delay
        self.calls = 0

    def read(self, frame, box):
        self.calls += 1
        time.sleep(self.delay)
        return f"attrs from {frame.shape[1]}x{frame.shape[0]} crop"


class OnePerson:
    def detect(self, frame):
        return [Detection((100, 50, 300, 500), 1, 0.9)]


class BackgroundVlmTest(unittest.TestCase):
    def test_video_does_not_wait_for_vlm(self):
        from concurrent.futures import ThreadPoolExecutor
        reader = SlowReader(0.5)
        processor = FrameProcessor(FramePreprocessor(0, 960), OnePerson(), vlm_par=reader,
                                   vlm_min_frames=2, vlm_pool=ThreadPoolExecutor(1))
        frame = np.zeros((540, 960, 3), np.uint8)
        started = time.perf_counter()
        results = [processor.process(frame).people[0].attributes for _ in range(5)]
        self.assertLess(time.perf_counter() - started, 0.4)  # never blocked on the 0.5 s read
        self.assertEqual(results, [None] * 5)
        time.sleep(0.7)
        self.assertEqual(processor.process(frame).people[0].attributes, "attrs from 200x450 crop")
        processor.process(frame)
        self.assertEqual(reader.calls, 1)  # read once per person

    def test_small_person_is_not_sent(self):
        from concurrent.futures import ThreadPoolExecutor

        class Tiny:
            def detect(self, frame):
                return [Detection((10, 10, 20, 40), 1, 0.9)]  # 30 px on a 540 px frame
        reader = SlowReader(0)
        processor = FrameProcessor(FramePreprocessor(0, 960), Tiny(), vlm_par=reader,
                                   vlm_min_frames=1, vlm_pool=ThreadPoolExecutor(1))
        for _ in range(3):
            processor.process(np.zeros((540, 960, 3), np.uint8))
        time.sleep(0.1)
        self.assertEqual(reader.calls, 0)


@unittest.skipUnless(MODELS_PRESENT, "models not downloaded")
class MultiCameraCliTest(TempOutput):
    def make_video(self, name, frames=40):
        path = self.tmp / f"{name}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (BUS.shape[1], BUS.shape[0]))
        for i in range(frames):
            writer.write(np.roll(BUS, i * 2, axis=1))
        writer.release()
        return str(path)

    def test_two_cameras_run_side_by_side(self):
        from person_detection import app
        config.SAVE_CSV = True
        factory = PipelineFactory()
        quit_event = threading.Event()
        workers = [app.CameraWorker(name, self.make_video(name), app.build_processor(factory, name), quit_event)
                   for name in ("camA", "camB")]
        for w in workers:
            w.start()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not all(w.latest(0)[0] is not None for w in workers):
            time.sleep(0.2)
        time.sleep(3)
        quit_event.set()
        for w in workers:
            w.join(10)
        for w in workers:
            frame, _ = w.latest(0)
            self.assertIsNotNone(frame, w.camera_name)
        rows = []
        for f in self.tmp.glob("person_log_*.csv"):
            rows += list(csv.DictReader(open(f, encoding="utf-8-sig")))
        self.assertEqual({r["camera"] for r in rows}, {"camA", "camB"})  # one file, told apart

    def test_detector_uses_cpu_without_cuda(self):
        processor = PipelineFactory().create(CameraOptions(save_csv=False, enable_vlm_par=False))
        import torch
        if not torch.cuda.is_available():
            self.assertEqual(processor.detector.device, "cpu")
            self.assertFalse(processor.detector.half)
        self.assertEqual(processor.detector.imgsz, config.YOLO_IMGSZ)

    def test_falls_back_to_source(self):
        from person_detection import app
        saved = config.CAMERAS
        try:
            config.CAMERAS = {}
            self.assertEqual(app.cameras(), {"cam1": config.SOURCE})
        finally:
            config.CAMERAS = saved


if __name__ == "__main__":
    unittest.main()
