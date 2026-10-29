"""Work-savers (person_detection/perf, smart faces, face votes, log cooldown).

Run from the `new` folder:   python -m unittest tests.test_performance -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

NEW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEW_DIR))

from person_detection import config  # noqa: E402
from person_detection.perf.motion import DetectionScheduler, MotionGate  # noqa: E402
from person_detection.storage.recorder import FaceIdentifier, PersonRecorder, TrackGate  # noqa: E402
from person_detection.vision.colors import Appearance  # noqa: E402
from person_detection.vision.detector import Detection  # noqa: E402

import ultralytics  # noqa: E402

ASSETS = Path(ultralytics.__file__).parent / "assets"
MODELS = Path(config.MODEL_PATH).exists() and Path(config.FACE_MODEL_PATH).exists()


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class MotionTest(unittest.TestCase):
    def test_still_picture_is_quiet_and_a_change_is_seen(self):
        gate = MotionGate()
        room = np.full((540, 960, 3), 90, np.uint8)
        self.assertTrue(gate.update(room))  # first frame: no background yet
        for _ in range(5):
            self.assertFalse(gate.update(room))
        noisy = cv2.add(room, np.random.default_rng(0).integers(0, 6, room.shape, dtype=np.uint8))
        self.assertFalse(gate.update(noisy))  # camera noise is not motion
        person = room.copy()
        cv2.rectangle(person, (400, 150), (520, 520), (25, 25, 25), -1)  # a dark coat
        self.assertTrue(gate.update(person))

    def test_scheduler(self):
        clock = Clock()
        sched = DetectionScheduler(MotionGate(), idle_check_sec=2.0, hold_sec=1.5, clock=clock)
        room = np.full((300, 400, 3), 90, np.uint8)
        runs = []
        for i in range(60):  # 6 s of an empty, still room at 10 fps
            clock.t = i * 0.1
            runs.append(sched.should_detect(room, had_people=False))
        # first frames (no background + hold), then only every 2 s
        self.assertLessEqual(sum(runs), 22)
        self.assertGreaterEqual(sum(runs), 3)
        clock.t = 6.1
        self.assertTrue(sched.should_detect(room, had_people=True))  # people last frame: always
        clock.t = 6.2
        self.assertTrue(sched.should_detect(room, had_people=False))  # still within hold time
        self.assertTrue(DetectionScheduler(None).should_detect(room, False))  # gate off


class FakeRecognizer:
    match_threshold = 0.363

    def __init__(self, scores):
        self.scores = list(scores)  # similarity to "Arm" of each face, in order
        self.calls = 0

    def embed(self, frame, raw):
        self.calls += 1
        v = np.zeros(128, np.float32)
        v[0] = self.scores[self.calls - 1]
        v[1] = 1.0
        return v

    def similarity(self, a, b):
        return float(a[0] / (np.linalg.norm(a) or 1)) if b is not None else 0.0


class Gallery:
    def gallery(self):
        return {"Arm": np.ones(128, np.float32)}


FACE = [((150, 60, 230, 150), np.zeros(15, np.float32))]
PERSON = Detection((100, 50, 300, 470), 4, 0.9)


class FaceVoteTest(unittest.TestCase):
    def ident(self, scores, **kw):
        return FaceIdentifier(FakeRecognizer(scores), Gallery(), **kw)

    def test_first_face_decides_with_one_vote(self):
        f = self.ident([0.3])
        self.assertEqual(f.identify(None, PERSON, FACE), "unknown")

    def test_votes_average_out_one_bad_face(self):
        # similarity of each single face: 0.20 (blurry), 0.50, 0.55 -> average decides "Arm"
        f = self.ident([0.2, 0.5, 0.55], votes=3, strong_margin=0.5)
        self.assertIsNone(f.identify(None, PERSON, FACE))
        self.assertTrue(f.pending(4))
        self.assertIsNone(f.identify(None, PERSON, FACE))
        self.assertEqual(f.identify(None, PERSON, FACE), "Arm")
        self.assertFalse(f.pending(4))
        self.assertEqual(f.candidates[4][0], "Arm")

    def test_clear_match_decides_at_once(self):
        f = self.ident([0.9], votes=3, strong_margin=0.12)
        self.assertEqual(f.identify(None, PERSON, FACE), "Arm")
        self.assertEqual(f.embeddings_made, 1)

    def test_small_faces_are_skipped(self):
        f = self.ident([0.9], min_face_px=100)
        self.assertIsNone(f.identify(None, PERSON, FACE))  # 80 px wide < 100
        self.assertFalse(f.pending(4))
        self.assertEqual(f.embeddings_made, 0)

    def test_finalize_with_what_is_there(self):
        f = self.ident([0.5], votes=3, strong_margin=0.5)
        f.identify(None, PERSON, FACE)
        self.assertEqual(f.finalize(4), "Arm")
        self.assertIsNone(f.finalize(99))


class Reader:
    def read(self, frame, box):
        return Appearance("red", "blue")


class Log:
    def __init__(self):
        self.rows = []

    def write(self, when, detection, appearance, name, *args, **kw):
        self.rows.append((name, kw.get("direction")))


class RecorderTest(unittest.TestCase):
    frame = np.zeros((480, 640, 3), np.uint8)

    def test_waits_for_the_vote_then_logs_the_name(self):
        log = Log()
        ident = FaceIdentifier(FakeRecognizer([0.5, 0.5, 0.5]), Gallery(), votes=3, strong_margin=0.5)
        rec = PersonRecorder(log, gate=TrackGate(1), reader=Reader(), face_identifier=ident)
        rec.observe(self.frame, PERSON, FACE)  # 1st face: pending, row waits
        self.assertEqual(log.rows, [])
        rec.observe(self.frame, PERSON, FACE)
        rec.observe(self.frame, PERSON, FACE)  # 3rd face decides
        self.assertEqual(log.rows, [("Arm", None)])

    def test_cooldown_skips_a_second_row_for_the_same_person(self):
        log, clock = Log(), Clock()
        ident = FaceIdentifier(FakeRecognizer([0.9] * 4), Gallery())
        rec = PersonRecorder(log, gate=TrackGate(1), reader=Reader(), face_identifier=ident,
                             log_cooldown_sec=30, clock=clock)
        rec.observe(self.frame, PERSON, FACE)
        clock.t = 10  # hidden for a moment, comes back with a new track id
        again = Detection(PERSON.box, 5, 0.9)
        rec.observe(self.frame, again, FACE)
        rec.record_direction(self.frame, again, "in")  # crossings are always written
        clock.t = 50
        later = Detection(PERSON.box, 6, 0.9)
        rec.observe(self.frame, later, FACE)
        self.assertEqual(log.rows, [("Arm", None), ("Arm", "in"), ("Arm", None)])
        self.assertEqual(rec.suppressed, 1)


@unittest.skipUnless(MODELS, "models not downloaded")
class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls._saved = (config.OUTPUT_DIR, config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH)
        config.OUTPUT_DIR = cls.tmp
        config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH = cls.tmp / "p.json", cls.tmp / "w.json"
        cls.scene = cv2.resize(cv2.imread(str(ASSETS / "zidane.jpg")), (960, 540))

    @classmethod
    def tearDownClass(cls):
        config.OUTPUT_DIR, config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH = cls._saved

    def test_face_regions_give_the_same_face(self):
        from person_detection.vision.face import FaceDetector, FaceRecognizer, head_region
        fd = FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE)
        fr = FaceRecognizer(config.FACE_RECOGNITION_MODEL_PATH)
        full = fd.detect_raw(self.scene)
        found = fd.detect_in_regions(self.scene, [head_region((640, 40, 860, 540))])
        self.assertEqual(len(found), 1)
        box, row = found[0]
        twin = min(full, key=lambda f: abs(f[0][0] - box[0]) + abs(f[0][1] - box[1]))
        self.assertLess(max(abs(a - b) for a, b in zip(twin[0], box)), 8)
        self.assertGreater(fr.similarity(fr.embed(self.scene, twin[1]), fr.embed(self.scene, row)), 0.8)
        self.assertEqual(fd.detect_in_regions(self.scene, [(0, 0, 10, 10)]), [])  # too small

    def test_idle_frames_skip_yolo(self):
        from person_detection.pipeline import CameraOptions, PipelineFactory
        empty = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg"))[0:300, :], (960, 540))  # building, no people
        processor = PipelineFactory().create(CameraOptions(enable_vlm_par=False, save_csv=False))
        processor.scheduler.clock = Clock()
        for i in range(40):
            processor.scheduler.clock.t = i / 10
            processor.process(empty.copy())
        stats = processor.perf()
        self.assertGreater(stats["idle_skips"], 20)
        self.assertLess(stats["detect_runs"], 20)

    def test_openvino_finds_the_same_people(self):
        from person_detection.perf.backend import openvino_available
        if not openvino_available():
            self.skipTest("openvino not installed")
        from person_detection.vision.detector import PersonDetector
        torch_det = PersonDetector(config.MODEL_PATH, 0.4, backend="pytorch")
        ov_det = PersonDetector(config.MODEL_PATH, 0.4, backend="openvino")
        self.assertEqual(ov_det.backend, "openvino")
        a, b = torch_det.detect(self.scene), ov_det.detect(self.scene)
        self.assertEqual(len(a), len(b))
        for da, db in zip(sorted(a, key=lambda d: d.box), sorted(b, key=lambda d: d.box)):
            self.assertLess(max(abs(x - y) for x, y in zip(da.box, db.box)), 6)


if __name__ == "__main__":
    unittest.main()
