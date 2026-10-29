"""Choosing models: face recognizer (SFace / ArcFace), person model, ReID, full-resolution faces.

Run from the `new` folder:   python -m unittest tests.test_models -v
ArcFace tests are skipped until models/insightface/w600k_mbf.onnx is downloaded
(choose ArcFace on the settings page, or: python -c "from person_detection.vision import
face_models; face_models.ensure_model('arcface_mbf')").
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

NEW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NEW_DIR))

from person_detection import config  # noqa: E402
from person_detection.perf.backend import tracker_config  # noqa: E402
from person_detection.storage.people_db import PeopleDB, people_db_path  # noqa: E402
from person_detection.vision import face_models  # noqa: E402
from person_detection.vision.detector import PERSON_MODELS, person_model_path  # noqa: E402

import ultralytics  # noqa: E402

ASSETS = Path(ultralytics.__file__).parent / "assets"
HAVE_FACE = Path(config.FACE_MODEL_PATH).exists() and Path(config.FACE_RECOGNITION_MODEL_PATH).exists()
HAVE_MBF = HAVE_FACE and face_models.is_downloaded("arcface_mbf")


class TempConfig(unittest.TestCase):
    KEYS = ("OUTPUT_DIR", "PEOPLE_DB_PATH", "WARDROBE_LOG_PATH", "PHOTOS_DIR", "FACE_RECOGNIZER")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._saved = {k: getattr(config, k) for k in self.KEYS}
        config.OUTPUT_DIR = self.tmp
        config.PEOPLE_DB_PATH = self.tmp / "people.json"
        config.WARDROBE_LOG_PATH = self.tmp / "wardrobe.json"
        config.PHOTOS_DIR = self.tmp / "photos"

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)


class RegistryTest(TempConfig):
    def test_choices_match_the_code(self):
        self.assertEqual(set(config.CHOICES["FACE_RECOGNIZER"]), set(face_models.FACE_MODELS))
        self.assertEqual(set(config.CHOICES["PERSON_MODEL"]), set(PERSON_MODELS))
        self.assertEqual(face_models.get("sface").dim, 128)
        self.assertEqual(face_models.get("arcface_r50").dim, 512)
        with self.assertRaises(ValueError):
            face_models.get("facenet")

    def test_thresholds_per_model(self):
        self.assertEqual(face_models.threshold_for("sface"), config.FACE_MATCH_THRESHOLD)
        self.assertEqual(face_models.threshold_for("arcface_mbf"), config.FACE_MATCH_THRESHOLD_ARCFACE)

    def test_gallery_file_per_model(self):
        self.assertEqual(people_db_path("sface"), self.tmp / "people.json")  # unchanged for old installs
        self.assertEqual(people_db_path("arcface_mbf"), self.tmp / "people_arcface_mbf.json")
        config.FACE_RECOGNIZER = "arcface_r50"
        self.assertEqual(people_db_path(), self.tmp / "people_arcface_r50.json")

    def test_person_model_path(self):
        self.assertEqual(person_model_path("yolo11n-pose"), config.MODEL_PATH)
        self.assertTrue(person_model_path("yolo26n-pose").endswith("yolo26n-pose.pt"))
        with self.assertRaises(ValueError):
            person_model_path("yolov3")

    def test_saved_settings_refuse_unknown_models(self):
        path = self.tmp / "settings.json"
        path.write_text(json.dumps({"FACE_RECOGNIZER": "facenet", "PERSON_MODEL": "yolo26n-pose"}))
        saved = {k: getattr(config, k) for k in ("FACE_RECOGNIZER", "PERSON_MODEL")}
        try:
            applied = config.apply_saved_settings(path)
            self.assertEqual(applied, {"PERSON_MODEL": "yolo26n-pose"})
            self.assertEqual(config.FACE_RECOGNIZER, saved["FACE_RECOGNIZER"])
        finally:
            for k, v in saved.items():
                setattr(config, k, v)

    def test_reid_tracker_file(self):
        plain = yaml.safe_load(Path(tracker_config(self.tmp, 60, True)).read_text())
        self.assertFalse(plain["with_reid"])
        path = tracker_config(self.tmp, 60, True, reid=True, reid_model="/m/yolo26n-reid.onnx")
        self.assertIn("_reid-yolo26n-reid", path)
        data = yaml.safe_load(Path(path).read_text())
        self.assertTrue(data["with_reid"])
        self.assertEqual(data["model"], "/m/yolo26n-reid.onnx")
        self.assertEqual(data["gmc_method"], "none")

    def test_vectors_of_two_models_never_match(self):
        sface = face_models.FaceRecognizer.__new__(face_models.FaceRecognizer)  # no file needed for this
        self.assertEqual(sface.similarity(np.ones(128), np.ones(512)), -1.0)
        arc = face_models.ArcFaceRecognizer.__new__(face_models.ArcFaceRecognizer)
        self.assertEqual(arc.similarity(np.ones(512), np.ones(128)), -1.0)
        self.assertAlmostEqual(arc.similarity(np.ones(512), np.ones(512) * 3), 1.0, places=5)


@unittest.skipUnless(HAVE_FACE, "face models not downloaded")
class AlignAndFullResTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from person_detection.vision.face import FaceDetector
        cls.fd = FaceDetector(config.FACE_MODEL_PATH, 0.6)
        cls.scene = cv2.imread(str(ASSETS / "zidane.jpg"))  # 1280 x 720

    def test_alignment_puts_landmarks_on_the_template(self):
        def error(landmarks):
            matrix, _ = cv2.estimateAffinePartial2D(landmarks, face_models.ARCFACE_TEMPLATE, method=cv2.LMEDS)
            return float(np.abs(cv2.transform(landmarks[None], matrix)[0] - face_models.ARCFACE_TEMPLATE).mean())

        for _, row in self.fd.detect_raw(self.scene):
            landmarks = np.asarray(row[4:14], np.float32).reshape(5, 2)
            # YuNet's order is ArcFace's order: much better fit than with the eyes/mouth swapped
            self.assertLess(error(landmarks), 6)
            self.assertLess(error(landmarks), error(landmarks[[1, 0, 2, 4, 3]]) / 2)
            self.assertEqual(face_models.align_arcface(self.scene, row).shape, (112, 112, 3))

    def test_faces_from_the_full_size_picture(self):
        small = cv2.resize(self.scene, (640, 360))
        region = (420, 20, 560, 170)  # around the face top right (processed-frame pixels)
        plain = self.fd.detect_in_regions(small, [region])
        hires = self.fd.detect_in_regions(small, [region], source=self.scene, scale=2.0)
        self.assertEqual(len(plain), 1)
        self.assertEqual(len(hires), 1)
        hit = hires[0]
        self.assertIs(hit.source, self.scene)
        self.assertLess(max(abs(a - b) for a, b in zip(hit[0], plain[0][0])), 8)  # same box, small frame
        self.assertAlmostEqual(float(hit.source_row[2]), 2 * float(hit[1][2]), delta=1)
        self.assertGreater(hit.source_width, 1.8 * (plain[0][0][2] - plain[0][0][0]))
        whole = self.fd.detect_full(small, self.scene, 2.0)
        self.assertTrue(all(h.source is self.scene for h in whole))

    def test_identifier_embeds_the_full_size_face(self):
        from person_detection.storage.recorder import FaceIdentifier
        from person_detection.vision.detector import Detection
        from person_detection.vision.face import FaceHit

        seen = []

        class Recognizer:
            match_threshold = 0.3

            def embed(self, frame, row):
                seen.append((frame.shape, float(row[2])))
                return np.ones(4, np.float32)

            def similarity(self, a, b):
                return 1.0

        class Gallery:
            def gallery(self):
                return {"A": np.ones(4, np.float32)}

        small_row = np.zeros(15, np.float32)
        small_row[2] = 20  # 20 px wide in the processed frame...
        big_row = small_row.copy()
        big_row[2] = 60  # ...60 px in the camera picture
        hit = FaceHit((100, 100, 120, 120), small_row, np.zeros((1080, 1920, 3), np.uint8), big_row)
        ident = FaceIdentifier(Recognizer(), Gallery(), min_face_px=40)
        self.assertEqual(ident.identify(np.zeros((540, 960, 3), np.uint8),
                                        Detection((50, 50, 200, 400), 1, 0.9), [hit]), "A")
        self.assertEqual(seen, [((1080, 1920, 3), 60.0)])  # big enough only at full size


@unittest.skipUnless(HAVE_MBF, "ArcFace MobileFaceNet not downloaded")
class ArcFaceTest(TempConfig):
    def setUp(self):
        super().setUp()
        from person_detection.vision.face import FaceDetector
        self.fd = FaceDetector(config.FACE_MODEL_PATH, 0.6)
        self.scene = cv2.imread(str(ASSETS / "zidane.jpg"))

    def test_same_face_is_close_other_faces_are_far(self):
        rec = face_models.create_recognizer("arcface_mbf")
        self.assertEqual(rec.model_key, "arcface_mbf")
        self.assertEqual(rec.match_threshold, config.FACE_MATCH_THRESHOLD_ARCFACE)
        faces = sorted(self.fd.detect_raw(self.scene), key=lambda f: f[0][0])[:2]
        small = cv2.resize(self.scene, (640, 360))
        faces_small = sorted(self.fd.detect_raw(small), key=lambda f: f[0][0])[:2]
        big = [rec.embed(self.scene, row) for _, row in faces]
        little = [rec.embed(small, row) for _, row in faces_small]
        self.assertEqual(big[0].shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(big[0])), 1.0, places=4)
        self.assertGreater(rec.similarity(big[0], little[0]), 0.7)  # same man, half the pixels
        self.assertLess(rec.similarity(big[0], big[1]), 0.3)  # two different men

    def test_switching_model_rebuilds_the_gallery_from_photos(self):
        from person_detection.enrollment import FaceEnroller, GalleryBuilder, known_names, missing_names
        crop = self.scene[100:720, 600:1000]
        sface_db = PeopleDB(people_db_path("sface"))
        FaceEnroller(self.fd, face_models.create_recognizer("sface"), sface_db, config.PHOTOS_DIR).enroll(
            "Zidane", [("z.jpg", cv2.imencode(".jpg", crop)[1].tobytes())])
        sface_db.enroll("NoPhotos", [np.ones(128, np.float32)])  # old enroll.py user, folder deleted
        arc_db = PeopleDB(people_db_path("arcface_mbf"))
        self.assertEqual(missing_names(arc_db, config.PHOTOS_DIR), ["NoPhotos", "Zidane"])
        enroller = FaceEnroller(self.fd, face_models.create_recognizer("arcface_mbf"), arc_db, config.PHOTOS_DIR)
        report = GalleryBuilder(enroller).run()
        self.assertEqual((report.added, report.no_photos, report.failed), (["Zidane"], ["NoPhotos"], {}))
        self.assertEqual(arc_db.gallery()["Zidane"].shape, (512,))
        self.assertEqual(sface_db.gallery()["Zidane"].shape, (128,))  # SFace gallery untouched
        # removing a person removes them under every model
        enroller.remove("Zidane")
        self.assertNotIn("Zidane", PeopleDB(people_db_path("sface")).names())
        self.assertNotIn("Zidane", PeopleDB(people_db_path("arcface_mbf")).names())
        self.assertEqual(known_names(arc_db, config.PHOTOS_DIR), ["NoPhotos"])

    def test_pipeline_uses_the_chosen_model(self):
        from person_detection.enrollment import FaceEnroller
        from person_detection.pipeline import CameraOptions, PipelineFactory
        config.FACE_RECOGNIZER = "arcface_mbf"
        factory = PipelineFactory()
        crop = self.scene[100:720, 600:1000]
        FaceEnroller(self.fd, face_models.create_recognizer(), factory.people_db(), config.PHOTOS_DIR).enroll(
            "Zidane", [("z.jpg", cv2.imencode(".jpg", crop)[1].tobytes())])
        processor = factory.create(CameraOptions(enable_vlm_par=False, save_csv=True, save_crops=False, long_side=640))
        big = cv2.resize(self.scene, (1920, 1080))  # 3x the processing size
        names = set()
        for _ in range(12):
            names |= {p.name for p in processor.process(big.copy()).people}
        perf = processor.perf()
        self.assertEqual(perf["face_model"], "arcface_mbf")
        self.assertTrue(perf["full_res"])
        self.assertIn("Zidane", names)


if __name__ == "__main__":
    unittest.main()
