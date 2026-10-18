"""Face enrollment from uploaded photos.  Run from `new`:  python -m unittest discover -s tests

Test photos are cut from ultralytics' zidane.jpg: person A is the man on the
left, person B the man on the right.
"""

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from person_detection import config  # noqa: E402
from person_detection.enrollment import EnrollError, FaceEnroller, clean_name, load_photo  # noqa: E402
from person_detection.pipeline import CameraOptions, PipelineFactory  # noqa: E402
from person_detection.storage.people_db import PeopleDB  # noqa: E402
from person_detection.vision.face import FaceDetector, FaceRecognizer  # noqa: E402

import ultralytics  # noqa: E402

SCENE = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
PERSON_A = SCENE[120:720, 380:860]
PERSON_B = SCENE[20:500, 840:1100]
MODELS_PRESENT = Path(config.FACE_MODEL_PATH).exists() and Path(config.FACE_RECOGNITION_MODEL_PATH).exists()


def jpeg(image, quality=92) -> bytes:
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def phone_photo_sideways(image) -> bytes:
    """Pixels stored sideways + EXIF 'rotate 90° to view', like many phones do."""
    from PIL import Image
    sideways = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    pil = Image.fromarray(cv2.cvtColor(sideways, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW to display
    out = io.BytesIO()
    pil.save(out, "JPEG", exif=exif.tobytes())
    return out.getvalue()


def photos_of_a():
    big = cv2.resize(PERSON_A, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)  # 1440x1800
    brighter = cv2.convertScaleAbs(PERSON_A, alpha=1.15, beta=20)
    return [("a1.jpg", jpeg(PERSON_A)), ("a2_big.jpg", jpeg(big)),
            ("a3_bright.png", cv2.imencode(".png", brighter)[1].tobytes()),
            ("a4_phone.jpg", phone_photo_sideways(PERSON_A))]


class NameTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(clean_name("  Arm   K "), "Arm K")
        self.assertEqual(clean_name("อุ๋ง"), "อุ๋ง")  # Thai with tone marks
        for bad in ["", "../x", "a/b", "con:", "x" * 41, ".hidden", "name."]:
            with self.assertRaises(EnrollError, msg=bad):
                clean_name(bad)

    def test_exif_rotation_is_applied(self):
        upright = load_photo(phone_photo_sideways(PERSON_A))
        self.assertEqual(upright.shape[:2], PERSON_A.shape[:2])


@unittest.skipUnless(MODELS_PRESENT, "face models not downloaded")
class EnrollTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = PeopleDB(self.tmp / "people.json")
        self.recognizer = FaceRecognizer(config.FACE_RECOGNITION_MODEL_PATH, config.FACE_MATCH_THRESHOLD)
        self.enroller = FaceEnroller(FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE),
                                     self.recognizer, self.db, self.tmp / "photos")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_four_photos_of_one_person(self):
        result = self.enroller.enroll("Arm", photos_of_a())
        self.assertEqual(result.used, 4, [(r.filename, r.message) for r in result.reports])
        self.assertEqual(len(result.saved), 4)
        self.assertEqual(self.db.names(), ["Arm"])
        self.assertTrue(all(r.thumbnail for r in result.reports))
        # stored photos are capped at 1600 px
        biggest = max(max(cv2.imread(str(p)).shape[:2]) for p in result.saved)
        self.assertLessEqual(biggest, 1600)

    def test_bad_files_are_reported_not_fatal(self):
        blank = jpeg(np.full((480, 640, 3), 200, np.uint8))
        result = self.enroller.enroll("Arm", photos_of_a()[:2] + [("blank.jpg", blank), ("notes.txt", b"hello")])
        by_name = {r.filename: r for r in result.reports}
        self.assertFalse(by_name["blank.jpg"].ok)
        self.assertIn("ไม่พบใบหน้า", by_name["blank.jpg"].message)
        self.assertFalse(by_name["notes.txt"].ok)
        self.assertEqual(result.used, 2)
        self.assertEqual(len(result.saved), 2)  # unusable files are not kept

    def test_nothing_usable_changes_nothing(self):
        with self.assertRaises(EnrollError):
            self.enroller.enroll("Arm", [("blank.jpg", jpeg(np.zeros((300, 300, 3), np.uint8)))])
        self.assertEqual(self.db.names(), [])
        self.assertFalse((self.tmp / "photos" / "Arm").exists())

    def test_someone_else_in_the_set_is_left_out(self):
        result = self.enroller.enroll("Arm", photos_of_a()[:3] + [("b.jpg", jpeg(PERSON_B))])
        by_name = {r.filename: r for r in result.reports}
        self.assertFalse(by_name["b.jpg"].ok, by_name["b.jpg"])
        self.assertEqual(result.used, 3)

    def test_two_photos_of_someone_else_are_both_left_out(self):
        # The other person's two photos agree with each other; they must not
        # vote themselves in (found while testing the web page).
        whole_scene = ("scene.jpg", jpeg(SCENE))  # biggest confident face here is person B
        result = self.enroller.enroll("Arm", photos_of_a()[:3] + [("b.jpg", jpeg(PERSON_B)), whole_scene])
        by_name = {r.filename: r for r in result.reports}
        self.assertFalse(by_name["b.jpg"].ok)
        self.assertFalse(by_name["scene.jpg"].ok)
        self.assertEqual(result.used, 3)

    def test_half_and_half_is_refused(self):
        b2 = cv2.resize(PERSON_B, None, fx=2, fy=2)
        with self.assertRaises(EnrollError):
            self.enroller.enroll("Arm", photos_of_a()[:2] + [("b.jpg", jpeg(PERSON_B)), ("b2.jpg", jpeg(b2))])
        self.assertEqual(self.db.names(), [])

    def test_two_different_people_rejected(self):
        with self.assertRaises(EnrollError):
            self.enroller.enroll("Arm", [("a.jpg", jpeg(PERSON_A)), ("b.jpg", jpeg(PERSON_B))])

    def test_adding_photos_later_uses_all_of_them(self):
        self.enroller.enroll("Arm", photos_of_a()[:2])
        result = self.enroller.enroll("Arm", photos_of_a()[2:])
        self.assertEqual(result.used, 4)
        self.assertEqual(sum(1 for r in result.reports if not r.is_new), 2)
        self.assertEqual(len(self.enroller.photos_of("Arm")), 4)

    def test_look_alike_warning(self):
        self.enroller.enroll("Arm", photos_of_a()[:2])
        result = self.enroller.enroll("Arm twin", photos_of_a()[2:])
        self.assertIsNotNone(result.looks_like)
        self.assertEqual(result.looks_like[0], "Arm")

    def test_remove_deletes_photos(self):
        self.enroller.enroll("Arm", photos_of_a()[:2])
        self.enroller.remove("Arm")
        self.assertEqual(self.db.names(), [])
        self.assertFalse((self.tmp / "photos" / "Arm").exists())

    def test_running_camera_recognises_new_person(self):
        saved = (config.OUTPUT_DIR, config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH)
        config.OUTPUT_DIR, config.PEOPLE_DB_PATH = self.tmp, self.tmp / "people.json"
        config.WARDROBE_LOG_PATH = self.tmp / "wardrobe.json"
        try:
            factory = PipelineFactory()
            factory._people_db = self.db  # the camera shares the gallery, as in the web portal
            processor = factory.create(CameraOptions(enable_vlm_par=False))
            self.enroller.enroll("Arm", photos_of_a())
            for _ in range(6):
                result = processor.process(cv2.resize(SCENE, (1920, 1080)))
            self.assertIn("Arm", [p.name for p in result.people])
        finally:
            config.OUTPUT_DIR, config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH = saved


if __name__ == "__main__":
    unittest.main()
