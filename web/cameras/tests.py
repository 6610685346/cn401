"""Portal tests.  Run from the web folder:   python manage.py test cameras

Uses the real models in new/models (VLM-PAR is switched off so the tests also
run on a machine without the 400 MB checkpoint).
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from django.contrib.auth.models import User
from django.test import TestCase

from person_detection import config

from . import services
from .models import Camera

ASSETS = Path(ultralytics.__file__).parent / "assets"


def jpeg(path, width=None, height=None) -> bytes:
    image = cv2.imread(str(path))
    if width:
        image = cv2.resize(image, (width, height))
    return cv2.imencode(".jpg", image)[1].tobytes()


class PortalTestBase(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.admin = User.objects.create_superuser("boss", "boss@example.com", "pw-boss-123")
        cls.viewer = User.objects.create_user("viewer", password="pw-viewer-123")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = (config.OUTPUT_DIR, config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH, config.PHOTOS_DIR)
        config.OUTPUT_DIR = Path(self.tmp)
        config.PEOPLE_DB_PATH = Path(self.tmp) / "people.json"
        config.WARDROBE_LOG_PATH = Path(self.tmp) / "wardrobe.json"
        config.PHOTOS_DIR = Path(self.tmp) / "photos"
        self._saved_settings_path = config.SETTINGS_PATH
        config.SETTINGS_PATH = Path(self.tmp) / "settings.json"
        services.reset_shared_state()

    def tearDown(self):
        config.OUTPUT_DIR, config.PEOPLE_DB_PATH, config.WARDROBE_LOG_PATH, config.PHOTOS_DIR = self._saved
        config.SETTINGS_PATH = self._saved_settings_path
        from . import system_settings
        system_settings.apply({})  # back to defaults for the next test
        services.reset_shared_state()
        services.workers.stop_all()
        services.sessions._sessions.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)


class AccessTest(PortalTestBase):
    def test_every_page_needs_login(self):
        for url in ["/", "/my-camera/", "/history/", "/people/", "/api/status/", "/cameras/add/"]:
            response = self.client.get(url)
            self.assertEqual(response.status_code, 302, url)
            self.assertIn("/login/", response["Location"], url)

    def test_viewer_can_look_but_not_manage(self):
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/history/").status_code, 200)
        self.assertEqual(self.client.get("/cameras/add/").status_code, 302)  # to admin login
        camera = Camera.objects.create(name="Gate", source="rtsp://a:b@1.2.3.4/x")
        self.client.post(f"/cameras/{camera.id}/toggle/", {"run": "1"})
        camera.refresh_from_db()
        self.assertFalse(camera.should_run)


class CameraFormTest(PortalTestBase):
    def test_add_validates_source(self):
        self.client.force_login(self.admin)
        response = self.client.post("/cameras/add/", {"name": "Bad", "source": "ftp://x", "rotate": 0, "long_side": 960})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Camera.objects.exists())
        response = self.client.post("/cameras/add/", {"name": "USB", "source": " 0 ", "rotate": 0, "long_side": 960})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(Camera.objects.get().source, "0")

    def test_password_never_shown(self):
        self.client.force_login(self.admin)
        Camera.objects.create(name="Gate", source="rtsp://admin:%23Secret99@192.168.1.13:554/media/video2")
        html = self.client.get("/").content.decode()
        self.assertNotIn("Secret99", html)
        self.assertIn("rtsp://admin:****@192.168.1.13", html)
        self.assertEqual(services.mask_passwords("x rtsp://u:p@h/y"), "x rtsp://u:****@h/y")


class BrowserFrameTest(PortalTestBase):
    URL = "/api/browser/frame/?session=test-session-1&vlm=0"

    def post(self, body, url=None):
        return self.client.post(url or self.URL, data=body, content_type="image/jpeg")

    def test_people_found_at_any_resolution(self):
        self.client.force_login(self.admin)
        for width, height in [(640, 360), (1280, 720), (3840, 2160)]:
            data = self.post(jpeg(ASSETS / "zidane.jpg", width, height)).json()
            self.assertEqual((data["source_width"], data["source_height"]), (width, height))
            self.assertEqual((data["width"], data["height"]), (960, 540))
            self.assertGreaterEqual(data["count"], 2, (width, height))
            for person in data["people"]:
                self.assertTrue(all(0 <= v <= 1 for v in person["box"]))

    def test_bad_requests(self):
        self.client.force_login(self.admin)
        self.assertEqual(self.post(b"not an image").status_code, 400)
        self.assertEqual(self.post(b"").status_code, 400)
        self.assertEqual(self.post(jpeg(ASSETS / "bus.jpg"), "/api/browser/frame/?session=x").status_code, 400)
        self.assertEqual(self.post(jpeg(ASSETS / "bus.jpg"), self.URL + "&long_side=123").status_code, 400)
        self.assertEqual(self.client.get(self.URL).status_code, 405)

    def test_session_limit(self):
        self.client.force_login(self.admin)
        services.sessions.max_sessions = 1
        try:
            image = jpeg(ASSETS / "bus.jpg")
            self.assertEqual(self.post(image).status_code, 200)
            other = self.post(image, "/api/browser/frame/?session=another-session&vlm=0")
            self.assertEqual(other.status_code, 429)
            self.client.post("/api/browser/close/?session=test-session-1")
            self.assertEqual(self.post(image, "/api/browser/frame/?session=another-session&vlm=0").status_code, 200)
        finally:
            services.sessions.max_sessions = 6
            services.sessions._sessions.clear()


class HistoryTest(PortalTestBase):
    def test_reads_old_and_new_csv_formats(self):
        old = Path(self.tmp) / "person_log_2026-09-30.csv"  # first version had no name column
        old.write_text("timestamp,track_id,shirt_color,pants_color,confidence\n"
                       "2026-09-30 10:00:00,1,red,blue,0.81\n", encoding="utf-8-sig")
        new = Path(self.tmp) / "person_log_2026-09-30_cam-Gate-HD.csv"
        new.write_text("timestamp,track_id,name,shirt_color,pants_color,confidence\n"
                       "2026-09-30 11:00:00,4,Oung,black,black,0.90\n", encoding="utf-8-sig")
        Camera.objects.create(name="Gate HD", source="http://1.2.3.4/video")
        self.client.force_login(self.viewer)
        html = self.client.get("/history/?date=2026-09-30").content.decode()
        self.assertIn("Oung", html)
        self.assertIn("กล้อง: Gate HD", html)
        self.assertIn("connect.py", html)
        self.assertIn("2 รายการ", html)
        html = self.client.get("/history/?date=2026-09-30&name=oung").content.decode()
        self.assertIn("1 รายการ", html)

    def test_download_only_log_files(self):
        (Path(self.tmp) / "person_log_2026-10-01.csv").write_text("timestamp\n", encoding="utf-8")
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get("/history/download/person_log_2026-10-01.csv").status_code, 200)
        self.assertEqual(self.client.get("/history/download/..%2Fsettings.py").status_code, 404)
        self.assertEqual(self.client.get("/history/download/people.json").status_code, 404)


class WorkerTest(PortalTestBase):
    def test_worker_runs_a_video_file_and_logs(self):
        # A short clip stands in for an IP camera; the worker loops it.
        clip = Path(self.tmp) / "clip.avi"
        image = cv2.resize(cv2.imread(str(ASSETS / "zidane.jpg")), (1920, 1080))
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1920, 1080))
        for _ in range(15):
            writer.write(image)
        writer.release()
        camera = Camera.objects.create(name="Clip", source="http://placeholder", enable_vlm_par=False)
        camera.source = str(clip)  # bypass form validation: files are for tests only
        services.workers.start(camera)
        worker = services.workers.get(camera.id)
        import time
        deadline = time.time() + 60
        while time.time() < deadline and not worker.recent:
            time.sleep(0.5)
        state = worker.snapshot()
        self.assertEqual(state["source_size"], (1920, 1080))
        self.assertEqual(state["process_size"], (960, 540))
        self.assertTrue(worker.jpeg)
        self.assertTrue(any(Path(self.tmp).glob("person_log_*_cam-Clip.csv")))
        services.workers.stop(camera.id)
        worker.join(5)
        self.assertFalse(worker.alive)


class EnrollViewTest(PortalTestBase):
    """The "คนที่ลงทะเบียน" page: upload photos, add more, delete."""

    def photos(self, count=4):
        from django.core.files.uploadedfile import SimpleUploadedFile
        scene = cv2.imread(str(ASSETS / "zidane.jpg"))
        face = scene[120:720, 380:860]
        variants = [face, cv2.resize(face, None, fx=2.5, fy=2.5), cv2.convertScaleAbs(face, alpha=1.1, beta=15),
                    cv2.convertScaleAbs(face, alpha=0.9, beta=-10)]
        return [SimpleUploadedFile(f"arm{i}.jpg", cv2.imencode(".jpg", img)[1].tobytes(), "image/jpeg")
                for i, img in enumerate(variants[:count], start=1)]

    def test_upload_four_photos_then_recognised(self):
        self.client.force_login(self.admin)
        response = self.client.post("/people/enroll/", {"name": "Arm", "photos": self.photos()})
        html = response.content.decode()
        self.assertEqual(response.status_code, 200)
        self.assertIn("ลงทะเบียน Arm แล้ว", html)
        self.assertIn("ใช้ 4 รูป", html)
        self.assertIn("data:image/jpeg;base64,", html)  # face crops shown back
        self.assertEqual(services.factory.people_db().names(), ["Arm"])
        self.assertEqual(len(list((Path(self.tmp) / "photos" / "Arm").glob("*.jpg"))), 4)
        # the browser-camera pipeline now puts the name on the person
        frame = jpeg(ASSETS / "zidane.jpg", 1920, 1080)
        for _ in range(6):
            data = self.client.post("/api/browser/frame/?session=enroll-check-1&vlm=0",
                                    data=frame, content_type="image/jpeg").json()
        self.assertIn("Arm", [p["name"] for p in data["people"]])

    def test_people_page_lists_photos_and_serves_thumbnails(self):
        self.client.force_login(self.admin)
        self.client.post("/people/enroll/", {"name": "อาร์ม", "photos": self.photos(2)})
        html = self.client.get("/people/").content.decode()
        self.assertIn("อาร์ม", html)
        self.assertIn("2 รูป", html)
        name = next((Path(self.tmp) / "photos" / "อาร์ม").glob("*.jpg")).name
        from urllib.parse import quote
        thumb = self.client.get(f"/people/{quote('อาร์ม')}/photos/{name}")
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb["Content-Type"], "image/jpeg")
        self.assertEqual(self.client.get(f"/people/{quote('อาร์ม')}/photos/..%2F..%2Fpeople.json").status_code, 404)
        self.assertEqual(self.client.get(f"/people/nobody/photos/{name}").status_code, 404)

    def test_errors_are_shown_and_nothing_saved(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        self.client.force_login(self.admin)
        blank = SimpleUploadedFile("blank.jpg", cv2.imencode(".jpg", np.zeros((300, 300, 3), np.uint8))[1].tobytes())
        response = self.client.post("/people/enroll/", {"name": "Arm", "photos": [blank]}, follow=True)
        self.assertIn("ไม่มีรูปที่ใช้ได้", response.content.decode())
        response = self.client.post("/people/enroll/", {"name": "../evil", "photos": self.photos(1)}, follow=True)
        self.assertIn("ชื่อใช้ได้เฉพาะ", response.content.decode())
        self.assertEqual(services.factory.people_db().names(), [])

    def test_delete_removes_person_and_photos(self):
        self.client.force_login(self.admin)
        self.client.post("/people/enroll/", {"name": "Arm", "photos": self.photos(2)})
        self.client.post("/people/delete/", {"name": "Arm"})
        self.assertEqual(services.factory.people_db().names(), [])
        self.assertFalse((Path(self.tmp) / "photos" / "Arm").exists())

    def test_viewer_cannot_enroll_or_delete(self):
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.post("/people/enroll/", {"name": "Arm", "photos": self.photos(1)}).status_code, 302)
        self.assertEqual(services.factory.people_db().names(), [])
        html = self.client.get("/people/").content.decode()
        self.assertNotIn("ลงทะเบียนคนใหม่", html)


class FloorCalibrationViewTest(PortalTestBase):
    IMAGE = [[0.2, 0.8], [0.8, 0.8], [0.7, 0.4], [0.3, 0.4]]
    FLOOR = [[0, 0], [3, 0], [3, 4], [0, 4]]

    def post(self, camera, body):
        import json
        return self.client.post(f"/api/cameras/{camera.id}/calibration/", json.dumps(body),
                                content_type="application/json")

    def test_save_check_and_clear(self):
        camera = Camera.objects.create(name="Gate", source="rtsp://a:b@1.2.3.4/x")
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get(f"/cameras/{camera.id}/calibrate/").status_code, 200)
        data = self.post(camera, {"image": self.IMAGE, "floor": self.FLOOR}).json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["grid"])
        camera.refresh_from_db()
        self.assertTrue(camera.is_calibrated)
        # wrong order of metres is refused and the saved calibration stays
        bad = self.post(camera, {"image": self.IMAGE, "floor": [[0, 0], [3, 0], [0, 4], [3, 4]]})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("ลำดับจุด", bad.json()["error"])
        self.assertEqual(self.post(camera, {"image": self.IMAGE, "floor": [["a", 0]] * 4}).status_code, 400)
        camera.refresh_from_db()
        self.assertTrue(camera.is_calibrated)
        self.assertTrue(self.post(camera, {"clear": True}).json()["cleared"])
        camera.refresh_from_db()
        self.assertFalse(camera.is_calibrated)

    def test_moving_the_picture_clears_calibration(self):
        camera = Camera.objects.create(name="Gate", source="rtsp://a:b@1.2.3.4/x")
        self.client.force_login(self.admin)
        self.post(camera, {"image": self.IMAGE, "floor": self.FLOOR})
        self.client.post(f"/cameras/{camera.id}/edit/", {"name": "Gate", "source": "rtsp://a:b@1.2.3.4/x",
                                                         "rotate": 90, "long_side": 960})
        camera.refresh_from_db()
        self.assertFalse(camera.is_calibrated)

    def test_viewer_cannot_calibrate_but_sees_map(self):
        camera = Camera.objects.create(name="Gate", source="rtsp://a:b@1.2.3.4/x")
        self.client.force_login(self.viewer)
        self.assertEqual(self.post(camera, {"image": self.IMAGE, "floor": self.FLOOR}).status_code, 302)
        self.assertEqual(self.client.get("/floor/").status_code, 200)
        self.assertEqual(self.client.get("/api/floor/").json(), {"people": [], "cameras": []})


class SystemSettingsTest(PortalTestBase):
    def form_data(self, **changes):
        from . import system_settings
        data = {}
        for setting in system_settings.SETTINGS:
            value = changes.get(setting.key, system_settings.DEFAULTS[setting.key])
            name = system_settings.field_name(setting.key)
            if setting.is_bool:
                if value:
                    data[name] = "on"
            else:
                data[name] = str(value)
        return data

    def test_viewer_cannot_open(self):
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get("/settings/").status_code, 302)

    def test_save_applies_live_and_writes_only_changes(self):
        import json
        from person_detection.multicam import hub as hub_module
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get("/settings/").status_code, 200)
        camera = Camera.objects.create(name="Clip", source="http://placeholder", enable_vlm_par=False, should_run=True)
        services.workers.start(camera)
        before = services.workers.get(camera.id)
        response = self.client.post("/settings/", self.form_data(**{
            "FRAME_SKIP": 3, "CONFIDENCE": 0.5, "hub.MATCH_RADIUS_M": 1.2, "SAVE_CROPS": False}), follow=True)
        html = response.content.decode()
        self.assertIn("บันทึกแล้ว", html)
        self.assertIn("รีสตาร์ทกล้อง 1 ตัว", html)  # CONFIDENCE needs new models
        self.assertEqual((config.FRAME_SKIP, config.CONFIDENCE, config.SAVE_CROPS), (3, 0.5, False))
        self.assertEqual(hub_module.MATCH_RADIUS_M, 1.2)
        self.assertIsNot(services.workers.get(camera.id), before)
        saved = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertEqual({k for k in saved if not k.startswith("_")},
                         {"FRAME_SKIP", "CONFIDENCE", "hub.MATCH_RADIUS_M", "SAVE_CROPS"})
        self.assertEqual(saved["_meta"]["updated_by"], "boss")
        # connect.py reads the same file at start-up (only its own keys)
        self.assertEqual(config.apply_saved_settings(config.SETTINGS_PATH),
                         {"FRAME_SKIP": 3, "CONFIDENCE": 0.5, "SAVE_CROPS": False})

    def test_hub_only_change_needs_no_restart(self):
        self.client.force_login(self.admin)
        camera = Camera.objects.create(name="Clip", source="http://placeholder", enable_vlm_par=False, should_run=True)
        services.workers.start(camera)
        before = services.workers.get(camera.id)
        html = self.client.post("/settings/", self.form_data(**{"hub.MERGE_AFTER": 10}), follow=True).content.decode()
        self.assertIn("มีผลทันที", html)
        self.assertIs(services.workers.get(camera.id), before)

    def test_bad_values_rejected_and_reset(self):
        self.client.force_login(self.admin)
        response = self.client.post("/settings/", self.form_data(FRAME_SKIP=0, CONFIDENCE=2))
        self.assertEqual(response.status_code, 200)
        self.assertIn("มีค่าที่ไม่ถูกต้อง", response.content.decode())
        self.assertFalse(config.SETTINGS_PATH.exists())
        self.client.post("/settings/", self.form_data(FRAME_SKIP=4))
        self.assertEqual(config.FRAME_SKIP, 4)
        self.client.post("/settings/", {"reset": "1"})
        from . import system_settings
        self.assertEqual(config.FRAME_SKIP, system_settings.DEFAULTS["FRAME_SKIP"])
        self.assertEqual(set(system_settings.read_saved()), {"_meta"})

    def test_hand_edited_bad_file_falls_back_to_defaults(self):
        from . import system_settings
        config.SETTINGS_PATH.write_text('{"FRAME_SKIP": "fast", "CONFIDENCE": 7}', encoding="utf-8")
        system_settings.load_and_apply()
        self.assertEqual(config.FRAME_SKIP, system_settings.DEFAULTS["FRAME_SKIP"])
        self.assertEqual(config.CONFIDENCE, system_settings.DEFAULTS["CONFIDENCE"])

    def test_frame_skip_reaches_the_worker(self):
        import time
        clip = Path(self.tmp) / "clip.avi"
        image = cv2.resize(cv2.imread(str(ASSETS / "zidane.jpg")), (640, 360))
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
        for _ in range(10):
            writer.write(image)
        writer.release()
        config.FRAME_SKIP = 1000  # never reached within the clip -> nothing processed
        camera = Camera.objects.create(name="Clip", source="http://placeholder", enable_vlm_par=False)
        camera.source = str(clip)
        services.workers.start(camera)
        time.sleep(4)
        worker = services.workers.get(camera.id)
        self.assertEqual(worker.status, "running")
        self.assertIsNone(worker.jpeg)
        config.FRAME_SKIP = 1  # read live: frames start coming through
        deadline = time.time() + 30
        while time.time() < deadline and worker.jpeg is None:
            time.sleep(0.5)
        self.assertIsNotNone(worker.jpeg)
