"""Door line page, "ใครอยู่ตอนนี้", and photos in the history.

Run from the web folder:   python manage.py test cameras.tests_presence
"""

import csv
import datetime as dt
import json
import os
import unittest
from pathlib import Path

import cv2
import numpy as np

from person_detection import config

from . import presence, services
from . import tests as base
from .models import Camera

TODAY = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).date().isoformat()
COLUMNS = ["timestamp", "camera", "track_id", "name", "shirt_color", "pants_color", "confidence",
           "global_id", "similarity", "direction", "snapshot"]


class PresenceBase(base.PortalTestBase):
    def setUp(self):
        super().setUp()
        self._crops = config.CROPS_DIR
        config.CROPS_DIR = Path(self.tmp) / "crops"

    def tearDown(self):
        config.CROPS_DIR = self._crops
        super().tearDown()

    def photo(self, name):
        path = Path(config.CROPS_DIR) / TODAY / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((80, 40, 3), 200, np.uint8))
        return f"{TODAY}/{name}"

    def write_log(self, rows, tag="cam-Gate"):
        path = Path(self.tmp) / f"person_log_{TODAY}_{tag}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for t, name, direction, snap in rows:
                w.writerow([f"{TODAY} {t}", "Gate", 1, name, "red", "blue", "0.9", "", "", direction, snap])


class DoorLineTest(PresenceBase):
    def setUp(self):
        super().setUp()
        self.camera = Camera.objects.create(name="Gate", source="http://1.2.3.4/video")

    def post(self, data):
        return self.client.post(f"/api/cameras/{self.camera.id}/door/", json.dumps(data), content_type="application/json")

    def test_staff_draws_and_clears_the_line(self):
        self.client.force_login(self.admin)
        html = self.client.get(f"/cameras/{self.camera.id}/door/").content.decode()
        self.assertIn("เส้นประตู: Gate", html)
        self.assertEqual(self.post({"a": [0.2, 0.6], "b": [0.8, 0.6], "inside": "right"}).status_code, 200)
        self.camera.refresh_from_db()
        self.assertEqual(self.camera.door_line, {"a": [0.2, 0.6], "b": [0.8, 0.6], "inside": "right"})
        self.assertTrue(self.camera.enable_direction)
        spec = services.CameraSpec.from_model(self.camera)
        self.assertEqual(spec.options.door_line["inside"], "right")
        self.assertIn("นับเข้า/ออกอยู่", self.client.get("/").content.decode())
        self.assertEqual(self.post({"clear": True}).status_code, 200)
        self.camera.refresh_from_db()
        self.assertIsNone(self.camera.door_line)

    def test_bad_lines_are_refused(self):
        self.client.force_login(self.admin)
        self.assertEqual(self.post({"a": [0.5, 0.5], "b": [0.505, 0.5]}).status_code, 400)  # too short
        self.assertEqual(self.post({"a": [0.1, 0.5], "b": [1.5, 0.5]}).status_code, 400)  # outside picture
        self.assertEqual(self.post({"a": [0.1, 0.5]}).status_code, 400)
        self.assertEqual(self.post({"a": [0.1, 0.5], "b": [0.9, 0.5], "inside": "up"}).status_code, 400)

    def test_viewer_cannot_change_it(self):
        self.client.force_login(self.viewer)
        self.assertEqual(self.post({"a": [0.2, 0.6], "b": [0.8, 0.6]}).status_code, 302)
        self.assertEqual(self.client.get(f"/cameras/{self.camera.id}/door/").status_code, 302)


class PresencePageTest(PresenceBase):
    def test_who_is_inside(self):
        arm = self.photo("Gate_person_1_080000_in.jpg")
        self.write_log([
            ("08:00:00", "Arm", "in", arm),
            ("08:00:01", "Arm", "", ""),
            ("08:10:00", "Oung", "in", ""),
            ("09:30:00", "Oung", "out", ""),
            ("10:00:00", "unknown", "in", ""),
            ("10:05:00", "unknown", "in", ""),
            ("10:20:00", "unknown", "out", ""),
            ("11:00:00", "Mew", "", ""),
        ])
        self.client.force_login(self.viewer)
        html = self.client.get("/presence/").content.decode()
        self.assertIn("ใครอยู่ในพื้นที่ตอนนี้", html)
        p = presence.summarize(presence._csv_events(TODAY))
        self.assertEqual((p["ins"], p["outs"], p["inside_total"]), (4, 2, 2))
        self.assertEqual([x["name"] for x in p["inside_named"]], ["Arm"])
        self.assertEqual(p["inside_unknown"], 1)
        self.assertEqual([x["name"] for x in p["left"]], ["Oung"])
        self.assertEqual([x["name"] for x in p["seen"]], ["Mew"])
        self.assertEqual(p["hourly"][10], {"hour": 10, "in": 2, "out": 1})
        self.assertIn("ออกแล้ว 09:30", html)
        self.assertIn("/history/photo/?f=", html)
        partial = self.client.get(f"/presence/?partial=1&date={TODAY}").content.decode()
        self.assertNotIn("<html", partial)
        self.assertIn("อยู่ข้างใน", partial)

    def test_no_crossings_explains_the_door_line(self):
        Camera.objects.create(name="Gate", source="http://1.2.3.4/video")
        self.write_log([("08:00:00", "Arm", "", "")])
        self.client.force_login(self.admin)
        html = self.client.get("/presence/").content.decode()
        self.assertIn("ยังไม่มีการเดินเข้า-ออกวันนี้", html)
        self.assertIn("ยังมีกล้อง 1 ตัวที่ยังไม่วาดเส้น", html)
        self.assertEqual(self.client.get("/presence/?date=not-a-date").status_code, 404)

    def test_presence_is_on_the_lan_page_too(self):
        from django.test import override_settings
        with override_settings(PORTAL_DEFAULT_MODE="web"):
            self.client.force_login(self.viewer)
            self.assertEqual(self.client.get("/presence/").status_code, 200)


class HistoryPhotoTest(PresenceBase):
    def test_photo_column_and_safe_serving(self):
        snap = self.photo("Gate_person_1_080000.jpg")
        self.write_log([("08:00:00", "Arm", "", snap)])
        (Path(self.tmp) / "secret.jpg").write_bytes(b"not for you")
        self.client.force_login(self.viewer)
        html = self.client.get(f"/history/?date={TODAY}").content.decode()
        self.assertIn('class="thumb-btn"', html)
        ok = self.client.get(f"/history/photo/?f={snap}")
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok["Content-Type"], "image/jpeg")
        self.assertEqual(self.client.get("/history/photo/?f=../secret.jpg").status_code, 404)
        self.assertEqual(self.client.get("/history/photo/?f=/etc/passwd").status_code, 404)
        self.assertEqual(self.client.get("/history/photo/?log=1").status_code, 404)  # no database here
        self.client.logout()
        self.assertEqual(self.client.get(f"/history/photo/?f={snap}").status_code, 302)  # login needed


URL = os.environ.get("TEST_DATABASE_URL", "")


@unittest.skipUnless(URL, "set TEST_DATABASE_URL to run the PostgreSQL tests")
class PresenceDatabaseTest(PresenceBase):
    def setUp(self):
        from person_detection.storage import database as dbm
        self._url = config.DATABASE_URL
        config.DATABASE_URL = URL
        db = dbm.Database(URL)
        dbm.Base.metadata.drop_all(db.engine)
        db.create_tables()
        db.dispose()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        config.DATABASE_URL = self._url
        services.reset_shared_state()

    def test_from_database(self):
        from person_detection.storage import database as dbm
        db = services.factory.database()
        people = services.factory.people_db()
        people.enroll("Arm", [np.ones(128, np.float32)])
        snap = Path(config.CROPS_DIR) / TODAY / "x_in.jpg"
        snap.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snap), np.zeros((20, 10, 3), np.uint8))
        writer = services.factory.log_writer()
        writer.add(camera="Gate", person_id=people.person_id("Arm"), direction="in", snapshot_path=str(snap))
        writer.add(camera="Gate", person_id=None, direction="in")
        self.assertTrue(writer.flush())
        p = presence.summarize(presence.day_events(TODAY)[0])
        self.assertEqual([x["name"] for x in p["inside_named"]], ["Arm"])
        self.assertEqual(p["inside_total"], 2)
        log_id = db.Session().scalars(dbm.select(dbm.DetectionLog.id).where(dbm.DetectionLog.person_id.is_not(None))).first()
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get(f"/history/photo/?log={log_id}").status_code, 200)
        self.assertIn("Arm", self.client.get("/presence/").content.decode())
