"""Choosing models on the settings page, the face-gallery rebuild, and comparing models in the video test.

Run from the web folder:   python manage.py test cameras.tests_models
"""

import datetime as dt
import json
from pathlib import Path
from unittest import mock

import numpy as np
from django.test import TestCase

from person_detection import config
from person_detection.storage.people_db import PeopleDB, people_db_path
from person_detection.vision import face_models

from . import evaluation, services
from . import tests as base
from .models import EvalEvent, EvalVideo


class SettingsModelTest(base.PortalTestBase):
    form_data = base.SystemSettingsTest.form_data

    def test_model_choices_are_on_the_page(self):
        self.client.force_login(self.admin)
        html = self.client.get("/settings/").content.decode()
        for text in ("โมเดลจดจำใบหน้า", "โมเดลตรวจจับคน (YOLO pose)", 'value="arcface_mbf"', 'value="yolo26n-pose"',
                     "ใช้ ReID ในตัวติดตามคน", "เกณฑ์ความเหมือนของใบหน้า (ArcFace)"):
            self.assertIn(text, html)

    def test_switching_face_model_starts_the_rebuild(self):
        self.client.force_login(self.admin)
        with mock.patch.object(services.gallery, "start", return_value=True) as start:
            html = self.client.post("/settings/", self.form_data(FACE_RECOGNIZER="arcface_mbf",
                                                                 PERSON_MODEL="yolo26n-pose"), follow=True).content.decode()
        start.assert_called_once_with(force=True)
        self.assertIn("บันทึกแล้ว", html)
        self.assertEqual((config.FACE_RECOGNIZER, config.PERSON_MODEL), ("arcface_mbf", "yolo26n-pose"))
        saved = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved["FACE_RECOGNIZER"], "arcface_mbf")
        self.assertEqual(face_models.threshold_for(), config.FACE_MATCH_THRESHOLD_ARCFACE)

    def test_unknown_model_is_refused(self):
        self.client.force_login(self.admin)
        with mock.patch.object(services.gallery, "start") as start:
            html = self.client.post("/settings/", self.form_data(FACE_RECOGNIZER="facenet")).content.decode()
        self.assertIn("มีค่าที่ไม่ถูกต้อง", html)
        self.assertEqual(config.FACE_RECOGNIZER, "sface")
        start.assert_not_called()

    def test_gallery_progress_and_missing_people(self):
        PeopleDB(people_db_path("sface")).enroll("Arm", [np.ones(128, np.float32)])
        config.FACE_RECOGNIZER = "arcface_mbf"
        self.client.force_login(self.admin)
        html = self.client.get("/people/").content.decode()
        self.assertIn("ยังไม่มีข้อมูลใบหน้าของโมเดล <b>arcface_mbf</b>", html)
        self.assertIn("<b>Arm</b>", html)
        services.gallery.state = {"model": "arcface_mbf", "running": True, "phase": "faces", "done": 1, "total": 3,
                                  "current": "Oung", "no_photos": ["Arm"], "failed": {}, "added": []}
        try:
            part = self.client.get("/api/gallery/?partial=1").content.decode()
            self.assertIn("กำลังสร้างข้อมูลใบหน้าด้วยโมเดลใหม่", part)
            self.assertIn("1 / 3 คน", part)
            self.assertIn("ต้องลงทะเบียนใหม่", part)
            self.assertEqual(self.client.get("/api/gallery/").json()["phase"], "faces")
            self.assertIn('id="gallery-status"', self.client.get("/settings/").content.decode())
        finally:
            services.gallery.state = None

    def test_rebuild_button_needs_staff(self):
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.post("/people/gallery/rebuild/").status_code, 302)
        self.client.force_login(self.admin)
        with mock.patch.object(services.gallery, "start", return_value=True) as start:
            self.client.post("/people/gallery/rebuild/")
        start.assert_called_once_with(force=True)


class Row:
    def __init__(self, kind, t, box=None):
        self.kind, self.time_s, self.box = kind, t, box
        self.gt_status = self.gt_identity = self.gt_direction = ""
        self.saved = False

    def save(self, update_fields=None):
        self.saved = True


class CarryGroundTruthTest(TestCase):
    def test_same_person_same_moment_gets_the_labels(self):
        old = [
            {"kind": "sighting", "time_s": 2.0, "box": [0.1, 0.1, 0.3, 0.9], "gt_status": "ok", "gt_identity": "Arm", "gt_direction": ""},
            {"kind": "sighting", "time_s": 2.1, "box": [0.6, 0.1, 0.8, 0.9], "gt_status": "ok", "gt_identity": "__none__", "gt_direction": ""},
            {"kind": "in", "time_s": 5.0, "box": None, "gt_status": "", "gt_identity": "", "gt_direction": "ok"},
            {"kind": "sighting", "time_s": 9.0, "box": [0.1, 0.1, 0.3, 0.9], "gt_status": "false", "gt_identity": "", "gt_direction": ""},
        ]
        new = [Row("sighting", 2.4, [0.62, 0.12, 0.81, 0.9]),  # the stranger, a bit later
               Row("sighting", 2.5, [0.11, 0.1, 0.31, 0.88]),  # Arm
               Row("in", 5.3, [0.1, 0.1, 0.3, 0.9]),
               Row("sighting", 12.0, [0.1, 0.1, 0.3, 0.9]),  # too far in time
               Row("out", 5.0)]  # other kind
        self.assertEqual(evaluation.carry_ground_truth(old, new), 3)
        self.assertEqual([r.gt_identity for r in new[:2]], ["__none__", "Arm"])
        self.assertEqual(new[2].gt_direction, "ok")
        self.assertEqual((new[3].gt_status, new[4].gt_direction), ("", ""))
        self.assertFalse(new[3].saved)

    def test_labels(self):
        self.assertEqual(evaluation.model_label({}), "yolo11n-pose + SFace")
        self.assertEqual(evaluation.model_label({"person_model": "yolo26n-pose", "face_model": "arcface_mbf",
                                                 "reid": True, "full_res": True}),
                         "yolo26n-pose + ArcFace-MBF + ReID + full-res")
        self.assertEqual(evaluation.model_threshold({"face_model": "arcface_r50"}), config.FACE_MATCH_THRESHOLD_ARCFACE)


class CompareModelsPageTest(base.PortalTestBase):
    def make_video(self, face_model):
        video = EvalVideo.objects.create(
            name="door", file="x.mp4", status="done", finished_at=dt.datetime(2026, 10, 29, tzinfo=dt.timezone.utc),
            stats={"face_model": face_model, "person_model": "yolo11n-pose", "ms_per_frame": 41.5, "frames_processed": 10},
            runs=[{"label": "yolo11n-pose + SFace", "finished_at": "2026-10-28T10:00:00", "threshold": 0.363,
                   "ms_per_frame": 60.2, "events": 4, "reviewed": 4, "precision": 0.75, "recall": 1.0, "f1": 0.857,
                   "far": 0.5, "frr": 0.0, "accuracy": 0.5, "coverage": 1.0, "latency_s": 0.8}])
        EvalEvent.objects.create(video=video, time_s=1, face_seen=True, candidate="Arm", score=0.40, gt_status="ok",
                                 gt_identity="Arm", box=[0.1, 0.1, 0.3, 0.9])
        return video

    def test_detail_uses_the_threshold_of_the_runs_face_model(self):
        video = self.make_video("arcface_mbf")
        self.client.force_login(self.admin)
        html = self.client.get(f"/evaluate/{video.id}/").content.decode()
        self.assertIn("เปรียบเทียบโมเดลกับวิดีโอนี้", html)
        self.assertIn("รอบนี้: yolo11n-pose + ArcFace-MBF", html)
        self.assertIn(f'name="t" value="{config.FACE_MATCH_THRESHOLD_ARCFACE}"', html)
        self.assertIn("yolo11n-pose + SFace", html)  # the earlier run
        self.assertIn("60.2", html)
        # score 0.40 >= ArcFace's threshold: counted as a correct name
        m = evaluation.video_metrics(video, evaluation.model_threshold(video.stats))
        self.assertEqual(m["face"]["correct"], 1)
        listing = self.client.get("/evaluate/").content.decode()
        self.assertIn("+1 รอบ", listing)

    def test_apply_threshold_goes_to_the_model_in_use(self):
        config.FACE_RECOGNIZER = "arcface_mbf"
        self.client.force_login(self.admin)
        self.client.post("/evaluate/threshold/", {"threshold": "0.31"})
        self.assertEqual(config.FACE_MATCH_THRESHOLD_ARCFACE, 0.31)
        self.assertEqual(config.FACE_MATCH_THRESHOLD, 0.363)  # SFace's own value untouched
        saved = json.loads(Path(config.SETTINGS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(saved["FACE_MATCH_THRESHOLD_ARCFACE"], 0.31)
