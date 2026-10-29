"""ทดสอบด้วยวิดีโอ: upload -> simulated run -> review -> metrics -> tune / enroll.

Run from the web folder:   python manage.py test cameras.tests_eval
"""

import json
from pathlib import Path

import cv2
import numpy as np
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase

from person_detection import config

from . import evaluation, services
from . import tests as base
from .models import EvalEvent, EvalVideo


def clip_bytes(tmp: Path, frames: int = 40) -> bytes:
    """A short video of the zidane test picture sliding sideways (two people, faces visible)."""
    scene = cv2.resize(cv2.imread(str(base.ASSETS / "zidane.jpg")), (960, 540))
    path = tmp / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
    for i in range(frames):
        writer.write(cv2.warpAffine(scene, np.float32([[1, 0, i - 20], [0, 1, 0]]), (960, 540)))
    writer.release()
    return path.read_bytes()


class VideoTestPages(TransactionTestCase):
    """TransactionTestCase: the video runs in a background thread with its own
    database connection, which cannot see rows of an open test transaction."""

    def setUp(self):
        self.admin = User.objects.create_superuser("boss", "boss@example.com", "pw-boss-123")
        self.viewer = User.objects.create_user("viewer", password="pw-viewer-123")
        base.PortalTestBase.setUp(self)

    def tearDown(self):
        evaluation.jobs.wait_idle(60)
        base.PortalTestBase.tearDown(self)

    def upload(self, split="test", **extra):
        data = {"name": f"clip {split}", "split": split, "lighting": "indoor", "distance": "0-3",
                "video": SimpleUploadedFile("clip.mp4", clip_bytes(Path(self.tmp)), "video/mp4"), **extra}
        response = self.client.post("/evaluate/", data)
        self.assertEqual(response.status_code, 302)
        self.assertTrue(evaluation.jobs.wait_idle(300))
        return EvalVideo.objects.latest("created_at")

    def enroll_arm(self):
        self.client.post("/people/enroll/", {"name": "Arm", "photos": base.EnrollViewTest.photos(self, 3)})

    def test_upload_runs_like_a_camera_and_writes_no_history(self):
        self.client.force_login(self.admin)
        self.enroll_arm()
        video = self.upload()
        self.assertEqual(video.status, "done", video.message)
        self.assertGreater(video.stats["frames_processed"], 0)
        events = list(video.events.all())
        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(any(e.name == "Arm" and e.face_seen and e.score > 0.363 for e in events))
        self.assertTrue(all(Path(e.snapshot).is_file() for e in events))
        self.assertTrue(any(e.face_photos for e in events))
        self.assertFalse(list(Path(self.tmp).glob("person_log_*.csv")))  # real history untouched
        html = self.client.get(f"/evaluate/{video.id}/").content.decode()
        self.assertIn("ตรวจผลทีละแถว", html)
        self.assertIn("ระบบบอกว่า <b>Arm</b>", html)
        self.assertEqual(self.client.get(f"/evaluate/event/{events[0].id}/snapshot.jpg").status_code, 200)

    def test_rerun_keeps_the_last_run_and_the_ground_truth(self):
        self.client.force_login(self.admin)
        self.enroll_arm()
        video = self.upload(realtime="")  # every frame: both runs see the same frames
        self.client.post(f"/evaluate/{video.id}/accept/")
        labelled = video.events.exclude(gt_status="").count()
        self.assertGreater(labelled, 0)
        self.client.post(f"/evaluate/{video.id}/rerun/")
        self.assertTrue(evaluation.jobs.wait_idle(300))
        video.refresh_from_db()
        self.assertEqual(video.status, "done")
        self.assertEqual(len(video.runs), 1)
        self.assertEqual(video.runs[0]["label"], "yolo11n-pose + SFace + full-res" if config.FACE_FULL_RES else "yolo11n-pose + SFace")
        self.assertEqual(video.runs[0]["reviewed"], labelled)
        self.assertEqual(video.events.exclude(gt_status="").count(), labelled)  # all labels found again
        self.assertIn("ใช้ผลตรวจเดิมกับ", video.message)
        self.assertTrue(all(e.box for e in video.events.all()))

    def test_review_gives_metrics(self):
        self.client.force_login(self.admin)
        self.enroll_arm()
        video = self.upload()
        arm = next(e for e in video.events.all() if e.name == "Arm")
        other = next(e for e in video.events.all() if e.id != arm.id)
        post = lambda e, data: self.client.post(f"/api/evaluate/event/{e.id}/review/", json.dumps(data),  # noqa: E731
                                                content_type="application/json")
        self.assertEqual(post(arm, {"gt_status": "ok", "gt_identity": "Arm"}).status_code, 200)
        self.assertEqual(post(other, {"gt_status": "ok", "gt_identity": "__none__"}).status_code, 200)
        self.assertEqual(post(other, {"gt_status": "maybe"}).status_code, 400)
        self.client.post(f"/evaluate/{video.id}/missed/", {"missed_people": 1, "missed_in": 0, "missed_out": 0})
        m = evaluation.video_metrics(EvalVideo.objects.get(id=video.id), config.FACE_MATCH_THRESHOLD)
        self.assertEqual(m["detection"]["tp"], 2)
        self.assertEqual(m["detection"]["fn"], 1)
        self.assertEqual(m["face"]["correct"], 1)
        self.assertAlmostEqual(m["face"]["far"], 0.0)
        self.assertAlmostEqual(m["face"]["frr"], 0.0)
        html = self.client.get(f"/api/evaluate/{video.id}/metrics/").content.decode()
        self.assertIn("Precision", html)
        self.assertIn("100.0%", html)
        csv_text = self.client.get(f"/evaluate/{video.id}/export.csv").content.decode("utf-8-sig")
        self.assertIn("gt_identity", csv_text.splitlines()[0])
        report = self.client.get("/evaluate/report.csv").content.decode("utf-8-sig")
        self.assertIn("ชุดทดสอบ", report)

    def test_train_split_suggests_and_applies_threshold(self):
        self.client.force_login(self.admin)
        self.enroll_arm()
        video = self.upload(split="train")
        for e in video.events.all():
            e.gt_status = "ok"
            e.gt_identity = "Arm" if e.name == "Arm" else "__none__"
            e.save()
        html = self.client.get("/evaluate/").content.decode()
        self.assertIn("ค่าที่ผิดพลาดรวมน้อยที่สุด", html)
        self.assertIn("ชุดจูน (train)", html)
        self.client.post("/evaluate/threshold/", {"threshold": "0.42"})
        self.assertEqual(config.FACE_MATCH_THRESHOLD, 0.42)
        saved = json.loads(Path(config.SETTINGS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(saved["FACE_MATCH_THRESHOLD"], 0.42)

    def test_enroll_from_video_and_bulk_accept(self):
        self.client.force_login(self.admin)
        video = self.upload()  # nobody registered yet
        event = next(e for e in video.events.all() if e.face_photos)
        response = self.client.post(f"/evaluate/event/{event.id}/enroll/", {"name": "FromVideo"}, follow=True)
        self.assertIn("ลงทะเบียน FromVideo จากวิดีโอแล้ว", response.content.decode())
        self.assertIn("FromVideo", services.factory.people_db().names())
        self.client.post(f"/evaluate/{video.id}/accept/")
        self.assertFalse(EvalEvent.objects.filter(video=video, kind="sighting", gt_status="").exists())

    def test_delete_and_permissions(self):
        self.client.force_login(self.admin)
        video = self.upload()
        folder = evaluation.video_dir(video.id)
        self.assertTrue(folder.is_dir())
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get("/evaluate/").status_code, 200)
        self.assertNotIn("อัปโหลดวิดีโอใหม่", self.client.get("/evaluate/").content.decode())
        event = video.events.first()
        self.assertEqual(self.client.post(f"/api/evaluate/event/{event.id}/review/", "{}",
                                          content_type="application/json").status_code, 302)
        self.assertEqual(self.client.post(f"/evaluate/{video.id}/delete/").status_code, 302)
        self.assertTrue(EvalVideo.objects.filter(id=video.id).exists())  # viewer cannot delete
        self.client.force_login(self.admin)
        self.client.post(f"/evaluate/{video.id}/delete/")
        self.assertFalse(EvalVideo.objects.filter(id=video.id).exists())
        self.assertFalse(folder.exists())

    def test_bad_file_is_refused(self):
        self.client.force_login(self.admin)
        response = self.client.post("/evaluate/", {"video": SimpleUploadedFile("notes.txt", b"hello")}, follow=True)
        self.assertIn("ใช้ไม่ได้", response.content.decode())
        self.assertFalse(EvalVideo.objects.exists())
