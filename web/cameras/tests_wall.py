"""ดูหลายกล้อง (several cameras at once) and the hidden video-test menu.

Run from the web folder:   python manage.py test cameras.tests_wall
"""

import threading
import time

from . import services
from . import tests as base
from .models import Camera


class FakeWorker:
    def __init__(self):
        self.jpeg, self.jpeg_id = b"\xff\xd8first", 1

    def snapshot(self):
        return {"id": 0, "status": "running", "people_now": 2, "fps": 7.5}


class WallTest(base.PortalTestBase):
    def setUp(self):
        super().setUp()
        self.gate = Camera.objects.create(name="Gate", source="http://1.2.3.4/a", should_run=True)
        self.hall = Camera.objects.create(name="Hall", source="http://1.2.3.4/b", should_run=False)
        self.worker = FakeWorker()
        services.workers._workers[self.gate.id] = self.worker

    def tearDown(self):
        services.workers._workers.pop(self.gate.id, None)
        super().tearDown()

    def test_every_camera_gets_a_tile(self):
        self.client.force_login(self.viewer)
        html = self.client.get("/wall/").content.decode()
        self.assertIn("ดูหลายกล้อง", html)
        self.assertIn(f'data-frame-url="/cameras/{self.gate.id}/frame.jpg"', html)
        self.assertIn(f'data-frame-url="/cameras/{self.hall.id}/frame.jpg"', html)
        self.assertIn("กล้องนี้หยุดอยู่", html)
        self.assertNotIn('name="run" value="1"', html)  # only staff may start a camera
        self.client.force_login(self.admin)
        self.assertIn('name="run" value="1"', self.client.get("/wall/").content.decode())
        self.assertIn('href="/wall/"', self.client.get("/").content.decode())  # menu + dashboard button

    def test_frame_waits_for_a_new_picture(self):
        self.client.force_login(self.viewer)
        first = self.client.get(f"/cameras/{self.gate.id}/frame.jpg")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first["Content-Type"], "image/jpeg")
        self.assertEqual(first["X-Frame-Id"], "1")

        def new_frame():
            time.sleep(0.3)
            self.worker.jpeg, self.worker.jpeg_id = b"\xff\xd8second", 2

        threading.Thread(target=new_frame).start()
        started = time.monotonic()
        second = self.client.get(f"/cameras/{self.gate.id}/frame.jpg?after=1")
        self.assertGreater(time.monotonic() - started, 0.2)  # waited for frame 2
        self.assertEqual((second["X-Frame-Id"], second.content), ("2", b"\xff\xd8second"))
        self.assertEqual(self.client.get(f"/cameras/{self.hall.id}/frame.jpg").status_code, 404)  # not running

    def test_login_needed(self):
        self.assertEqual(self.client.get("/wall/").status_code, 302)
        self.assertEqual(self.client.get(f"/cameras/{self.gate.id}/frame.jpg").status_code, 302)


class EvalMenuTest(base.PortalTestBase):
    def test_hidden_until_switched_on(self):
        self.client.force_login(self.admin)
        self.assertNotIn('href="/evaluate/"', self.client.get("/").content.decode())
        self.assertEqual(self.client.get("/evaluate/").status_code, 200)  # page still works
        form = base.SystemSettingsTest.form_data(self, **{"web.SHOW_EVAL_MENU": True})
        self.client.post("/settings/", form)
        self.assertTrue(services.SHOW_EVAL_MENU)
        self.assertIn('href="/evaluate/"', self.client.get("/").content.decode())
