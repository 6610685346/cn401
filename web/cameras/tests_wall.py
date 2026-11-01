"""ดูหลายกล้อง (several cameras at once), the hidden video-test menu and the floor-plan switch.

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
    def test_menu_shown_and_can_be_hidden(self):
        self.client.force_login(self.admin)
        self.assertIn('href="/evaluate/"', self.client.get("/").content.decode())
        form = base.SystemSettingsTest.form_data(self, **{"web.SHOW_EVAL_MENU": False})
        self.client.post("/settings/", form)
        self.assertFalse(services.SHOW_EVAL_MENU)
        self.assertNotIn('href="/evaluate/"', self.client.get("/").content.decode())
        self.assertEqual(self.client.get("/evaluate/").status_code, 200)  # page still works by URL

    def test_train_part_off(self):
        services.EVAL_TRAINING = False
        self.client.force_login(self.admin)
        html = self.client.get("/evaluate/").content.decode()
        self.assertNotIn("จูนเกณฑ์ความเหมือนใบหน้าจากชุด train", html)
        self.assertNotIn('name="split" value="train"', html)
        self.assertIn('<input type="hidden" name="split" value="test">', html)
        self.assertEqual(self.client.post("/evaluate/threshold/", {"threshold": "0.4"}).status_code, 404)
        services.EVAL_TRAINING = True
        html = self.client.get("/evaluate/").content.decode()
        self.assertIn("จูนเกณฑ์ความเหมือนใบหน้าจากชุด train", html)


class FloorMapSwitchTest(base.PortalTestBase):
    def test_switch_off_and_on(self):
        from . import system_settings
        camera = Camera.objects.create(name="ประตู", source="0")
        self.client.force_login(self.admin)
        self.assertIn('href="/floor/"', self.client.get("/").content.decode())
        self.assertIn("ปิดระบบแผนผัง", self.client.get("/settings/").content.decode())

        self.client.post("/settings/floor-map/", {"on": "0"})
        self.assertFalse(services.FLOOR_MAP)
        self.assertEqual(system_settings.read_saved().get("web.FLOOR_MAP"), False)
        home = self.client.get("/").content.decode()
        self.assertNotIn('href="/floor/"', home)
        self.assertNotIn(f"/cameras/{camera.id}/calibrate/", home)
        for url in ("/floor/", "/api/floor/", f"/cameras/{camera.id}/calibrate/"):
            self.assertEqual(self.client.get(url).status_code, 404, url)
        self.assertEqual(self.client.post(f"/api/cameras/{camera.id}/calibration/", "{}",
                                          content_type="application/json").status_code, 404)
        self.assertIn("เปิดระบบแผนผัง", self.client.get("/settings/").content.decode())

        # saving the settings form (which has no field for it) keeps it off
        self.client.post("/settings/", base.SystemSettingsTest.form_data(self))
        self.assertFalse(services.FLOOR_MAP)

        self.client.post("/settings/floor-map/", {"on": "1"})
        self.assertTrue(services.FLOOR_MAP)
        self.assertEqual(self.client.get("/floor/").status_code, 200)

    def test_viewer_cannot_switch(self):
        self.client.force_login(self.viewer)
        self.client.post("/settings/floor-map/", {"on": "0"})
        self.assertTrue(services.FLOOR_MAP)

    def test_link_leaves_the_hub_when_off(self):
        from types import SimpleNamespace
        from person_detection.multicam.hub import IdentityHub
        from person_detection.multicam.link import CameraLink
        on = [True]
        hub = IdentityHub()
        link = CameraLink(hub, "a", enabled=lambda: on[0])
        person = SimpleNamespace(track_id=1, foot=None, name=None, global_id=None, cameras=[])
        link(SimpleNamespace(people=[person]))
        self.assertIsNotNone(person.global_id)
        self.assertIsNotNone(link.lookup(1))
        on[0] = False
        later = SimpleNamespace(track_id=1, foot=None, name=None, global_id=None, cameras=[])
        link(SimpleNamespace(people=[later]))
        self.assertIsNone(later.global_id)
        self.assertIsNone(link.lookup(1))
        self.assertEqual(hub.snapshot(), [])
