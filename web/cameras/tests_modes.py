"""Desktop app vs LAN web page (portal/modes.py).

    python manage.py test cameras

The test client has no ASGI scope, so the mode comes from
settings.PORTAL_DEFAULT_MODE ("app" while testing); WEB-mode tests override it.
ScopeTest checks the real mechanism: the server marks requests in the scope.
"""

import asyncio
from pathlib import Path

from django.contrib.auth.models import User
from django.core.asgi import get_asgi_application
from django.test import SimpleTestCase, override_settings

from portal.modes import APP, WEB, with_mode

from .models import Camera
from .tests import PortalTestBase

TOKEN = "test-token-1234567890"

# Every URL of the app that must NOT exist on the LAN page.
APP_ONLY_URLS = [
    "/cameras/add/", "/cameras/1/", "/cameras/1/edit/", "/cameras/1/stream.mjpg",
    "/cameras/1/calibrate/", "/cameras/1/snapshot.jpg", "/my-camera/", "/people/",
    "/settings/", "/floor/", "/api/floor/", "/api/status/", "/admin/", f"/app-login/{TOKEN}/",
]


@override_settings(PORTAL_DEFAULT_MODE=WEB, PORTAL_APP_TOKEN=TOKEN)
class WebPageTest(PortalTestBase):
    """Other computers in the LAN: log in and read the history, nothing else."""

    def test_needs_login_and_lands_on_history(self):
        response = self.client.get("/history/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response["Location"])
        response = self.client.post("/login/", {"username": "viewer", "password": "pw-viewer-123"})
        self.assertRedirects(response, "/history/", fetch_redirect_response=False)
        self.assertRedirects(self.client.get("/"), "/history/", fetch_redirect_response=False)

    def test_history_page_shows_only_history(self):
        (Path(self.tmp) / "person_log_2026-10-01_cam-Gate.csv").write_text(
            "timestamp,track_id,name,shirt_color,pants_color,confidence,global_id\n"
            "2026-10-01 09:00:00,3,Arm,gray,white,0.90,G1\n", encoding="utf-8-sig")
        self.client.force_login(self.admin)  # even an admin gets only the history
        html = self.client.get("/history/?date=2026-10-01").content.decode()
        self.assertIn("Arm", html)
        self.assertIn("ออกจากระบบ", html)
        for app_only in ("กล้อง IP", "ตั้งค่าระบบ", "คนที่ลงทะเบียน", "ผู้ดูแล", "แผนผัง"):
            self.assertNotIn(app_only, html)
        response = self.client.get("/history/download/person_log_2026-10-01_cam-Gate.csv")
        self.assertEqual(response.status_code, 200)

    def test_app_pages_do_not_exist(self):
        camera = Camera.objects.create(name="Gate", source="rtsp://a:b@1.2.3.4/x")
        urls = [u.replace("/1/", f"/{camera.id}/") for u in APP_ONLY_URLS]
        self.client.force_login(self.admin)
        # contrast: the same admin on the app server reaches these pages
        # (stream / snapshot are 404 there too while the camera is stopped)
        with override_settings(PORTAL_DEFAULT_MODE=APP):
            for url in urls:
                if not url.endswith((".mjpg", ".jpg")):
                    self.assertNotEqual(self.client.get(url).status_code, 404, url)
        self.client.force_login(self.admin)  # app-login above switched the user
        for url in urls:
            self.assertEqual(self.client.get(url).status_code, 404, url)
        for url in ("/api/browser/frame/?session=abcdefgh12", "/cameras/1/toggle/",
                    "/people/enroll/", "/api/cameras/1/calibration/"):
            self.assertEqual(self.client.post(url).status_code, 404, url)

    def test_app_account_cannot_log_in_on_the_web(self):
        with override_settings(PORTAL_DEFAULT_MODE=APP):
            self.client.get(f"/app-login/{TOKEN}/")  # creates the account
        self.client.logout()
        response = self.client.post("/login/", {"username": "cn401-app", "password": ""})
        self.assertEqual(response.status_code, 200)  # form shown again, not logged in
        self.assertNotIn("_auth_user_id", self.client.session)


@override_settings(PORTAL_APP_TOKEN=TOKEN)
class AppLoginTest(PortalTestBase):
    """The desktop window signs in with the secret made by desktop.py."""

    def test_right_secret_signs_in_as_the_app_account(self):
        response = self.client.get(f"/app-login/{TOKEN}/")
        self.assertRedirects(response, "/", fetch_redirect_response=False)
        html = self.client.get("/").content.decode()
        self.assertIn("แอปบนเครื่องนี้", html)
        self.assertNotIn("ออกจากระบบ", html)  # nowhere to log in again from
        self.assertIn("ตั้งค่าระบบ", html)
        user = User.objects.get(username="cn401-app")
        self.assertTrue(user.is_staff and user.is_superuser)
        self.assertFalse(user.has_usable_password())
        self.client.get(f"/app-login/{TOKEN}/")  # a second start reuses the account
        self.assertEqual(User.objects.filter(username="cn401-app").count(), 1)

    def test_own_camera_is_logged_as_this_pc(self):
        self.client.get(f"/app-login/{TOKEN}/")
        self.assertIn("กล้องของเครื่องนี้", self.client.get("/my-camera/").content.decode())
        (Path(self.tmp) / "person_log_2026-10-02_this-pc.csv").write_text(
            "timestamp,track_id,name,shirt_color,pants_color,confidence\n"
            "2026-10-02 10:00:00,1,Arm,gray,white,0.9\n", encoding="utf-8-sig")
        html = self.client.get("/history/?date=2026-10-02").content.decode()
        self.assertIn("กล้องของเครื่องนี้", html)
        self.assertNotIn("this-pc<", html)

    def test_wrong_or_missing_secret(self):
        self.assertEqual(self.client.get("/app-login/wrong-token-123456/").status_code, 404)
        with override_settings(PORTAL_APP_TOKEN=""):
            self.assertEqual(self.client.get("/app-login/x/").status_code, 404)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_secret_is_useless_on_the_web_server(self):
        with override_settings(PORTAL_DEFAULT_MODE=WEB):
            self.assertEqual(self.client.get(f"/app-login/{TOKEN}/").status_code, 404)
        self.assertFalse(User.objects.filter(username="cn401-app").exists())

    def test_dashboard_shows_the_lan_address(self):
        from . import services
        saved = dict(services.web_info)
        try:
            services.web_info.update(managed=True, enabled=True, urls=["https://192.168.1.50:8000/"])
            self.client.get(f"/app-login/{TOKEN}/")
            html = self.client.get("/").content.decode()
            self.assertIn("https://192.168.1.50:8000/", html)
            services.web_info.update(enabled=False, urls=[], error="เปิดพอร์ต 8000 ไม่ได้")
            self.assertIn("เปิดพอร์ต 8000 ไม่ได้", self.client.get("/").content.decode())
        finally:
            services.web_info.clear()
            services.web_info.update(saved)


async def asgi_get(app, path: str) -> int:
    """Send one GET straight to an ASGI app, the way uvicorn does; returns the status."""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
        "root_path": "", "headers": [(b"host", b"127.0.0.1")],
        "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000),
    }
    sent, body_given, never = [], False, asyncio.Event()

    async def receive():
        nonlocal body_given
        if not body_given:
            body_given = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await never.wait()  # the client stays connected

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


class ScopeTest(SimpleTestCase):
    """The listener a request came in on decides the mode, whatever the default."""

    async def test_scope_mark_wins(self):
        application = get_asgi_application()
        # app server: the page exists (asks for a login); web server: no such page
        self.assertEqual(await asgi_get(with_mode(application, APP), "/people/"), 302)
        self.assertEqual(await asgi_get(with_mode(application, WEB), "/people/"), 404)
        self.assertEqual(await asgi_get(with_mode(application, WEB), "/history/"), 302)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            with_mode(get_asgi_application(), "everything")
