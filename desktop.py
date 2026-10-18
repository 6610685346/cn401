"""CN401 desktop app: the camera system in its own window.

    python desktop.py            (or double-click "CN401 App.bat")

What it does
------------
* Starts the Django project from web/ inside this process, on 127.0.0.1 only
  (a random free port), and shows it in a desktop window (pywebview, which on
  Windows uses Microsoft Edge WebView2). The window signs itself in with a
  secret made fresh on every start, so there is no login screen.
  Everything is in the window: IP cameras, the computer's own camera, people
  (face enrollment), floor calibration, the floor map, settings, admin.
* Also serves the LAN web page on port 8000 (HTTPS): other computers can only
  log in and read / download the history. See web/portal/modes.py.
* Cameras marked "running" start with the app and keep recording while the
  window is open (minimised is fine). Closing the window stops them.

Options:  --web-port 8000   --web-http   --no-web   --no-autostart
          --browser (use an Edge app window instead of pywebview)   --debug
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
WEBVIEW_DATA = WEB_DIR / ".webview"  # window cookies / remembered choices (not in git)
TITLE = "CN401 ตรวจจับคน"
INSTANCE_PORT = 47401  # held while the app runs, so it cannot be opened twice
DEFAULT_WEB_PORT = 8000

log = logging.getLogger("desktop")


# ------------------------------------------------------------------- basics
def setup_logging(debug: bool = False) -> None:
    """Log to logs/app.log. Without a console (pythonw) print() goes there too."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > 5 * 1024 * 1024:
        try:
            os.replace(LOG_FILE, LOG_FILE.with_suffix(".log.1"))
        except OSError:
            pass
    handlers: list[logging.Handler] = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
    if sys.stdout is None or sys.stderr is None:  # started with pythonw: no console
        stream = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    else:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)


def claim_single_instance() -> socket.socket | None:
    """Hold a local port for the app's lifetime; None if another copy holds it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", INSTANCE_PORT))
        sock.listen(1)
    except OSError:
        sock.close()
        return None
    return sock


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def message_box(text: str, title: str = TITLE) -> None:
    """A plain Windows message box (works before any window exists)."""
    log.warning("%s", text)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, title, 0x40)
            return
        except Exception:
            pass
    print(text)


def page(title: str, body: str) -> str:
    """Small self-contained HTML page for the window (loading / error)."""
    return f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><style>
    body{{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
    font-family:"Segoe UI","Leelawadee UI",system-ui,sans-serif;background:#f4f6fa;color:#1d2433}}
    @media (prefers-color-scheme:dark){{body{{background:#14171f;color:#e6e9ef}}}}
    .box{{max-width:640px;padding:32px;text-align:center}} h1{{font-size:22px;margin:0 0 10px}}
    p{{opacity:.8;line-height:1.6}} pre{{text-align:left;white-space:pre-wrap;font-size:12px;
    background:rgba(127,127,127,.12);padding:12px;border-radius:8px;max-height:40vh;overflow:auto}}
    .spin{{width:36px;height:36px;margin:0 auto 18px;border:4px solid rgba(127,127,127,.25);
    border-top-color:#2459d6;border-radius:50%;animation:s 1s linear infinite}}
    @keyframes s{{to{{transform:rotate(360deg)}}}}</style></head>
    <body><div class="box"><h1>{title}</h1>{body}</div></body></html>"""


LOADING_HTML = page("CN401 ตรวจจับคน", '<div class="spin"></div><p>กำลังเปิดโปรแกรมและโหลดโมเดล…<br>'
                                        "ครั้งแรกอาจใช้เวลาสักครู่</p>")


# ------------------------------------------------------------------ servers
class ServerThread:
    """One uvicorn server running in a background thread."""

    def __init__(self, app, host: str, port: int, name: str, **ssl):
        import uvicorn

        config = uvicorn.Config(app, host=host, port=port, log_level="warning",
                                lifespan="off", access_log=False, **ssl)
        self.server = uvicorn.Server(config)
        self.name = name
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, name=f"server-{name}", daemon=True)

    def _run(self) -> None:
        try:
            self.server.run()
        except SystemExit:  # uvicorn exits like this when it cannot open the port
            self.error = self.error or "เปิดพอร์ตไม่ได้"
        except Exception as exc:
            self.error = f"{exc.__class__.__name__}: {exc}"
            log.exception("%s server crashed", self.name)

    def start(self, timeout: float = 20) -> bool:
        """Start and wait until it is listening; False (with .error) if it failed."""
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.server.started:
                return True
            if not self._thread.is_alive():
                self.error = self.error or "เซิร์ฟเวอร์หยุดทำงาน"
                return False
            time.sleep(0.05)
        self.error = "เริ่มเซิร์ฟเวอร์ไม่ทันเวลา"
        return False

    def stop(self, timeout: float = 5) -> None:
        self.server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout)


class Backend:
    """Django + the two servers: the app's private one and the LAN history page."""

    def __init__(self, args):
        self.args = args
        self.token = secrets.token_urlsafe(32)
        self.app_server: ServerThread | None = None
        self.web_server: ServerThread | None = None
        self.app_url = ""

    def start(self) -> str:
        """Set everything up; returns the URL the window should open."""
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "portal.settings")
        os.environ["PORTAL_APP_TOKEN"] = self.token
        os.environ["PORTAL_DEFAULT_MODE"] = "web"  # anything not on the app's own server
        os.environ["PORTAL_AUTOSTART"] = "0" if self.args.no_autostart else "1"
        if str(WEB_DIR) not in sys.path:
            sys.path.insert(0, str(WEB_DIR))

        import django
        django.setup()
        from django.core.management import call_command
        log.info("checking the database")
        call_command("migrate", interactive=False, verbosity=0)

        from portal.asgi import application  # also starts the saved cameras
        from portal.modes import APP, WEB, with_mode

        port = free_port()
        self.app_server = ServerThread(with_mode(application, APP), "127.0.0.1", port, "app")
        if not self.app_server.start():
            raise RuntimeError(f"เปิดเซิร์ฟเวอร์ของแอปไม่ได้: {self.app_server.error}")
        self.app_url = f"http://127.0.0.1:{port}"
        log.info("app server on %s", self.app_url)

        self._start_web(with_mode(application, WEB))
        return f"{self.app_url}/app-login/{self.token}/"

    def _start_web(self, web_app) -> None:
        from cameras import services
        from cameras.management.commands.runportal import ensure_certificate, lan_addresses

        info = services.web_info
        info.update(managed=True, enabled=False, urls=[], error="")
        if self.args.no_web:
            info["error"] = "ปิดไว้ (เปิดโปรแกรมด้วย --no-web)"
            return
        port = self.args.web_port
        try:
            addresses = lan_addresses()
            ssl = {} if self.args.web_http else ensure_certificate(addresses)
        except Exception as exc:
            info["error"] = f"สร้างใบรับรอง HTTPS ไม่ได้: {exc}"
            log.exception("certificate")
            return
        scheme = "http" if self.args.web_http else "https"
        server = ServerThread(web_app, "0.0.0.0", port, "web", **ssl)
        if server.start():
            self.web_server = server
            info.update(enabled=True, urls=[f"{scheme}://{ip}:{port}/" for ip in addresses])
            log.info("LAN history page on %s", info["urls"] or f"port {port}")
        else:
            info["error"] = (f"เปิดพอร์ต {port} ไม่ได้ (อาจมีโปรแกรมอื่นใช้อยู่ เช่น runportal "
                             f"ที่เปิดค้างไว้) · {server.error}")
            log.warning("LAN history page not started: %s", server.error)

    def stop(self) -> None:
        log.info("shutting down")
        try:
            from cameras import services
            services.workers.stop_all()  # cameras finish their current frame
        except Exception:
            log.exception("stopping cameras")
        for server in (self.web_server, self.app_server):
            if server:
                server.stop()


# ------------------------------------------------------------------- window
def allow_camera_for(window, origin: str) -> None:
    """Let our own page use the webcam without WebView2's permission prompt.

    Best effort: if anything here fails WebView2 simply asks the user, which
    also works.
    """
    try:
        from Microsoft.Web.WebView2.Core import CoreWebView2PermissionState  # pythonnet, Windows only
        from System import Action

        form = window.native
        core = form.browser.webview.CoreWebView2

        def on_permission(sender, args):
            if str(args.Uri).startswith(origin):
                args.State = CoreWebView2PermissionState.Allow

        def attach():  # WebView2 objects must be touched on the window's thread
            core.PermissionRequested += on_permission

        form.Invoke(Action(attach))
        log.info("camera permission granted to %s", origin)
    except Exception as exc:
        log.info("camera permission hook not installed (%s); WebView2 will ask", exc)


def run_window(backend: Backend, debug: bool) -> None:
    """pywebview window: shows 'loading', starts the backend, then the app."""
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True  # history CSV download
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    window = webview.create_window(
        TITLE, html=LOADING_HTML, width=1320, height=860, min_size=(960, 620),
        confirm_close=True, text_select=True, background_color="#f4f6fa")
    hooked = []

    def on_loaded():
        if backend.app_url and not hooked:
            hooked.append(True)
            allow_camera_for(window, backend.app_url)

    window.events.loaded += on_loaded

    def boot():
        try:
            url = backend.start()
        except Exception as exc:
            log.exception("start-up failed")
            window.load_html(page("เปิดโปรแกรมไม่สำเร็จ",
                                  f"<p>{exc}</p><pre>{traceback.format_exc()}</pre>"
                                  f"<p>รายละเอียดอยู่ใน {LOG_FILE}</p>"))
            return
        window.load_url(url)

    WEBVIEW_DATA.mkdir(parents=True, exist_ok=True)
    webview.start(boot, debug=debug, private_mode=False, storage_path=str(WEBVIEW_DATA),
                  localization={
                      "global.quitConfirmation": "ปิดโปรแกรม? กล้องทั้งหมดจะหยุดบันทึก",
                      "global.quit": "ปิดโปรแกรม",
                      "global.cancel": "ยกเลิก",
                      "global.saveFile": "บันทึกไฟล์",
                  })


def find_edge() -> str | None:
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"),
                 os.environ.get("LOCALAPPDATA")):
        if base:
            path = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            if path.exists():
                return str(path)
    return None


def run_browser_fallback(backend: Backend) -> None:
    """No pywebview / WebView2: use an Edge app window (or the default browser)."""
    url = backend.start()
    edge = find_edge()
    if edge:
        profile = WEB_DIR / ".edge-app"
        process = subprocess.Popen([edge, f"--app={url}", f"--user-data-dir={profile}",
                                    "--no-first-run", "--window-size=1320,860"])
        log.info("opened Edge app window (pid %s)", process.pid)
        process.wait()  # its own profile, so this returns when the window closes
        return
    import webbrowser
    webbrowser.open(url)
    try:  # keep running until the user closes this small control window
        import tkinter as tk
        root = tk.Tk()
        root.title(TITLE)
        tk.Label(root, text="CN401 กำลังทำงานอยู่ในเบราว์เซอร์\nปิดหน้าต่างนี้เพื่อหยุดกล้องและปิดโปรแกรม",
                 padx=24, pady=16).pack()
        tk.Button(root, text="ปิดโปรแกรม", command=root.destroy, padx=12).pack(pady=(0, 16))
        root.mainloop()
    except Exception:
        print("CN401 is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(3600)


# --------------------------------------------------------------------- main
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CN401 desktop app")
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT, help="LAN history page port")
    parser.add_argument("--web-http", action="store_true", help="LAN history page over plain HTTP")
    parser.add_argument("--no-web", action="store_true", help="do not serve the LAN history page")
    parser.add_argument("--no-autostart", action="store_true", help="do not start saved cameras")
    parser.add_argument("--browser", action="store_true", help="Edge app window instead of pywebview")
    parser.add_argument("--debug", action="store_true", help="more logging, developer tools")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.debug)
    lock = claim_single_instance()
    if lock is None:
        message_box("CN401 เปิดอยู่แล้ว ดูที่แถบงาน (taskbar)\nเปิดซ้ำจะทำให้กล้องบันทึกซ้ำสองชุด")
        return 1

    backend = Backend(args)
    code = 0
    try:
        try:
            if args.browser:
                raise ImportError("--browser")
            import webview  # noqa: F401
        except ImportError as exc:
            log.info("using the browser window (%s)", exc)
            run_browser_fallback(backend)
        else:
            run_window(backend, args.debug)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log.exception("app failed")
        message_box(f"CN401 ทำงานผิดพลาด:\n{exc}\n\nรายละเอียดใน {LOG_FILE}")
        code = 1
    finally:
        backend.stop()
        lock.close()
    logging.shutdown()
    # Model threads (torch, camera readers) are daemons; do not wait for them.
    os._exit(code)


if __name__ == "__main__":
    main()
