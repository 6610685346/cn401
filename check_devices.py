"""CN401 - check a computer: GPU, cameras on it, IP cameras, network, devices on the LAN.

Portable: copy check_devices.py + check_devices.bat to ANY Windows computer
(no .venv, no project folder needed) and double-click the .bat. Only Python 3.8+
is needed (python.org); nothing is installed or changed.

    python check_devices.py                  everything
    python check_devices.py --quick          do not open the cameras (only list them)
    python check_devices.py --no-lan         do not look for devices on the LAN
    python check_devices.py --rtsp URL       also test this camera URL (repeat for more cameras)
    python check_devices.py --project PATH   the CN401 folder (found by itself when this
                                             file is inside it or next to it)

Prints, and saves to a .txt report:
  1. the computer      Windows, CPU, RAM, Python
  2. GPU               graphics cards, NVIDIA (nvidia-smi), PyTorch CUDA / OpenVINO if installed
  3. cameras on this PC   webcams / USB cameras Windows sees; with OpenCV installed also
                       opens each one (resolution, fps)
  4. IP cameras        those added in the CN401 app / config.py (only on a computer with the
                       project): network reachable? user / password / path right? (RTSP DESCRIBE,
                       no OpenCV needed); with OpenCV also reads one picture
  5. network           this computer's IP addresses
  6. packages, models  (with the project) versions and model files
  7. LAN devices       cameras / routers on this computer's networks (only their x.y.z.1-254):
                       ONVIF discovery, camera ports, ARP table -> IP, MAC, brand, RTSP URL guess

Camera passwords are never printed or saved.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import base64
import hashlib
import ipaddress
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # no OpenCV warnings for cameras that are not there

IS_WINDOWS = os.name == "nt"
QUICK = "--quick" in sys.argv
SCRIPT_DIR = Path(__file__).resolve().parent


def find_project() -> Path | None:
    """The CN401 folder (the one holding new/person_detection/config.py), if this PC has it."""
    if "--project" in sys.argv:
        i = sys.argv.index("--project")
        candidates = [Path(sys.argv[i + 1])] if i + 1 < len(sys.argv) else []
    else:
        candidates = [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent, Path.cwd()]
    for folder in candidates:
        if (folder / "new" / "person_detection" / "config.py").exists():
            return folder.resolve()
    return None


ROOT = find_project()  # None on a computer without the project
HERE = ROOT / "new" if ROOT else None

lines: list[str] = []
summary: list[tuple[str, str]] = []  # (ok/warn/bad, text)


def out(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def head(title: str) -> None:
    out()
    out(f"=== {title} " + "=" * max(0, 60 - len(title)))


def note(level: str, text: str) -> None:
    summary.append((level, text))


def run(cmd: list[str], timeout: float = 20) -> str:
    """Output of a command, or '' if it is missing / fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return result.stdout.strip()
    except Exception:
        return ""


def powershell(script: str, timeout: float = 30) -> str:
    if not IS_WINDOWS or not shutil.which("powershell"):
        return ""
    return run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + script], timeout)


def mask(url: str) -> str:
    """Hide the password in rtsp://user:pass@host/..."""
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1****@", str(url))


# ------------------------------------------------------------------ 1. computer
def check_computer() -> None:
    head("1. เครื่องคอมพิวเตอร์")
    out(f"ระบบ      : {platform.system()} {platform.release()} ({platform.version()})")
    cpu = platform.processor()
    if IS_WINDOWS:
        cpu = powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name") or cpu
    elif Path("/proc/cpuinfo").exists():
        found = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        cpu = found.group(1) if found else cpu
    out(f"CPU       : {cpu.strip()} · {os.cpu_count()} threads")
    ram = _ram_gb()
    if ram:
        out(f"RAM       : {ram:.1f} GB")
        if ram < 7.5:
            note("warn", f"RAM {ram:.1f} GB ค่อนข้างน้อย: เปิดได้ราว 1-2 กล้อง ควรปิด VLM-PAR")
    out(f"Python    : {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        note("warn", "Python เก่ากว่า 3.10: รันตัวตรวจนี้ได้ แต่โปรแกรม CN401 ต้องใช้ 3.10 ขึ้นไป")


def _ram_gb() -> float | None:
    if IS_WINDOWS:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 2**30
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


# ------------------------------------------------------------------ 2. GPU
def check_gpu() -> None:
    head("2. การ์ดจอ / ตัวเร่งความเร็ว")
    if IS_WINDOWS:
        cards = powershell("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }")
        for card in cards.splitlines():
            out(f"การ์ดจอ   : {card.strip()}")
    smi = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]) \
        if shutil.which("nvidia-smi") else ""
    if smi:
        out(f"NVIDIA    : {smi}")
    else:
        out("NVIDIA    : ไม่พบ (ไม่มีการ์ด NVIDIA หรือยังไม่ได้ลงไดรเวอร์)")

    try:
        import torch
        cuda = torch.cuda.is_available()
        out(f"PyTorch   : {torch.__version__} · CUDA {'ใช้ได้' if cuda else 'ใช้ไม่ได้'}"
            + (f" · {torch.cuda.get_device_name(0)} · CUDA {torch.version.cuda}" if cuda else ""))
        if cuda:
            note("ok", f"YOLO รันบน GPU: {torch.cuda.get_device_name(0)}")
        elif smi:
            note("bad", "มีการ์ด NVIDIA แต่ PyTorch เป็นรุ่น CPU: ลง PyTorch แบบ CUDA ตามหัวข้อ GPU ใน how _to_use.txt "
                        "(เร็วขึ้นหลายเท่า)")
        else:
            note("warn", "ไม่มี GPU NVIDIA: YOLO รันบน CPU (ใช้ OpenVINO ช่วยได้)")
    except ImportError:
        out("PyTorch   : ไม่ได้ติดตั้งใน Python นี้ (ไม่เป็นไรสำหรับการตรวจ)")
        if ROOT and smi:
            note("warn", "ตรวจด้วย Python ที่ไม่มี PyTorch: เช็ก CUDA ไม่ได้ (รันด้วย .venv ของโปรเจกต์จะเห็นครบ)")
    except Exception as exc:
        out(f"PyTorch   : ติดตั้งแล้วแต่ใช้ไม่ได้ ({exc.__class__.__name__}: {exc})")
        note("bad", "PyTorch เสีย: รัน setup_and_run.bat ใหม่")

    try:
        import openvino as ov
        devices = ov.Core().available_devices
        out(f"OpenVINO  : {ov.__version__} · อุปกรณ์ {', '.join(devices)}")
        if not _cuda_ok():
            note("ok", "มี OpenVINO: YOLO บน CPU เร็วขึ้น ~1.5 เท่า (YOLO_BACKEND=auto)")
    except ImportError:
        out("OpenVINO  : ไม่ได้ติดตั้ง (ไม่บังคับ: pip install openvino)")
        if ROOT and not smi:
            note("warn", "ไม่มี GPU และไม่มี OpenVINO: pip install openvino (ใน .venv ของโปรเจกต์) ให้ YOLO เร็วขึ้น")
    except Exception as exc:
        out(f"OpenVINO  : ติดตั้งแล้วแต่ใช้ไม่ได้ ({exc})")


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------------------------------------------ 3. local cameras
def check_local_cameras() -> None:
    head("3. กล้องที่ต่อกับเครื่องนี้ (webcam / USB)")
    names = []
    if IS_WINDOWS:
        listed = powershell("Get-PnpDevice -PresentOnly -Class Camera,Image -ErrorAction SilentlyContinue | "
                            "ForEach-Object { $_.Status + '|' + $_.FriendlyName }")
        for row in listed.splitlines():
            status, _, name = row.partition("|")
            if name.strip():
                names.append(name.strip())
                out(f"Windows เห็น: {name.strip()}  [{status.strip()}]")
    if IS_WINDOWS and not names:
        out("Windows ไม่พบกล้องในเครื่อง")
    if QUICK:
        out("(--quick: ไม่ได้ลองเปิดกล้อง)")
        return
    try:
        import cv2
    except ImportError:
        out("(Python นี้ไม่มี OpenCV: บอกได้แค่ชื่อกล้อง ถ้าอยากลองเปิดภาพ: pip install opencv-python)")
        if names:
            note("ok", f"Windows เห็นกล้องในเครื่อง {len(names)} ตัว")
        return
    cv2.setLogLevel(0) if hasattr(cv2, "setLogLevel") else None
    backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
    working = 0
    misses = 0
    for index in range(10):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            misses += 1
            if misses >= 2 and index >= len(names):
                break  # indices are consecutive: two empty in a row = no more cameras
            continue
        misses = 0
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # ask for the best it can do
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ok, frame = False, None
        for _ in range(10):  # some cameras give black/empty frames first
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                break
            time.sleep(0.05)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if ok and frame is not None:
            working += 1
            height, width = frame.shape[:2]
            dark = " (ภาพมืด: ฝาปิดเลนส์/ไฟ?)" if frame.mean() < 8 else ""
            out(f"OpenCV index {index}: ใช้ได้ {width}x{height}" + (f" @ {fps:.0f} fps" if fps and fps < 240 else "") + dark)
        else:
            out(f"OpenCV index {index}: เปิดได้แต่ไม่มีภาพ (โปรแกรมอื่นใช้อยู่? เช่น Zoom/Teams/แอปกล้อง)")
    if working:
        note("ok", f"กล้องในเครื่อง {working} ตัว ใช้ได้กับหน้า \"กล้องของเครื่องนี้\"")
    else:
        out("ไม่พบกล้องที่ OpenCV เปิดได้")
        if names:
            note("warn", "Windows เห็นกล้อง แต่ OpenCV เปิดไม่ได้: ปิดโปรแกรมที่ใช้กล้องอยู่ "
                         "และเช็ก Settings > Privacy > Camera ให้แอป desktop ใช้กล้องได้")


# ------------------------------------------------------------------ 4. IP cameras
def ip_camera_sources() -> list[tuple[str, str, str]]:
    """(where it is set, name, URL) of every IP camera the system knows."""
    found = [("--rtsp", f"กล้อง {i}", url) for i, url in enumerate(
        (sys.argv[j + 1] for j, a in enumerate(sys.argv[:-1]) if a == "--rtsp"), start=1)]
    if ROOT is None:
        return found
    db = ROOT / "web" / "db.sqlite3"
    if db.exists():
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                for name, source, run_flag in conn.execute(
                        "SELECT name, source, should_run FROM cameras_camera ORDER BY name"):
                    found.append(("แอป" + (" (รันอยู่)" if run_flag else ""), name, source))
        except sqlite3.Error as exc:
            out(f"อ่านรายชื่อกล้องในแอปไม่ได้: {exc}")
    try:
        sys.path.insert(0, str(HERE))
        from person_detection import config
        for name, source in (getattr(config, "CAMERAS", None) or {}).items():
            if not any(src == source for _, _, src in found):
                found.append(("config.py", name, source))
    except Exception as exc:
        out(f"อ่าน config.py ไม่ได้: {exc.__class__.__name__}")
    return found


def check_ip_cameras() -> None:
    head("4. กล้อง IP")
    cameras = ip_camera_sources()
    if ROOT is None and not cameras:
        out("เครื่องนี้ไม่มีโฟลเดอร์ CN401 · ทดสอบกล้องเองได้: check_devices.bat --rtsp \"rtsp://user:pass@IP:554/stream1\" "
            "· กล้องในวง LAN ดูข้อ 7")
        return
    if not cameras:
        out("ยังไม่มีกล้อง IP (เพิ่มได้ในแอป หน้า \"กล้อง IP\")")
        return
    reachable = 0
    for where, name, source in cameras:
        out(f"- {name} [{where}]  {mask(source)}")
        parts = urlsplit(str(source))
        if parts.scheme not in ("rtsp", "rtsps", "http", "https") or not parts.hostname:
            out("    (ไม่ใช่ URL เครือข่าย ข้าม)")
            continue
        host = parts.hostname
        port = parts.port or {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}[parts.scheme]
        if "X" in host.upper().split(".")[-1]:
            out(f"    IP ยังเป็นตัวอย่าง ({host}): แก้เป็น IP จริงของกล้อง")
            note("warn", f"กล้อง {name}: IP ยังไม่ได้แก้ ({host})")
            continue
        started = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=3):
                ms = (time.perf_counter() - started) * 1000
            out(f"    เครือข่าย: ต่อ {host}:{port} ได้ ({ms:.0f} ms)")
        except OSError as exc:
            out(f"    เครือข่าย: ต่อ {host}:{port} ไม่ได้ ({exc.__class__.__name__})")
            note("bad", f"กล้อง {name}: ต่อ {host}:{port} ไม่ได้ - กล้องเปิดอยู่ไหม / อยู่วง LAN เดียวกับคอมไหม")
            continue
        if parts.password == "" or (parts.username and not parts.password):
            out("    ไม่มีรหัสผ่านใน URL (config.py ใช้ CAM_PASSWORD: ตั้งก่อนรัน)")
        if parts.scheme.startswith("rtsp"):
            code, why = rtsp_describe(str(source))
            out(f"    RTSP: {why}")
            if code != 200:
                note("bad", f"กล้อง {name}: {why}")
                continue
        if QUICK:
            reachable += 1
            continue
        shape = _open_stream(source)
        if shape:
            reachable += 1
            out(f"    สตรีม: เปิดได้ ภาพ {shape[1]}x{shape[0]}")
        elif shape is None and parts.scheme.startswith("rtsp"):
            reachable += 1  # DESCRIBE said yes; no OpenCV here to read a picture
        else:
            out("    สตรีม: เปิดไม่ได้ (กล้องอาจจำกัดจำนวนคนดูอยู่)")
            note("bad", f"กล้อง {name}: ต่อเครือข่ายได้แต่อ่านภาพไม่ได้")
    if reachable:
        note("ok", f"กล้อง IP ใช้ได้ {reachable}/{len(cameras)} ตัว")


def rtsp_describe(url: str, timeout: float = 4.0) -> tuple[int, str]:
    """Ask the camera for the stream description (RTSP DESCRIBE), logging in with the
    URL's user / password (Basic or Digest). Tells apart: wrong password, wrong path, OK.
    No OpenCV needed, nothing is changed on the camera."""
    parts = urlsplit(url)
    user, password = unquote(parts.username or ""), unquote(parts.password or "")
    host_port = parts.hostname + (f":{parts.port}" if parts.port else "")
    clean = urlunsplit((parts.scheme, host_port, parts.path or "/", parts.query, ""))
    port = parts.port or 554

    def ask(sock, cseq, auth=None):
        request = (f"DESCRIBE {clean} RTSP/1.0\r\nCSeq: {cseq}\r\nAccept: application/sdp\r\n"
                   f"User-Agent: CN401-check\r\n" + (f"Authorization: {auth}\r\n" if auth else "") + "\r\n")
        sock.sendall(request.encode())
        data = b""
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
        head_text = data.split(b"\r\n\r\n")[0].decode("latin-1", "replace")
        m = re.match(r"RTSP/1\.\d\s+(\d+)", head_text)
        return (int(m.group(1)) if m else 0), head_text

    def authorization(head_text):
        challenges = re.findall(r"^WWW-Authenticate:\s*(.+)$", head_text, re.MULTILINE | re.IGNORECASE)
        digest = next((c for c in challenges if c.lower().startswith("digest")), None)
        if digest:
            fields = dict(re.findall(r'(\w+)="?([^",]*)"?', digest[6:]))
            realm, nonce = fields.get("realm", ""), fields.get("nonce", "")
            ha1 = hashlib.md5(f"{user}:{realm}:{password}".encode()).hexdigest()
            ha2 = hashlib.md5(f"DESCRIBE:{clean}".encode()).hexdigest()
            if "auth" in fields.get("qop", ""):
                cnonce, nc = uuid.uuid4().hex[:16], "00000001"
                response = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}".encode()).hexdigest()
                return (f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{clean}", '
                        f'response="{response}", qop=auth, nc={nc}, cnonce="{cnonce}"')
            response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
            return f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{clean}", response="{response}"'
        return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

    try:
        with socket.create_connection((parts.hostname, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            code, head_text = ask(sock, 1)
            if code == 401:
                if not user:
                    return 401, "กล้องต้องการ user / รหัสผ่าน แต่ใน URL ไม่มี"
                code, head_text = ask(sock, 2, authorization(head_text))
    except OSError as exc:
        return 0, f"คุยกับกล้องไม่ได้ ({exc.__class__.__name__})"
    if code == 200:
        return 200, "user / รหัสผ่าน และ path ถูก (กล้องส่งรายละเอียดสตรีมมา)"
    if code == 401:
        return 401, "รหัสผ่านหรือ user ผิด"
    if code == 404:
        return 404, f"path ผิด ({parts.path or '/'}) ดู path ที่ถูกในคู่มือกล้อง หรือข้อ 7"
    if code == 0:
        return 0, "พอร์ตนี้ไม่ใช่ RTSP"
    return code, f"กล้องตอบ RTSP {code}"


def _open_stream(source: str):
    """(height, width) of one picture; None when OpenCV isn't installed; () when it failed."""
    try:
        import cv2
    except ImportError:
        return None
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000])
    try:
        if not cap.isOpened():
            return ()
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame.shape[:2]
        return ()
    except Exception:
        return ()
    finally:
        cap.release()


# ------------------------------------------------------------------ 5. network
def check_network() -> None:
    head("5. เครือข่าย")
    addresses = []
    if IS_WINDOWS:
        rows = powershell("Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | "
                          "Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | "
                          "ForEach-Object { $_.InterfaceAlias + '|' + $_.IPAddress }")
        for row in rows.splitlines():
            alias, _, ip = row.partition("|")
            addresses.append(ip.strip())
            out(f"{alias.strip():<28} {ip.strip()}")
    if not addresses:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.168.1.1", 9))  # no packet is sent: only picks the outgoing interface
                addresses.append(s.getsockname()[0])
                out(f"IP หลัก: {addresses[-1]}")
        except OSError:
            out("ไม่พบ IP ในวง LAN")
    if addresses:
        out("เครื่องอื่นในวง LAN เปิดเว็บดูประวัติได้ที่ https://<IP ด้านบน>:8000 (ตอนแอปเปิดอยู่)")


# ------------------------------------------------------------------ 6. packages + models
def check_software() -> None:
    head("6. แพ็กเกจและโมเดล")
    if ROOT is None:
        out("เครื่องนี้ไม่มีโฟลเดอร์ CN401 (ข้าม)")
        return
    out(f"โฟลเดอร์โปรเจกต์: {ROOT}")
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    if Path(sys.executable).resolve() != venv.resolve():
        out("(ตรวจด้วย Python ของเครื่อง ไม่ใช่ .venv ของโปรเจกต์: แพ็กเกจด้านล่างคือของ Python นี้)")
    for module, label in (("cv2", "OpenCV"), ("ultralytics", "Ultralytics"), ("django", "Django"),
                          ("transformers", "Transformers"), ("sqlalchemy", "SQLAlchemy"),
                          ("webview", "pywebview"), ("onnxruntime", "onnxruntime")):
        try:
            mod = __import__(module)
            out(f"{label:<13}: {getattr(mod, '__version__', 'ok')}")
        except Exception:
            out(f"{label:<13}: ไม่ได้ติดตั้ง")
            if module in ("cv2", "ultralytics", "django") and Path(sys.executable).resolve() == venv.resolve():
                note("bad", f"{label} ยังไม่ได้ติดตั้ง: รัน setup_and_run.bat")
    models = HERE / "models"
    wanted = ["yolo11n-pose.pt", "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx",
              "vlm_par.pt"]
    for name in wanted:
        path = models / name
        out(f"{name:<38}: " + (f"มี ({path.stat().st_size / 2**20:.1f} MB)" if path.exists() else "ยังไม่มี"))
        if not path.exists():
            note("warn", f"ยังไม่มีโมเดล {name}: setup_and_run.bat จะดาวน์โหลดให้")
    extra = sorted(p.name for p in models.glob("*") if p.name not in wanted and p.name not in ("photos", "insightface")) if models.exists() else []
    extra += sorted(f"insightface/{p.name}" for p in (models / "insightface").glob("*.onnx"))
    if extra:
        out("โมเดลอื่นที่มี: " + ", ".join(extra))


# ------------------------------------------------------------------ 7. LAN
# (ONVIF WS-Discovery, camera ports, ARP table; only this PC's own networks, x.y.z.1-254)

PORTS = {
    554: "RTSP (สตรีมกล้อง)",
    8554: "RTSP",
    80: "เว็บ",
    443: "เว็บ HTTPS",
    8000: "Hikvision SDK",
    8080: "เว็บ",
    37777: "Dahua",
    34567: "XMEye/DVR",
    2020: "ONVIF (Tapo)",
}
SKIP_ADAPTERS = re.compile(r"vEthernet|VirtualBox|VMware|Hyper-V|WSL|Loopback|Bluetooth|Tailscale|ZeroTier|Npcap|TAP-",
                           re.IGNORECASE)

# Common RTSP paths, by what the camera says it is
RTSP_GUESS = [
    (re.compile(r"hikvision|hik", re.I), "/Streaming/Channels/101  (ย่อย: /Streaming/Channels/102)"),
    (re.compile(r"dahua|imou", re.I), "/cam/realmonitor?channel=1&subtype=0  (ย่อย: subtype=1)"),
    (re.compile(r"tapo|tp-?link|tp-ipc|vigi", re.I), "/stream1  (ย่อย: /stream2)"),
    (re.compile(r"axis", re.I), "/axis-media/media.amp"),
    (re.compile(r"reolink", re.I), "/h264Preview_01_main"),
    (re.compile(r"ezviz", re.I), "/H.264  หรือ /Streaming/Channels/101"),
]


@dataclass
class Device:
    ip: str
    mac: str = ""
    ports: list[int] = field(default_factory=list)
    rtsp_server: str = ""
    web_server: str = ""
    web_title: str = ""
    onvif: str = ""  # brand / model from WS-Discovery
    onvif_urls: list[str] = field(default_factory=list)
    is_this_pc: bool = False

    @property
    def kind(self) -> str:
        text = " ".join([self.rtsp_server, self.web_server, self.web_title, self.onvif]).lower()
        if self.is_this_pc:
            return "เครื่องนี้"
        if self.onvif or 554 in self.ports or 8554 in self.ports or 2020 in self.ports:
            return "กล้อง IP"
        if any(w in text for w in ("camera", "ipc", "nvr", "dvr", "hikvision", "dahua", "tapo")):
            return "กล้อง / เครื่องบันทึก"
        if any(w in text for w in ("router", "gateway", "tp-link", "asus", "mikrotik", "huawei", "zte")):
            return "เราเตอร์"
        if 80 in self.ports or 443 in self.ports:
            return "อุปกรณ์มีหน้าเว็บ"
        return "อุปกรณ์"

    def rtsp_guess(self) -> str:
        text = " ".join([self.rtsp_server, self.web_server, self.web_title, self.onvif])
        port = 554 if 554 in self.ports else 8554 if 8554 in self.ports else None
        if port is None:
            return ""
        base = f"rtsp://USER:PASSWORD@{self.ip}:{port}"
        for pattern, path in RTSP_GUESS:
            if pattern.search(text):
                return base + path
        if 2020 in self.ports:  # TP-Link Tapo/VIGI answer ONVIF on 2020
            return base + "/stream1  (ย่อย: /stream2)"
        return base + "/  (path ดูในคู่มือกล้อง เช่น /stream1, /live, /h264)"


# ------------------------------------------------------------------ networks
def local_networks() -> list[tuple[str, str, ipaddress.IPv4Network]]:
    """(adapter name, this PC's IP, the /24 around it) for each real, private network."""
    found = []
    if IS_WINDOWS:
        rows = run(["powershell", "-NoProfile", "-Command",
                    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-NetIPAddress -AddressFamily IPv4 "
                    "-ErrorAction SilentlyContinue | ForEach-Object { $_.InterfaceAlias + '|' + $_.IPAddress + '|' + "
                    "$_.PrefixLength }"])
        for row in rows.splitlines():
            parts = row.strip().split("|")
            if len(parts) == 3:
                found.append((parts[0], parts[1], int(parts[2] or 24)))
    else:
        for row in run(["ip", "-4", "-o", "addr"]).splitlines():
            m = re.search(r"^\d+:\s+(\S+)\s+inet\s+([\d.]+)/(\d+)", row)
            if m:
                found.append((m.group(1), m.group(2), int(m.group(3))))
    networks = []
    for name, ip, prefix in found:
        address = ipaddress.ip_address(ip)
        if not address.is_private or address.is_loopback or address.is_link_local or SKIP_ADAPTERS.search(name):
            continue
        # big networks (/16 ...) are scanned only around this PC: x.y.z.1-254
        network = ipaddress.ip_network(f"{ip}/{max(prefix, 24)}", strict=False)
        networks.append((name, ip, network))
    return networks


# ------------------------------------------------------------------ ONVIF discovery
PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
 xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<e:Header><w:MessageID>uuid:{id}</w:MessageID><w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>"""


def parse_onvif(xml: str) -> tuple[list[str], str]:
    """(service URLs, 'brand model') from a WS-Discovery ProbeMatch."""
    urls = re.findall(r"https?://[^\s<]+", (re.search(r"XAddrs>([^<]*)<", xml) or [None, ""])[1])
    scopes = (re.search(r"Scopes>([^<]*)<", xml) or [None, ""])[1]
    info = {}
    for scope in scopes.split():
        m = re.match(r"onvif://www\.onvif\.org/(name|hardware|mfr|manufacturer)/(.+)", scope)
        if m:
            info.setdefault(m.group(1), re.sub(r"%20|_", " ", m.group(2)))
    label = " ".join(v for v in (info.get("mfr") or info.get("manufacturer") or info.get("name"),
                                 info.get("hardware")) if v)
    return urls, label or "ONVIF"


def onvif_discover(local_ips: list[str], wait: float = 3.0) -> dict[str, tuple[list[str], str]]:
    found: dict[str, tuple[list[str], str]] = {}
    sockets = []
    for ip in local_ips:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
            s.bind((ip, 0))
            s.settimeout(0.3)
            s.sendto(PROBE.format(id=uuid.uuid4()).encode(), ("239.255.255.250", 3702))
            sockets.append(s)
        except OSError:
            continue
    end = time.monotonic() + wait
    while sockets and time.monotonic() < end:
        for s in sockets:
            try:
                data, (sender, _) = s.recvfrom(65535)
                found[sender] = parse_onvif(data.decode("utf-8", "replace"))
            except OSError:
                pass
    for s in sockets:
        s.close()
    return found


# ------------------------------------------------------------------ per host
def open_ports(ip: str, ports=PORTS, timeout: float = 0.5) -> list[int]:
    result = []
    for port in ports:
        try:
            with socket.create_connection((ip, port), timeout=timeout):
                result.append(port)
        except OSError:
            pass
    return result


def rtsp_server(ip: str, port: int) -> str:
    """'Server:' of an RTSP OPTIONS answer (no login needed for OPTIONS on most cameras)."""
    try:
        with socket.create_connection((ip, port), timeout=2) as s:
            s.settimeout(2)
            s.sendall(f"OPTIONS rtsp://{ip}:{port}/ RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: CN401-check\r\n\r\n".encode())
            reply = s.recv(2048).decode("latin-1", "replace")
    except OSError:
        return ""
    if not reply.startswith("RTSP/"):
        return ""
    server = re.search(r"^Server:\s*(.+)$", reply, re.MULTILINE | re.IGNORECASE)
    return server.group(1).strip() if server else "RTSP"


def web_info(ip: str, port: int) -> tuple[str, str]:
    """(Server header, page <title>) of http://ip:port/ ."""
    try:
        with socket.create_connection((ip, port), timeout=2) as s:
            s.settimeout(2)
            s.sendall(f"GET / HTTP/1.0\r\nHost: {ip}\r\nUser-Agent: CN401-check\r\n\r\n".encode())
            data = b""
            while len(data) < 16384:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
    except OSError:
        return "", ""
    text = data.decode("utf-8", "replace")
    server = re.search(r"^Server:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    title = re.search(r"<title[^>]*>([^<]{0,80})", text, re.IGNORECASE)
    return (server.group(1).strip() if server else ""), (title.group(1).strip() if title else "")


def arp_table() -> dict[str, str]:
    """ip -> MAC of devices this PC has seen."""
    table = {}
    text = run(["arp", "-a"]) if IS_WINDOWS else run(["ip", "neigh"])
    for m in re.finditer(r"(\d+\.\d+\.\d+\.\d+)\D+?(([0-9a-f]{2}[-:]){5}[0-9a-f]{2})", text, re.IGNORECASE):
        mac = m.group(2).replace("-", ":").upper()
        if mac not in ("FF:FF:FF:FF:FF:FF", "00:00:00:00:00:00") and not mac.startswith("01:00:5E"):
            table[m.group(1)] = mac
    return table


def lan_scan(networks=None, progress=print) -> list[Device]:
    networks = local_networks() if networks is None else networks
    if not networks:
        return []
    mine = {ip for _, ip, _ in networks if ip}
    onvif = onvif_discover(sorted(mine))
    hosts = [str(h) for _, _, net in networks for h in net.hosts()]
    progress(f"กำลังตรวจ {len(hosts)} ที่อยู่ ...")
    with ThreadPoolExecutor(max_workers=96) as pool:
        port_map = dict(zip(hosts, pool.map(open_ports, hosts)))
    arp = arp_table()
    devices = {}
    for ip in set(hosts) | set(onvif):
        in_scope = any(ipaddress.ip_address(ip) in net for _, _, net in networks)
        if not in_scope:
            continue
        ports = port_map.get(ip, [])
        if ip in mine or ports or ip in onvif or ip in arp:
            devices[ip] = Device(ip, arp.get(ip, ""), ports, is_this_pc=ip in mine)
    for ip, (urls, label) in onvif.items():
        if ip in devices:
            devices[ip].onvif, devices[ip].onvif_urls = label, urls

    def details(device: Device) -> None:
        for port in (554, 8554):
            if port in device.ports and not device.is_this_pc:
                device.rtsp_server = rtsp_server(device.ip, port)
                break
        for port in (80, 8080):
            if port in device.ports and not device.is_this_pc:
                device.web_server, device.web_title = web_info(device.ip, port)
                break

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(details, devices.values()))
    return sorted(devices.values(), key=lambda d: tuple(int(x) for x in d.ip.split(".")))


def lan_report(devices: list[Device], networks, known_ips: dict[str, str] | None = None, out=print) -> list[tuple[str, str]]:
    """Print the devices; returns (level, text) notes for check_devices.py's summary."""
    notes = []
    known_ips = known_ips or {}
    for name, ip, net in networks:
        extra = " · Internet Connection Sharing: อุปกรณ์ที่เสียบสาย LAN ตรงกับเครื่องนี้จะได้ 192.168.137.x" \
            if ip == "192.168.137.1" else ""
        out(f"เครือข่าย {name}: เครื่องนี้ {ip} · ตรวจ {net}{extra}")
    if not devices:
        out("ไม่พบอุปกรณ์")
        return notes
    cameras = []
    for d in devices:
        tags = []
        if d.ports:
            tags.append("พอร์ต " + ", ".join(f"{p} {PORTS[p]}" for p in d.ports))
        for label, value in (("ONVIF", d.onvif), ("RTSP", d.rtsp_server), ("เว็บ", d.web_server),
                             ("หน้าเว็บ", d.web_title)):
            if value:
                tags.append(f"{label}: {value}")
        out(f"- {d.ip:<15} {d.mac or '':<17}  [{d.kind}]")
        for tag in tags:
            out(f"      {tag}")
        if d.kind.startswith("กล้อง") and not d.is_this_pc:
            cameras.append(d)
            guess = d.rtsp_guess()
            if guess:
                out(f"      ลองใส่ในแอป: {guess}")
            if d.ip in known_ips:
                out(f"      อยู่ในแอปแล้ว: {known_ips[d.ip]}")
    if cameras:
        new = [d.ip for d in cameras if d.ip not in known_ips]
        notes.append(("ok", f"พบกล้องในวง LAN {len(cameras)} ตัว: " + ", ".join(d.ip for d in cameras)))
        if new:
            notes.append(("warn", "กล้องที่ยังไม่ได้เพิ่มในแอป: " + ", ".join(new)))
    else:
        notes.append(("warn", "ไม่พบกล้องในวง LAN (เช็กสาย/ไฟกล้อง และว่ากล้องอยู่วงเดียวกับเครื่องนี้)"))
    found = {d.ip for d in devices}
    for ip, name in known_ips.items():
        if any(ipaddress.ip_address(ip) in net for _, _, net in networks) and ip not in found:
            notes.append(("bad", f"กล้อง {name} ({ip}) ที่ตั้งไว้ในแอป ไม่ตอบในวง LAN"))
    return notes



def check_lan() -> None:
    head("7. อุปกรณ์ในวง LAN (สาย LAN / สวิตช์ / Wi-Fi)")
    if "--no-lan" in sys.argv:
        out("(--no-lan: ข้าม)")
        return
    networks = local_networks()
    if not networks:
        out("เครื่องนี้ไม่ได้ต่อ LAN / Wi-Fi วงไหนเลย")
        note("warn", "ไม่ได้ต่อเครือข่าย: กล้อง IP ใช้ไม่ได้")
        return
    known = {}
    for _, name, source in ip_camera_sources():
        host = urlsplit(str(source)).hostname
        if host and re.fullmatch(r"\d+\.\d+\.\d+\.\d+", host):
            known[host] = name
    started = time.monotonic()
    devices = lan_scan(networks, progress=out)
    for level, text in lan_report(devices, networks, known, out=out):
        note(level, text)
    out(f"(ใช้เวลา {time.monotonic() - started:.0f} วินาที)")


# ------------------------------------------------------------------ main
def main() -> None:
    if IS_WINDOWS:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        os.system("")  # lets the console show UTF-8 / colours
    out(f"CN401 ตรวจอุปกรณ์ · {dt.datetime.now():%Y-%m-%d %H:%M} · {socket.gethostname()}")
    out(f"โปรเจกต์ CN401: {ROOT if ROOT else 'ไม่พบในเครื่องนี้ (ตรวจเฉพาะเครื่องและเครือข่าย)'}")
    for step in (check_computer, check_gpu, check_local_cameras, check_ip_cameras, check_network, check_software,
                 check_lan):
        try:
            step()
        except Exception as exc:  # one broken check must not hide the others
            out(f"(ตรวจส่วนนี้ไม่สำเร็จ: {exc.__class__.__name__}: {exc})")

    head("สรุป")
    marks = {"ok": "[OK]  ", "warn": "[!]   ", "bad": "[X]   "}
    for level in ("bad", "warn", "ok"):
        for lvl, text in summary:
            if lvl == level:
                out(marks[level] + text)
    if not summary:
        out("ไม่มีอะไรต้องแก้")

    folder = HERE / "output" if HERE else SCRIPT_DIR
    report = folder / f"device_check_{socket.gethostname()}_{dt.datetime.now():%Y%m%d_%H%M}.txt"
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nบันทึกรายงานไว้ที่ {report}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
