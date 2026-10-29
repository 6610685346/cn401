"""Check what this computer has for the CN401 system.

    python check_devices.py            (or double-click check_devices.bat in the V02 folder)
    python check_devices.py --quick    skip opening the cameras (only lists them)
    python check_devices.py --no-lan   skip looking for devices on the LAN (section 7)

Looks at, and prints a summary of:
  1. the computer      Windows version, CPU, RAM, Python
  2. GPU               NVIDIA card (nvidia-smi), whether PyTorch can use it (CUDA), OpenVINO devices
  3. cameras on this PC   webcams / USB cameras: name, and the picture each OpenCV index gives
  4. IP cameras        the ones added in the app (web/db.sqlite3) and in config.py CAMERAS:
                       is the camera on the network, and can its stream be opened
  5. network           this computer's IP addresses (for the LAN history page)
  6. packages, models  versions, and which model files are already downloaded
  7. LAN devices       cameras / routers plugged into this PC's networks (lan_scan.py):
                       IP, MAC, open ports, brand, and a guess of the RTSP URL for the app

Nothing is changed. A copy of the report is saved to new/output/device_check_<date>.txt
(camera passwords are never printed or saved).
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
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # no OpenCV warnings for cameras that are not there

HERE = Path(__file__).resolve().parent  # new/
ROOT = HERE.parent  # V02/
IS_WINDOWS = os.name == "nt"
QUICK = "--quick" in sys.argv

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
        note("bad", "Python เก่ากว่า 3.10: โปรแกรมใช้ไม่ได้")


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
            note("warn", "ไม่มี GPU NVIDIA: YOLO รันบน CPU (ใช้ OpenVINO ช่วยได้, ดูข้อ OpenVINO ด้านล่าง)")
    except Exception as exc:
        out(f"PyTorch   : ใช้ไม่ได้ ({exc.__class__.__name__}: {exc})")
        note("bad", "PyTorch ยังไม่ได้ติดตั้ง/เสีย: รัน setup_and_run.bat")

    try:
        import openvino as ov
        devices = ov.Core().available_devices
        out(f"OpenVINO  : {ov.__version__} · อุปกรณ์ {', '.join(devices)}")
        if not _cuda_ok():
            note("ok", "มี OpenVINO: YOLO บน CPU เร็วขึ้น ~1.5 เท่า (YOLO_BACKEND=auto)")
    except ImportError:
        out("OpenVINO  : ไม่ได้ติดตั้ง (ไม่บังคับ: pip install openvino)")
        if not _cuda_ok():
            note("warn", "ไม่มี GPU และไม่มี OpenVINO: ลอง pip install openvino ใน .venv ให้ YOLO เร็วขึ้น")
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
        out("OpenCV ไม่ได้ติดตั้ง: ลองเปิดกล้องไม่ได้")
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
    found = []
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
        if QUICK:
            reachable += 1
            continue
        shape = _open_stream(source)
        if shape:
            reachable += 1
            out(f"    สตรีม: เปิดได้ ภาพ {shape[1]}x{shape[0]}")
        else:
            out("    สตรีม: เปิดไม่ได้ (รหัสผ่าน/ path ผิด หรือกล้องจำกัดจำนวนคนดู)")
            note("bad", f"กล้อง {name}: ต่อเครือข่ายได้แต่เปิดภาพไม่ได้ - เช็ก user/รหัสผ่าน และ path เช่น /stream1")
    if reachable:
        note("ok", f"กล้อง IP ใช้ได้ {reachable}/{len(cameras)} ตัว")


def _open_stream(source: str):
    try:
        import cv2
    except ImportError:
        return None
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000])
    try:
        if not cap.isOpened():
            return None
        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame.shape[:2]
        return None
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
    for module, label in (("cv2", "OpenCV"), ("ultralytics", "Ultralytics"), ("django", "Django"),
                          ("transformers", "Transformers"), ("sqlalchemy", "SQLAlchemy"),
                          ("webview", "pywebview"), ("onnxruntime", "onnxruntime")):
        try:
            mod = __import__(module)
            out(f"{label:<13}: {getattr(mod, '__version__', 'ok')}")
        except Exception:
            out(f"{label:<13}: ไม่ได้ติดตั้ง")
            if module in ("cv2", "ultralytics", "django"):
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
def check_lan() -> None:
    head("7. อุปกรณ์ในวง LAN (สาย LAN / สวิตช์ / Wi-Fi)")
    if "--no-lan" in sys.argv:
        out("(--no-lan: ข้าม)")
        return
    import lan_scan

    networks = lan_scan.local_networks()
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
    devices = lan_scan.scan(networks, progress=out)
    for level, text in lan_scan.report(devices, networks, known, out=out):
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

    report = HERE / "output" / f"device_check_{dt.datetime.now():%Y%m%d_%H%M}.txt"
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nบันทึกรายงานไว้ที่ {report}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
