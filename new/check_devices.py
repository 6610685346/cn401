"""Check this computer: GPU, cameras on this PC, IP cameras, network, packages.

Run from the `new` folder (check_devices.bat does this):
    python check_devices.py            full check, opens each camera once
    python check_devices.py --quick    only list the cameras, do not open them

Nothing is changed. The report is also saved to output/device_check_<date>.txt.
IP cameras are found by scanning every local /24 subnet for RTSP ports.
"""

import argparse
import datetime
import importlib.metadata
import ipaddress
import os
import platform
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from person_detection import config  # also sets the FFmpeg RTSP options

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # empty camera indexes warn otherwise

RTSP_PORTS = (554, 8554)
WEB_PORTS = (80, 8080)
CONNECT_TIMEOUT_SEC = 0.5
OPEN_TIMEOUT_MSEC = 5000
LOCAL_CAMERA_INDEXES = range(5)
PACKAGES = ("opencv-python", "numpy", "ultralytics", "lap", "torch", "torchvision",
            "transformers", "huggingface_hub", "pywebview", "Django", "uvicorn")

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def heading(title):
    say()
    say(f"=== {title} ===")


def port_open(host, port, timeout=CONNECT_TIMEOUT_SEC) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def try_import(name):
    try:
        return __import__(name)
    except Exception:  # missing, or a broken install (e.g. torch DLL errors)
        return None


# ------------------------------------------------------------------ sections

def check_gpu():
    heading("GPU")
    torch = try_import("torch")
    if torch is not None:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                say(f"  CUDA {i}: {torch.cuda.get_device_name(i)}  (torch {torch.__version__})")
        else:
            say(f"  torch {torch.__version__} sees no CUDA GPU - YOLO / VLM-PAR will run on the CPU")
    else:
        say("  torch is not installed")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        for row in out.stdout.strip().splitlines():
            say(f"  nvidia-smi: {row}")
    except (OSError, subprocess.TimeoutExpired):
        say("  nvidia-smi not found (no NVIDIA driver)")


def local_ipv4() -> list:
    """IPv4 addresses of this PC's adapters, without loopback / link-local."""
    try:
        found = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        found = []
    keep = []
    for ip in found:
        addr = ipaddress.ip_address(ip)
        if not (addr.is_loopback or addr.is_link_local) and ip not in keep:
            keep.append(ip)
    return keep


def check_network() -> list:
    heading("Network")
    ips = local_ipv4()
    if not ips:
        say("  no network address - is the cable / Wi-Fi connected?")
    for ip in ips:
        say(f"  this PC: {ip}  (scanning {ipaddress.ip_network(ip + '/24', strict=False)})")
    return ips


def list_local_cameras_windows():
    if platform.system() != "Windows":
        return
    cmd = ("Get-PnpDevice -Class Camera,Image -PresentOnly -ErrorAction SilentlyContinue"
           " | ForEach-Object { $_.FriendlyName + ' [' + $_.Status + ']' }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                             capture_output=True, text=True, timeout=20)
        names = [n for n in out.stdout.strip().splitlines() if n.strip()]
    except (OSError, subprocess.TimeoutExpired):
        names = []
    say("  Windows camera devices: " + (", ".join(names) if names else "none"))


def check_local_cameras(quick: bool):
    heading("Cameras on this PC (USB / built-in)")
    list_local_cameras_windows()
    if quick:
        say("  --quick: not opening the cameras")
        return
    cv2 = try_import("cv2")
    if cv2 is None:
        say("  opencv is not installed - cannot open the cameras")
        return
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    for index in LOCAL_CAMERA_INDEXES:
        cap = cv2.VideoCapture(index, backend)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            say(f"  index {index}: OK {w}x{h}   -> use SOURCE = \"{index}\"")


def scan_subnet(ip):
    """(host, open ports) for every host in ip's /24 with an RTSP port open."""
    hosts = [str(h) for h in ipaddress.ip_network(ip + "/24", strict=False).hosts() if str(h) != ip]
    jobs = [(h, p) for h in hosts for p in RTSP_PORTS]
    with ThreadPoolExecutor(max_workers=256) as pool:
        results = pool.map(lambda job: port_open(*job), jobs)
        found = {}
        for (host, port), ok in zip(jobs, results):
            if ok:
                found.setdefault(host, []).append(port)
    return found


def check_ip_cameras(ips: list) -> dict:
    heading("IP cameras on the LAN (RTSP ports " + ", ".join(map(str, RTSP_PORTS)) + ")")
    cameras = {}
    for ip in ips:
        cameras.update(scan_subnet(ip))
    if not cameras:
        say("  none found. Check the camera has power, is on the same network as this PC,")
        say("  and that its IP is in one of the subnets above.")
    for host, ports in sorted(cameras.items(), key=lambda kv: ipaddress.ip_address(kv[0])):
        web = [p for p in WEB_PORTS if port_open(host, p)]
        extra = f", web page http://{host}" + (f":{web[0]}" if web[0] != 80 else "") if web else ""
        say(f"  {host}: RTSP port {', '.join(map(str, ports))}{extra}")
    return cameras


def check_configured_cameras(quick: bool, found: dict):
    heading("Cameras in person_detection/config.py")
    cameras = dict(getattr(config, "CAMERAS", None) or {"SOURCE": config.SOURCE})
    if not config.CAMERA_PASSWORD:
        say("  CAM_PASSWORD is not set - most cameras refuse RTSP without it")
    cv2 = None if quick else try_import("cv2")
    for name, source in cameras.items():
        source = str(source)
        if source.strip().isdigit():
            say(f"  {name}: local camera index {source}")
            continue
        parts = urlsplit(source)
        try:
            host, port = parts.hostname, parts.port or 554
            ipaddress.ip_address(host or "")
        except ValueError:
            say(f"  {name}: '{parts.hostname}' is not a valid IP address - fix it in config.py")
            continue
        if not port_open(host, port, timeout=2):
            hint = f" (cameras found: {', '.join(found)})" if found else ""
            say(f"  {name}: {host}:{port} not reachable{hint}")
            continue
        if cv2 is None:
            say(f"  {name}: {host}:{port} reachable")
            continue
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MSEC])
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            say(f"  {name}: {host} stream OK {frame.shape[1]}x{frame.shape[0]}")
        else:
            say(f"  {name}: {host}:{port} reachable but the stream did not open -"
                " check CAM_USER / CAM_PASSWORD and the stream path")


def check_packages():
    heading("Python packages")
    say(f"  Python {platform.python_version()}  ({sys.executable})")
    for name in PACKAGES:
        try:
            say(f"  {name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            say(f"  {name} MISSING")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="only list the cameras, do not open them")
    args = parser.parse_args()

    say(f"CN401 device check  {datetime.datetime.now():%Y-%m-%d %H:%M}  on {socket.gethostname()}")
    check_gpu()
    ips = check_network()
    check_local_cameras(args.quick)
    found = check_ip_cameras(ips)
    check_configured_cameras(args.quick, found)
    check_packages()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = config.OUTPUT_DIR / f"device_check_{datetime.date.today()}.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved to {report}")


if __name__ == "__main__":
    main()
