"""Find the devices plugged into this computer's LAN (cameras on a LAN cable, switch, router).

    python lan_scan.py               scan every wired / Wi-Fi network this PC is on (only its /24)
    python lan_scan.py 192.168.137   scan only 192.168.137.1-254

Used by check_devices.py (section 7). Three ways to find devices, because
cameras often ignore ping:

  * ONVIF WS-Discovery  - asks "any IP cameras here?" on the network (UDP 3702);
                          ONVIF cameras answer with their address, brand and model
  * port check          - tries the ports cameras / routers listen on (RTSP 554, web 80 ...)
  * ARP table           - every device this PC has talked to (also ones with no open port)

For each device it shows the MAC address, the open ports, the RTSP / web server
name, and for cameras a guess of the stream URL to put in the app.
Only this PC's own networks are scanned, nothing is changed, no password is tried.

Tip: a camera plugged straight into the laptop with "Internet Connection
Sharing" gets an address in 192.168.137.x (the laptop is 192.168.137.1).
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

IS_WINDOWS = os.name == "nt"

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
def run(cmd: list[str], timeout: float = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                              errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except Exception:
        return ""


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


def scan(networks=None, progress=print) -> list[Device]:
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


def report(devices: list[Device], networks, known_ips: dict[str, str] | None = None, out=print) -> list[tuple[str, str]]:
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


def main() -> None:
    if IS_WINDOWS:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    networks = local_networks()
    if len(sys.argv) > 1 and re.fullmatch(r"\d+\.\d+\.\d+(\.\d+)?", sys.argv[1]):
        base = ".".join(sys.argv[1].split(".")[:3])
        net = ipaddress.ip_network(f"{base}.0/24")
        mine = next((ip for _, ip, n in networks if n == net), "")
        networks = [("(ระบุเอง)", mine, net)]
    if not networks:
        print("ไม่พบเครือข่าย LAN / Wi-Fi ที่ใช้อยู่")
        return
    started = time.monotonic()
    devices = scan(networks)
    for level, text in report(devices, networks):
        print(f"[{level}] {text}")
    print(f"(ใช้เวลา {time.monotonic() - started:.0f} วินาที)")


if __name__ == "__main__":
    main()
