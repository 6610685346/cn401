"""python manage.py runportal [--port 8000] [--http] [--full]

Runs the LAN history page on its own (no cameras): other computers log in and
read / download the detection history. The desktop app (desktop.py) already
serves this page while it is open, so this command is for when the app is
closed, or to put the history page on a different computer.

HTTPS is on by default so passwords are not sent in clear text over the LAN.
A self-signed certificate is made on first run in web/certs/; each browser
warns once ("connection not private") - choose Advanced -> Proceed.

--full serves the whole portal like V01 (cameras, settings, ...) to the LAN.
For development only: in V02 that part belongs to the desktop app.
"""

import datetime
import ipaddress
import os
import socket
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand


def lan_addresses() -> list[str]:
    """IPv4 addresses other computers can use to reach this one."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except socket.gaierror:
        pass
    try:  # the address of the interface that routes to the LAN
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.168.0.1", 9))
            found.add(probe.getsockname()[0])
    except OSError:
        pass
    return sorted(ip for ip in found if not ip.startswith("127."))


def make_certificate(cert_path: Path, key_path: Path, hosts: list[str]) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CN401 portal")])
    alt_names = [x509.DNSName("localhost"), x509.DNSName(socket.gethostname())]
    for host in hosts + ["127.0.0.1"]:
        alt_names.append(x509.IPAddress(ipaddress.ip_address(host)))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def ensure_certificate(addresses: list[str], fresh: bool = False) -> dict:
    """uvicorn ssl options, making web/certs/portal.crt first if needed."""
    cert = Path(settings.BASE_DIR) / "certs" / "portal.crt"
    key = Path(settings.BASE_DIR) / "certs" / "portal.key"
    if fresh or not cert.exists() or not key.exists():
        make_certificate(cert, key, addresses)
    return {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}


class Command(BaseCommand):
    help = "Run the LAN history page (uvicorn, HTTPS by default). Cameras live in desktop.py."

    def add_arguments(self, parser):
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--http", action="store_true", help="plain HTTP instead of HTTPS")
        parser.add_argument("--new-cert", action="store_true", help="make a fresh certificate (e.g. new IP)")
        parser.add_argument("--full", action="store_true",
                            help="development: serve the whole portal (cameras too) like V01")
        parser.add_argument("--no-autostart", action="store_true", help="with --full: do not start saved cameras")

    def handle(self, *args, **options):
        import uvicorn

        from portal.modes import APP, WEB, with_mode

        full = options["full"]
        # Only the full portal touches cameras; must be set before portal.asgi loads.
        os.environ["PORTAL_AUTOSTART"] = "1" if full and not options["no_autostart"] else "0"
        from portal.asgi import application

        addresses = lan_addresses()
        ssl = {}
        scheme = "http"
        if not options["http"]:
            self.stdout.write("Using the HTTPS certificate in web/certs/ ...")
            ssl = ensure_certificate(addresses, options["new_cert"])
            scheme = "https"

        port = options["port"]
        what = "FULL portal (development)" if full else "history page"
        self.stdout.write(self.style.SUCCESS(f"\nCN401 {what} is starting. Open one of these:"))
        self.stdout.write(f"  this computer : {scheme}://localhost:{port}/")
        for ip in addresses:
            self.stdout.write(f"  other computers in the LAN : {scheme}://{ip}:{port}/")
        if scheme == "https":
            self.stdout.write("  (first visit: the browser warns about the certificate -> Advanced -> Proceed)")
        self.stdout.write("Press Ctrl+C to stop.\n")

        uvicorn.run(with_mode(application, APP if full else WEB), host=options["host"], port=port,
                    log_level="warning", workers=1, lifespan="off", **ssl)
