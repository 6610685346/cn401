"""ASGI entry point, used by desktop.py and `python manage.py runportal`.

Which pages a server shows is decided by wrapping this application with
portal.modes.with_mode(application, "app" | "web").
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "portal.settings")

application = get_asgi_application()

# The desktop app starts the cameras marked "running" (PORTAL_AUTOSTART=1) so
# they keep recording even when the window is minimised. The history-only
# web server does not touch the cameras.
if os.environ.get("PORTAL_AUTOSTART", "0") == "1":
    from cameras.services import start_background

    start_background()
