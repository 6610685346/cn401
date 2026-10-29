"""Two faces of one Django project: the desktop APP and the LAN WEB.

* APP  - everything (cameras, settings, enrollment, floor calibration, admin).
         Served only on 127.0.0.1 to the desktop window started by desktop.py.
* WEB  - read-only history for other computers in the LAN: log in, look at
         the detection history, download the CSV files. Nothing else exists
         there (the other URLs are simply not routed).

The server marks each connection with its mode in the ASGI scope (see
`with_mode`), so the decision is made by which listener the request came in
on, not by anything the client sends. A request without a mark (plain
`runserver`, WSGI) gets settings.PORTAL_DEFAULT_MODE, which is WEB unless
running the test suite, so the full app is never exposed by accident.
"""

from __future__ import annotations

from django.conf import settings

APP = "app"
WEB = "web"
MODES = (APP, WEB)
WEB_URLCONF = "portal.urls_web"


def with_mode(asgi_app, mode: str):
    """Wrap an ASGI app so every request it serves is marked with `mode`."""
    if mode not in MODES:
        raise ValueError(f"unknown portal mode: {mode}")

    async def app(scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            scope = {**scope, "portal_mode": mode}
        await asgi_app(scope, receive, send)

    return app


def request_mode(request) -> str:
    scope = getattr(request, "scope", None) or {}
    mode = scope.get("portal_mode") or getattr(settings, "PORTAL_DEFAULT_MODE", WEB)
    return mode if mode in MODES else WEB


class PortalModeMiddleware:
    """Routes WEB requests to the small history-only URL list."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.portal_mode = request_mode(request)
        if request.portal_mode == WEB:
            request.urlconf = WEB_URLCONF
        return self.get_response(request)


def context(request) -> dict:
    """Template variables: portal_mode, is_app and (app only) web_info."""
    mode = getattr(request, "portal_mode", None) or request_mode(request)
    data = {"portal_mode": mode, "is_app": mode == APP}
    if mode == APP:
        from cameras import services  # imported lazily: loads the detection code
        data["web_info"] = services.web_info
        data["show_eval_menu"] = services.SHOW_EVAL_MENU  # "ตั้งค่าระบบ" → เมนู
    return data
