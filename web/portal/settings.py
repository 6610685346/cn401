"""Django settings for CN401: the desktop app and the LAN history page.

Meant to run on one computer inside the LAN, next to the cameras.
    python desktop.py              the desktop app (also serves the LAN page)
    python manage.py runportal     only the LAN history page
"""

import os
import secrets
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # .../V01/web
PROJECT_ROOT = BASE_DIR.parent  # .../V01

# The detection code lives in V01/new/person_detection; make it importable.
DETECTION_DIR = PROJECT_ROOT / "new"
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))


def _secret_key() -> str:
    """Random key created on first run and kept in web/.secret_key (not in git)."""
    if os.environ.get("PORTAL_SECRET_KEY"):
        return os.environ["PORTAL_SECRET_KEY"]
    path = BASE_DIR / ".secret_key"
    if not path.exists():
        path.write_text(secrets.token_urlsafe(50), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


SECRET_KEY = _secret_key()
DEBUG = os.environ.get("PORTAL_DEBUG", "0") == "1"

# LAN use: reachable by IP or computer name. Login protects every page.
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "cameras",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # Desktop app (everything) vs LAN web (history only); see portal/modes.py
    "portal.modes.PortalModeMiddleware",
    # Every page needs a logged-in user unless the view is marked
    # @login_not_required (only the login page is).
    "django.contrib.auth.middleware.LoginRequiredMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "portal.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "portal.modes.context",
            ],
        },
    },
]

WSGI_APPLICATION = "portal.wsgi.application"
ASGI_APPLICATION = "portal.asgi.application"

# Users and the camera list only. Detection history stays in the CSV/JSON
# files under V01/new/output, shared with connect.py.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        "OPTIONS": {"timeout": 20},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "login"

LANGUAGE_CODE = "th"
TIME_ZONE = "Asia/Bangkok"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
WHITENOISE_USE_FINDERS = True  # serve static files without collectstatic

# Browser-camera frames are small JPEGs, but leave room for a big camera.
DATA_UPLOAD_MAX_MEMORY_SIZE = 8 * 1024 * 1024

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---- portal-specific -------------------------------------------------------
# "app" = desktop window (everything), "web" = LAN page (history only).
# Each server marks its requests (portal/modes.py); this is only the fallback
# for unmarked requests, e.g. `manage.py runserver`. Tests use the full app.
_RUNNING_TESTS = len(sys.argv) > 1 and sys.argv[1] == "test"
PORTAL_DEFAULT_MODE = os.environ.get("PORTAL_DEFAULT_MODE") or ("app" if _RUNNING_TESTS else "web")
# One-time secret the desktop window uses to sign in (set by desktop.py).
PORTAL_APP_TOKEN = os.environ.get("PORTAL_APP_TOKEN", "")
# The local account the desktop app signs in as (no password: cannot be used
# to log in to the LAN web page).
PORTAL_APP_USERNAME = "cn401-app"

# Start the "always running" IP cameras when the server starts. runportal
# sets this; plain management commands (migrate, createsuperuser) do not.
PORTAL_AUTOSTART = os.environ.get("PORTAL_AUTOSTART", "0") == "1"
# Most browser-camera sessions processed at the same time (GPU memory).
PORTAL_MAX_BROWSER_SESSIONS = int(os.environ.get("PORTAL_MAX_BROWSER_SESSIONS", "6"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {"cameras": {"handlers": ["console"], "level": "INFO"}},
}
