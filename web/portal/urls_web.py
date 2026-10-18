"""URLs of the LAN web page: log in and read the history, nothing else.

Cameras, settings, enrollment and calibration live in the desktop app
(desktop.py), which uses portal/urls.py.
"""

from django.contrib.auth import views as auth_views
from django.urls import path
from django.views.generic import RedirectView

from cameras import views

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="history"), name="home"),
    path("login/", auth_views.LoginView.as_view(next_page="history"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(next_page="login"), name="logout"),
    path("history/", views.history, name="history"),
    path("history/download/<str:filename>", views.history_download, name="history_download"),
]
