"""URLs of the desktop app (everything). The LAN web page uses urls_web.py."""

from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path

from cameras import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("app-login/<str:token>/", views.app_login, name="app_login"),
    path("login/", auth_views.LoginView.as_view(), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("admin/", admin.site.urls),

    path("cameras/add/", views.camera_add, name="camera_add"),
    path("cameras/<int:pk>/", views.camera_live, name="camera_live"),
    path("cameras/<int:pk>/edit/", views.camera_edit, name="camera_edit"),
    path("cameras/<int:pk>/delete/", views.camera_delete, name="camera_delete"),
    path("cameras/<int:pk>/toggle/", views.camera_toggle, name="camera_toggle"),
    path("cameras/<int:pk>/stream.mjpg", views.camera_mjpeg, name="camera_mjpeg"),
    path("cameras/<int:pk>/calibrate/", views.camera_calibrate, name="camera_calibrate"),
    path("cameras/<int:pk>/snapshot.jpg", views.camera_snapshot, name="camera_snapshot"),
    path("api/cameras/<int:pk>/calibration/", views.api_calibration, name="api_calibration"),
    path("floor/", views.floor_map, name="floor_map"),
    path("settings/", views.settings_page, name="settings_page"),
    path("api/floor/", views.api_floor, name="api_floor"),

    path("my-camera/", views.browser_camera, name="browser_camera"),
    path("api/status/", views.api_status, name="api_status"),
    path("api/browser/frame/", views.api_browser_frame, name="api_browser_frame"),
    path("api/browser/close/", views.api_browser_close, name="api_browser_close"),

    path("history/", views.history, name="history"),
    path("history/download/<str:filename>", views.history_download, name="history_download"),
    path("people/", views.people, name="people"),
    path("people/enroll/", views.person_enroll, name="person_enroll"),
    path("people/delete/", views.person_delete, name="person_delete"),
    path("people/<str:name>/photos/<str:filename>", views.person_photo, name="person_photo"),
]
