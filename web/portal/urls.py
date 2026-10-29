"""URLs of the desktop app (everything). The LAN web page uses urls_web.py."""

from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path

from cameras import views, views_eval

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
    path("cameras/<int:pk>/frame.jpg", views.camera_frame, name="camera_frame"),
    path("wall/", views.camera_wall, name="camera_wall"),
    path("cameras/<int:pk>/calibrate/", views.camera_calibrate, name="camera_calibrate"),
    path("cameras/<int:pk>/snapshot.jpg", views.camera_snapshot, name="camera_snapshot"),
    path("cameras/<int:pk>/door/", views.camera_door, name="camera_door"),
    path("api/cameras/<int:pk>/door/", views.api_door, name="api_door"),
    path("api/cameras/<int:pk>/calibration/", views.api_calibration, name="api_calibration"),
    path("floor/", views.floor_map, name="floor_map"),
    path("settings/", views.settings_page, name="settings_page"),
    path("api/gallery/", views.api_gallery_status, name="api_gallery_status"),
    path("people/gallery/rebuild/", views.gallery_rebuild, name="gallery_rebuild"),
    path("api/floor/", views.api_floor, name="api_floor"),

    path("my-camera/", views.browser_camera, name="browser_camera"),
    path("api/status/", views.api_status, name="api_status"),
    path("api/browser/frame/", views.api_browser_frame, name="api_browser_frame"),
    path("api/browser/close/", views.api_browser_close, name="api_browser_close"),

    path("presence/", views.presence_page, name="presence"),
    path("history/", views.history, name="history"),
    path("history/download/<str:filename>", views.history_download, name="history_download"),
    path("history/export.csv", views.history_export, name="history_export"),
    path("history/photo/", views.history_photo, name="history_photo"),
    path("evaluate/", views_eval.evaluate, name="evaluate"),
    path("evaluate/report.csv", views_eval.eval_report, name="eval_report"),
    path("evaluate/threshold/", views_eval.eval_apply_threshold, name="eval_apply_threshold"),
    path("evaluate/<int:pk>/", views_eval.evaluate_detail, name="evaluate_detail"),
    path("evaluate/<int:pk>/preview.jpg", views_eval.eval_preview, name="eval_preview"),
    path("evaluate/<int:pk>/export.csv", views_eval.eval_export, name="eval_export"),
    path("evaluate/<int:pk>/missed/", views_eval.eval_missed, name="eval_missed"),
    path("evaluate/<int:pk>/accept/", views_eval.eval_bulk_accept, name="eval_bulk_accept"),
    path("evaluate/<int:pk>/rerun/", views_eval.eval_rerun, name="eval_rerun"),
    path("evaluate/<int:pk>/cancel/", views_eval.eval_cancel, name="eval_cancel"),
    path("evaluate/<int:pk>/update/", views_eval.eval_update, name="eval_update"),
    path("evaluate/<int:pk>/delete/", views_eval.eval_delete, name="eval_delete"),
    path("evaluate/event/<int:pk>/snapshot.jpg", views_eval.eval_image, name="eval_snapshot"),
    path("evaluate/event/<int:pk>/face/<int:index>.jpg", views_eval.eval_image, {"which": "face"}, name="eval_face"),
    path("evaluate/event/<int:pk>/enroll/", views_eval.eval_enroll, name="eval_enroll"),
    path("api/evaluate/<int:pk>/status/", views_eval.api_eval_status, name="api_eval_status"),
    path("api/evaluate/<int:pk>/metrics/", views_eval.api_eval_metrics, name="api_eval_metrics"),
    path("api/evaluate/event/<int:pk>/review/", views_eval.api_eval_review, name="api_eval_review"),
    path("people/", views.people, name="people"),
    path("people/enroll/", views.person_enroll, name="person_enroll"),
    path("people/delete/", views.person_delete, name="person_delete"),
    path("people/<str:name>/photos/<str:filename>", views.person_photo, name="person_photo"),
]
