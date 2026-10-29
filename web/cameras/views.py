import asyncio
import base64
import csv
import json
import re
import secrets
from urllib.parse import quote
from pathlib import Path

import cv2

from django.conf import settings as django_settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model, login
from django.contrib.auth.decorators import login_not_required
from django.http import (FileResponse, Http404, HttpResponse, HttpResponseBadRequest, JsonResponse,
                         StreamingHttpResponse)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from person_detection import config
from person_detection.core.geometry import scale_to_long_side
from person_detection.enrollment import EnrollError
from person_detection.multicam.floor import CalibrationError, FloorCalibration
from person_detection.pipeline import CameraOptions
from person_detection.storage.recorder import safe_slug

from portal.modes import APP

from . import services, system_settings
from .forms import CameraForm
from .models import LONG_SIDE_CHOICES, ROTATE_CHOICES, Camera

# ------------------------------------------------------------- desktop app


def app_user():
    """The local account the desktop window signs in as (created on first use).

    It has no usable password, so it can only be used through the desktop
    window, never to log in to the LAN web page.
    """
    User = get_user_model()
    user, created = User.objects.get_or_create(
        username=django_settings.PORTAL_APP_USERNAME,
        defaults={"is_staff": True, "is_superuser": True},
    )
    if created:
        user.set_unusable_password()
        user.save()
    elif not (user.is_staff and user.is_superuser and user.is_active):
        user.is_staff = user.is_superuser = user.is_active = True
        user.save(update_fields=["is_staff", "is_superuser", "is_active"])
    return user


@login_not_required
def app_login(request, token):
    """The desktop window opens /app-login/<secret>/ once to sign itself in.

    The secret is made fresh by desktop.py on every start and only works on
    the app's own 127.0.0.1 server.
    """
    expected = django_settings.PORTAL_APP_TOKEN
    if (getattr(request, "portal_mode", None) != APP or not expected
            or not secrets.compare_digest(token.encode(), expected.encode())):
        raise Http404
    login(request, app_user(), backend="django.contrib.auth.backends.ModelBackend")
    return redirect("dashboard")


# ------------------------------------------------------------------ cameras


def dashboard(request):
    cameras = Camera.objects.all()
    for camera in cameras:
        camera.state = services.workers.snapshot(camera.id)
    return render(request, "cameras/dashboard.html", {
        "cameras": cameras,
        "device": services.device_info(),
        "storage": services.storage_info(),
        "browser_sessions": services.sessions.count(),
    })


@staff_member_required
def camera_add(request):
    form = CameraForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        camera = form.save()
        if camera.should_run:
            services.workers.start(camera)
        messages.success(request, f"เพิ่มกล้อง {camera.name} แล้ว")
        return redirect("dashboard")
    return render(request, "cameras/camera_form.html", {"form": form, "title": "เพิ่มกล้อง IP"})


@staff_member_required
def camera_edit(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    old_view = (camera.rotate, camera.source)
    form = CameraForm(request.POST or None, instance=camera)
    if request.method == "POST" and form.is_valid():
        camera = form.save(commit=False)
        if camera.floor_calibration and (camera.rotate, camera.source) != old_view:
            # the picture changed, so the clicked floor points no longer match
            camera.floor_calibration = None
            messages.warning(request, "เปลี่ยนการหมุนภาพหรือ URL แล้ว ต้องปรับเทียบพื้นใหม่")
        camera.save()
        # restart with the new settings (or stop if switched off)
        if camera.should_run:
            services.workers.start(camera)
        else:
            services.workers.stop(camera.id)
        messages.success(request, f"บันทึก {camera.name} แล้ว")
        return redirect("dashboard")
    return render(request, "cameras/camera_form.html",
                  {"form": form, "title": f"แก้ไข {camera.name}", "camera": camera})


@staff_member_required
@require_POST
def camera_delete(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    services.workers.stop(camera.id)
    camera.delete()
    messages.success(request, f"ลบกล้อง {camera.name} แล้ว (ไฟล์ CSV เดิมยังอยู่)")
    return redirect("dashboard")


@staff_member_required
@require_POST
def camera_toggle(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    camera.should_run = request.POST.get("run") == "1"
    camera.save(update_fields=["should_run"])
    if camera.should_run:
        services.workers.start(camera)
    else:
        services.workers.stop(camera.id)
    return redirect(request.POST.get("next") or "dashboard")


def camera_live(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    return render(request, "cameras/live.html", {"camera": camera})


async def camera_mjpeg(request, pk):
    """Live annotated picture as MJPEG: works in a plain <img> tag."""
    worker = services.workers.get(pk)
    if worker is None:
        raise Http404("กล้องนี้ไม่ได้รันอยู่")

    async def stream():
        last_id = -1
        idle = 0.0
        while True:
            current = services.workers.get(pk)
            if current is not worker:  # stopped or restarted
                return
            if worker.jpeg is not None and worker.jpeg_id != last_id:
                last_id, jpeg, idle = worker.jpeg_id, worker.jpeg, 0.0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            else:
                await asyncio.sleep(0.04)
                idle += 0.04
                if idle > 60:  # camera dead for a minute: let the page reconnect
                    return

    response = StreamingHttpResponse(stream(), content_type="multipart/x-mixed-replace; boundary=frame")
    response["Cache-Control"] = "no-store"
    return response


def camera_wall(request):
    """Several cameras at once (each tile asks for its next picture: frame.jpg)."""
    cameras = list(Camera.objects.all())
    for camera in cameras:
        camera.state = services.workers.snapshot(camera.id)
    return render(request, "cameras/wall.html", {"cameras": cameras})


async def camera_frame(request, pk):
    """The newest annotated picture of a camera, as one JPEG.

    ?after=<id> waits (up to 2 s) until a picture newer than <id> exists, so a
    tile that asks again right away gets the next frame, never the same one.
    One short request per picture, instead of one endless MJPEG stream per
    camera: a browser opens at most 6 connections to one server, which a wall
    of MJPEG streams would use up.
    """
    worker = services.workers.get(pk)
    if worker is None:
        raise Http404("กล้องนี้ไม่ได้รันอยู่")
    try:
        after = int(request.GET.get("after", "-1"))
    except ValueError:
        after = -1
    waited = 0.0
    while (worker.jpeg is None or worker.jpeg_id == after) and waited < 2.0:
        await asyncio.sleep(0.03)
        waited += 0.03
        if services.workers.get(pk) is not worker:  # stopped or restarted
            break
    jpeg, frame_id = worker.jpeg, worker.jpeg_id
    if jpeg is None:
        response = HttpResponse(status=204)
    else:
        response = HttpResponse(jpeg, content_type="image/jpeg")
        response["X-Frame-Id"] = str(frame_id)
    response["Cache-Control"] = "no-store"
    return response


def api_status(request):
    data = {"device": services.device_info(), "browser_sessions": services.sessions.count(), "cameras": {}}
    for camera in Camera.objects.all():
        data["cameras"][camera.id] = services.workers.snapshot(camera.id) or {
            "id": camera.id, "name": camera.name, "status": "stopped"}
    return JsonResponse(data)


# --------------------------------------------------------- browser camera


THIS_PC_TAG = "this-pc"  # CSV tag of the desktop app's own camera page


def _browser_tag(user) -> str:
    if user.get_username() == django_settings.PORTAL_APP_USERNAME:
        return THIS_PC_TAG
    return f"browser-{user.get_username()}"


def browser_camera(request):
    return render(request, "cameras/browser.html", {
        "csv_source": _describe_tag(_browser_tag(request.user), {}),
        "max_fps": services.BROWSER_MAX_FPS,
        "rotate_choices": ROTATE_CHOICES,
        "long_side_choices": LONG_SIDE_CHOICES,
    })


def _browser_options(request, user) -> CameraOptions:
    def flag(name, default):
        value = request.GET.get(name)
        return default if value is None else value == "1"

    try:
        rotate = int(request.GET.get("rotate", 0))
        long_side = int(request.GET.get("long_side", 960))
    except ValueError:
        raise ValueError("bad rotate/long_side")
    if rotate not in dict(ROTATE_CHOICES) or long_side not in dict(LONG_SIDE_CHOICES):
        raise ValueError("bad rotate/long_side")
    return CameraOptions(
        rotate=rotate,
        long_side=long_side,
        enable_face_detection=True,
        enable_face_recognition=flag("face", True),
        enable_vlm_par=flag("vlm", True),
        enable_direction=False,
        save_csv=config.SAVE_CSV,
        save_crops=config.SAVE_CROPS,
        csv_tag=_browser_tag(user),
    )


SESSION_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


@require_POST
async def api_browser_frame(request):
    """Browser sends one JPEG frame, gets back the people found in it."""
    user = await request.auser()
    session_id = request.GET.get("session", "")
    if not SESSION_ID.match(session_id):
        return HttpResponseBadRequest("bad session id")
    try:
        options = _browser_options(request, user)
    except ValueError as exc:
        return HttpResponseBadRequest(str(exc))
    body = request.body
    if not body:
        return HttpResponseBadRequest("empty frame")
    try:
        # model work happens in a worker thread so the server stays responsive
        data = await asyncio.to_thread(services.sessions.process, user, session_id, body, options)
    except services.SessionLimitError as exc:
        return JsonResponse({"error": str(exc)}, status=429)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse(data)


@require_POST
def api_browser_close(request):
    session_id = request.GET.get("session", "")
    if SESSION_ID.match(session_id):
        services.sessions.close(request.user, session_id)
    return JsonResponse({"ok": True})


# ------------------------------------------------------------ system settings


@staff_member_required
def settings_page(request):
    if request.method == "POST" and request.POST.get("reset") == "1":
        changed = system_settings.save({}, request.user)
        _after_settings_change(request, changed, "กลับเป็นค่าเริ่มต้นแล้ว")
        return redirect("settings_page")

    form = system_settings.SettingsForm(request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            changed = system_settings.save(form.values(), request.user)
            _after_settings_change(request, changed, "บันทึกแล้ว")
            return redirect("settings_page")
        messages.error(request, "มีค่าที่ไม่ถูกต้อง ดูข้อความสีแดงด้านล่าง")
    saved = system_settings.read_saved()
    return render(request, "cameras/settings.html", {
        "form": form,
        "meta": saved.get("_meta", {}),
        "overridden": [k for k in saved if k in system_settings.BY_KEY],
        "settings_path": system_settings.path(),
        "cameras_running": len(services.workers.running_ids()),
        "gallery": services.gallery.status(),
    })


@staff_member_required
@require_POST
def gallery_rebuild(request):
    """Make the missing face vectors of the current face model from the photos."""
    if services.gallery.start(force=True):
        messages.success(request, "กำลังสร้างข้อมูลใบหน้าจากรูปที่ลงทะเบียนไว้")
    else:
        messages.info(request, "กำลังทำอยู่แล้ว")
    return redirect("people")


def api_gallery_status(request):
    """Progress of making face vectors for a newly chosen face model."""
    state = services.gallery.status()
    if request.GET.get("partial"):
        if not state:
            return HttpResponse("")
        return render(request, "cameras/gallery_status.html", {"gallery": state})
    return JsonResponse(state or {})


def _after_settings_change(request, changed: list[str], done: str) -> None:
    if not changed:
        messages.info(request, "ไม่มีค่าที่เปลี่ยน")
        return
    if "FACE_RECOGNIZER" in changed:
        try:  # download the new face model and make everyone's face vector with it
            services.gallery.start(force=True)
        except Exception as exc:
            messages.error(request, f"เตรียมโมเดลใบหน้าไม่สำเร็จ: {exc}")
    if system_settings.needs_restart(changed):
        restarted = 0
        for camera in Camera.objects.filter(pk__in=services.workers.running_ids()):
            services.workers.start(camera)  # rebuild its models with the new values
            restarted += 1
        services.sessions.clear()
        services.reset_enroller()
        extra = f" · รีสตาร์ทกล้อง {restarted} ตัวเพื่อใช้ค่าใหม่" if restarted else ""
    else:
        extra = " · มีผลทันที"
    labels = ", ".join(system_settings.BY_KEY[k].label for k in changed)
    messages.success(request, f"{done}: {labels}{extra}")


# ------------------------------------------------------- floor / multi-camera


@staff_member_required
def camera_calibrate(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    calibration = FloorCalibration.from_dict(camera.floor_calibration)
    others = [{"name": c.name, "floor": c.floor_calibration["floor"]}
              for c in Camera.objects.exclude(pk=pk) if c.floor_calibration]
    return render(request, "cameras/calibrate.html", {
        "camera": camera,
        "running": services.workers.get(camera.id) is not None,
        "initial": json.dumps({
            "image": calibration.image_points if calibration else [],
            "floor": calibration.floor_points if calibration else [],
            "errors": calibration.point_errors_m if calibration else [],
            "grid": calibration.grid_lines() if calibration else [],
            "error_m": calibration.error_m if calibration else None,
        }),
        "others": others,
    })


def camera_snapshot(request, pk):
    """Latest clean picture (no boxes drawn) for clicking floor points."""
    worker = services.workers.get(pk)
    if worker is None or not worker.snapshot_jpeg:
        raise Http404("ยังไม่มีภาพ")
    response = HttpResponse(worker.snapshot_jpeg, content_type="image/jpeg")
    response["Cache-Control"] = "no-store"
    return response


@staff_member_required
@require_POST
def api_calibration(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    try:
        data = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "ข้อมูลไม่ถูกต้อง"}, status=400)
    if data.get("clear"):
        camera.floor_calibration = None
        camera.save(update_fields=["floor_calibration"])
        services.workers.set_calibration(camera.id, None)
        return JsonResponse({"ok": True, "cleared": True})
    try:
        calibration = FloorCalibration.fit(data.get("image", []), data.get("floor", []))
    except CalibrationError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except (TypeError, ValueError):
        return JsonResponse({"error": "ค่าเมตรต้องเป็นตัวเลข"}, status=400)
    camera.floor_calibration = calibration.to_dict()
    camera.save(update_fields=["floor_calibration"])
    services.workers.set_calibration(camera.id, calibration)
    return JsonResponse({
        "ok": True,
        "error_m": calibration.error_m,
        "errors": calibration.point_errors_m,
        "grid": calibration.grid_lines(),
    })


# ------------------------------------------------------------------ door line
@staff_member_required
def camera_door(request, pk):
    camera = get_object_or_404(Camera, pk=pk)
    return render(request, "cameras/door.html", {
        "camera": camera,
        "running": services.workers.get(camera.id) is not None,
        "initial": json.dumps(camera.door_line or None),
    })


@staff_member_required
@require_POST
def api_door(request, pk):
    from person_detection.counting.line import DoorLine
    camera = get_object_or_404(Camera, pk=pk)
    try:
        data = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "ข้อมูลไม่ถูกต้อง"}, status=400)
    if data.get("clear"):
        camera.door_line = None
        camera.save(update_fields=["door_line"])
    else:
        try:
            a, b = data["a"], data["b"]
            if not all(0 <= float(v) <= 1 for v in (*a, *b)):
                raise ValueError("จุดต้องอยู่ในภาพ")
            line = DoorLine(a, b, data.get("inside", "left"))
        except (KeyError, TypeError, IndexError):
            return JsonResponse({"error": "ต้องมีจุด 2 จุด"}, status=400)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        camera.door_line = line.to_dict()
        camera.enable_direction = True  # drawing a door means: count it
        camera.save(update_fields=["door_line", "enable_direction"])
    if services.workers.get(camera.id) is not None:
        services.workers.start(camera)  # restart so the processor uses the new line
    return JsonResponse({"ok": True, "door_line": camera.door_line, "enabled": camera.enable_direction})


def floor_map(request):
    cameras = Camera.objects.all()
    return render(request, "cameras/floor.html", {
        "calibrated": [c for c in cameras if c.is_calibrated],
        "uncalibrated": [c for c in cameras if not c.is_calibrated],
    })


def api_floor(request):
    cameras = []
    for camera in Camera.objects.all():
        if camera.floor_calibration:
            worker = services.workers.get(camera.id)
            cameras.append({"name": camera.name, "floor": camera.floor_calibration["floor"],
                            "running": worker is not None and worker.status == "running"})
    return JsonResponse({"people": services.hub.snapshot(), "cameras": cameras})


# ------------------------------------------------------------------ presence
def presence_page(request):
    """ใครอยู่ตอนนี้: who is inside, in/out counts, hourly chart, recent crossings."""
    import datetime as dt

    from . import presence
    date = request.GET.get("date") or presence.today()
    try:
        dt.date.fromisoformat(date)
    except ValueError:
        raise Http404
    events, store = presence.day_events(date)
    context = {"p": presence.summarize(events), "date": date, "is_today": date == presence.today(),
               "store": store, "cameras_without_door": Camera.objects.filter(door_line__isnull=True).count()
               if request.user.is_staff else 0}
    if request.GET.get("partial"):
        return render(request, "cameras/presence_body.html", context)
    return render(request, "cameras/presence.html", context)


# ------------------------------------------------------------------ history

LOG_NAME = re.compile(r"^person_log_(\d{4}-\d{2}-\d{2})(?:_(.+))?\.csv$")


def _describe_tag(tag: str | None, camera_names: dict[str, str]) -> str:
    if not tag:
        return "connect.py"
    if tag in camera_names:
        return f"กล้อง: {camera_names[tag]}"
    if tag.startswith("cam-"):
        return f"กล้อง: {tag[4:]}"
    if tag.startswith("browser-"):
        return f"เบราว์เซอร์: {tag[8:]}"
    if tag == THIS_PC_TAG:
        return "กล้องของเครื่องนี้"
    return tag


def _log_files() -> list[dict]:
    # file tags are made file-name safe ("Gate HD" -> "cam-Gate-HD"); map back
    camera_names = {safe_slug(c.csv_tag): c.name for c in Camera.objects.all()}
    out_dir = Path(config.OUTPUT_DIR)
    files = []
    if out_dir.exists():
        for path in out_dir.glob("person_log_*.csv"):
            match = LOG_NAME.match(path.name)
            if match:
                files.append({"path": path, "file": path.name, "date": match[1],
                              "tag": match[2] or "", "source": _describe_tag(match[2], camera_names)})
    files.sort(key=lambda f: (f["date"], f["source"]), reverse=True)
    return files


def _history_db(request, database):
    """History read from PostgreSQL (DATABASE_URL set)."""
    from person_detection.storage.database import log_dates, log_rows
    dates = log_dates(database)
    date = request.GET.get("date") or (dates[0] if dates else "")
    source = request.GET.get("source", "")
    name_query = request.GET.get("name", "").strip()
    rows, total, cameras = ([], 0, [])
    if date:
        try:
            rows, total, cameras = log_rows(database, date, source, name_query)
        except ValueError:
            raise Http404
    for row in rows:
        row["source"] = row["camera"]
        if row.get("snapshot_path"):
            row["photo_url"] = f"{reverse('history_photo')}?log={row['id']}"
    return render(request, "cameras/history.html", {
        "dates": dates, "date": date, "source": source, "name": request.GET.get("name", ""),
        "sources": [(c, c) for c in cameras], "from_db": True,
        "rows": rows, "total": total,
    })


def history_export(request):
    """One day of the database history as a CSV download."""
    database = services.factory.database()
    if database is None:
        raise Http404
    from person_detection.storage.database import log_rows
    date = request.GET.get("date", "")
    try:
        rows, _, _ = log_rows(database, date, request.GET.get("source", ""),
                              request.GET.get("name", "").strip(), limit=None)
    except ValueError:
        raise Http404
    columns = ["timestamp", "camera", "name", "surname", "id_number", "direction", "similarity",
               "shirt_color", "pants_color", "confidence", "global_id", "track_id"]
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="detections_{date}.csv"'
    response.write("\ufeff")  # so Excel reads Thai correctly
    writer = csv.writer(response)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row.get(c, "") for c in columns])
    return response


def history(request):
    database = services.factory.database()
    if database is not None:
        return _history_db(request, database)
    files = _log_files()
    dates = sorted({f["date"] for f in files}, reverse=True)
    date = request.GET.get("date") or (dates[0] if dates else "")
    source = request.GET.get("source", "")
    name_query = request.GET.get("name", "").strip().lower()

    rows = []
    day_files = [f for f in files if f["date"] == date]
    for f in day_files:
        if source and f["tag"] != source:
            continue
        with open(f["path"], newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                name = row.get("name") or "-"  # older files have no name column
                if name_query and name_query not in name.lower():
                    continue
                photo = (row.get("snapshot") or "").strip()
                rows.append({**row, "name": name, "source": f["source"],
                             "photo_url": f"{reverse('history_photo')}?f={quote(photo)}" if photo else ""})
    rows.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    return render(request, "cameras/history.html", {
        "dates": dates, "date": date, "source": source, "name": request.GET.get("name", ""),
        "sources": [(f["tag"], f["source"]) for f in day_files],
        "day_files": day_files,
        "rows": rows[:500], "total": len(rows),
    })


def _crop_path(relative: str) -> Path | None:
    """A photo under the crops folder (never outside it)."""
    root = Path(config.CROPS_DIR).resolve()
    try:
        path = (root / relative).resolve()
    except (OSError, ValueError):
        return None
    if root not in path.parents or path.suffix.lower() != ".jpg" or not path.is_file():
        return None
    return path


def history_photo(request):
    """Photo of one history row: ?log=<detection_logs id> (database) or ?f=<file> (CSV)."""
    path = None
    if request.GET.get("log"):
        database = services.factory.database()
        if database is None:
            raise Http404
        from person_detection.storage.database import DetectionLog
        try:
            log_id = int(request.GET["log"])
        except ValueError:
            raise Http404
        with database.Session() as session:
            row = session.get(DetectionLog, log_id)
        if row is None or not row.snapshot_path:
            raise Http404
        candidate = Path(row.snapshot_path)
        root = Path(config.CROPS_DIR).resolve()
        if candidate.is_file() and root in candidate.resolve().parents:
            path = candidate
    elif request.GET.get("f"):
        path = _crop_path(request.GET["f"])
    if path is None:
        raise Http404
    response = FileResponse(open(path, "rb"), content_type="image/jpeg")
    response["Cache-Control"] = "private, max-age=86400"
    return response


def history_download(request, filename):
    if not LOG_NAME.match(filename):
        raise Http404
    path = Path(config.OUTPUT_DIR) / filename
    if not path.is_file():
        raise Http404
    return FileResponse(open(path, "rb"), as_attachment=True, filename=filename)


# ------------------------------------------------------------------- people

MAX_PHOTOS_PER_UPLOAD = 20
MAX_PHOTO_BYTES = 15 * 1024 * 1024
PHOTO_NAME = re.compile(r"^[A-Za-z0-9ก-๙ _.-]+\.(jpe?g|png|webp|bmp)$", re.IGNORECASE)


def people(request):
    db = services.factory.people_db()
    wardrobe = services.factory.wardrobe()
    from person_detection.enrollment import photos_of
    entries = []
    details = {}
    if hasattr(db, "details"):  # the PostgreSQL gallery knows surname / ID
        details = {d["name"]: d for d in db.details()}
    for name in sorted(db.names(), key=str.lower):
        days = wardrobe.history(name)
        extra = details.get(name, {})
        entries.append({
            "name": name,
            "surname": extra.get("surname", ""),
            "id_number": extra.get("id_number") or "",
            "photos": [p.name for p in photos_of(config.PHOTOS_DIR, name)],
            "days": sorted(days.items(), reverse=True)[:7],
        })
    missing = []
    try:
        missing = services.gallery.missing()
    except Exception:
        pass
    return render(request, "cameras/people.html", {
        "people": entries,
        "gallery": services.gallery.status(),
        "face_model": config.FACE_RECOGNIZER,
        "missing": missing,
        "max_photos": MAX_PHOTOS_PER_UPLOAD,
        "uses_database": bool(config.DATABASE_URL),
    })


def _uploaded_photos(request) -> list[tuple[str, bytes]]:
    files = request.FILES.getlist("photos")
    if len(files) > MAX_PHOTOS_PER_UPLOAD:
        raise EnrollError(f"อัปโหลดได้ครั้งละไม่เกิน {MAX_PHOTOS_PER_UPLOAD} รูป")
    uploads = []
    for f in files:
        if f.size > MAX_PHOTO_BYTES:
            raise EnrollError(f"{f.name} ใหญ่เกิน 15 MB")
        uploads.append((f.name, f.read()))
    return uploads


@staff_member_required
@require_POST
def person_enroll(request):
    """Register a new person, or add photos to an existing one."""
    name = request.POST.get("name", "")
    try:
        result = services.enroller().enroll(
            name, _uploaded_photos(request),
            surname=request.POST.get("surname", "").strip()[:100] or None,
            id_number=request.POST.get("id_number", "").strip()[:20] or None)
    except EnrollError as exc:
        messages.error(request, str(exc))
        return redirect("people")
    reports = [{
        "filename": r.filename,
        "ok": r.ok,
        "message": r.message,
        "is_new": r.is_new,
        "similarity": r.similarity,
        "thumbnail": base64.b64encode(r.thumbnail).decode() if r.thumbnail else "",
    } for r in result.reports]
    return render(request, "cameras/enroll_result.html", {
        "result": result, "reports": reports,
        "new_ok": sum(1 for r in result.reports if r.ok and r.is_new),
        "new_total": sum(1 for r in result.reports if r.is_new),
    })


@staff_member_required
@require_POST
def person_delete(request):
    name = request.POST.get("name", "")
    if name not in services.factory.people_db().names():
        raise Http404
    services.enroller().remove(name)
    messages.success(request, f"ลบ {name} และรูปของเขาแล้ว")
    return redirect("people")


def person_photo(request, name, filename):
    """Small preview of an enrollment photo."""
    if not PHOTO_NAME.match(filename) or name not in services.factory.people_db().names():
        raise Http404
    try:
        from person_detection.enrollment import person_dir
        path = person_dir(config.PHOTOS_DIR, name) / filename
    except EnrollError:
        raise Http404
    if not path.is_file():
        raise Http404
    image = cv2.imread(str(path))
    if image is None:
        raise Http404
    image = scale_to_long_side(image, 240) if max(image.shape[:2]) > 240 else image
    data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    response = HttpResponse(data, content_type="image/jpeg")
    response["Cache-Control"] = "private, max-age=300"
    return response
