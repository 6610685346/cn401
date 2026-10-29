"""Pages of "ทดสอบด้วยวิดีโอ": upload, run, review, metrics, threshold tuning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from person_detection import config
from person_detection.enrollment import EnrollError
from person_detection.evaluation import metrics as M

from . import evaluation, services, system_settings
from .models import Camera
from .models import (DISTANCE_CHOICES, LIGHTING_CHOICES, LONG_SIDE_CHOICES, ROTATE_CHOICES, SPLIT_CHOICES,
                     EvalEvent, EvalVideo)
from .views import _after_settings_change

LIGHTING = dict(LIGHTING_CHOICES)
DISTANCE = dict(DISTANCE_CHOICES)
SPLIT = dict(SPLIT_CHOICES)


def _current_threshold() -> float:
    """Threshold of the face model chosen in settings."""
    from person_detection.vision import face_models
    return face_models.threshold_for()


def _threshold(request, default: float | None = None) -> float:
    try:
        value = float(request.GET.get("t", ""))
        if 0.0 < value < 1.0:
            return round(value, 3)
    except ValueError:
        pass
    return _current_threshold() if default is None else default


def _chart(sweep: dict, current: float) -> dict | None:
    """Coordinates for the FAR / FRR line chart (viewBox 0 0 520 220)."""
    rows = [r for r in sweep["rows"] if r["far"] is not None or r["frr"] is not None]
    if not rows:
        return None
    left, right, top, bottom = 44, 506, 14, 186
    t0, t1 = sweep["rows"][0]["threshold"], sweep["rows"][-1]["threshold"]
    x = lambda t: left + (t - t0) / (t1 - t0) * (right - left)  # noqa: E731
    y = lambda v: bottom - v * (bottom - top)  # noqa: E731

    def line(key):
        pts = [(x(r["threshold"]), y(r[key])) for r in rows if r[key] is not None]
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in pts), [{"x": a, "y": b} for a, b in pts]

    far, far_pts = line("far")
    frr, frr_pts = line("frr")
    best = sweep.get("best")
    return {
        "far": far, "frr": frr, "far_pts": far_pts, "frr_pts": frr_pts,
        "xticks": [{"x": x(r["threshold"]), "label": f"{r['threshold']:.2f}"} for r in sweep["rows"][::2]],
        "yticks": [{"y": y(v), "label": f"{int(v * 100)}%"} for v in (0, 0.25, 0.5, 0.75, 1)],
        "best_x": x(best["threshold"]) if best else None,
        "current_x": x(current) if t0 <= current <= t1 else None,
        "left": left, "right": right, "top": top, "bottom": bottom,
    }


def _groups(videos, threshold, fixed=False):
    """Results per split, and test results per lighting / distance (report tables)."""
    done = [v for v in videos if v.status in ("done", "cancelled")]
    groups = []
    for key, label in SPLIT_CHOICES:
        part = [v for v in done if v.split == key]
        if part:
            groups.append({"label": label, "videos": len(part), "m": evaluation.group_metrics(part, threshold)})
    # Same videos, different models (each at its own model's threshold unless ?t= is given)
    labels = {}
    for v in done:
        labels.setdefault(evaluation.model_label(v.stats), []).append(v)
    if len(labels) > 1:
        for label, part in sorted(labels.items()):
            t = threshold if fixed else evaluation.model_threshold(part[0].stats)
            groups.append({"label": f"โมเดล: {label} (τ {t:g})", "videos": len(part),
                           "m": evaluation.group_metrics(part, t)})
    tests = [v for v in done if v.split == "test"]
    by_condition = []
    for field, names in (("lighting", LIGHTING), ("distance", DISTANCE)):
        for key, label in names.items():
            part = [v for v in tests if key and getattr(v, field) == key]
            if part:
                by_condition.append({"label": ("แสง: " if field == "lighting" else "ระยะ: ") + label,
                                     "videos": len(part), "m": evaluation.group_metrics(part, threshold)})
    return groups, by_condition


def evaluate(request):
    videos = list(EvalVideo.objects.prefetch_related("events"))
    threshold = _threshold(request)
    if request.method == "POST":
        if not request.user.is_staff:
            raise Http404
        return _upload(request)
    groups, by_condition = _groups(videos, threshold, fixed="t" in request.GET)
    # The sweep needs scores of one model: the train videos run with the face model in use now
    face_model = config.FACE_RECOGNIZER
    train = [v for v in videos if v.split == "train" and v.status in ("done", "cancelled")
             and ((v.stats or {}).get("face_model") or "sface") == face_model]
    sweep = evaluation.threshold_sweep(train, face_model) if train else None
    for v in videos:
        v.model_label = evaluation.model_label(v.stats) if v.stats else ""
    return render(request, "cameras/evaluate.html", {
        "videos": videos, "groups": groups, "by_condition": by_condition, "threshold": threshold,
        "current_threshold": _current_threshold(), "face_model": face_model,
        "model_label": evaluation.model_label({"face_model": face_model, "person_model": config.PERSON_MODEL,
                                               "reid": config.TRACKER_REID, "full_res": config.FACE_FULL_RES}),
        "sweep": sweep, "chart": _chart(sweep, threshold) if sweep else None,
        "split_choices": SPLIT_CHOICES, "lighting_choices": LIGHTING_CHOICES, "distance_choices": DISTANCE_CHOICES,
        "rotate_choices": ROTATE_CHOICES, "long_side_choices": LONG_SIDE_CHOICES,
        "accept": ",".join(sorted(evaluation.VIDEO_TYPES)),
        "door_cameras": [c for c in Camera.objects.all() if c.door_line],
    })


def _upload(request):
    upload = request.FILES.get("video")
    if not upload:
        messages.error(request, "กรุณาเลือกไฟล์วิดีโอ")
        return redirect("evaluate")
    suffix = Path(upload.name).suffix.lower()
    if suffix not in evaluation.VIDEO_TYPES:
        messages.error(request, f"ไฟล์ {suffix or '(ไม่มีนามสกุล)'} ใช้ไม่ได้ ใช้ได้: " + ", ".join(sorted(evaluation.VIDEO_TYPES)))
        return redirect("evaluate")
    if upload.size > evaluation.MAX_VIDEO_BYTES:
        messages.error(request, "ไฟล์ใหญ่เกิน 4 GB")
        return redirect("evaluate")

    def choice(name, choices, default):
        value = request.POST.get(name, default)
        return value if value in dict(choices) else default

    def number(name, choices, default):
        try:
            value = int(request.POST.get(name, default))
        except ValueError:
            return default
        return value if value in dict(choices) else default

    video = EvalVideo.objects.create(
        name=(request.POST.get("name") or Path(upload.name).stem).strip()[:120],
        split=choice("split", SPLIT_CHOICES, "test"),
        lighting=choice("lighting", LIGHTING_CHOICES, ""),
        distance=choice("distance", DISTANCE_CHOICES, ""),
        notes=request.POST.get("notes", "")[:2000],
        options={
            "rotate": number("rotate", ROTATE_CHOICES, 0),
            "long_side": number("long_side", LONG_SIDE_CHOICES, config.PROCESS_LONG_SIDE),
            "realtime": request.POST.get("realtime") == "on",
            "direction": request.POST.get("direction") == "on",
            "door_line": _door_line(request.POST.get("door_camera")),
            "vlm": request.POST.get("vlm") == "on",
        },
        created_by=request.user.get_username(), file="")
    folder = evaluation.video_dir(video.id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"video{suffix}"
    with open(path, "wb") as handle:
        for chunk in upload.chunks():
            handle.write(chunk)
    video.file = str(path)
    video.message = "รอคิว"
    video.save(update_fields=["file", "message"])
    evaluation.jobs.submit(video.id)
    messages.success(request, f"อัปโหลด {video.name} แล้ว ระบบกำลังประมวลผล")
    return redirect("evaluate_detail", video.id)


def _door_line(camera_id):
    """Copy of a camera's door line, so the test counts in/out the same way."""
    try:
        camera = Camera.objects.filter(pk=int(camera_id)).first()
    except (TypeError, ValueError):
        return None
    return camera.door_line if camera else None


def evaluate_detail(request, pk):
    video = get_object_or_404(EvalVideo, pk=pk)
    run_threshold = evaluation.model_threshold(video.stats)  # of the face model this run used
    threshold = _threshold(request, run_threshold)
    events = list(video.events.all())
    gallery = sorted(services.factory.people_db().names(), key=str.lower)
    names = sorted(set(gallery) | {e.candidate for e in events if e.candidate}, key=str.lower)
    rows = []
    for e in events:
        predicted = M.predicted_name(e.as_metric(), threshold)
        rows.append({"e": e, "predicted": predicted, "faces": len(e.face_photos or [])})
    return render(request, "cameras/evaluate_detail.html", {
        "video": video, "rows": rows, "names": names, "gallery": gallery, "threshold": threshold,
        "current_threshold": run_threshold, "model_label": evaluation.model_label(video.stats),
        "runs": list(reversed(video.runs or [])),
        "this_run": evaluation.run_summary(video, threshold) if video.stats and events else None,
        "m": evaluation.video_metrics(video, threshold), "stats": video.stats or {},
        "lighting": LIGHTING.get(video.lighting, ""), "distance": DISTANCE.get(video.distance, ""),
        "split": SPLIT.get(video.split, video.split), "NONE": M.NONE, "SKIP": M.SKIP,
        "running": video.status in ("queued", "running"),
    })


def api_eval_status(request, pk):
    video = get_object_or_404(EvalVideo, pk=pk)
    return JsonResponse({"status": video.status, "progress": video.progress, "message": video.message,
                         "events": video.events.count(), "stats": video.stats or {},
                         "has_preview": evaluation.jobs.preview(video.id) is not None})


def eval_preview(request, pk):
    data = evaluation.jobs.preview(int(pk))
    if not data:
        raise Http404
    response = HttpResponse(data, content_type="image/jpeg")
    response["Cache-Control"] = "no-store"
    return response


def eval_image(request, pk, which="snapshot", index=0):
    event = get_object_or_404(EvalEvent, pk=pk)
    if which == "face":
        photos = event.face_photos or []
        if not 0 <= index < len(photos):
            raise Http404
        path = Path(photos[index])
    else:
        path = Path(event.snapshot) if event.snapshot else None
    if not path or not path.is_file() or evaluation.eval_dir().resolve() not in path.resolve().parents:
        raise Http404
    response = FileResponse(open(path, "rb"), content_type="image/jpeg")
    response["Cache-Control"] = "private, max-age=3600"
    return response


def api_eval_metrics(request, pk):
    """The metric cards, re-rendered after a review change."""
    video = get_object_or_404(EvalVideo, pk=pk)
    threshold = _threshold(request)
    html = render_to_string("cameras/eval_metrics.html", {
        "m": evaluation.video_metrics(video, threshold), "stats": video.stats or {}, "video": video}, request)
    return HttpResponse(html)


REVIEW_VALUES = {
    "gt_status": {"", "ok", "false", "duplicate"},
    "gt_direction": {"", "ok", "wrong", "false"},
}


@staff_member_required
@require_POST
def api_eval_review(request, pk):
    event = get_object_or_404(EvalEvent, pk=pk)
    try:
        data = json.loads(request.body or b"{}")
    except ValueError:
        return JsonResponse({"error": "bad json"}, status=400)
    changed = []
    for field in ("gt_status", "gt_direction"):
        if field in data:
            if data[field] not in REVIEW_VALUES[field]:
                return JsonResponse({"error": f"bad {field}"}, status=400)
            setattr(event, field, data[field])
            changed.append(field)
    if "gt_identity" in data:
        event.gt_identity = str(data["gt_identity"] or "")[:100]
        changed.append("gt_identity")
    if changed:
        event.save(update_fields=changed)
    return JsonResponse({"ok": True, "event": event.id, **{f: getattr(event, f) for f in changed}})


@staff_member_required
@require_POST
def eval_bulk_accept(request, pk):
    """Mark every unreviewed row as "the system was right"."""
    video = get_object_or_404(EvalVideo, pk=pk)
    threshold = _threshold(request)
    count = 0
    for e in video.events.all():
        if e.kind == "sighting" and not e.gt_status:
            e.gt_status = "ok"
            if not e.gt_identity:
                predicted = M.predicted_name(e.as_metric(), threshold)
                e.gt_identity = predicted if predicted and predicted != M.UNKNOWN else (M.NONE if predicted else "")
            e.save(update_fields=["gt_status", "gt_identity"])
            count += 1
        elif e.kind in ("in", "out") and not e.gt_direction:
            e.gt_direction = "ok"
            e.save(update_fields=["gt_direction"])
            count += 1
    messages.success(request, f"ยืนยันตามระบบ {count} แถว (แก้แถวที่ผิดได้ต่อ)")
    return redirect("evaluate_detail", video.id)


@staff_member_required
@require_POST
def eval_missed(request, pk):
    video = get_object_or_404(EvalVideo, pk=pk)
    for field in ("missed_people", "missed_in", "missed_out"):
        try:
            setattr(video, field, max(0, int(request.POST.get(field, 0) or 0)))
        except ValueError:
            pass
    video.save(update_fields=["missed_people", "missed_in", "missed_out"])
    if request.headers.get("x-requested-with") == "fetch":
        return JsonResponse({"ok": True})
    messages.success(request, "บันทึกจำนวนที่ระบบพลาดแล้ว")
    return redirect("evaluate_detail", video.id)


@staff_member_required
@require_POST
def eval_enroll(request, pk):
    """Register a person from the photos a video event collected."""
    event = get_object_or_404(EvalEvent, pk=pk)
    name = request.POST.get("name", "").strip()
    paths = [Path(p) for p in (event.face_photos or []) if Path(p).is_file()]
    if not paths and event.snapshot and Path(event.snapshot).is_file():
        paths = [Path(event.snapshot)]
    if not paths:
        messages.error(request, "แถวนี้ไม่มีรูปหน้าให้ใช้")
        return redirect("evaluate_detail", event.video_id)
    try:
        result = services.enroller().enroll(name, [(f"video{event.video_id}_{p.name}", p.read_bytes()) for p in paths])
    except EnrollError as exc:
        messages.error(request, str(exc))
        return redirect("evaluate_detail", event.video_id)
    event.gt_status = event.gt_status or "ok"
    event.gt_identity = result.name
    event.save(update_fields=["gt_status", "gt_identity"])
    messages.success(request, f"ลงทะเบียน {result.name} จากวิดีโอแล้ว (ใช้ {result.used} รูป) · "
                              "ถ้าจะวัดผลด้วยวิดีโอนี้ ให้กด \"ประมวลผลใหม่\" เพราะตอนรันยังไม่รู้จักคนนี้")
    return redirect("evaluate_detail", event.video_id)


@staff_member_required
@require_POST
def eval_apply_threshold(request):
    try:
        value = round(float(request.POST.get("threshold", "")), 3)
    except ValueError:
        value = None
    from person_detection.vision import face_models
    key = face_models.get().threshold_setting  # the face model in use now
    setting = system_settings.BY_KEY[key]
    if value is None or not setting.min <= value <= setting.max:
        messages.error(request, "ค่า threshold ไม่ถูกต้อง")
        return redirect("evaluate")
    values = system_settings.current()
    values[key] = value
    changed = system_settings.save(values, request.user)
    _after_settings_change(request, changed, f"ตั้งเกณฑ์ความเหมือนใบหน้าเป็น {value}")
    return redirect("evaluate")


@staff_member_required
@require_POST
def eval_rerun(request, pk):
    video = get_object_or_404(EvalVideo, pk=pk)
    if video.status in ("queued", "running"):
        return redirect("evaluate_detail", video.id)
    video.status, video.progress, video.message = "queued", 0, "รอคิว"
    video.save(update_fields=["status", "progress", "message"])
    evaluation.jobs.submit(video.id)
    messages.success(request, "เริ่มประมวลผลใหม่ (ผลตรวจเดิมของวิดีโอนี้จะถูกล้าง)")
    return redirect("evaluate_detail", video.id)


@staff_member_required
@require_POST
def eval_cancel(request, pk):
    video = get_object_or_404(EvalVideo, pk=pk)
    evaluation.jobs.cancel(video.id)
    if video.status == "queued":
        video.status, video.message = "cancelled", "ยกเลิกก่อนเริ่ม"
        video.save(update_fields=["status", "message"])
    return redirect("evaluate_detail", video.id)


@staff_member_required
@require_POST
def eval_update(request, pk):
    video = get_object_or_404(EvalVideo, pk=pk)
    if request.POST.get("split") in dict(SPLIT_CHOICES):
        video.split = request.POST["split"]
    if request.POST.get("lighting", None) in LIGHTING:
        video.lighting = request.POST["lighting"]
    if request.POST.get("distance", None) in DISTANCE:
        video.distance = request.POST["distance"]
    if request.POST.get("name"):
        video.name = request.POST["name"].strip()[:120]
    video.save(update_fields=["split", "lighting", "distance", "name"])
    messages.success(request, "บันทึกข้อมูลวิดีโอแล้ว")
    return redirect("evaluate_detail", video.id)


@staff_member_required
@require_POST
def eval_delete(request, pk):
    video = get_object_or_404(EvalVideo, pk=pk)
    name = video.name
    evaluation.delete_video(video)
    messages.success(request, f"ลบ {name} แล้ว")
    return redirect("evaluate")


def _csv_response(filename: str) -> tuple[HttpResponse, csv.writer]:
    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.write("﻿")
    return response, csv.writer(response)


def eval_export(request, pk):
    """Every event of one video with its review, for the report appendix."""
    video = get_object_or_404(EvalVideo, pk=pk)
    threshold = _threshold(request)
    response, writer = _csv_response(f"video_test_{video.id}.csv")
    writer.writerow(["time_s", "kind", "track_id", "system_name", f"name_at_{threshold}", "face_seen", "candidate",
                     "score", "shirt", "pants", "confidence", "latency_s", "gt_status", "gt_identity", "gt_direction"])
    for e in video.events.all():
        predicted = M.predicted_name(e.as_metric(), threshold)
        writer.writerow([e.time_s, e.kind, e.track_id, e.name, predicted or "", e.face_seen, e.candidate,
                         "" if e.score is None else f"{e.score:.4f}", e.shirt_color, e.pants_color,
                         f"{e.confidence:.2f}", "" if e.latency_s is None else e.latency_s,
                         e.gt_status, e.gt_identity, e.gt_direction])
    return response


def _fmt(v, pct=True):
    if v is None:
        return ""
    return f"{v * 100:.1f}" if pct else (f"{v:.3f}" if isinstance(v, float) else v)


def eval_report(request):
    """Summary table (per split and per condition) for the report."""
    threshold = _threshold(request)
    videos = list(EvalVideo.objects.prefetch_related("events"))
    groups, by_condition = _groups(videos, threshold)
    response, writer = _csv_response(f"video_test_summary_t{threshold}.csv")
    writer.writerow(["group", "videos", "events", "reviewed", "TP", "FP", "FN", "precision_%", "recall_%", "F1_%",
                     "genuine", "impostor", "FAR_%", "FRR_%", "accuracy_%", "misID_%", "face_coverage_%",
                     "in_precision_%", "in_recall_%", "out_precision_%", "out_recall_%", "occupancy_error",
                     "latency_s", "threshold"])
    for g in groups + by_condition:
        m = g["m"]
        d, f, r = m["detection"], m["face"], m["direction"]
        writer.writerow([g["label"], g["videos"], m["events"], m["reviewed"], d["tp"], d["fp"], d["fn"],
                         _fmt(d["precision"]), _fmt(d["recall"]), _fmt(d["f1"]), f["genuine"], f["impostor"],
                         _fmt(f["far"]), _fmt(f["frr"]), _fmt(f["accuracy"]), _fmt(f["misid"]), _fmt(f["coverage"]),
                         _fmt(r["in"]["precision"]), _fmt(r["in"]["recall"]), _fmt(r["out"]["precision"]),
                         _fmt(r["out"]["recall"]), "" if r["occupancy_error"] is None else r["occupancy_error"],
                         "" if m["latency_s"] is None else f"{m['latency_s']:.2f}", threshold])
    return response
