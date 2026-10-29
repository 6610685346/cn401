""""ใครอยู่ตอนนี้": who is inside, from the day's in/out rows.

Works with both stores: PostgreSQL (DATABASE_URL) or the daily CSV files.
A person is *inside* when their last crossing of the day was "in". People
the system could not name are only counted (unknown in - unknown out).
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from urllib.parse import quote

from django.urls import reverse

from person_detection import config

from . import services

BANGKOK = dt.timezone(dt.timedelta(hours=7))
UNKNOWN = "unknown"


def today() -> str:
    return dt.datetime.now(BANGKOK).date().isoformat()


def _csv_events(date: str) -> list[dict]:
    rows = []
    for path in sorted(Path(config.OUTPUT_DIR).glob(f"person_log_{date}*.csv")):
        tag = path.stem[len(f"person_log_{date}"):].lstrip("_")
        try:
            with open(path, newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    photo = (row.get("snapshot") or "").strip()
                    rows.append({
                        "timestamp": row.get("timestamp", ""),
                        "camera": row.get("camera") or tag or "connect.py",
                        "name": (row.get("name") or UNKNOWN).strip() or UNKNOWN,
                        "surname": "",
                        "direction": (row.get("direction") or "").strip(),
                        "photo_url": f"{reverse('history_photo')}?f={quote(photo)}" if photo else "",
                    })
        except OSError:
            continue
    return rows


def _db_events(database, date: str) -> list[dict]:
    from person_detection.storage.database import log_rows
    rows, _, _ = log_rows(database, date, limit=None)
    for row in rows:
        row["photo_url"] = f"{reverse('history_photo')}?log={row['id']}" if row.get("snapshot_path") else ""
    return rows


def day_events(date: str) -> tuple[list[dict], str]:
    database = services.factory.database()
    if database is not None:
        return _db_events(database, date), "PostgreSQL"
    return _csv_events(date), "CSV"


def summarize(events: list[dict]) -> dict:
    events = sorted(events, key=lambda e: e.get("timestamp", ""))
    crossings = [e for e in events if e.get("direction") in ("in", "out")]
    ins = sum(1 for e in crossings if e["direction"] == "in")
    outs = len(crossings) - ins

    people: dict[str, dict] = {}
    for e in events:
        name = e.get("name") or UNKNOWN
        if name == UNKNOWN:
            continue
        p = people.setdefault(name, {"name": name, "surname": e.get("surname", ""), "first_seen": e["timestamp"],
                                     "ins": 0, "outs": 0, "state": "seen", "last_in": "", "last_out": "",
                                     "camera": "", "photo_url": "", "last_seen": ""})
        p["last_seen"] = e["timestamp"]
        p["camera"] = e.get("camera", "")
        if e.get("photo_url"):
            p["photo_url"] = e["photo_url"]
        if e.get("direction") == "in":
            p["ins"] += 1
            p["state"], p["last_in"] = "inside", e["timestamp"]
        elif e.get("direction") == "out":
            p["outs"] += 1
            p["state"], p["last_out"] = "left", e["timestamp"]

    unknown_in = sum(1 for e in crossings if e["direction"] == "in" and (e.get("name") or UNKNOWN) == UNKNOWN)
    unknown_out = sum(1 for e in crossings if e["direction"] == "out" and (e.get("name") or UNKNOWN) == UNKNOWN)
    named_inside = [p for p in people.values() if p["state"] == "inside"]
    named_inside.sort(key=lambda p: p["last_in"], reverse=True)
    left = sorted((p for p in people.values() if p["state"] == "left"), key=lambda p: p["last_out"], reverse=True)
    seen = sorted((p for p in people.values() if p["state"] == "seen"), key=lambda p: p["last_seen"], reverse=True)

    hourly = [{"hour": h, "in": 0, "out": 0} for h in range(24)]
    for e in crossings:
        try:
            hour = int(e["timestamp"][11:13])
        except (ValueError, IndexError):
            continue
        hourly[hour][e["direction"]] += 1

    inside_total = max(0, ins - outs, len(named_inside))
    return {
        "events": len(events), "crossings": len(crossings), "ins": ins, "outs": outs,
        "inside_total": inside_total, "inside_named": named_inside,
        "inside_unknown": max(0, inside_total - len(named_inside), unknown_in - unknown_out),
        "left": left, "seen": seen, "people_today": len(people),
        "recent": list(reversed(crossings))[:12], "hourly": hourly, "chart": _chart(hourly),
    }


def _chart(hourly: list[dict]) -> dict | None:
    """Bars per hour for an SVG of 640 x 200 (in = left bar, out = right bar)."""
    busy = [h["hour"] for h in hourly if h["in"] or h["out"]]
    if not busy:
        return None
    first, last = min(min(busy), 7), max(max(busy), 19)
    hours = hourly[first:last + 1]
    top = max(max(h["in"], h["out"]) for h in hours) or 1
    left, right, base, height = 34, 630, 172, 150
    step = (right - left) / len(hours)
    bar = max(3.0, min(14.0, step * 0.32))
    bars, labels = [], []
    for i, h in enumerate(hours):
        cx = left + step * (i + 0.5)
        for kind, offset in (("in", -bar - 1), ("out", 1)):
            value = h[kind]
            if value:
                bh = height * value / top
                bars.append({"x": round(cx + offset, 1), "y": round(base - bh, 1), "w": round(bar, 1),
                             "h": round(bh, 1), "kind": kind, "value": value, "hour": h["hour"]})
        if i % 2 == 0 or len(hours) <= 13:
            labels.append({"x": round(cx, 1), "text": f"{h['hour']:02d}"})
    return {"bars": bars, "labels": labels, "base": base, "left": left, "right": right,
            "top_label": top, "top_y": base - height}
