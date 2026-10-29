"""Evaluation metrics, as defined in the report (หัวข้อตัวชี้วัด).

Unit: one *event* = one person walking through the picture once = one log row.

Reviewed events carry the ground truth a person typed in on the web page:

    sighting  gt_status    "ok" (a real person, first row for them)
                           "false" (not a person) / "duplicate" (same visit logged again)
              gt_identity  a registered name / NONE (person not registered) / SKIP (cannot tell)
    in / out  gt_direction "ok" / "wrong" (they went the other way) / "false" (no crossing)

plus counts per video of what the system missed: people, crossings in, crossings out.

Face results are recomputed for any threshold τ from the stored best candidate
and score, so one run gives the whole FAR / FRR curve.
"""

from __future__ import annotations

from collections.abc import Iterable

NONE = "__none__"  # a real person who is not registered (impostor)
SKIP = "__skip__"  # identity cannot be told from the video
UNKNOWN = "unknown"


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def predicted_name(event: dict, threshold: float) -> str | None:
    """Name the system gives this event at threshold τ; None = no face was seen."""
    if not event.get("face_seen"):
        return None
    candidate, score = event.get("candidate"), event.get("score")
    if candidate and score is not None and score >= threshold:
        return candidate
    return UNKNOWN


def face_counts(events: Iterable[dict], threshold: float) -> dict:
    """Genuine / impostor outcomes over reviewed real people whose face was seen."""
    c = dict(genuine=0, impostor=0, correct=0, false_reject=0, misidentified=0, false_accept=0, correct_reject=0)
    for e in events:
        if e.get("kind") != "sighting" or e.get("gt_status") != "ok":
            continue
        truth = e.get("gt_identity") or ""
        if not truth or truth == SKIP:
            continue
        predicted = predicted_name(e, threshold)
        if predicted is None:
            continue  # no face: counted in face coverage, not FAR/FRR
        if truth == NONE:
            c["impostor"] += 1
            c["correct_reject" if predicted == UNKNOWN else "false_accept"] += 1
        else:
            c["genuine"] += 1
            if predicted == truth:
                c["correct"] += 1
            elif predicted == UNKNOWN:
                c["false_reject"] += 1
            else:
                c["misidentified"] += 1
    return c


def face_rates(c: dict) -> dict:
    return {
        "far": _ratio(c["false_accept"], c["impostor"]),
        "frr": _ratio(c["false_reject"], c["genuine"]),
        "accuracy": _ratio(c["correct"], c["genuine"]),
        "misid": _ratio(c["misidentified"], c["genuine"]),
    }


def compute(videos: Iterable[tuple[list[dict], dict]], threshold: float) -> dict:
    """Metrics over one or more videos: [(events, missed)], counts summed (micro average).

    missed = {"people": FN, "in": missed crossings in, "out": missed crossings out}
    """
    videos = list(videos)
    events = [e for evs, _ in videos for e in evs]
    missed = {k: sum(int((m or {}).get(k) or 0) for _, m in videos) for k in ("people", "in", "out")}

    sightings = [e for e in events if e.get("kind") == "sighting"]
    reviewed = [e for e in sightings if e.get("gt_status")]
    tp = sum(1 for e in reviewed if e["gt_status"] == "ok")
    fp = sum(1 for e in reviewed if e["gt_status"] in ("false", "duplicate"))
    fn = missed["people"]
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None

    real = [e for e in reviewed if e["gt_status"] == "ok"]
    with_face = sum(1 for e in real if e.get("face_seen"))
    counts = face_counts(events, threshold)

    directions = {}
    crossings = [e for e in events if e.get("kind") in ("in", "out")]
    for d, other in (("in", "out"), ("out", "in")):
        mine = [e for e in crossings if e["kind"] == d and e.get("gt_direction")]
        tp_d = sum(1 for e in mine if e["gt_direction"] == "ok")
        fp_d = sum(1 for e in mine if e["gt_direction"] in ("wrong", "false"))
        # a crossing logged as the other direction was really this one, and was missed as such
        fn_d = missed[d] + sum(1 for e in crossings if e["kind"] == other and e.get("gt_direction") == "wrong")
        directions[d] = {"tp": tp_d, "fp": fp_d, "fn": fn_d, "precision": _ratio(tp_d, tp_d + fp_d),
                         "recall": _ratio(tp_d, tp_d + fn_d), "logged": sum(1 for e in crossings if e["kind"] == d)}
    predicted_net = directions["in"]["logged"] - directions["out"]["logged"]
    true_in = directions["in"]["tp"] + directions["in"]["fn"]
    true_out = directions["out"]["tp"] + directions["out"]["fn"]
    reviewed_crossings = sum(1 for e in crossings if e.get("gt_direction"))

    latencies = [e["latency_s"] for e in real if e.get("latency_s") is not None]
    return {
        "threshold": threshold,
        "events": len(sightings),
        "reviewed": len(reviewed),
        "detection": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        "face": {**counts, **face_rates(counts), "coverage": _ratio(with_face, tp), "with_face": with_face},
        "direction": {**directions, "crossings": len(crossings), "reviewed": reviewed_crossings,
                      "occupancy_error": abs(predicted_net - (true_in - true_out)) if crossings or true_in or true_out else None},
        "latency_s": sum(latencies) / len(latencies) if latencies else None,
    }


DEFAULT_THRESHOLDS = [round(0.30 + 0.02 * i, 2) for i in range(11)]  # 0.30 ... 0.50


def sweep(events: list[dict], thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> dict:
    """FAR / FRR for each threshold, and the one with the fewest errors.

    The suggestion minimises FAR + FRR (both weighted equally); on a tie the
    higher threshold wins, because a wrong name is worse than "unknown".
    """
    rows = []
    for t in thresholds:
        c = face_counts(events, t)
        rows.append({"threshold": t, **c, **face_rates(c)})
    usable = [r for r in rows if r["far"] is not None or r["frr"] is not None]
    best = None
    if usable:
        best = min(usable, key=lambda r: ((r["far"] or 0) + (r["frr"] or 0), -r["threshold"]))
    return {"rows": rows, "best": best,
            "genuine": rows[0]["genuine"] if rows else 0, "impostor": rows[0]["impostor"] if rows else 0}
