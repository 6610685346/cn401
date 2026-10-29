"""Manage the PostgreSQL database (see DATABASE.md).

Set DATABASE_URL first (Windows cmd):
    set DATABASE_URL=postgresql+psycopg://postgres:PASSWORD@localhost:5432/person_detection

    python register.py init                 create the tables (safe to run again)
    python register.py status               connection, row counts, newest log row
    python register.py list                 registered people
    python register.py delete NAME          soft delete: hide from matching, delete face vectors
    python register.py import-people        copy output/people.json into the database
    python register.py import-csv           copy output/person_log_*.csv into detection_logs
    python register.py purge --days 90      delete log rows (and snapshots) older than 90 days

To add a person use enroll.py (or the web page): it writes to the database
when DATABASE_URL is set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

from person_detection import config


def _database():
    if not config.DATABASE_URL:
        sys.exit("DATABASE_URL is not set. Example (Windows cmd):\n"
                 "  set DATABASE_URL=postgresql+psycopg://postgres:PASSWORD@localhost:5432/person_detection")
    from person_detection.storage.database import Database
    return Database(config.DATABASE_URL, config.DB_TIMEZONE)


def cmd_init(args) -> None:
    db = _database()
    db.create_tables()
    try:
        db.set_database_timezone()
        print(f"Database time zone set to {db.timezone}")
    except Exception as exc:  # not the owner: the program still sets it per connection
        print(f"(could not set the database time zone: {str(exc).splitlines()[0]})")
    print("Tables ready: persons, face_embeddings, cameras, detection_logs")


def cmd_status(args) -> None:
    from sqlalchemy import func, select

    from person_detection.storage.database import Camera, DetectionLog, FaceEmbedding, Person
    db = _database()
    if not db.ping():
        sys.exit("Cannot connect to the database. Is PostgreSQL (Docker) running?")
    with db.Session() as s:
        print("persons (active):", s.scalar(select(func.count()).where(Person.deleted_at.is_(None))))
        print("face_embeddings: ", s.scalar(select(func.count()).select_from(FaceEmbedding)))
        print("cameras:         ", ", ".join(s.scalars(select(Camera.camera_name)).all()) or "-")
        print("detection_logs:  ", s.scalar(select(func.count()).select_from(DetectionLog)))
        print("newest log row:  ", s.scalar(select(func.max(DetectionLog.detected_at))) or "-")


def cmd_list(args) -> None:
    from person_detection.storage.database import SqlPeopleDB
    people = SqlPeopleDB(_database()).details()
    if not people:
        print("Nobody registered yet. Use enroll.py or the web page.")
    for p in people:
        full = f"{p['name']} {p['surname']}".strip()
        print(f"{p['id']:>4}  {full:<30} {p['id_number'] or '-':<14} {p['photos']} photo(s)")


def cmd_delete(args) -> None:
    from person_detection.storage.database import SqlPeopleDB
    people = SqlPeopleDB(_database())
    if args.name not in people.names():
        sys.exit(f"'{args.name}' is not registered")
    people.remove(args.name)
    print(f"Removed '{args.name}' from matching and deleted their face vectors. "
          f"Their photos are still in {config.PHOTOS_DIR / args.name} (delete them from the web page "
          f"or by hand if they should go too).")


def cmd_import_people(args) -> None:
    """people.json holds one averaged vector per name; it becomes one row.
    For one row per photo, run enroll.py again on models/photos/<name>/."""
    import numpy as np

    from person_detection.storage.database import SqlPeopleDB
    path = Path(args.file or config.PEOPLE_DB_PATH)
    if not path.is_file():
        sys.exit(f"{path} not found")
    raw = json.loads(path.read_text(encoding="utf-8"))
    people = SqlPeopleDB(_database())
    existing = set(people.names())
    added = 0
    for name, vector in raw.items():
        if name in existing and not args.replace:
            print(f"  skip {name} (already in the database; --replace to overwrite)")
            continue
        people.enroll(name, [np.asarray(vector, dtype=np.float32)])
        added += 1
        print(f"  added {name}")
    print(f"{added} person(s) imported from {path}")


def _parse_time(text: str, zone) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone)
    except ValueError:
        return None


def cmd_import_csv(args) -> None:
    """Old CSV rows -> detection_logs. Rows already imported (same camera and
    second) are skipped, so running it twice does not double the history."""
    from sqlalchemy import insert, select

    from person_detection.storage.database import DetectionLog, SqlPeopleDB, _zone
    db = _database()
    people = SqlPeopleDB(db)
    zone = _zone(db.timezone)
    files = sorted(Path(args.dir or config.OUTPUT_DIR).glob("person_log_*.csv"))
    if not files:
        sys.exit("No person_log_*.csv files found")
    total = skipped = 0
    for path in files:
        tag = path.stem[len("person_log_YYYY-MM-DD"):].lstrip("_")
        rows = []
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                when = _parse_time(row.get("timestamp", ""), zone)
                if when is None:
                    continue
                camera = (row.get("camera") or tag or "connect.py").strip()
                name = (row.get("name") or "").strip()
                attributes = {k: row[k] for k in ("track_id", "shirt_color", "pants_color",
                                                  "confidence", "global_id") if row.get(k)}
                attributes["name_source"] = "csv_import"
                if name and name != "unknown" and people.person_id(name) is None:
                    attributes["csv_name"] = name  # person not registered in the database
                similarity = row.get("similarity") or None
                direction = (row.get("direction") or "").strip() or None
                rows.append({
                    "camera_id": db.camera_id(camera),
                    "person_id": people.person_id(name),
                    "similarity": float(similarity) if similarity else None,
                    "direction": direction if direction in ("in", "out") else None,
                    "attributes": attributes,
                    "snapshot_path": None,
                    "detected_at": when,
                })
        if not rows:
            continue
        with db.engine.begin() as conn:
            first = min(r["detected_at"] for r in rows)
            last = max(r["detected_at"] for r in rows)
            seen = set(conn.execute(select(DetectionLog.camera_id, DetectionLog.detected_at).where(
                DetectionLog.detected_at >= first, DetectionLog.detected_at <= last)).all())
            new = [r for r in rows if (r["camera_id"], r["detected_at"]) not in seen]
            if new:
                conn.execute(insert(DetectionLog), new)
        total += len(new)
        skipped += len(rows) - len(new)
        print(f"  {path.name}: {len(new)} new, {len(rows) - len(new)} already there")
    print(f"Imported {total} row(s), skipped {skipped}")


def cmd_purge(args) -> None:
    from person_detection.storage.database import purge_older_than
    rows, files = purge_older_than(_database(), args.days, delete_snapshots=not args.keep_snapshots)
    print(f"Deleted {rows} log row(s) older than {args.days} days and {files} snapshot file(s)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Manage the PostgreSQL database.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the tables").set_defaults(func=cmd_init)
    sub.add_parser("status", help="connection and row counts").set_defaults(func=cmd_status)
    sub.add_parser("list", help="registered people").set_defaults(func=cmd_list)
    p = sub.add_parser("delete", help="soft-delete a person")
    p.add_argument("name")
    p.set_defaults(func=cmd_delete)
    p = sub.add_parser("import-people", help="people.json -> persons + face_embeddings")
    p.add_argument("--file")
    p.add_argument("--replace", action="store_true")
    p.set_defaults(func=cmd_import_people)
    p = sub.add_parser("import-csv", help="person_log_*.csv -> detection_logs")
    p.add_argument("--dir")
    p.set_defaults(func=cmd_import_csv)
    p = sub.add_parser("purge", help="delete old log rows (PDPA retention)")
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--keep-snapshots", action="store_true")
    p.set_defaults(func=cmd_purge)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
