"""Enroll a person's face. Usage: python enroll.py "Alice" photos/alice/"""

import sys
from pathlib import Path

import cv2

from person_detection import config
from person_detection.vision.face import FaceDetector, FaceRecognizer
from person_detection.storage.people_db import PeopleDB


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python enroll.py <name> <photos_dir>")
        sys.exit(1)

    name, photos_dir = sys.argv[1], Path(sys.argv[2])
    detector = FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE)
    recognizer = FaceRecognizer(
        config.FACE_RECOGNITION_MODEL_PATH, config.FACE_MATCH_THRESHOLD)
    db = PeopleDB(config.PEOPLE_DB_PATH)

    embeddings = []
    for path in sorted(photos_dir.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable file: {path}")
            continue
        faces = detector.detect_raw(image)
        if not faces:
            print(f"No face found in {path}, skipping")
            continue
        box, raw_row = max(faces, key=lambda bf: (
            bf[0][2] - bf[0][0]) * (bf[0][3] - bf[0][1]))
        embeddings.append(recognizer.embed(image, raw_row))
        print(f"Used {path.name}")

    if not embeddings:
        print("No usable photos found; nothing enrolled.")
        sys.exit(1)

    db.enroll(name, embeddings)
    print(
        f"Enrolled '{name}' from {len(embeddings)} photo(s) -> {config.PEOPLE_DB_PATH}")


if __name__ == "__main__":
    main()
