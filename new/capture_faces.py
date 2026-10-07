"""Capture face photos from the live camera for enrollment.

Usage:  python capture_faces.py "Oung"
Press SPACE to save a photo when a face box is shown, 'q' to quit.
Saves into models/photos/<name>/
"""

import sys
import time
from pathlib import Path

import cv2

from person_detection import config
from person_detection.vision.face import FaceDetector
from person_detection.sources.stream import VideoStream
from person_detection.core.geometry import FramePreprocessor


def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: python capture_faces.py "Name"')
        sys.exit(1)

    name = sys.argv[1]
    out_dir = Path(config.MODELS_DIR) / "photos" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(config.FACE_MODEL_PATH, config.FACE_CONFIDENCE)
    preprocess = FramePreprocessor(
        config.ROTATE, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

    saved = 0
    with VideoStream(config.SOURCE, config.MAX_READ_FAILURES, config.RECONNECT_DELAY_SEC) as stream:
        print("Press SPACE to save a photo when a green face box is shown, 'q' to quit.")
        for frame in stream.frames():
            frame = preprocess(frame)
            faces = detector.detect(frame)
            display = frame.copy()
            for x1, y1, x2, y2 in faces:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Saved: {saved}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Capture Faces", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and faces:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = out_dir / f"{name}_{stamp}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                print(f"Saved {path}")

    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} photo(s) to {out_dir}")


if __name__ == "__main__":
    main()
