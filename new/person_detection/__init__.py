"""Person detection from a phone video stream, webcam, or video file.

Layout, one job per folder:
    config.py    settings you edit (camera source, thresholds, output paths)
    app.py       wires the pieces together
    core/        geometry: Box/Frame types, box helpers, FramePreprocessor
    sources/     stream: video input with reconnect
    vision/      detector (YOLO people), face (YuNet), colors (clothing)
    counting/    direction: zone-crossing Up/Down (off by default)
    storage/     recorder: CSV log + person crops
    ui/          drawing (overlay), display (OpenCV window)
"""

import os

# FFmpeg options must be set before any module imports cv2, and every
# submodule is imported through this package, so this is the place.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # mute MJPEG "overread" spam
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
