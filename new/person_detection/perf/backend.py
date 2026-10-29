"""Faster YOLO on computers without an NVIDIA GPU, and a tracker setup for fixed cameras.

OpenVINO (Intel's inference engine) runs the same YOLO11n-pose model about
25-40 % faster than PyTorch on a laptop CPU. The model is converted once
(a few seconds) into models/<name>_<size>_openvino_model/ and reused after that.
With an NVIDIA GPU, PyTorch + CUDA (fp16) stays the fastest and is kept.

The tracker file turns off BoT-SORT's camera-motion compensation (GMC): it
estimates how the *camera* moved by optical flow on every frame, which a
camera bolted to the wall never does. Lost tracks are also kept a little
longer, so a person hidden for a moment keeps their ID (fewer duplicate rows).
"""

from __future__ import annotations

import threading
from pathlib import Path

_export_lock = threading.Lock()


def openvino_available() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except Exception:
        return False


def resolve_model(model_path: str, backend: str, device: str, imgsz: int) -> tuple[str, str]:
    """(path to load, backend name actually used)."""
    backend = (backend or "auto").lower()
    use_openvino = backend == "openvino" or (backend == "auto" and device == "cpu")
    if not use_openvino or not openvino_available() or not str(model_path).endswith(".pt"):
        return model_path, "pytorch"
    source = Path(model_path)
    # Ultralytics recognises an OpenVINO model by the "_openvino_model" folder suffix
    target = source.with_name(f"{source.stem}_{imgsz}_openvino_model")
    with _export_lock:  # several cameras start at once: convert only one time
        if not any(target.glob("*.xml")):
            try:
                from ultralytics import YOLO
                exported = Path(YOLO(model_path).export(format="openvino", imgsz=imgsz, half=False, verbose=False))
                if exported.resolve() != target.resolve():
                    if target.exists():
                        import shutil
                        shutil.rmtree(target)
                    exported.rename(target)
            except Exception as exc:  # keep working with PyTorch
                print(f"OpenVINO export failed, using PyTorch: {str(exc).splitlines()[0] if str(exc) else exc!r}")
                return model_path, "pytorch"
    return str(target), "openvino"


def tracker_config(directory: Path, track_buffer: int = 60, static_camera: bool = True,
                   reid: bool = False, reid_model: str = "auto") -> str:
    """Write a BoT-SORT file for fixed cameras and return its path.

    reid: also compare what people look like (ReID) when matching tracks, so a
    person who was hidden behind someone keeps their ID more often.
    reid_model "auto" reuses features YOLO already computed (almost free);
    "yolo26n-reid.onnx" etc. is a separate appearance model run on every
    person box (better at telling similar people apart, costs more).
    """
    import ultralytics
    import yaml

    base = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
    settings = yaml.safe_load(base.read_text(encoding="utf-8"))
    settings["track_buffer"] = int(track_buffer)
    if static_camera:
        settings["gmc_method"] = "none"
    settings["with_reid"] = bool(reid)
    tag = ""
    if reid:
        settings["model"] = reid_model or "auto"
        tag = "_reid-" + Path(settings["model"]).stem
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"botsort_{'static' if static_camera else 'moving'}_{int(track_buffer)}{tag}.yaml"
    text = yaml.safe_dump(settings, sort_keys=False)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return str(path)
