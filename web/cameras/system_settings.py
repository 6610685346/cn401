"""System settings that can be changed on the "ตั้งค่าระบบ" page.

Each setting points at a module-level value used by the running code
(person_detection.config, the multi-camera hub, or this web app). Changed
values are saved to new/settings.json, which config.py also reads at start-up,
so connect.py follows the same settings. Only values that differ from the
defaults are written, so later edits to config.py still apply to the rest.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from django import forms

from person_detection import config
from person_detection.multicam import hub as hub_module

from . import services


@dataclass(frozen=True)
class Setting:
    key: str  # name in settings.json
    target: object  # module/object holding the live value
    attr: str
    kind: type  # int, float, bool or str (str needs `choices`)
    group: str
    label: str
    help: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    restart: bool = True  # running cameras restart to pick it up
    cli: bool = False  # connect.py uses it too
    choices: tuple = ()  # ((value, label), ...) for str settings

    @property
    def is_bool(self) -> bool:
        return self.kind is bool

    @property
    def is_choice(self) -> bool:
        return self.kind is str

    def choice_label(self, value) -> str:
        return dict(self.choices).get(value, str(value))


def _face_model_choices():
    from person_detection.vision.face_models import CHOICES
    return tuple(CHOICES)


def _person_model_choices():
    from person_detection.vision.detector import PERSON_MODELS
    return tuple(PERSON_MODELS.items())


REID_CHOICES = (
    ("auto", "auto · ใช้ feature ของ YOLO เอง (แทบไม่เพิ่มเวลา)"),
    ("yolo26n-reid.onnx", "yolo26n-reid · โมเดล ReID แยก (แม่นกว่า ช้ากว่า)"),
    ("yolo26s-reid.onnx", "yolo26s-reid · โมเดล ReID แยก ใหญ่กว่า"),
)

SETTINGS = [
    # -- models (vision/face_models.py, vision/detector.py) -------------------
    Setting("PERSON_MODEL", config, "PERSON_MODEL", str, "โมเดล",
            "โมเดลตรวจจับคน (YOLO pose)",
            "รุ่นใหญ่ (s/m) เจอคนไกล/โดนบังได้ดีกว่าแต่ช้ากว่า · YOLO26 เป็นรุ่นใหม่ที่เร็วบน CPU "
            "· ดาวน์โหลดอัตโนมัติครั้งแรก · วัดผลเทียบกันได้ในหน้า \"ทดสอบด้วยวิดีโอ\"",
            cli=True, choices=_person_model_choices()),
    Setting("FACE_RECOGNIZER", config, "FACE_RECOGNIZER", str, "โมเดล",
            "โมเดลจดจำใบหน้า",
            "ArcFace แม่นกว่า SFace มาก (ดาวน์โหลด ~120-280 MB ครั้งแรก, ใช้เพื่อการศึกษาเท่านั้น) "
            "· เปลี่ยนแล้วระบบจะสร้างข้อมูลใบหน้าของทุกคนใหม่จากรูปที่ลงทะเบียนไว้ ชื่อ/รหัสคนไม่เปลี่ยน",
            cli=True, choices=_face_model_choices()),
    Setting("FACE_FULL_RES", config, "FACE_FULL_RES", bool, "โมเดล",
            "หาและจำหน้าจากภาพความละเอียดเต็มของกล้อง",
            "หน้าคนที่อยู่ไกลจะมีพิกเซลมากขึ้น 2-4 เท่า (กล้อง 1080p-4K) จำชื่อได้แม่นขึ้น ใช้เวลาเพิ่มเล็กน้อย", cli=True),
    Setting("TRACKER_REID", config, "TRACKER_REID", bool, "โมเดล",
            "ใช้ ReID ในตัวติดตามคน (BoT-SORT)",
            "เทียบลักษณะภายนอกด้วย ทำให้คนที่โดนบังแล้วโผล่มายังได้ ID เดิมบ่อยขึ้น (บันทึกซ้ำน้อยลง)", cli=True),
    Setting("REID_MODEL", config, "REID_MODEL", str, "โมเดล",
            "โมเดล ReID", "ใช้เมื่อเปิดข้อด้านบน", cli=True, choices=REID_CHOICES),
    # -- speed --------------------------------------------------------------
    Setting("FRAME_SKIP", config, "FRAME_SKIP", int, "ความเร็ว",
            "ประมวลผลทุก N เฟรม (frame skip)",
            "1 = ทุกเฟรมที่กล้องส่งมา · 2 = เว้นทีละเฟรม ... ค่ามากขึ้นใช้ CPU/GPU น้อยลง "
            "แต่การติดตามคนต่อเนื่องน้อยลง (คนเดินเร็วอาจได้ ID ใหม่)", 1, 30, 1, restart=False, cli=True),
    Setting("web.BROWSER_MAX_FPS", services, "BROWSER_MAX_FPS", int, "ความเร็ว",
            "กล้องจากเบราว์เซอร์: ส่งภาพสูงสุดกี่เฟรมต่อวินาที",
            "ลดลงถ้ามีหลายคนใช้หน้า \"กล้องของเครื่องนี้\" พร้อมกันแล้วเครื่องช้า", 1, 30, 1, restart=False),
    Setting("web.MAX_BROWSER_SESSIONS", services.sessions, "max_sessions", int, "ความเร็ว",
            "กล้องจากเบราว์เซอร์: จำนวนหน้าต่างพร้อมกันสูงสุด",
            "แต่ละหน้าต่างใช้โมเดลชุดของตัวเอง (หน่วยความจำ GPU)", 1, 30, 1, restart=False),
    # -- work-savers (PERFORMANCE.md) ---------------------------------------
    Setting("MOTION_GATE", config, "MOTION_GATE", bool, "ประสิทธิภาพ",
            "พัก YOLO เมื่อภาพนิ่งและไม่มีคน",
            "เช็กการเคลื่อนไหวจากภาพเล็ก (~1 ms) ถ้าไม่มีคนและไม่มีอะไรขยับ จะไม่รัน YOLO "
            "(ยังตรวจทุก ๆ ช่วงเวลาด้านล่าง เผื่อมีคนยืนนิ่ง)", cli=True),
    Setting("IDLE_CHECK_SEC", config, "IDLE_CHECK_SEC", float, "ประสิทธิภาพ",
            "ตอนพัก ยังตรวจจับทุก (วินาที)", "น้อย = เจอคนที่ยืนนิ่งเร็วขึ้น · มาก = ประหยัดเครื่องกว่า",
            0.5, 10.0, 0.5, cli=True),
    Setting("SMART_FACES", config, "SMART_FACES", bool, "ประสิทธิภาพ",
            "หาหน้าเฉพาะรอบศีรษะคนที่ยังไม่รู้ชื่อ",
            "ไม่ต้องรัน YuNet ทั้งภาพทุกเฟรม · คนที่รู้ชื่อแล้วจะไม่ถูกหาหน้าซ้ำ", cli=True),
    Setting("FACE_EVERY", config, "FACE_EVERY", int, "ประสิทธิภาพ",
            "หาหน้าทุก N เฟรม", "ใช้เมื่อเปิดข้อด้านบน · 1 = ทุกเฟรม", 1, 10, 1, cli=True),
    Setting("FACE_VOTES", config, "FACE_VOTES", int, "ประสิทธิภาพ",
            "ใช้กี่ใบหน้าก่อนตัดสินชื่อ",
            "เฉลี่ยหลายใบหน้าจากหลายเฟรม ลดการทายชื่อผิดจากหน้าเบลอ/หันข้าง · ถ้าเหมือนมากจะตัดสินทันที · 1 = แบบเดิม",
            1, 7, 1, cli=True),
    Setting("FACE_MIN_SIZE", config, "FACE_MIN_SIZE", int, "ประสิทธิภาพ",
            "ขนาดหน้าเล็กสุดที่ใช้จำ (px)", "หน้าที่เล็กกว่านี้ SFace มักผิด จึงรอให้คนเดินเข้ามาใกล้ก่อน · 0 = ใช้ทุกขนาด",
            0, 120, 2, cli=True),
    Setting("LOG_COOLDOWN_SEC", config, "LOG_COOLDOWN_SEC", int, "ประสิทธิภาพ",
            "ไม่บันทึกคนเดิมซ้ำภายใน (วินาที)",
            "คนที่รู้ชื่อ กล้องเดียวกัน ถ้าถูกบันทึกไปแล้วไม่นาน (เช่น โดนบังแล้วได้ ID ใหม่) จะไม่บันทึกแถวพบคนซ้ำ "
            "· แถวเข้า/ออกบันทึกเสมอ · 0 = ปิด", 0, 600, 5, cli=True),
    Setting("TRACKER_STATIC", config, "TRACKER_STATIC", bool, "ประสิทธิภาพ",
            "กล้องติดตั้งอยู่กับที่", "ปิดการชดเชยการขยับของกล้องใน BoT-SORT (ไม่ต้องคำนวณ optical flow ทุกเฟรม)", cli=True),
    Setting("TRACK_BUFFER", config, "TRACK_BUFFER", int, "ประสิทธิภาพ",
            "จำคนที่หายจากภาพได้นาน (เฟรม)", "คนที่โดนบังชั่วครู่จะได้ ID เดิม ลดการบันทึกซ้ำ · ค่าเดิมของ Ultralytics = 30",
            10, 300, 10, cli=True),
    # -- detection ------------------------------------------------------------
    Setting("CONFIDENCE", config, "CONFIDENCE", float, "การตรวจจับคน",
            "ความมั่นใจขั้นต่ำในการตรวจจับคน",
            "ต่ำ = เจอคนมากขึ้นแต่มีกรอบผิดมากขึ้น · สูง = แม่นขึ้นแต่อาจพลาดคนที่อยู่ไกล/โดนบัง",
            0.1, 0.95, 0.05, cli=True),
    Setting("YOLO_IMGSZ", config, "YOLO_IMGSZ", int, "การตรวจจับคน",
            "ขนาดภาพที่ YOLO ใช้ตรวจจับคน (px)",
            "เล็ก = เร็วขึ้น (เช่น 480 ตอนเปิดหลายกล้อง) · ใหญ่ = เจอคนที่อยู่ไกล/ตัวเล็กได้ดีขึ้น "
            "· ค่ามาตรฐานของ YOLO คือ 640 (ใช้ทวีคูณของ 32)", 320, 1280, 32, cli=True),
    Setting("MIN_FRAMES_BEFORE_LOG", config, "MIN_FRAMES_BEFORE_LOG", int, "การตรวจจับคน",
            "ต้องเห็นคนกี่เฟรมก่อนบันทึกลง CSV",
            "กันกรอบที่โผล่แวบเดียวแล้วหาย (ตรวจผิด) ไม่ให้ถูกบันทึก", 1, 100, 1, cli=True),
    Setting("VLM_PAR_MIN_FRAMES", config, "VLM_PAR_MIN_FRAMES", int, "การตรวจจับคน",
            "VLM-PAR: รอกี่เฟรมก่อนอ่านเสื้อผ้า",
            "รอให้ติดตามคนได้นิ่งก่อนค่อยใช้โมเดลใหญ่ (อ่านครั้งเดียวต่อคน)", 1, 60, 1, cli=True),
    # -- faces ------------------------------------------------------------------
    Setting("FACE_CONFIDENCE", config, "FACE_CONFIDENCE", float, "ใบหน้า",
            "ความมั่นใจขั้นต่ำในการเจอใบหน้า",
            "ต่ำ = เจอหน้าที่เล็ก/เอียงได้มากขึ้น แต่อาจไปจับสิ่งที่ไม่ใช่หน้า", 0.3, 0.99, 0.05, cli=True),
    Setting("FACE_MATCH_THRESHOLD", config, "FACE_MATCH_THRESHOLD", float, "ใบหน้า",
            "เกณฑ์ความเหมือนของใบหน้าที่จะนับว่าเป็นคนเดียวกัน",
            "ใช้กับ SFace · ค่ามาตรฐานของ SFace คือ 0.363 · สูงขึ้น = ทายชื่อผิดน้อยลง แต่ขึ้น unknown บ่อยขึ้น",
            0.2, 0.9, 0.01, cli=True),
    Setting("FACE_MATCH_THRESHOLD_ARCFACE", config, "FACE_MATCH_THRESHOLD_ARCFACE", float, "ใบหน้า",
            "เกณฑ์ความเหมือนของใบหน้า (ArcFace)",
            "ใช้กับ ArcFace ทั้งสองรุ่น · คะแนนของ ArcFace ต่ำกว่า SFace จึงใช้ค่าต่ำกว่า "
            "· ควรจูนจากวิดีโอชุด train ในหน้า \"ทดสอบด้วยวิดีโอ\"", 0.1, 0.9, 0.01, cli=True),
    # -- saving ---------------------------------------------------------------
    Setting("SAVE_CSV", config, "SAVE_CSV", bool, "การบันทึก", "บันทึกประวัติลง CSV",
            "ปิด = ไม่บันทึกอะไรเลย (ยังดูภาพสดได้)", cli=True),
    Setting("SAVE_CROPS", config, "SAVE_CROPS", bool, "การบันทึก", "บันทึกรูปของแต่ละแถว",
            "รูปเล็ก 1 รูปต่อแถวในประวัติ (ดูได้ในหน้าประวัติและหน้า \"ใครอยู่ตอนนี้\") เก็บใน new/output/cropped_persons/<วันที่> "
            "· เป็นข้อมูลส่วนบุคคล ควรลบของเก่าเป็นระยะ", cli=True),
    # -- menu ------------------------------------------------------------------
    Setting("web.SHOW_EVAL_MENU", services, "SHOW_EVAL_MENU", bool, "เมนู",
            "แสดงเมนู \"ทดสอบด้วยวิดีโอ\"",
            "ซ่อนไว้ก่อนได้ ผลทดสอบและวิดีโอเดิมยังอยู่ครบ เปิดเมื่อจะวัดผล/เทียบโมเดล", restart=False),
    # -- multi-camera -------------------------------------------------------------
    Setting("hub.MATCH_RADIUS_M", hub_module, "MATCH_RADIUS_M", float, "รวมคนข้ามกล้อง",
            "ระยะห่างสูงสุดบนพื้นที่ยังนับว่าเป็นคนเดียวกัน (เมตร)",
            "มากขึ้น = รวมได้ง่ายขึ้นแม้ปรับเทียบไม่เป๊ะ แต่คนที่ยืนชิดกันอาจถูกรวมผิด",
            0.2, 3.0, 0.1, restart=False),
    Setting("hub.MAX_TIME_GAP_S", hub_module, "MAX_TIME_GAP_S", float, "รวมคนข้ามกล้อง",
            "ภาพจากสองกล้องห่างกันได้ไม่เกิน (วินาที)",
            "เพิ่มถ้ากล้องตัวหนึ่งดีเลย์มาก", 0.2, 5.0, 0.1, restart=False),
    Setting("hub.MERGE_AFTER", hub_module, "MERGE_AFTER", int, "รวมคนข้ามกล้อง",
            "ต้องยืนตรงกันต่อเนื่องกี่ครั้งก่อนรวม",
            "มากขึ้น = รวมช้าลงแต่ผิดน้อยลง", 2, 50, 1, restart=False),
    Setting("hub.FORGET_AFTER_S", hub_module, "FORGET_AFTER_S", float, "รวมคนข้ามกล้อง",
            "ลืมคนที่หายไปจากทุกกล้องเกิน (วินาที)",
            "หลังจากนี้ถ้ากลับมาจะได้ G-เลขใหม่", 5, 600, 5, restart=False),
]
BY_KEY = {s.key: s for s in SETTINGS}


def _file_default(setting: Setting):
    if setting.target is config:
        return config.FILE_DEFAULTS.get(setting.attr, getattr(config, setting.attr))
    return getattr(setting.target, setting.attr)


# Defaults = values in the code before any saved settings were applied. config.py
# applied settings.json at import but remembered the originals in FILE_DEFAULTS;
# the other targets are not changed until apply() runs.
DEFAULTS = {s.key: _file_default(s) for s in SETTINGS}

_lock = threading.Lock()


def path() -> Path:
    return Path(config.SETTINGS_PATH)


def current() -> dict:
    return {s.key: getattr(s.target, s.attr) for s in SETTINGS}


def read_saved() -> dict:
    try:
        data = json.loads(path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _coerce(setting: Setting, value):
    if setting.kind is str:
        if value not in dict(setting.choices):
            raise ValueError
        return value
    if setting.kind is bool:
        if not isinstance(value, bool):
            raise ValueError
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    value = setting.kind(value)
    if setting.min is not None and value < setting.min or setting.max is not None and value > setting.max:
        raise ValueError
    return value


def apply(values: dict) -> list[str]:
    """Set live values (defaults for missing keys); returns keys that changed."""
    changed = []
    for setting in SETTINGS:
        try:
            value = _coerce(setting, values[setting.key]) if setting.key in values else DEFAULTS[setting.key]
        except ValueError:
            value = DEFAULTS[setting.key]  # a hand-edited file with a bad value
        if getattr(setting.target, setting.attr) != value:
            setattr(setting.target, setting.attr, value)
            changed.append(setting.key)
    return changed


def load_and_apply() -> None:
    """At server start: apply settings.json to every target."""
    apply(read_saved())


def save(values: dict, user=None) -> list[str]:
    """Store values that differ from the defaults, apply them, return changed keys."""
    with _lock:
        overrides = {k: v for k, v in values.items() if k in BY_KEY and v != DEFAULTS[k]}
        data = dict(overrides)
        data["_meta"] = {"updated_by": getattr(user, "username", None),
                         "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        target = path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        return apply(overrides)


def needs_restart(keys) -> bool:
    return any(BY_KEY[k].restart for k in keys)


# ------------------------------------------------------------------ the form
def field_name(key: str) -> str:
    return key.replace(".", "__")


class SettingsForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        values = current()
        for s in SETTINGS:
            name = field_name(s.key)
            if s.kind is bool:
                field = forms.BooleanField(required=False)
            elif s.kind is str:
                field = forms.ChoiceField(choices=s.choices)
            elif s.kind is int:
                field = forms.IntegerField(min_value=s.min, max_value=s.max)
                field.widget.attrs["step"] = s.step or 1
            else:
                field = forms.FloatField(min_value=s.min, max_value=s.max)
                field.widget.attrs["step"] = s.step or "any"
            field.label = s.label
            field.help_text = s.help
            field.initial = values[s.key]
            field.setting = s
            field.default = DEFAULTS[s.key]
            self.fields[name] = field

    def values(self) -> dict:
        return {s.key: self.cleaned_data[field_name(s.key)] for s in SETTINGS}

    def groups(self):
        """[(group name, [bound fields])] in display order."""
        out = {}
        for s in SETTINGS:
            out.setdefault(s.group, []).append(self[field_name(s.key)])
        return list(out.items())
