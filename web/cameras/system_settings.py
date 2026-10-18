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
    kind: type  # int, float or bool
    group: str
    label: str
    help: str
    min: float | None = None
    max: float | None = None
    step: float | None = None
    restart: bool = True  # running cameras restart to pick it up
    cli: bool = False  # connect.py uses it too

    @property
    def is_bool(self) -> bool:
        return self.kind is bool


SETTINGS = [
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
    # -- detection ------------------------------------------------------------
    Setting("CONFIDENCE", config, "CONFIDENCE", float, "การตรวจจับคน",
            "ความมั่นใจขั้นต่ำในการตรวจจับคน",
            "ต่ำ = เจอคนมากขึ้นแต่มีกรอบผิดมากขึ้น · สูง = แม่นขึ้นแต่อาจพลาดคนที่อยู่ไกล/โดนบัง",
            0.1, 0.95, 0.05, cli=True),
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
            "ค่ามาตรฐานของ SFace คือ 0.363 · สูงขึ้น = ทายชื่อผิดน้อยลง แต่ขึ้น unknown บ่อยขึ้น",
            0.2, 0.9, 0.01, cli=True),
    # -- saving ---------------------------------------------------------------
    Setting("SAVE_CSV", config, "SAVE_CSV", bool, "การบันทึก", "บันทึกประวัติลง CSV",
            "ปิด = ไม่บันทึกอะไรเลย (ยังดูภาพสดได้)", cli=True),
    Setting("SAVE_CROPS", config, "SAVE_CROPS", bool, "การบันทึก", "บันทึกรูปของแต่ละคน",
            "เก็บรูปตัดเฉพาะคนไว้ใน new/output/cropped_persons (ใช้พื้นที่ดิสก์ และเป็นข้อมูลส่วนบุคคล)", cli=True),
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
