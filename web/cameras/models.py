import re

from django.core.exceptions import ValidationError
from django.db import models

ROTATE_CHOICES = [(0, "0°"), (90, "90° ตามเข็ม"), (180, "180°"), (270, "270° ตามเข็ม")]
LONG_SIDE_CHOICES = [(640, "640 (เร็วสุด)"), (960, "960 (แนะนำ)"), (1280, "1280 (เห็นหน้าไกลขึ้น)"),
                     (1920, "1920 (ละเอียดสุด ใช้ GPU มาก)")]


def validate_source(value: str) -> None:
    value = value.strip()
    if value.isdigit():
        return  # webcam index on the server computer
    if not re.match(r"^(rtsp|rtsps|http|https)://", value, re.IGNORECASE):
        raise ValidationError(
            "ใส่ URL ที่ขึ้นต้นด้วย rtsp:// หรือ http(s):// "
            "หรือเลข 0, 1 สำหรับกล้อง USB ที่เสียบกับเครื่องเซิร์ฟเวอร์")


class Camera(models.Model):
    """A network (IP) camera, or a USB camera plugged into the server itself.

    Cameras plugged into a *user's* computer are not stored here: the browser
    opens them on the "กล้องของเครื่องนี้" page.
    """

    name = models.CharField("ชื่อกล้อง", max_length=60, unique=True)
    source = models.CharField(
        "URL กล้อง", max_length=500, validators=[validate_source],
        help_text="เช่น rtsp://user:pass@192.168.1.13:554/media/video2 หรือ "
                  "http://192.168.1.20:8080/video  (อักขระพิเศษในรหัสผ่าน เช่น # ให้เขียนเป็น %23)")
    rotate = models.IntegerField("หมุนภาพ", choices=ROTATE_CHOICES, default=0)
    long_side = models.IntegerField(
        "ขนาดภาพที่ใช้ประมวลผล (ด้านยาว, px)", choices=LONG_SIDE_CHOICES, default=960,
        help_text="กล้องความละเอียดเท่าไรก็ได้ ภาพจะถูกย่อ/ขยายเป็นขนาดนี้ก่อนตรวจจับ")
    enable_face_recognition = models.BooleanField("จดจำใบหน้า (ชื่อคน)", default=True)
    enable_vlm_par = models.BooleanField("VLM-PAR (สีเสื้อผ้า/กระเป๋า/หมวก)", default=True)
    enable_direction = models.BooleanField(
        "นับคนเดินเข้า/ออก", default=False,
        help_text="ใช้เส้นประตูที่วาดในหน้า \"เส้นประตู\" ของกล้องนี้ (ถ้ายังไม่วาด จะใช้โซนเดิมในโค้ด)")
    door_line = models.JSONField(
        "เส้นประตู", null=True, blank=True, editable=False,
        help_text="จุด a, b (0..1 ของภาพ) และด้านที่เป็นข้างใน สำหรับนับเข้า/ออก")
    floor_calibration = models.JSONField(
        "การปรับเทียบพื้น", null=True, blank=True, editable=False,
        help_text="จุดบนพื้น 4+ จุด สำหรับรวมคนข้ามกล้อง (ตั้งในหน้า ปรับเทียบพื้น)")
    should_run = models.BooleanField(
        "รันอยู่", default=False,
        help_text="เปิดค้างไว้ = เริ่มทำงานเองทุกครั้งที่เปิดเซิร์ฟเวอร์")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "กล้อง"
        verbose_name_plural = "กล้อง"

    def __str__(self) -> str:
        return self.name

    @property
    def masked_source(self) -> str:
        """URL with the password hidden, for showing on pages."""
        return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1****@", self.source)

    @property
    def has_door(self) -> bool:
        return bool(self.door_line)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.floor_calibration)

    @property
    def csv_tag(self) -> str:
        return f"cam-{self.name}"


# ------------------------------------------------------------- video tests
SPLIT_CHOICES = [("test", "ชุดทดสอบ (test)"), ("train", "ชุดจูน (train)")]
LIGHTING_CHOICES = [("", "-"), ("day", "กลางวัน"), ("night", "กลางคืน"), ("indoor", "ไฟในห้อง")]
DISTANCE_CHOICES = [("", "-"), ("0-3", "0–3 ม."), ("3-6", "3–6 ม."), ("6-10", "6–10 ม."), ("mixed", "หลายระยะ")]


class EvalVideo(models.Model):
    """A recorded video played through the pipeline to measure the system."""

    STATUS = [("queued", "รอคิว"), ("running", "กำลังประมวลผล"), ("done", "เสร็จ"),
              ("error", "ผิดพลาด"), ("cancelled", "ยกเลิก")]

    name = models.CharField("ชื่อ", max_length=120)
    file = models.CharField(max_length=500, editable=False)  # path of the uploaded video
    split = models.CharField("ชุดข้อมูล", max_length=10, choices=SPLIT_CHOICES, default="test")
    lighting = models.CharField("สภาพแสง", max_length=10, choices=LIGHTING_CHOICES, blank=True, default="")
    distance = models.CharField("ระยะจากกล้อง", max_length=10, choices=DISTANCE_CHOICES, blank=True, default="")
    notes = models.TextField("หมายเหตุ", blank=True, default="")
    options = models.JSONField(default=dict, blank=True)  # rotate, long_side, realtime, direction, vlm
    status = models.CharField(max_length=12, choices=STATUS, default="queued")
    progress = models.FloatField(default=0)  # 0..1
    message = models.CharField(max_length=300, blank=True, default="")
    stats = models.JSONField(default=dict, blank=True)  # EvalStats.to_dict() (includes the models used)
    # Summaries of earlier runs of this video (other models / settings), newest last
    runs = models.JSONField(default=list, blank=True)
    # what the system missed, typed in while reviewing (false negatives)
    missed_people = models.PositiveIntegerField("คนที่ระบบไม่ได้บันทึก", default=0)
    missed_in = models.PositiveIntegerField("การเดินเข้าที่ระบบพลาด", default=0)
    missed_out = models.PositiveIntegerField("การเดินออกที่ระบบพลาด", default=0)
    created_by = models.CharField(max_length=150, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "วิดีโอทดสอบ"
        verbose_name_plural = "วิดีโอทดสอบ"

    def __str__(self) -> str:
        return self.name

    @property
    def reviewed_count(self) -> int:
        return sum(1 for e in self.events.all()
                   if (e.kind == "sighting" and e.gt_status) or (e.kind != "sighting" and e.gt_direction))

    @property
    def missed(self) -> dict:
        return {"people": self.missed_people, "in": self.missed_in, "out": self.missed_out}


class EvalEvent(models.Model):
    """One row the system would have logged while playing an EvalVideo."""

    KINDS = [("sighting", "พบคน"), ("in", "เดินเข้า"), ("out", "เดินออก")]

    video = models.ForeignKey(EvalVideo, on_delete=models.CASCADE, related_name="events")
    kind = models.CharField(max_length=10, choices=KINDS, default="sighting")
    track_id = models.IntegerField(default=-1)
    time_s = models.FloatField(default=0)
    latency_s = models.FloatField(null=True, blank=True)
    name = models.CharField(max_length=100, default="unknown")  # what the system said at run time
    face_seen = models.BooleanField(default=False)
    candidate = models.CharField(max_length=100, blank=True, default="")  # best match at any threshold
    score = models.FloatField(null=True, blank=True)
    shirt_color = models.CharField(max_length=30, blank=True, default="")
    pants_color = models.CharField(max_length=30, blank=True, default="")
    confidence = models.FloatField(default=0)
    snapshot = models.CharField(max_length=500, blank=True, default="")
    face_photos = models.JSONField(default=list, blank=True)  # file paths, for enrolling
    # person box, 0..1 of the frame: matches this row to the same person in a later run
    box = models.JSONField(null=True, blank=True)
    # ground truth, typed in on the review page
    gt_status = models.CharField(max_length=10, blank=True, default="")  # ok / false / duplicate
    gt_identity = models.CharField(max_length=100, blank=True, default="")  # name / __none__ / __skip__
    gt_direction = models.CharField(max_length=10, blank=True, default="")  # ok / wrong / false

    class Meta:
        ordering = ["time_s", "id"]

    def as_metric(self) -> dict:
        return {
            "kind": self.kind, "face_seen": self.face_seen, "candidate": self.candidate or None,
            "score": self.score, "latency_s": self.latency_s, "gt_status": self.gt_status,
            "gt_identity": self.gt_identity, "gt_direction": self.gt_direction,
        }
