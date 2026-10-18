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
    enable_direction = models.BooleanField("บอกทิศทางเดินผ่านโซน (ทดลอง)", default=False)
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
    def is_calibrated(self) -> bool:
        return bool(self.floor_calibration)

    @property
    def csv_tag(self) -> str:
        return f"cam-{self.name}"
