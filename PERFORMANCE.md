# ระบบเสริมเพื่อความเร็ว/ความเสถียร และการเลือกโมเดล (CN401)

เอกสารนี้อธิบายโค้ดที่เพิ่มเข้ามา 2 ส่วน

1. **ระบบเสริมลดงานของโมเดล** — ไม่รันโมเดลในเฟรมที่ไม่จำเป็น เลือก backend ที่เร็วกว่า และลดการบันทึกซ้ำ
2. **การเลือกโมเดล** — เปลี่ยนโมเดลตรวจจับคน (YOLO) โมเดลจดจำใบหน้า (SFace / ArcFace) และเปิด ReID ได้จากหน้า "ตั้งค่าระบบ"
   แล้วนำไปวัดผลเทียบกันในหน้า "ทดสอบด้วยวิดีโอ"

ทุกอย่างปรับได้จากหน้า **ตั้งค่าระบบ** (กลุ่ม "โมเดล" และ "ประสิทธิภาพ") และ `connect.py` ใช้ค่าเดียวกัน (จาก `settings.json`)
ค่าเริ่มต้นทำงานเหมือนเดิมทุกอย่าง ยกเว้นระบบเสริมที่เปิดไว้ (ปิดได้ทุกตัว)

---

## 1. ผลลัพธ์โดยสรุป

วัดด้วย `python benchmark.py` (คลิปทดสอบ 360 เฟรม: ห้องว่าง → คน 2 คน → ห้องว่าง → คนหลายคน → ห้องว่าง,
ประมวลผลทุกเฟรม, CPU 2 คอร์ ไม่มี GPU)

| การตั้งค่า | ms/เฟรม | fps | YOLO รัน (ครั้ง) | YuNet รัน (ครั้ง) | จำหน้า (ครั้ง) | แถวที่บันทึก |
|---|---:|---:|---:|---:|---:|---:|
| เดิม (ก่อนปรับ) | 174.9 | 5.7 | 359 | 359 | 4 | 7 |
| ระบบเสริม (PyTorch) | 67.5 | 14.8 | 222 | 32 | 10 | 7 |
| ระบบเสริม + OpenVINO | **51.2** | **19.5** | 222 | 32 | 10 | 7 |

- เร็วขึ้น **3.4 เท่า** ต่อเฟรม โดยบันทึกคนได้เท่าเดิม (7 แถว) และยังจำชื่อ "Demo" ได้
- YOLO ทำงานน้อยลง 38% (ช่วงห้องว่าง), YuNet ทำงานน้อยลง 91% (หาหน้าเฉพาะรอบศีรษะคนที่ยังไม่รู้ชื่อ)
- การจำหน้าเพิ่มจาก 4 เป็น 10 ครั้ง เพราะระบบเก็บหลายใบหน้าต่อคนเพื่อ "โหวต" ชื่อ (แม่นขึ้น แลกกับ ~10 ms ต่อคน)

เปรียบเทียบโมเดล (`python benchmark.py --models`, ระบบเสริมเปิด, OpenVINO):

| โมเดล | ms/เฟรม | fps | หมายเหตุ |
|---|---:|---:|---|
| YOLO11n + SFace (ค่าเริ่มต้น) | 49.0 | 20.4 | |
| YOLO11n + ArcFace-MBF | 53.2 | 18.8 | แม่นกว่า SFace เวลาเพิ่มนิดเดียว (~12–18 ms/หน้า) |
| YOLO11n + ArcFace-R50 | 63.0 | 15.9 | แม่นที่สุด ~140–210 ms/หน้า บน CPU (ทำเฉพาะตอนยังไม่รู้ชื่อ) |
| YOLO26n + SFace | 54.3 | 18.4 | YOLO รุ่นใหม่ ไม่ใช้ NMS |
| YOLO11s + SFace | 115.3 | 8.7 | เจอคนไกล/โดนบังดีกว่า แต่ช้ากว่า ~2 เท่า ควรมี GPU |
| YOLO11n + SFace + ReID | 63.4 | 15.8 | ReID แยก (yolo26n-reid) เพราะ OpenVINO ใช้ feature ของ YOLO ไม่ได้ |

> ตัวเลขนี้คือ **ความเร็ว** เท่านั้น ส่วน **ความแม่น** (Precision/Recall/FAR/FRR) ต้องวัดจากวิดีโอจริงในหน้า
> "ทดสอบด้วยวิดีโอ" (หัวข้อ 4) เพราะคลิปทดสอบมีคนลงทะเบียนเพียงคนเดียว

---

## 2. ระบบเสริมลดงานของโมเดล

### 2.1 พัก YOLO เมื่อภาพนิ่งและไม่มีคน — `person_detection/perf/motion.py` (ไฟล์ใหม่)

**ปัญหา:** กล้องหน้าประตูเห็นทางเดินว่างเกือบทั้งวัน แต่ YOLO ถูกรันทุกเฟรมเท่ากับตอนมีคน

- `MotionGate` ย่อภาพเป็นขาวดำกว้าง 160 px (~1 ms) แล้วเทียบกับภาพพื้นหลังที่อัปเดตช้า ๆ
  ถ้าพิกเซลที่เปลี่ยนเกิน 0.4% ของภาพถือว่า "มีการเคลื่อนไหว" (ไม่นับ noise ของกล้อง)
  - ขณะที่ YOLO เพิ่งตรวจแล้วไม่เจอใคร พื้นหลังจะเรียนรู้เร็วขึ้น (`quiet_learn_rate` 0.3) เพื่อลบ "เงา"
    ของคนที่เพิ่งเดินออกไป — ทำให้ YOLO พักได้เร็วขึ้นหลังคนออก (YOLO ลดลงจาก 13% เป็น 38%)
- `DetectionScheduler` ตัดสินใจว่าเฟรมนี้ต้องรัน YOLO หรือไม่
  - เฟรมก่อนหน้ามีคน → รันเสมอ (tracker ต้องการทุกเฟรม)
  - มีการเคลื่อนไหว → รัน และรันต่อไปอีก 1.5 วินาที
  - ไม่มีอะไรขยับ ไม่มีคน → รันแค่ทุก `IDLE_CHECK_SEC` (2 วินาที) เผื่อมีคนยืนนิ่ง
- ตั้งค่า: `MOTION_GATE`, `IDLE_CHECK_SEC`

### 2.2 YOLO เร็วขึ้นบน CPU และ tracker สำหรับกล้องติดตั้งอยู่กับที่ — `person_detection/perf/backend.py` (ไฟล์ใหม่)

- `resolve_model()` — ถ้าเครื่องไม่มี NVIDIA GPU และติดตั้ง `openvino` ไว้ จะแปลง YOLO เป็น OpenVINO ครั้งเดียว
  (`models/<ชื่อ>_<imgsz>_openvino_model/`) แล้วใช้ตัวนั้น เร็วกว่า PyTorch บน CPU ~25–40%
  ถ้ามี GPU ยังใช้ PyTorch + CUDA (fp16) เหมือนเดิม · ตั้งค่า `YOLO_BACKEND` ใน `config.py` (`auto`/`pytorch`/`openvino`)
- `tracker_config()` — เขียนไฟล์ตั้งค่า BoT-SORT เอง
  - ปิด GMC (`gmc_method: none`) — การชดเชยกล้องขยับด้วย optical flow ทุกเฟรม ซึ่งกล้องที่ยึดผนังไม่ต้องใช้ (`TRACKER_STATIC`)
  - `track_buffer` 60 เฟรม (เดิม 30) คนที่โดนบังชั่วครู่ได้ ID เดิม ลดการบันทึกซ้ำ (`TRACK_BUFFER`)
  - เปิด ReID ได้ (หัวข้อ 3.3)

### 2.3 หาหน้าเฉพาะคนที่ยังไม่รู้ชื่อ — `vision/face.py`, `pipeline.py`

- `FaceDetector.detect_in_regions()` รัน YuNet เฉพาะบริเวณศีรษะ (`head_region()`: ครึ่งบนของกรอบคน)
  ของคนที่ยังไม่รู้ชื่อ แทนการรันทั้งภาพทุกเฟรม คนที่รู้ชื่อแล้วจะไม่ถูกหาหน้าซ้ำ
  ถ้าบริเวณรวมกันเกินครึ่งภาพจะรันทั้งภาพแทน (คุ้มกว่า)
- รันทุก `FACE_EVERY` เฟรม · ตั้งค่า: `SMART_FACES`, `FACE_EVERY`

### 2.4 โหวตชื่อจากหลายใบหน้า — `storage/recorder.py` (`FaceIdentifier`)

- หน้าที่เล็กกว่า `FACE_MIN_SIZE` px ไม่ใช้ (โมเดลมักทายผิด รอให้คนเดินใกล้ก่อน)
- ถ้าเหมือนคนในฐานข้อมูลชัดมาก (คะแนน ≥ เกณฑ์ + 0.12) ตัดสินทันที
- ไม่อย่างนั้นเก็บ `FACE_VOTES` ใบหน้าจากหลายเฟรม เฉลี่ยเวกเตอร์ แล้วค่อยตัดสิน — หน้าเบลอ/หันข้างใบเดียวทำให้ทายผิดไม่ได้
- `PersonRecorder` รอผลโหวตก่อนบันทึก (สูงสุด 10 เฟรม) ชื่อในประวัติจึงไม่ขึ้น unknown ก่อนแล้วค่อยเปลี่ยน

### 2.5 ไม่บันทึกคนเดิมซ้ำ — `storage/recorder.py` (`PersonRecorder`)

- คนที่รู้ชื่อ กล้องเดียวกัน ถ้าถูกบันทึกแถว "พบคน" ไปแล้วไม่เกิน `LOG_COOLDOWN_SEC` (30 วินาที) จะไม่บันทึกซ้ำ
  (เช่น โดนบังแล้ว tracker ให้ ID ใหม่) · แถวเดินเข้า/ออก **บันทึกเสมอ** · จำนวนที่ข้ามแสดงในหน้าภาพสด

### 2.6 ตัวนับงาน — `pipeline.py` (`FrameProcessor.perf()`), หน้า "ภาพสด"

- นับว่า YOLO/YuNet/โมเดลจำหน้าทำงานกี่ครั้ง พักกี่เฟรม ข้ามการบันทึกกี่แถว และใช้โมเดลอะไรอยู่
- แสดงในกล่อง "งานที่ประหยัดได้" ของหน้าภาพสด (`web/static/status.js`, `live.html`)
  ป้ายด้านบนบอกโมเดลที่ใช้ เช่น `yolo11n-pose · OpenVINO · ArcFace-MBF · full-res`

---

## 3. การเลือกโมเดล

### 3.1 โมเดลจดจำใบหน้า — `person_detection/vision/face_models.py` (ไฟล์ใหม่)

| ค่า `FACE_RECOGNIZER` | โมเดล | ขนาดเวกเตอร์ | ไฟล์ | เวลา/หน้า (CPU) |
|---|---|---:|---|---:|
| `sface` (ค่าเริ่มต้น) | OpenCV SFace | 128 | `models/face_recognition_sface_2021dec.onnx` (มีอยู่แล้ว) | ~27 ms |
| `arcface_mbf` | ArcFace MobileFaceNet (InsightFace buffalo_s, w600k_mbf) | 512 | `models/insightface/w600k_mbf.onnx` 13 MB | ~12–18 ms |
| `arcface_r50` | ArcFace ResNet-50 (InsightFace buffalo_l, w600k_r50) | 512 | `models/insightface/w600k_r50.onnx` 174 MB | ~140–210 ms |

- `FACE_MODELS` — ตารางโมเดล (ชื่อ, ขนาดเวกเตอร์, ไฟล์, URL, SHA-256, ชื่อค่าเกณฑ์ที่ใช้)
- `ensure_model()` — ถ้ายังไม่มีไฟล์ ดาวน์โหลด zip จาก GitHub release ของ InsightFace (v0.7) แตกเฉพาะไฟล์ที่ใช้
  ตรวจ SHA-256 แล้วเก็บใน `models/insightface/` (ดาวน์โหลดครั้งเดียว; มี lock กันหลายกล้องโหลดพร้อมกัน)
- `align_arcface()` — หมุน/ย่อหน้าเป็น 112×112 ให้ตา จมูก มุมปาก (5 จุดจาก YuNet) ตรงกับตำแหน่งมาตรฐานของ ArcFace
  (`estimateAffinePartial2D`) — ลำดับจุดของ YuNet ตรงกับ ArcFace อยู่แล้ว (ตรวจในเทสต์)
- `ArcFaceRecognizer` — interface เดียวกับ `FaceRecognizer` (SFace): `embed()`, `similarity()`, `best_match()`
  รันด้วย `cv2.dnn` ของ OpenCV **ไม่ต้องติดตั้งแพ็กเกจเพิ่ม**
- `create_recognizer()` — สร้างโมเดลตามที่เลือกไว้ (ใช้ทั้งกล้อง, หน้าลงทะเบียน, ทดสอบวิดีโอ, `enroll.py`)
- เกณฑ์ความเหมือน: SFace ใช้ `FACE_MATCH_THRESHOLD` (0.363) · ArcFace ทั้งสองรุ่นใช้ `FACE_MATCH_THRESHOLD_ARCFACE`
  (เริ่มที่ 0.35 — คะแนนของ ArcFace ต่ำกว่า SFace) **ควรจูนจากวิดีโอชุด train** (หัวข้อ 4)
- เวกเตอร์ของต่างโมเดลเทียบกันไม่ได้: `similarity()` คืน -1 ถ้าขนาดไม่เท่ากัน (ไม่มีทางจับคู่ผิดโมเดล)
- ⚠️ โมเดลของ InsightFace อนุญาตให้ใช้เพื่อการวิจัย/การศึกษาที่ไม่ใช่เชิงพาณิชย์ (เหมาะกับโปรเจกต์นี้)

### 3.2 โมเดลตรวจจับคน — `vision/detector.py` (`PERSON_MODELS`, `person_model_path()`)

`PERSON_MODEL`: `yolo11n-pose` (ค่าเริ่มต้น), `yolo11s-pose`, `yolo11m-pose`, `yolo26n-pose`, `yolo26s-pose`
ทั้งหมดเป็นโมเดล pose (ใช้ข้อเท้าหาตำแหน่งเท้าสำหรับเส้นประตู) Ultralytics ดาวน์โหลดไฟล์ `.pt` ให้เองครั้งแรก
ถ้าใช้ OpenVINO จะแปลงแยกโฟลเดอร์ตามชื่อรุ่น

### 3.3 ReID ใน tracker — `perf/backend.py` (`tracker_config(reid=...)`), `pipeline.py` (`_tracker_for`)

- `TRACKER_REID` เปิดการเทียบลักษณะภายนอกใน BoT-SORT: คนที่โดนบังแล้วโผล่มาได้ ID เดิมบ่อยขึ้น
- `REID_MODEL` = `auto` ใช้ feature ที่ YOLO คำนวณอยู่แล้ว (PyTorch, แทบไม่เพิ่มเวลา ~+10 ms)
  แต่ OpenVINO ไม่มี feature ให้ใช้ ระบบจึงสลับไปใช้ `yolo26n-reid.onnx` ให้อัตโนมัติ (~+14 ms)
- ไฟล์ ReID ถูกดาวน์โหลดลง `models/` (Ultralytics จะติดตั้ง `onnxruntime` ให้เองถ้ายังไม่มี)

### 3.4 หาและจำหน้าจากภาพความละเอียดเต็ม — `vision/face.py` (`FaceHit`), `core/geometry.py`, `pipeline.py`

**ปัญหา:** ภาพกล้องถูกย่อเหลือด้านยาว 960 px ก่อนประมวลผล หน้าคนที่อยู่ไกลจึงเหลือ 20–30 px ซึ่งจำไม่ได้
ทั้งที่กล้อง 1080p–4K เห็นหน้านั้นใหญ่กว่า 2–4 เท่า

- `FramePreprocessor` แยกเป็น `upright()` (หมุนอย่างเดียว) + `scale()` (ย่อ) เพื่อให้ pipeline เก็บภาพเต็มไว้
- `detect_in_regions(source=, scale=)` ตัดบริเวณศีรษะจากภาพเต็ม (ย่อเหลือไม่เกิน 480 px ให้ YuNet)
  แล้วคืน `FaceHit` = (กรอบ, จุด landmark ในภาพที่ย่อแล้ว) + `source`/`source_row` (ภาพเต็มและจุดในภาพเต็ม)
  กรอบที่วาด/จับคู่กับคนยังเป็นพิกัดเดิม ส่วน `FaceIdentifier` ใช้ภาพเต็มในการจำหน้า และวัด `FACE_MIN_SIZE` จากภาพเต็ม
- ตั้งค่า: `FACE_FULL_RES` (เปิดเป็นค่าเริ่มต้น; ไม่มีผลถ้ากล้องละเอียดไม่เกินขนาดประมวลผล)

### 3.5 ข้อมูลใบหน้าแยกตามโมเดล — `storage/people_db.py`, `storage/database.py`, `enrollment.py`

- ไม่มีฐานข้อมูล: SFace ใช้ `output/people.json` (เหมือนเดิม) โมเดลอื่นใช้ `output/people_<model>.json` (`people_db_path()`)
- PostgreSQL: ตาราง `face_embeddings` มีคอลัมน์ `model` และรับเวกเตอร์ 128 หรือ 512 ตัวเลข
  `SqlPeopleDB(db, model)` อ่านเฉพาะแถวของโมเดลนั้น · ฐานข้อมูลเก่าถูกอัปเกรดอัตโนมัติ (`Database.upgrade()`)
- `PipelineFactory.people_db(model)` เก็บแกลเลอรีแยกต่อโมเดล
- `GalleryBuilder` (+ `known_names()`, `missing_names()`, `prepare_gallery()`) — เมื่อเปลี่ยนโมเดล
  สร้างเวกเตอร์ของทุกคนใหม่จากรูปลงทะเบียนใน `models/photos/<ชื่อ>/` (เหมือนลงทะเบียนใหม่ แต่ใช้รูปเดิม)
  - เว็บ: `services.GalleryJob` ทำในเบื้องหลังทันทีที่บันทึกการตั้งค่า (ดาวน์โหลดโมเดล → สร้างเวกเตอร์)
    มีแถบความคืบหน้าในหน้าตั้งค่า (`gallery_status.html`, `/api/gallery/`) และหน้า "คนที่ลงทะเบียน"
    ถ้ามีคนที่ยังขาดจะมีปุ่ม "สร้างจากรูปที่ลงทะเบียนไว้" (`/people/gallery/rebuild/`)
  - `connect.py`: ทำตอนเปิดโปรแกรม (`app.py` → `prepare_gallery()`)
  - ลบคน = ลบออกจากแกลเลอรีของทุกโมเดล

#### เปลี่ยนโมเดลแล้ว ID จะเปลี่ยนไหม?

| สิ่งที่เรียกว่า "ID" | เปลี่ยนไหม | เหตุผล |
|---|---|---|
| Track ID ของ tracker (ตัวเลขบนกรอบ, `track_id` ใน CSV) | ไม่เกี่ยวกับโมเดลใบหน้า | เริ่มนับใหม่ทุกครั้งที่กล้องเริ่มอยู่แล้ว ใช้แค่ภายในการรันครั้งนั้น (เปลี่ยน YOLO/ReID อาจทำให้แบ่ง track ต่างจากเดิมได้) |
| ชื่อคน / `persons.id` / รหัสนักศึกษา | **ไม่เปลี่ยน** | ยังเป็นแถวเดิมใน `persons` ประวัติเก่าใน `detection_logs` ยังชี้คนเดิม |
| เวกเตอร์ใบหน้า (`face_embeddings`, `people*.json`) | **ต่างกันต่อโมเดล** | SFace (128) กับ ArcFace (512) เทียบกันไม่ได้ จึงเก็บแยกและสร้างใหม่จากรูปเดิม ของเดิมยังอยู่ — สลับกลับเป็น SFace ได้ทันที |
| G-ID ข้ามกล้อง | ไม่เกี่ยว | ได้จากตำแหน่งบนพื้น ไม่ได้จากใบหน้า |

ข้อควรระวัง: คนที่ลงทะเบียนแบบไม่มีรูปเหลืออยู่ (เช่น ใช้ `enroll.py` จากโฟลเดอร์ที่ลบไปแล้ว) สร้างเวกเตอร์ใหม่ไม่ได้
ระบบจะแสดงชื่อไว้ ให้ลงทะเบียนใหม่ด้วยรูป · ถ้าเพิ่มรูปให้คนเดิมภายหลัง จะอัปเดตเฉพาะโมเดลที่ใช้อยู่
(โมเดลอื่นยังใช้รูปชุดก่อน; กดสร้างใหม่ได้)

---

## 4. วัดผลเทียบโมเดลในหน้า "ทดสอบด้วยวิดีโอ"

- `evaluation/runner.py` บันทึกโมเดลที่ใช้ลงใน `stats` ของแต่ละรอบ (`person_model`, `face_model`, `backend`, `reid`, `full_res`)
  และกรอบของคนในแต่ละแถว (`EvalEvent.box`, migration `0005_eval_models`)
- **ประมวลผลใหม่หลังเปลี่ยนโมเดล:** `web/cameras/evaluation.py`
  - `run_summary()` เก็บสรุปผลรอบก่อนลง `EvalVideo.runs` (สูงสุด 10 รอบ)
  - `carry_ground_truth()` ใส่ผลตรวจที่กรอกไว้ให้แถวของรอบใหม่ที่เป็น **คนเดิมในเวลาใกล้กัน**
    (ชนิดแถวเดียวกัน ห่างไม่เกิน 1 วินาที กรอบซ้อนกัน IoU ≥ 0.3) — ไม่ต้องตรวจวิดีโอเดิมใหม่ทั้งหมด
    เหลือตรวจเฉพาะแถวที่ไม่ตรงกับรอบก่อน และจำนวนคนที่ระบบพลาด
  - หน้าวิดีโอมีตาราง **"เปรียบเทียบโมเดลกับวิดีโอนี้"** (ms/เฟรม, Precision, Recall, F1, FAR, FRR, Accuracy, Coverage, หน่วง)
- เกณฑ์ τ ของแต่ละรอบใช้ค่าของโมเดลใบหน้าที่รอบนั้นใช้ (`model_threshold()`) เพราะคะแนน SFace กับ ArcFace คนละสเกล
- หน้ารวม: คอลัมน์ "โมเดล" ในรายการวิดีโอ และแถว "โมเดล: ..." ในตารางสรุปเมื่อมีหลายโมเดล
- จูนเกณฑ์: ใช้เฉพาะวิดีโอชุด train ที่รันด้วยโมเดลใบหน้าที่ใช้อยู่ (ArcFace กวาด τ 0.20–0.50)
  ปุ่ม "ใช้ค่านี้กับระบบ" ตั้งค่าเกณฑ์ของโมเดลที่ใช้อยู่ (`FACE_MATCH_THRESHOLD` หรือ `FACE_MATCH_THRESHOLD_ARCFACE`)

**ขั้นตอนวัดผลแนะนำ**

1. อัปโหลดวิดีโอจากกล้องจริง (ชุด test และ train) ด้วยโมเดลเดิม แล้วตรวจผลทีละแถว + ใส่จำนวนคนที่พลาด
2. หน้าตั้งค่า → เปลี่ยนโมเดล (เช่น ArcFace-MBF) → รอแถบ "สร้างข้อมูลใบหน้า" เสร็จ
3. เปิดวิดีโอเดิม → "ประมวลผลใหม่" → ตรวจเฉพาะแถวที่ยังไม่มีผล → ดูตารางเปรียบเทียบ
4. ใช้ชุด train จูน τ ของโมเดลใหม่ → "ใช้ค่านี้กับระบบ" → ดูผลชุด test

---

## 5. ไฟล์ที่เพิ่ม/แก้ และหน้าที่

### ไฟล์ใหม่

| ไฟล์ | หน้าที่ |
|---|---|
| `new/person_detection/perf/__init__.py` | แพ็กเกจระบบเสริม |
| `new/person_detection/perf/motion.py` | `MotionGate`, `DetectionScheduler` — พัก YOLO เมื่อภาพนิ่งและไม่มีคน |
| `new/person_detection/perf/backend.py` | `resolve_model()` OpenVINO, `tracker_config()` BoT-SORT สำหรับกล้องนิ่ง + ReID |
| `new/person_detection/vision/face_models.py` | ตารางโมเดลใบหน้า, ดาวน์โหลด, `align_arcface()`, `ArcFaceRecognizer`, `create_recognizer()` |
| `new/benchmark.py` | วัดความเร็ว: ก่อน/หลังระบบเสริม และ `--models` เทียบโมเดล |
| `new/tests/test_performance.py` | เทสต์ระบบเสริม (motion, scheduler, โหวต, cooldown, หาหน้าเฉพาะบริเวณ, OpenVINO ได้ผลเท่ากัน) |
| `new/tests/test_models.py` | เทสต์การเลือกโมเดล (ตาราง/เกณฑ์, ไฟล์แกลเลอรี, ReID, alignment, full-res, ArcFace, สร้างแกลเลอรีใหม่) |
| `web/cameras/migrations/0005_eval_models.py` | `EvalEvent.box`, `EvalVideo.runs` |
| `web/cameras/tests_models.py` | เทสต์หน้าเว็บ: ตั้งค่าโมเดล, งานสร้างแกลเลอรี, ย้ายผลตรวจ, ตารางเปรียบเทียบ, เกณฑ์ต่อโมเดล |
| `web/templates/cameras/gallery_status.html` | แถบความคืบหน้าดาวน์โหลดโมเดล/สร้างข้อมูลใบหน้า |

### ไฟล์ที่แก้

| ไฟล์ | สิ่งที่เปลี่ยน |
|---|---|
| `new/person_detection/config.py` | ค่า performance, `PERSON_MODEL`, `FACE_RECOGNIZER`, `FACE_MATCH_THRESHOLD_ARCFACE`, `FACE_FULL_RES`, `TRACKER_REID`, `REID_MODEL`, `CHOICES` (ค่าข้อความที่อนุญาต) |
| `new/person_detection/pipeline.py` | scheduler/idle-skip, `_faces()`, `perf()`, ภาพเต็มสำหรับหน้า, `people_db(model)`, `_tracker_for()`, สร้างโมเดลตามการตั้งค่า |
| `new/person_detection/core/geometry.py` | `FramePreprocessor.upright()` / `scale()` |
| `new/person_detection/vision/detector.py` | backend OpenVINO, ไฟล์ tracker, `PERSON_MODELS`, `person_model_path()` |
| `new/person_detection/vision/face.py` | `FaceHit`, `detect_in_regions()` (+ภาพเต็ม), `detect_full()`, `head_region()`, กันเทียบต่างโมเดล |
| `new/person_detection/storage/recorder.py` | `FaceIdentifier` (โหวต, ขนาดขั้นต่ำ, ใช้ภาพเต็ม), `PersonRecorder` (cooldown, รอผลโหวต) |
| `new/person_detection/storage/people_db.py` | `people_db_path(model)`, `open_people_db(model)` |
| `new/person_detection/storage/database.py` | คอลัมน์ `model`, เวกเตอร์ 128/512, `upgrade()`, `SqlPeopleDB(db, model)`, `all_names()` |
| `new/person_detection/enrollment.py` | `person_dir()`/`photos_of()` แบบไม่ต้องโหลดโมเดล, ลบคนทุกโมเดล, `GalleryBuilder`, `prepare_gallery()` |
| `new/person_detection/evaluation/runner.py` | ใช้โมเดลที่เลือก, บันทึกโมเดลใน stats, กรอบคนในแต่ละแถว, cooldown/scheduler ตามเวลาในวิดีโอ |
| `new/person_detection/app.py` | `connect.py` พิมพ์โมเดลที่ใช้ และสร้างแกลเลอรีของโมเดลใหม่ตอนเริ่ม |
| `new/enroll.py` | ใช้โมเดลที่เลือก (`--model` เลือกเองได้) |
| `new/tests/test_database.py` | เทสต์แกลเลอรีแยกต่อโมเดลใน PostgreSQL และอัปเกรดฐานข้อมูลเก่า |
| `new/tests/test_resolution.py` | บังคับ PyTorch เพื่อผลคงที่, ยอมให้จำนวนคนต่างได้ ±1 |
| `web/cameras/system_settings.py` | รองรับค่าแบบเลือก (str + choices), กลุ่ม "โมเดล" และ "ประสิทธิภาพ" |
| `web/cameras/services.py` | `enroller()` ตามโมเดล, `GalleryJob`, ส่ง `perf` ไปหน้าภาพสด |
| `web/cameras/views.py` | สถานะแกลเลอรีในหน้าตั้งค่า/คนที่ลงทะเบียน, `api_gallery_status`, `gallery_rebuild` |
| `web/cameras/evaluation.py` | `run_summary`, `carry_ground_truth`, `model_label`, `model_threshold`, กวาด τ ตามโมเดล |
| `web/cameras/views_eval.py` | เกณฑ์ต่อโมเดล, กลุ่มตามโมเดล, ตารางเปรียบเทียบ, ปุ่มใช้เกณฑ์กับโมเดลที่ใช้อยู่ |
| `web/cameras/models.py` | `EvalEvent.box`, `EvalVideo.runs` |
| `web/cameras/tests_eval.py` | เทสต์ "ประมวลผลใหม่" เก็บรอบเดิมและย้ายผลตรวจ |
| `web/portal/urls.py` | `/api/gallery/`, `/people/gallery/rebuild/` |
| `web/templates/cameras/settings.html`, `people.html`, `live.html`, `evaluate.html`, `evaluate_detail.html` | UI ตามข้างบน |
| `web/static/status.js`, `web/static/portal.css` | กล่อง "งานที่ประหยัดได้" และป้ายโมเดล |
| `DATABASE.md`, `requirements.txt` | คอลัมน์ `model`; แพ็กเกจเสริม (openvino, onnxruntime) แบบไม่บังคับ |

---

## 6. ทดสอบ

```
cd new
python -m unittest discover -s tests          # 107 tests (ArcFace tests ข้ามถ้ายังไม่ได้ดาวน์โหลด)
python benchmark.py                           # ก่อน/หลังระบบเสริม
python benchmark.py --models                  # เทียบโมเดล (ดาวน์โหลดโมเดลที่ยังไม่มี)
python benchmark.py path\to\camera_clip.mp4   # วัดกับวิดีโอจริง
cd ..\web
python manage.py test cameras                 # 66 tests
```

ผลล่าสุด: engine 107 tests ผ่าน, web 66 tests ผ่าน (รวม PostgreSQL ด้วย `TEST_DATABASE_URL`)
