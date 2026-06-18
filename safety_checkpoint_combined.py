# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
================================================================
COMBINED SAFETY CHECKPOINT — Helmet + Mask + Alcohol Detection
================================================================
Access is granted ONLY when ALL THREE are true:
  1. Helmet is detected                       (YOLO)
  2. Mask is detected                          (OpenCV colour analysis)
  3. Alcohol sensor reading is BELOW threshold (MQ-3, live from Arduino)

Flow:
  - Webcam continuously checks for helmet + mask.
  - In the background, a serial reader thread keeps the latest
    MQ-3 reading updated at all times (Arduino streams "SENSOR:xxx"
    every 300ms on its own — this matches the confirmed-working
    alcohol_gate_control_working.ino protocol).
  - Once helmet + mask are both confirmed for CONFIRM_FRAMES
    consecutive frames, the current live alcohol reading is used
    immediately (no waiting/prompting needed) and a single combined
    decision is sent to Arduino: ACCESS_GRANTED or ACCESS_DENIED.

Hardware (all on one Arduino Uno):
  Pin 2   -> Red LED    (220 ohm resistor to GND)
  Pin 3   -> Green LED  (220 ohm resistor to GND)
  Pin 4   -> Buzzer     (active buzzer: + to pin 4, - to GND)
  Pin 5   -> Servo      (signal/orange wire; red->5V, black/brown->GND)
  Pin A1  -> MQ-3 alcohol sensor analog output
  USB     -> COM5 (check Device Manager, or set serial_port to "AUTO")

Detection methods:
  Helmet  -> YOLOv8 model (helmet_model.pt, place in same folder)
  Mask    -> OpenCV colour analysis on face region (no extra model)
  Alcohol -> MQ-3 analog reading streamed live by Arduino, threshold-based

Install requirements:
  pip install ultralytics opencv-python pyserial numpy

Model (download once, place in same folder as this script):
  https://huggingface.co/keremberke/yolov8m-hard-hat-detection/resolve/main/best.pt
  -> rename to: helmet_model.pt

Upload safety_checkpoint_combined.ino to the Arduino FIRST,
then run this script.
================================================================
"""

import cv2
import serial
import serial.tools.list_ports
import time
import os
import sys
import json
import threading
import datetime
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these if needed
# ──────────────────────────────────────────────────────────────────────
CONFIG = {
    "serial_port"            : "COM5",   # "AUTO" to auto-detect, or e.g. "COM3"
    "baud_rate"               : 9600,
    "model_path"              : "helmet_model.pt",
    "conf_threshold"          : 0.40,    # YOLO detection confidence (0.0-1.0)
    "confirm_frames"          : 15,      # Consecutive frames needed to confirm helmet+mask
    "alcohol_threshold"       : 400,     # MQ-3 raw value above this = alcohol detected
    "sensor_warmup_seconds"   : 3,       # Extra warm-up wait after connecting
    "log_file"                : "access_log.json",
    "camera_index"            : 0,       # 0 = default webcam
    "cooldown_seconds"        : 5,       # Wait time after each decision before next scan cycle
    "detection_timer_seconds" : 13,      # Time window to detect helmet+mask before DENY
}

# ──────────────────────────────────────────────────────────────────────
# COLOURS (BGR)
# ──────────────────────────────────────────────────────────────────────
GREEN  = (0, 210,   0)
RED    = (0,   0, 220)
ORANGE = (0, 140, 255)
YELLOW = (0, 200, 220)
CYAN   = (220, 220,  0)
WHITE  = (240, 240, 240)
DARK   = (20,  20,  20)
GREY   = (120, 120, 120)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# ──────────────────────────────────────────────────────────────────────
# CHECK MODEL EXISTS
# ──────────────────────────────────────────────────────────────────────
if not os.path.exists(CONFIG["model_path"]):
    print("=" * 55)
    print(f"ERROR: Model file '{CONFIG['model_path']}' not found!")
    print()
    print("Download it here (paste in browser):")
    print("https://huggingface.co/keremberke/yolov8m-hard-hat-detection/resolve/main/best.pt")
    print()
    print(f"Rename it to '{CONFIG['model_path']}' and place it in:")
    print(f"  {os.path.abspath('.')}")
    print("=" * 55)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# LOAD FACE CASCADE (for mask region detection)
# ──────────────────────────────────────────────────────────────────────
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("[ERROR] Could not load face cascade. Check OpenCV install.")
    sys.exit(1)
print("[OK] Face detector loaded.")

# ──────────────────────────────────────────────────────────────────────
# LOAD YOLO MODEL
# ──────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading YOLO model: {CONFIG['model_path']} ...")
model = YOLO(CONFIG["model_path"])
print(f"[OK] YOLO model loaded. Classes: {list(model.names.values())}")

HELMET_ON_CLASSES  = {'hardhat', 'helmet', 'hard-hat', 'hard hat', 'safety helmet'}
HELMET_OFF_CLASSES = {'no-hardhat', 'no hardhat', 'no-helmet', 'no helmet',
                      'without helmet', 'no hard-hat'}

# ──────────────────────────────────────────────────────────────────────
# ARDUINO CONNECTION — push-based sensor protocol (confirmed working)
# ──────────────────────────────────────────────────────────────────────
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(k in desc for k in ["arduino", "uno", "ch340", "cp210x", "ftdi"]):
            return p.device
        if "2341:0043" in hwid or "2341:0001" in hwid:
            return p.device
    return ports[0].device if ports else None

class ArduinoController:
    def __init__(self):
        self.ser = None
        self.connected = False
        self._latest_sensor = 0
        self._lock = threading.Lock()
        self._reader_thread = None
        self._connect()

    def _connect(self):
        port = CONFIG["serial_port"]
        if port == "AUTO":
            port = find_arduino_port()
        if not port:
            print("[ERROR] Arduino not found.")
            return
        try:
            self.ser = serial.Serial(port, CONFIG["baud_rate"], timeout=1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            deadline = time.time() + 12
            while time.time() < deadline:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    print(f"  [Arduino init] {line}")
                    if "ARDUINO_READY" in line:
                        print(f"[OK] Arduino connected on {port}")
                        self.connected = True
                        self._start_reader()
                        return
            print("[WARN] No READY signal - starting anyway.")
            self.connected = True
            self._start_reader()
        except serial.SerialException as e:
            print(f"[ERROR] Serial: {e}")

    def _start_reader(self):
        self._reader_thread = threading.Thread(target=self._serial_reader, daemon=True)
        self._reader_thread.start()

    def _serial_reader(self):
        """Background thread: continuously reads serial lines and keeps
        the latest MQ-3 sensor value updated, since Arduino streams it
        unprompted every 300ms."""
        while True:
            try:
                if self.ser and self.ser.is_open:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("SENSOR:"):
                        try:
                            val = int(line.split(":")[1])
                            with self._lock:
                                self._latest_sensor = val
                        except ValueError:
                            pass
                    elif line.startswith("STATUS:"):
                        print(f"  [Arduino] {line}")
                else:
                    time.sleep(0.1)
            except Exception:
                time.sleep(0.1)

    def get_sensor_value(self):
        """Return the latest live MQ-3 reading (already streaming in background)."""
        with self._lock:
            return self._latest_sensor

    def send_command(self, command):
        if self.ser and self.connected:
            try:
                # NOTE: Do NOT hold self._lock here.
                # self._lock guards only _latest_sensor (see get_sensor_value).
                # Holding it during a serial write would deadlock with the
                # _serial_reader background thread that also acquires _lock.
                # pyserial write/flush is thread-safe on its own.
                self.ser.reset_input_buffer()          # flush stale SENSOR: lines
                self.ser.write((command + "\n").encode("utf-8"))
                self.ser.flush()
                print(f"  [-> Arduino] Sent: {command}")
            except Exception as e:
                print(f"[ERROR] Send failed: {e}")
        else:
            print(f"[SIM] -> Arduino: {command}")

    def close(self):
        if self.ser:
            self.ser.close()

# ──────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────
def log_event(helmet_ok, mask_ok, alcohol_value, alcohol_ok, granted):
    entry = {
        "timestamp"      : datetime.datetime.now().isoformat(),
        "helmet_ok"      : helmet_ok,
        "mask_ok"        : mask_ok,
        "alcohol_value"  : alcohol_value,
        "alcohol_ok"     : alcohol_ok,
        "access_granted" : granted,
    }
    logs = []
    if os.path.exists(CONFIG["log_file"]):
        with open(CONFIG["log_file"], "r") as f:
            try:
                logs = json.load(f)
            except Exception:
                pass
    logs.append(entry)
    with open(CONFIG["log_file"], "w") as f:
        json.dump(logs, f, indent=2)
    return entry

# ──────────────────────────────────────────────────────────────────────
# MASK DETECTION — checks nose/mouth band of a face ROI for mask colors
# ──────────────────────────────────────────────────────────────────────
def has_mask(roi):
    """
    Colour-based mask detector.  Works on two kinds of ROI:
      A) A full face crop (no-helmet case) — scans the lower 40-90% band
         where nose+mouth sit.
      B) The strip directly below the helmet YOLO box — the mask fills
         most of this strip, so we scan the whole thing.
    Detects: light-blue surgical masks, white N95/KN95, black, green, grey masks.
    Returns True if mask-coloured pixels dominate the scan area.
    """
    rh, rw = roi.shape[:2]
    if rh < 15 or rw < 15:
        return False

    def _count_mask_pixels(crop):
        if crop.size == 0:
            return 0, 0
        hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = crop.shape[0] * crop.shape[1]

        # ── Mask colour ranges (HSV) ──────────────────────────────────────
        # Light-blue surgical mask  (most common; hue 85-120, low-medium sat)
        lb    = cv2.inRange(hsv, np.array([85,  15,  80]), np.array([120, 200, 255]))
        # Deeper/darker blue
        db    = cv2.inRange(hsv, np.array([100, 40,  40]), np.array([135, 255, 255]))
        # White (N95, cloth)
        wh    = cv2.inRange(hsv, np.array([0,   0,  150]), np.array([179,  50, 255]))
        # Black / very dark
        bk    = cv2.inRange(hsv, np.array([0,   0,    0]), np.array([179, 100,  70]))
        # Green (reusable masks)
        gn    = cv2.inRange(hsv, np.array([35,  30,  40]), np.array([90,  255, 255]))
        # Grey
        gy    = cv2.inRange(hsv, np.array([0,   0,   80]), np.array([179,  40, 210]))

        mpx = (cv2.countNonZero(lb) + cv2.countNonZero(db) + cv2.countNonZero(wh) +
               cv2.countNonZero(bk) + cv2.countNonZero(gn) + cv2.countNonZero(gy))

        skin = cv2.inRange(hsv, np.array([0, 18, 60]), np.array([25, 200, 255]))
        spx  = cv2.countNonZero(skin)
        return mpx, total

    x1 = int(rw * 0.05);  x2 = int(rw * 0.95)

    # Band A — whole ROI (used when roi IS the below-helmet strip)
    mpx_full, total_full = _count_mask_pixels(roi[0:rh, x1:x2])
    ratio_full = mpx_full / max(total_full, 1)

    # Band B — lower 40-90% (used when roi is a full face crop)
    y1 = int(rh * 0.40);  y2 = int(rh * 0.90)
    mpx_low, total_low = _count_mask_pixels(roi[y1:y2, x1:x2])
    ratio_low = mpx_low / max(total_low, 1)

    # Pass if EITHER band has strong mask coverage
    return ratio_full > 0.20 or ratio_low > 0.18

# ──────────────────────────────────────────────────────────────────────
# DRAW HELPERS (for alcohol gauge panel)
# ──────────────────────────────────────────────────────────────────────
def put_text_center(img, text, cx, cy, font, scale, color, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (cx - tw//2, cy + th//2), font, scale, color, thickness, cv2.LINE_AA)

def draw_gauge(img, x, y, r, value, max_val, threshold, label):
    cv2.ellipse(img, (x, y), (r, r), 180, 0, 180, (50, 50, 70), 14)
    angle = int(180 * min(value, max_val) / max_val)
    color = GREEN if value < threshold else RED
    if angle > 0:
        cv2.ellipse(img, (x, y), (r, r), 180, 0, angle, color, 10)
    put_text_center(img, str(value), x, y - 10, FONT_BOLD, 0.8, WHITE, 2)
    put_text_center(img, label,      x, y + 18, FONT, 0.42, GREY, 1)
    tick_angle = int(180 * threshold / max_val)
    ta = np.radians(180 + tick_angle)
    tx = int(x + r * np.cos(ta))
    ty = int(y + r * np.sin(ta))
    cv2.circle(img, (tx, ty), 5, YELLOW, -1)

# ──────────────────────────────────────────────────────────────────────
# CONNECT TO ARDUINO
# ──────────────────────────────────────────────────────────────────────
arduino = ArduinoController()
if not arduino.connected:
    print("[FATAL] Cannot connect to Arduino. Check COM port / cable / driver.")
    sys.exit(1)

print(f"\nWarming up MQ-3 sensor ({CONFIG['sensor_warmup_seconds']}s)...", end="", flush=True)
for _ in range(CONFIG["sensor_warmup_seconds"]):
    time.sleep(1)
    print(".", end="", flush=True)
print(" Ready!\n")

# ──────────────────────────────────────────────────────────────────────
# OPEN CAMERA
# ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CONFIG["camera_index"], cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ──────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────
CONFIRM_FRAMES = CONFIG["confirm_frames"]
confirm_count       = 0          # consecutive frames with helmet+mask both OK
result_until        = 0          # timestamp until which a result/cooldown is displayed
last_result_color   = WHITE      # color of the cooldown bar (matches last decision)

# Detection timer: 13-second window to detect helmet+mask
detection_start     = time.time()   # when the current 13s window started
detection_timed_out = False         # True while showing the timeout-deny result

print("[START] Combined safety checkpoint running. Press Q to quit.\n")

# ──────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera read failed.")
        break

    now  = time.time()
    h, w = frame.shape[:2]

    # ── 1. HELMET DETECTION (YOLO) ────────────────────────────────────
    helmet_on  = False
    helmet_off = False

    results = model(frame, stream=True, verbose=False, conf=CONFIG["conf_threshold"])

    best_helmet_on  = None   # (conf, x1, y1, x2, y2)
    best_helmet_off = None

    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])].lower().strip()
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name in HELMET_ON_CLASSES:
                if best_helmet_on is None or conf > best_helmet_on[0]:
                    best_helmet_on = (conf, x1, y1, x2, y2)
            elif cls_name in HELMET_OFF_CLASSES:
                if best_helmet_off is None or conf > best_helmet_off[0]:
                    best_helmet_off = (conf, x1, y1, x2, y2)

    # Geometry sanity: a "NO HELMET" box whose top is in the upper 55% of the
    # frame almost certainly covers the head/helmet, not the below-chin region
    # a true "no helmet" person would produce.  Discard it to prevent the model
    # misclassifying the helmet brim+face area as "no hardhat".
    if best_helmet_off is not None:
        _nhc, _nx1, _ny1, _nx2, _ny2 = best_helmet_off
        box_top_frac = _ny1 / h
        box_h_frac   = (_ny2 - _ny1) / h
        if box_top_frac < 0.55 or box_h_frac > 0.55:
            best_helmet_off = None   # geometry says this is the helmet being mislabelled

    # Only trust whichever side (helmet-on vs helmet-off) is more confident overall,
    # so the model never shows both a "HELMET" and "NO HELMET" box on the same head.
    if best_helmet_on and best_helmet_off:
        if best_helmet_on[0] >= best_helmet_off[0]:
            best_helmet_off = None
        else:
            best_helmet_on = None

    if best_helmet_on:
        conf, x1, y1, x2, y2 = best_helmet_on
        helmet_on = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 3)
        cv2.putText(frame, f"HELMET {conf:.0%}", (x1, y1 - 10),
                    FONT, 0.65, GREEN, 2, cv2.LINE_AA)
    elif best_helmet_off:
        conf, x1, y1, x2, y2 = best_helmet_off
        helmet_off = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 3)
        cv2.putText(frame, f"NO HELMET {conf:.0%}", (x1, y1 - 10),
                    FONT, 0.65, RED, 2, cv2.LINE_AA)

    # ── 2. MASK DETECTION ─────────────────────────────────────────────
    # Key insight: the YOLO helmet box bottom edge sits at ~eye level.
    # The mask (nose+mouth) is BELOW that edge.  When a helmet is found
    # we skip the face cascade entirely and colour-scan the region directly
    # below the helmet box — this is robust regardless of head pose.
    mask_on    = False
    mask_col   = ORANGE
    mask_label = "NO MASK"
    scan_box   = None   # (rx1, ry1, rw, rh) drawn on screen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if best_helmet_on is not None:
        # ── Helmet present: scan the strip below the helmet YOLO box ─────
        _hc, _hx1, _hy1, _hx2, _hy2 = best_helmet_on
        helm_w = _hx2 - _hx1
        helm_h = _hy2 - _hy1

        # Extend x slightly for mask straps; scan 0.1–1.1× helm_h below brim
        margin_x = int(helm_w * 0.12)
        rx1 = max(0, _hx1 - margin_x)
        rx2 = min(w, _hx2 + margin_x)
        ry1 = max(0, int(_hy2 + helm_h * 0.05))   # just below brim
        ry2 = min(h, int(_hy2 + helm_h * 1.10))   # down to chin/neck

        if ry2 > ry1 + 10 and rx2 > rx1 + 10:
            scan_roi = frame[ry1:ry2, rx1:rx2]
            mask_on  = has_mask(scan_roi)
            scan_box = (rx1, ry1, rx2 - rx1, ry2 - ry1)

    else:
        # ── No helmet: standard face cascade → colour scan ────────────────
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            lower_gray  = gray[h // 3:, :]
            loose_faces = face_cascade.detectMultiScale(lower_gray, scaleFactor=1.05,
                                                        minNeighbors=3, minSize=(50, 50))
            if len(loose_faces) > 0:
                lx, ly, lw, lh = max(loose_faces, key=lambda f: f[2] * f[3])
                faces = np.array([[lx, h // 3 + ly, lw, lh]])

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_roi = frame[fy:fy + fh, fx:fx + fw]
            mask_on  = has_mask(face_roi)
            scan_box = (fx, fy, fw, fh)

    if scan_box is not None:
        rx1, ry1, rw, rh = scan_box
        mask_col   = GREEN  if mask_on else ORANGE
        mask_label = "MASK ON" if mask_on else "NO MASK"
        cv2.rectangle(frame, (rx1, ry1), (rx1 + rw, ry1 + rh), mask_col, 2)
        cv2.putText(frame, mask_label, (rx1, ry1 + rh + 22),
                    FONT, 0.65, mask_col, 2, cv2.LINE_AA)

    # ── 3. LIVE ALCOHOL READING (always available — streamed in background) ──
    sensor_value = arduino.get_sensor_value()
    alcohol_ok   = sensor_value < CONFIG["alcohol_threshold"]

    # ── 4. CONFIRMATION COUNTER (only while in active scan window) ────────
    # During cooldown (now <= result_until) we skip counting so the timer
    # doesn't run down while the result screen is displayed.
    in_cooldown = (now <= result_until)

    if not in_cooldown:
        if helmet_on and mask_on:
            confirm_count += 1
        else:
            confirm_count = max(0, confirm_count - 1)

    # ── 5A. DETECTION TIMER EXPIRY: helmet+mask NOT confirmed in 13 seconds ──
    detection_elapsed = now - detection_start
    timer_seconds     = CONFIG["detection_timer_seconds"]

    if (not in_cooldown
            and detection_elapsed >= timer_seconds
            and confirm_count < CONFIRM_FRAMES):
        # Person failed to present helmet+mask within the allotted window
        confirm_count = 0
        print("[TIMEOUT] 13s elapsed — helmet/mask not confirmed -> ACCESS DENIED")

        result_frame = frame.copy()
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), DARK, -1)
        cv2.addWeighted(overlay, 0.55, result_frame, 0.45, 0, result_frame)

        put_text_center(result_frame, "ACCESS DENIED", w//2, h//2 - 30,
                        FONT_BOLD, 1.4, RED, 3)
        put_text_center(result_frame, "HELMET / MASK NOT DETECTED IN TIME", w//2, h//2 + 20,
                        FONT, 0.75, ORANGE, 2)

        log_event(False, False, sensor_value, alcohol_ok, False)
        arduino.send_command("ACCESS_DENIED")

        cv2.imshow("Safety Checkpoint", result_frame)
        cv2.waitKey(1)

        last_result_color = RED
        result_until  = now + CONFIG["cooldown_seconds"]
        detection_start = result_until   # restart 13s window AFTER cooldown
        frame = result_frame

    # ── 5B. HELMET + MASK CONFIRMED — pause timer, check alcohol ──────────
    elif (not in_cooldown and confirm_count >= CONFIRM_FRAMES):
        confirm_count = 0
        granted = alcohol_ok   # helmet+mask already confirmed; alcohol is the final gate

        result_frame = frame.copy()
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), DARK, -1)
        cv2.addWeighted(overlay, 0.55, result_frame, 0.45, 0, result_frame)

        draw_gauge(result_frame, w//2, 220, 90, sensor_value, 1023,
                   CONFIG["alcohol_threshold"], "MQ-3 raw (0-1023)")

        if granted:
            result_text  = "ACCESS GRANTED"
            result_color = GREEN
            last_result_color = GREEN
            print(f"[PASS] Helmet OK + Mask OK + Alcohol CLEAR ({sensor_value}) -> Access granted")
        else:
            result_text  = f"ACCESS DENIED: ALCOHOL DETECTED ({sensor_value})"
            result_color = RED
            last_result_color = RED
            print(f"[FAIL] Alcohol detected ({sensor_value}) -> Access denied")

        put_text_center(result_frame, "HELMET + MASK: OK", w//2, h//2 - 60,
                        FONT, 0.75, GREEN, 2)
        put_text_center(result_frame, result_text, w//2, h - 80,
                        FONT_BOLD, 1.0, result_color, 2)

        log_event(True, True, sensor_value, alcohol_ok, granted)
        arduino.send_command("ACCESS_GRANTED" if granted else "ACCESS_DENIED")

        cv2.imshow("Safety Checkpoint", result_frame)
        cv2.waitKey(1)

        last_result_color = result_color
        result_until  = now + CONFIG["cooldown_seconds"]
        detection_start = result_until   # restart 13s window AFTER cooldown
        frame = result_frame

    # ── 6. DRAW HUD (skipped while a result is being held on screen) ──────
    if not in_cooldown:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "SAFETY CHECKPOINT -- HELMET + MASK + ALCOHOL",
                    (20, 30), FONT_BOLD, 0.65, CYAN, 1, cv2.LINE_AA)

        # Detection countdown
        time_left = max(0.0, timer_seconds - detection_elapsed)
        timer_col = GREEN if time_left > 5 else (YELLOW if time_left > 2 else RED)

        if confirm_count > 0:
            pct = min(100, int(confirm_count / CONFIRM_FRAMES * 100))
            status_text  = f"HELMET + MASK OK -- Confirming {pct}%  |  Timer: {time_left:.1f}s"
            status_color = GREEN
        else:
            status_text  = f"SCANNING... STEP INTO FRAME  |  Timer: {time_left:.1f}s"
            status_color = YELLOW
        cv2.putText(frame, status_text, (20, 68), FONT, 0.65, status_color, 2, cv2.LINE_AA)

        # Detection timer bar (fills left-to-right, shrinks as time runs out)
        bar_max = w - 40
        bar_w   = int((time_left / timer_seconds) * bar_max)
        cv2.rectangle(frame, (20, 78), (w - 20, 84), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 78), (20 + bar_w, 84), timer_col, -1)

        conn_text = "ARDUINO: CONNECTED" if arduino.connected else "ARDUINO: DISCONNECTED"
        conn_col  = GREEN if arduino.connected else RED
        cv2.putText(frame, conn_text, (w - 300, 30), FONT, 0.5, conn_col, 1, cv2.LINE_AA)

        h_icon_col = GREEN if helmet_on else (RED if helmet_off else YELLOW)
        m_icon_col = GREEN if mask_on else YELLOW
        a_icon_col = GREEN if alcohol_ok else RED
        cv2.putText(frame, f"[Helmet]: {'ON' if helmet_on else ('OFF' if helmet_off else '?')}",
                    (w - 300, 55), FONT, 0.48, h_icon_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"[Mask]: {'ON' if mask_on else '?'}",
                    (w - 170, 55), FONT, 0.48, m_icon_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"[Alcohol]: {sensor_value}",
                    (w - 300, 75), FONT, 0.48, a_icon_col, 1, cv2.LINE_AA)
    else:
        # ── Cooldown countdown bar (shown while result_until is in the future) ──
        cooldown_left = max(0.0, result_until - now)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 42), (w, h), DARK, -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

        bar_w = int((cooldown_left / CONFIG["cooldown_seconds"]) * (w - 40))
        cv2.rectangle(frame, (20, h - 28), (w - 20, h - 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 28), (20 + bar_w, h - 10), last_result_color, -1)
        cv2.putText(frame, f"Next scan in: {cooldown_left:.1f}s",
                    (20, h - 32), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    cv2.imshow("Safety Checkpoint", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ──────────────────────────────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
arduino.close()
print("\n[DONE] System shut down.")
