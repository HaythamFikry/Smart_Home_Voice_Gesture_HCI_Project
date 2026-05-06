"""
╔══════════════════════════════════════════════════════════════════╗
║         ENHANCED SMART HOME HCI — Gesture + Voice Control        ║
║                                                                  ║
║  Devices  : Light (on/off) | Fan (speed 1-3) | AC | TV           ║
║  Gestures : 0=All OFF | 1=Fan Spd1 | 2=Fan Spd2 | 3=Fan Spd3     ║
║             4=AC Toggle | 5=Light ON | ✌=TV Toggle              ║
║             Two hands (5+5)=ALL ON | (0+0)=ALL OFF               ║
║  Voice    : Natural language — "turn on the lights", "fan speed  ║
║             two", "everything off", etc.  (non-blocking thread)  ║
╚══════════════════════════════════════════════════════════════════╝
Dependencies:
    pip install opencv-python mediapipe SpeechRecognition pyaudio
"""

import cv2
import mediapipe as mp
import speech_recognition as sr
import threading
import time
import pyaudio
from collections import deque
from datetime import datetime

# ──────────────────────────────────────────────
#  MediaPipe Setup
# ──────────────────────────────────────────────
mp_hands  = mp.solutions.hands
hands_det = mp_hands.Hands(
    max_num_hands=2,          # ← detect up to TWO hands
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6,
)
mp_draw = mp.solutions.drawing_utils
HAND_STYLE = mp_draw.DrawingSpec(color=(0, 220, 255), thickness=2, circle_radius=4)
CONN_STYLE = mp_draw.DrawingSpec(color=(0, 120, 255), thickness=2)

# ──────────────────────────────────────────────
#  Speech Recogniser (runs in a background thread)
# ──────────────────────────────────────────────
recognizer   = sr.Recognizer()
recognizer.energy_threshold        = 300
recognizer.dynamic_energy_threshold = True

# ──────────────────────────────────────────────
#  Device State
# ──────────────────────────────────────────────
class SmartHome:
    def __init__(self):
        self.light_on  = False
        self.fan_speed = 0          # 0 = OFF, 1-3 = speeds
        self.ac_on     = False
        self.tv_on     = False
        self._lock     = threading.Lock()

    # ── helpers ──────────────────────────────
    @property
    def fan_on(self):
        return self.fan_speed > 0

    def all_on(self):
        with self._lock:
            self.light_on  = True
            self.fan_speed = 3
            self.ac_on     = True
            self.tv_on     = True

    def all_off(self):
        with self._lock:
            self.light_on  = False
            self.fan_speed = 0
            self.ac_on     = False
            self.tv_on     = False

    def set_fan_speed(self, speed: int):
        """Speed: 0-3"""
        with self._lock:
            self.fan_speed = max(0, min(3, speed))

    def snapshot(self):
        """Thread-safe copy of all states."""
        with self._lock:
            return {
                "light":     self.light_on,
                "fan_speed": self.fan_speed,
                "fan_on":    self.fan_speed > 0,
                "ac":        self.ac_on,
                "tv":        self.tv_on,
            }

home = SmartHome()

# ──────────────────────────────────────────────
#  Command Log  (last N entries shown on screen)
# ──────────────────────────────────────────────
MAX_LOG = 6
log: deque = deque(maxlen=MAX_LOG)

def log_event(src: str, msg: str):
    ts  = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {src}: {msg}"
    log.append(entry)
    print(entry)

# ──────────────────────────────────────────────
#  Gesture Debounce
#  A gesture must be held for DEBOUNCE_FRAMES consecutive frames
#  before it fires — prevents accidental triggers.
# ──────────────────────────────────────────────
DEBOUNCE_FRAMES  = 8          # ~0.25 s at 30 fps
COOLDOWN_SECONDS = 1.0        # minimum gap between gesture fires

gesture_buffer   = deque(maxlen=DEBOUNCE_FRAMES)
last_gesture_ts  = 0.0
last_gesture_key = None       # remember what just fired to avoid repeats

def debounce_gesture(key):
    """
    Push current gesture key into buffer.
    Return True (fire!) only when:
      • all DEBOUNCE_FRAMES slots agree on the same key
      • it differs from the last fired gesture  OR  cooldown has elapsed
    """
    global last_gesture_ts, last_gesture_key

    gesture_buffer.append(key)

    if len(gesture_buffer) < DEBOUNCE_FRAMES:
        return False
    if len(set(gesture_buffer)) != 1:       # not all the same
        return False

    now = time.time()
    if key == last_gesture_key and (now - last_gesture_ts) < COOLDOWN_SECONDS:
        return False                         # still in cooldown

    last_gesture_ts  = now
    last_gesture_key = key
    return True

# ──────────────────────────────────────────────
#  Finger Counting
# ──────────────────────────────────────────────
FINGERTIP_IDS = [4, 8, 12, 16, 20]

def count_fingers(lms, handedness_label: str) -> int:
    """
    Returns 0-5.
    handedness_label: 'Left' or 'Right' (MediaPipe reports mirrored labels
    because we flip the frame, so we swap them internally).
    """
    fingers = []
    # ── Thumb (x-axis comparison; direction depends on hand) ──
    tip, pip = lms.landmark[4], lms.landmark[3]
    # After horizontal flip: 'Left' label == visually right hand
    if handedness_label == "Left":
        fingers.append(1 if tip.x > pip.x else 0)
    else:
        fingers.append(1 if tip.x < pip.x else 0)

    # ── Four fingers (y-axis: tip above pip2 = extended) ──
    for i in range(1, 5):
        tip_y = lms.landmark[FINGERTIP_IDS[i]].y
        pip_y = lms.landmark[FINGERTIP_IDS[i] - 2].y
        fingers.append(1 if tip_y < pip_y else 0)

    return sum(fingers)

def is_peace_sign(lms) -> bool:
    """
    True when index + middle extended, ring + pinky folded.
    (Victory / peace gesture → TV toggle)
    """
    idx_up  = lms.landmark[8].y  < lms.landmark[6].y
    mid_up  = lms.landmark[12].y < lms.landmark[10].y
    rng_dwn = lms.landmark[16].y > lms.landmark[14].y
    pnk_dwn = lms.landmark[20].y > lms.landmark[18].y
    return idx_up and mid_up and rng_dwn and pnk_dwn

# ──────────────────────────────────────────────
#  Apply Gesture
# ──────────────────────────────────────────────
def apply_gesture(key, counts: list[int]):
    """
    key     canonical gesture identifier (string)
    counts  list of finger counts for each detected hand
    """
    s = home.snapshot()

    if key == "two_all_on":
        home.all_on()
        log_event("Gesture", "ALL DEVICES ON")

    elif key == "two_all_off":
        home.all_off()
        log_event("Gesture", "ALL DEVICES OFF")

    elif key == "peace":
        with home._lock:
            home.tv_on = not s["tv"]
        state = "ON" if not s["tv"] else "OFF"
        log_event("Gesture", f"TV {state}")

    elif key == "five":
        with home._lock:
            home.light_on = True
        log_event("Gesture", "Light ON")

    elif key == "zero":
        with home._lock:
            home.light_on = False
        log_event("Gesture", "Light OFF")

    elif key == "one":
        home.set_fan_speed(1)
        log_event("Gesture", "Fan Speed 1")

    elif key == "two":
        home.set_fan_speed(2)
        log_event("Gesture", "Fan Speed 2")

    elif key == "three":
        home.set_fan_speed(3)
        log_event("Gesture", "Fan Speed 3")

    elif key == "four":
        with home._lock:
            home.ac_on = not s["ac"]
        state = "ON" if not s["ac"] else "OFF"
        log_event("Gesture", f"AC {state}")

# ──────────────────────────────────────────────
#  Voice Command Parser
# ──────────────────────────────────────────────
VOICE_ALIASES = {
    # Light
    "light on":       lambda: setattr_thread(home, "light_on", True),
    "lights on":      lambda: setattr_thread(home, "light_on", True),
    "turn on light":  lambda: setattr_thread(home, "light_on", True),
    "turn on lights": lambda: setattr_thread(home, "light_on", True),
    "switch on light":lambda: setattr_thread(home, "light_on", True),
    "light off":      lambda: setattr_thread(home, "light_on", False),
    "lights off":     lambda: setattr_thread(home, "light_on", False),
    "turn off light": lambda: setattr_thread(home, "light_on", False),
    "turn off lights":lambda: setattr_thread(home, "light_on", False),
    # Fan on/off
    "fan on":         lambda: home.set_fan_speed(1),
    "start fan":      lambda: home.set_fan_speed(1),
    "turn on fan":    lambda: home.set_fan_speed(1),
    "fan off":        lambda: home.set_fan_speed(0),
    "stop fan":       lambda: home.set_fan_speed(0),
    "turn off fan":   lambda: home.set_fan_speed(0),
    # Fan speed
    "fan speed one":   lambda: home.set_fan_speed(1),
    "fan speed 1":     lambda: home.set_fan_speed(1),
    "fan speed two":   lambda: home.set_fan_speed(2),
    "fan speed 2":     lambda: home.set_fan_speed(2),
    "fan speed three": lambda: home.set_fan_speed(3),
    "fan speed 3":     lambda: home.set_fan_speed(3),
    "fan high":        lambda: home.set_fan_speed(3),
    "fan low":         lambda: home.set_fan_speed(1),
    "fan medium":      lambda: home.set_fan_speed(2),
    # AC
    "ac on":          lambda: setattr_thread(home, "ac_on", True),
    "air on":         lambda: setattr_thread(home, "ac_on", True),
    "turn on ac":     lambda: setattr_thread(home, "ac_on", True),
    "ac off":         lambda: setattr_thread(home, "ac_on", False),
    "air off":        lambda: setattr_thread(home, "ac_on", False),
    "turn off ac":    lambda: setattr_thread(home, "ac_on", False),
    # TV
    "tv on":          lambda: setattr_thread(home, "tv_on", True),
    "television on":  lambda: setattr_thread(home, "tv_on", True),
    "turn on tv":     lambda: setattr_thread(home, "tv_on", True),
    "tv off":         lambda: setattr_thread(home, "tv_on", False),
    "television off": lambda: setattr_thread(home, "tv_on", False),
    "turn off tv":    lambda: setattr_thread(home, "tv_on", False),
    # All
    "all on":         home.all_on,
    "everything on":  home.all_on,
    "all off":        home.all_off,
    "everything off": home.all_off,
    "turn everything off": home.all_off,
    "turn everything on":  home.all_on,
    "goodnight":      home.all_off,
    "good night":     home.all_off,
}

def setattr_thread(obj, attr, val):
    """Thread-safe single-attribute setter."""
    with obj._lock:
        setattr(obj, attr, val)

def parse_voice_command(text: str) -> bool:
    """
    Try every alias (longest match first) against the recognised text.
    Returns True if a command was executed.
    """
    text = text.lower().strip()
    # sort by length descending → prefer more-specific phrases
    for phrase in sorted(VOICE_ALIASES.keys(), key=len, reverse=True):
        if phrase in text:
            VOICE_ALIASES[phrase]()
            log_event("Voice", f'"{text}" → {phrase}')
            return True
    log_event("Voice", f'Not understood: "{text}"')
    return False

# ──────────────────────────────────────────────
#  Non-blocking Voice Listener Thread
# ──────────────────────────────────────────────
voice_active = threading.Event()   # set() to trigger a single listen cycle
voice_result = {"text": "", "ts": 0}

def voice_worker():
    while True:
        voice_active.wait()        # block until triggered
        voice_active.clear()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                print("Listening…")
                audio = recognizer.listen(source, timeout=4, phrase_time_limit=5)
            text = recognizer.recognize_google(audio).lower()
            voice_result["text"] = text
            voice_result["ts"]   = time.time()
            parse_voice_command(text)
        except sr.WaitTimeoutError:
            log_event("Voice", "Timeout — no speech detected")
        except sr.UnknownValueError:
            log_event("Voice", "Could not understand audio")
        except Exception as e:
            log_event("Voice", f"Error: {e}")

voice_thread = threading.Thread(target=voice_worker, daemon=True)
voice_thread.start()

# ──────────────────────────────────────────────
#  HUD Rendering Helpers
# ──────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

C_ON    = (80, 255, 120)    # green
C_OFF   = (60, 60, 220)     # red-ish
C_GOLD  = (30, 210, 255)    # amber/gold
C_CYAN  = (255, 220, 0)     # cyan
C_PANEL = (15, 15, 15)      # near-black panel
C_WHITE = (240, 240, 240)


def draw_panel(img, x, y, w, h, alpha=0.55):
    """Semi-transparent dark rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_PANEL, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_device_row(img, x, y, label: str, is_on: bool, detail: str = ""):
    indicator = "*" if is_on else "-"
    color     = C_ON if is_on else C_OFF
    status    = detail if detail else ("ON" if is_on else "OFF")
    cv2.putText(img, f"{indicator} {label:<8} {status}", (x, y),
                FONT_SMALL, 0.62, color, 2)


def draw_hud(frame, state: dict, fingers_list: list[int], listening: bool):
    H, W = frame.shape[:2]

    # ── Device Status Panel (top-left) ──────────────────────
    draw_panel(frame, 10, 10, 250, 145)
    cv2.putText(frame, "SMART HOME", (20, 38), FONT, 0.7, C_GOLD, 2)
    fan_detail = f"SPD {state['fan_speed']}" if state["fan_on"] else "OFF"
    draw_device_row(frame, 22,  65, "Light",  state["light"],   "")
    draw_device_row(frame, 22,  90, "Fan",    state["fan_on"],  fan_detail)
    draw_device_row(frame, 22, 115, "AC",     state["ac"],      "")
    draw_device_row(frame, 22, 140, "TV",     state["tv"],      "")

    # ── Gesture Legend (top-right) ───────────────────────────
    legend = [
        ("0 fingers", "Light OFF"),
        ("1 finger ", "Fan Spd 1"),
        ("2 fingers", "Fan Spd 2"),
        ("3 fingers", "Fan Spd 3"),
        ("4 fingers", "AC Toggle"),
        ("5 fingers", "Light ON"),
        ("* Peace " , "TV Toggle"),
        ("* Both  0", "All OFF"),
        ("* Both 10", "All ON"),
    ]
    panel_w = 230
    panel_h = 30 + len(legend) * 23
    draw_panel(frame, W - panel_w - 10, 10, panel_w, panel_h)
    cv2.putText(frame, "GESTURES", (W - panel_w, 32), FONT, 0.6, C_GOLD, 2)
    for i, (g, desc) in enumerate(legend):
        cv2.putText(frame, f"{g}  {desc}",
                    (W - panel_w, 55 + i * 23),
                    FONT_SMALL, 0.5, C_WHITE, 1)

    # ── Finger Count Badge ───────────────────────────────────
    total_fingers = sum(fingers_list)
    badge_txt = f"{'|'.join(str(f) for f in fingers_list) if fingers_list else '-'}"
    cv2.putText(frame, f"Fingers: {badge_txt}", (20, H - 60),
                FONT_SMALL, 0.9,
                C_ON if total_fingers > 0 else C_OFF, 2)

    # ── Voice Button Prompt ──────────────────────────────────
    v_color = (0, 180, 255) if listening else C_CYAN
    v_label = "  LISTENING…" if listening else "  [V] Voice Command"
    cv2.putText(frame, v_label, (20, H - 30), FONT_SMALL, 0.7, v_color, 2)

    # ── Command Log (bottom-right) ───────────────────────────
    log_entries = list(log)
    log_h = 12 + len(log_entries) * 20
    draw_panel(frame, W - 420, H - log_h - 30, 410, log_h)
    for i, entry in enumerate(reversed(log_entries)):
        alpha_col = max(60, 240 - i * 30)
        cv2.putText(frame, entry[-55:],         # truncate long lines
                    (W - 415, H - 40 - i * 20),
                    FONT_SMALL, 0.42,
                    (alpha_col, alpha_col, alpha_col), 1)

    # ── ESC hint ────────────────────────────────────────────
    cv2.putText(frame, "[ESC] Quit", (W - 130, H - 10),
                FONT_SMALL, 0.5, C_GOLD, 1)


# ──────────────────────────────────────────────
#  Main Loop
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

log_event("System", "Started press V for voice, ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)          # mirror

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_det.process(rgb)

    fingers_list  = []   # one entry per detected hand
    gesture_keys  = []   # candidate keys this frame

    # ── Hand Processing ──────────────────────────────────────
    if result.multi_hand_landmarks:
        hand_data = list(zip(result.multi_hand_landmarks,
                             result.multi_handedness))

        for lms, handedness in hand_data:
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS,
                                   HAND_STYLE, CONN_STYLE)
            label   = handedness.classification[0].label   # 'Left' or 'Right'
            count   = count_fingers(lms, label)
            fingers_list.append(count)

            # Individual-hand gesture keys
            if is_peace_sign(lms):
                gesture_keys.append("peace")
            elif count == 0:
                gesture_keys.append("zero")
            elif count == 1:
                gesture_keys.append("one")
            elif count == 2:
                gesture_keys.append("two")
            elif count == 3:
                gesture_keys.append("three")
            elif count == 4:
                gesture_keys.append("four")
            elif count == 5:
                gesture_keys.append("five")

        # ── Two-hand composite gestures ──────────────────────
        if len(fingers_list) == 2:
            a, b = sorted(fingers_list)
            if a == 0 and b == 0:
                gesture_keys = ["two_all_off"]
            elif a == 5 and b == 5:
                gesture_keys = ["two_all_on"]

    # ── Debounce & Fire ──────────────────────────────────────
    composite_key = "_".join(sorted(gesture_keys)) if gesture_keys else "none"
    if gesture_keys and debounce_gesture(composite_key):
        # Use the first (or most significant) key
        primary = gesture_keys[0] if len(gesture_keys) == 1 else composite_key
        apply_gesture(primary, fingers_list)
    elif not gesture_keys:
        debounce_gesture("none")   # keep buffer honest

    # ── Draw HUD ─────────────────────────────────────────────
    state     = home.snapshot()
    listening = voice_active.is_set()
    draw_hud(frame, state, fingers_list, listening)

    cv2.imshow("Smart Home HCI — Enhanced", frame)

    # ── Key Handling ─────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('v') or key == ord('V'):
        if not voice_active.is_set():
            voice_active.set()           # trigger background listener

    elif key == 27:                      # ESC
        break

# ──────────────────────────────────────────────
#  Cleanup
# ──────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("Session ended.")