"""
╔══════════════════════════════════════════════════════════════════╗
║         ENHANCED SMART HOME HCI  Gesture + Voice Control        ║
║  Devices  : Light (on/off) | Fan (speed 1-3) | AC | TV          ║
║  Gestures : 0=Light OFF | 1=Fan Spd1 | 2=Fan Spd2 | 3=Fan Spd3 ║
║             4=AC Toggle  | 5=Light ON | Peace=TV Toggle         ║
║             Two hands: (5+5)=ALL ON | (0+0)=ALL OFF             ║
║  Voice    : Vosk OFFLINE STT  no internet required             ║
╚══════════════════════════════════════════════════════════════════╝

Install dependencies:
    pip install opencv-python mediapipe pyaudio vosk

Download a Vosk model from: https://alphacephei.com/vosk/models
Recommended: vosk-model-small-en-us-0.15  (40 MB, fast)
Place the model folder next to this script and set MODEL_PATH below.
"""

# ── Imports ──────────────────────────────────────────────────────
import cv2                              # OpenCV: camera capture, image drawing, window display
import mediapipe as mp                  # MediaPipe: real-time ML hand landmark detection
import pyaudio                          # PyAudio: low-level microphone audio stream (raw PCM)
import threading                        # threading: run voice listener in a non-blocking daemon thread
import time                             # time: wall-clock timestamps for gesture cooldown comparisons
import json                             # json: parse Vosk result dictionaries from JSON strings
from collections import deque           # deque: fixed-size queue for debounce buffer and on-screen log
from datetime import datetime           # datetime: human-readable HH:MM:SS timestamps in the log
from vosk import Model, KaldiRecognizer # Vosk: fully offline speech-to-text engine (no internet needed)

# ──────────────────────────────────────────────────────────────────
#  SECTION 1  Vosk Offline Speech Recognition Setup
#
#  Why Vosk instead of Google STT?
#    • Works 100% offline  no API key, no internet dependency
#    • Lower latency (no network round-trip)
#    • Vocabulary restriction  fewer false-positives for our commands
#    • KaldiRecognizer streams audio chunk-by-chunk (no blocking listen)
# ──────────────────────────────────────────────────────────────────
MODEL_PATH = "ch9\Smart_Home_Voice_Gesture_HCI_Project\model"                    # Path to the Vosk model folder (relative to this script)
                                        # Download from: https://alphacephei.com/vosk/models

# Build the restricted vocabulary list  only the words our commands need.
# Vosk will only recognise words in this list  greatly reduces false matches.
VOCAB = json.dumps([
    "light", "lights", "on", "off",     # Light commands
    "fan", "start", "stop",             # Fan on/off
    "speed", "one", "two", "three",     # Fan speed words
    "high", "medium", "low",            # Fan speed shorthands
    "air", "ac",                        # AC commands
    "tv", "television",                 # TV commands
    "turn", "switch",                   # Verb prefixes
    "all", "everything",                # Bulk commands
    "good", "night", "goodnight",       # Goodnight shortcut
])

vosk_model = Model(MODEL_PATH)          # Load acoustic model from disk (happens once at startup)
vosk_rec   = KaldiRecognizer(           # Create a streaming recogniser bound to the loaded model
    vosk_model,                         # The acoustic model to use for inference
    16000,                              # Sample rate in Hz  must match PyAudio stream below
    VOCAB,                              # Restrict recognition to this JSON vocabulary list
)

# ── PyAudio microphone stream ──────────────────────────────────────
CHUNK      = 4096                       # Audio buffer size in frames per read (larger = fewer reads)
SAMPLE_RATE = 16000                     # 16 kHz  standard for Vosk models
pa         = pyaudio.PyAudio()          # Initialise the PyAudio interface to the OS audio layer
mic_stream = pa.open(                   # Open a raw PCM microphone input stream
    format   = pyaudio.paInt16,         # 16-bit signed integer samples (required by Vosk)
    channels = 1,                       # Mono audio (Vosk does not use stereo)
    rate     = SAMPLE_RATE,             # Must match the rate used to initialise KaldiRecognizer
    input    = True,                    # This is an input (microphone) stream, not output
    frames_per_buffer = CHUNK,          # How many frames to buffer before read() returns
)
mic_stream.start_stream()               # Begin capturing audio from the microphone immediately

# ──────────────────────────────────────────────────────────────────
#  SECTION 2  MediaPipe Hand Detection Setup
# ──────────────────────────────────────────────────────────────────
mp_hands  = mp.solutions.hands              # Access the MediaPipe Hands solution module
hands_det = mp_hands.Hands(                 # Create the hand landmark detector object
    max_num_hands=2,                        # Detect up to 2 hands  enables two-hand composite gestures
    min_detection_confidence=0.75,          # Minimum confidence to start tracking a new hand (0–1)
    min_tracking_confidence=0.6,            # Minimum confidence to continue tracking an existing hand
)
mp_draw   = mp.solutions.drawing_utils     # Utility that draws the hand skeleton on the camera frame

# Custom colours for the hand skeleton overlay (OpenCV uses BGR, not RGB)
HAND_STYLE = mp_draw.DrawingSpec(color=(0, 220, 255), thickness=2, circle_radius=4)  # Cyan joint dots
CONN_STYLE = mp_draw.DrawingSpec(color=(0, 120, 255), thickness=2)                   # Blue bone lines

# ──────────────────────────────────────────────────────────────────
#  SECTION 3  SmartHome State Class
#  All device states live in one place. A threading.Lock prevents
#  race conditions when the voice thread writes simultaneously with
#  the gesture loop running on the main thread.
# ──────────────────────────────────────────────────────────────────
class SmartHome:
    def __init__(self):
        self.light_on  = False          # Light starts OFF
        self.fan_speed = 0              # Fan starts OFF (0=OFF, 1=slow, 2=medium, 3=fast)
        self.ac_on     = False          # AC starts OFF
        self.tv_on     = False          # TV starts OFF
        self._lock     = threading.Lock()   # Mutex: only one thread modifies state at a time

    @property
    def fan_on(self):
        return self.fan_speed > 0       # Computed read-only property: True when fan is running

    def all_on(self):
        with self._lock:                # Acquire lock  prevents concurrent writes from voice thread
            self.light_on  = True       # Light  ON
            self.fan_speed = 3          # Fan  maximum speed
            self.ac_on     = True       # AC  ON
            self.tv_on     = True       # TV  ON

    def all_off(self):
        with self._lock:                # Acquire lock for thread safety
            self.light_on  = False      # Light  OFF
            self.fan_speed = 0          # Fan  OFF (speed 0)
            self.ac_on     = False      # AC  OFF
            self.tv_on     = False      # TV  OFF

    def set_fan_speed(self, speed: int):
        with self._lock:                            # Acquire lock
            self.fan_speed = max(0, min(3, speed))  # Clamp 0–3: prevents invalid values

    def snapshot(self):
        """Return a thread-safe dictionary copy of all current device states."""
        with self._lock:                # Lock before reading to prevent torn reads
            return {
                "light":     self.light_on,     # Current light state (True/False)
                "fan_speed": self.fan_speed,    # Current fan speed (0-3)
                "fan_on":    self.fan_speed > 0,# Convenience on/off boolean
                "ac":        self.ac_on,        # Current AC state (True/False)
                "tv":        self.tv_on,        # Current TV state (True/False)
            }

home = SmartHome()  # Single global instance shared by gesture loop and voice thread

# ──────────────────────────────────────────────────────────────────
#  SECTION 4  Command Log
#  Keeps the last MAX_LOG entries for the on-screen HUD panel.
# ──────────────────────────────────────────────────────────────────
MAX_LOG = 6                                 # Maximum visible log lines on screen at once
log: deque = deque(maxlen=MAX_LOG)          # deque auto-drops the oldest entry when full

def log_event(src: str, msg: str):
    """Append a timestamped entry to the rotating log and echo it to the terminal."""
    ts    = datetime.now().strftime("%H:%M:%S")     # Format current time as HH:MM:SS
    entry = f"[{ts}] {src}: {msg}"                 # Compose the full log string
    log.append(entry)                               # Add to deque (oldest removed when full)
    print(entry)                                    # Echo to terminal for debugging

# ──────────────────────────────────────────────────────────────────
#  SECTION 5  Gesture Debounce System
#
#  Problem: hand position fluctuates frame-to-frame  false triggers.
#  Solution: require the SAME gesture key for DEBOUNCE_FRAMES
#  consecutive frames before acting.  Then enforce COOLDOWN_SECONDS
#  gap before the same gesture can fire again.
# ──────────────────────────────────────────────────────────────────
DEBOUNCE_FRAMES  = 8        # Frames gesture must be stable before firing (~0.25 s at 30 fps)
COOLDOWN_SECONDS = 1.0      # Minimum seconds between identical gesture fires

gesture_buffer   = deque(maxlen=DEBOUNCE_FRAMES)    # Sliding window of last N gesture keys
last_gesture_ts  = 0.0                              # Timestamp of the last fired gesture
last_gesture_key = None                             # Key of the last fired gesture

def debounce_gesture(key: str) -> bool:
    """
    Push this frame's gesture key into the buffer.
    Return True (fire!) only when:
      1. All DEBOUNCE_FRAMES slots hold the same key (gesture is stable), AND
      2. The key differs from the last fired gesture OR the cooldown has elapsed.
    """
    global last_gesture_ts, last_gesture_key
    gesture_buffer.append(key)                      # Record this frame's key in the sliding window

    if len(gesture_buffer) < DEBOUNCE_FRAMES:       # Buffer not yet full  too early to decide
        return False
    if len(set(gesture_buffer)) != 1:               # Mixed keys  gesture is still unstable
        return False

    now = time.time()                               # Current wall-clock time in seconds
    if key == last_gesture_key and (now - last_gesture_ts) < COOLDOWN_SECONDS:
        return False                                # Same gesture is in cooldown  suppress it

    last_gesture_ts  = now                          # Record when this gesture fired
    last_gesture_key = key                          # Record which gesture fired
    return True                                     # All checks passed  caller should fire it

# ──────────────────────────────────────────────────────────────────
#  SECTION 6  Finger Counting
# ──────────────────────────────────────────────────────────────────
FINGERTIP_IDS = [4, 8, 12, 16, 20]     # MediaPipe landmark indices: thumb tip  pinky tip

def count_fingers(lms, handedness_label: str) -> int:
    """
    Count extended fingers (0–5) using MediaPipe landmark positions.
    Thumb: compared on X-axis (moves left/right); direction depends on hand.
    Index–Pinky: compared on Y-axis (tip above pip2 joint = extended).
    Note: frame is horizontally flipped so 'Left'/'Right' labels are swapped.
    """
    fingers = []                                        # 1=extended, 0=folded per finger

    # Thumb: compare tip (landmark 4) vs IP joint (landmark 3) on the X-axis
    tip, pip = lms.landmark[4], lms.landmark[3]
    if handedness_label == "Left":                      # After flip: 'Left' = user's right hand
        fingers.append(1 if tip.x > pip.x else 0)      # Right hand: tip is to the RIGHT of pip
    else:                                               # 'Right' label = user's left hand after flip
        fingers.append(1 if tip.x < pip.x else 0)      # Left hand: tip is to the LEFT of pip

    # Index  Pinky: compare tip Y vs the knuckle 2 joints below on the Y-axis
    for i in range(1, 5):                               # Loop over fingers 1–4 (index to pinky)
        tip_y = lms.landmark[FINGERTIP_IDS[i]].y       # Y-coordinate of this finger's tip
        pip_y = lms.landmark[FINGERTIP_IDS[i] - 2].y   # Y-coordinate of knuckle 2 joints below
        fingers.append(1 if tip_y < pip_y else 0)       # Tip above pip  finger is extended

    return sum(fingers)                                 # Total extended fingers (0–5)


def is_peace_sign(lms) -> bool:
    """Detect peace/victory sign: index+middle extended, ring+pinky folded  TV toggle."""
    idx_up  = lms.landmark[8].y  < lms.landmark[6].y   # Index tip above middle joint  extended
    mid_up  = lms.landmark[12].y < lms.landmark[10].y  # Middle tip above middle joint  extended
    rng_dwn = lms.landmark[16].y > lms.landmark[14].y  # Ring tip below middle joint  folded
    pnk_dwn = lms.landmark[20].y > lms.landmark[18].y  # Pinky tip below middle joint  folded
    return idx_up and mid_up and rng_dwn and pnk_dwn   # All four conditions must be True

# ──────────────────────────────────────────────────────────────────
#  SECTION 7  Gesture  Device Command Mapping
# ──────────────────────────────────────────────────────────────────
def apply_gesture(key: str, counts: list):
    """Execute the device command corresponding to the confirmed gesture key."""
    s = home.snapshot()                         # Thread-safe read (used for toggle logic)

    if key == "two_all_on":                     # Both hands showing 5 fingers
        home.all_on()
        log_event("Gesture", "ALL DEVICES ON")

    elif key == "two_all_off":                  # Both hands showing 0 fingers (fists)
        home.all_off()
        log_event("Gesture", "ALL DEVICES OFF")

    elif key == "peace":                        # Peace/victory sign  toggle TV
        with home._lock:
            home.tv_on = not s["tv"]            # Flip TV state (ONOFF or OFFON)
        state = "ON" if not s["tv"] else "OFF"
        log_event("Gesture", f"TV {state}")

    elif key == "five":                         # Open hand (all 5 fingers)  Light ON
        with home._lock:
            home.light_on = True
        log_event("Gesture", "Light ON")

    elif key == "zero":                         # Closed fist (0 fingers)  Light OFF
        with home._lock:
            home.light_on = False
        log_event("Gesture", "Light OFF")

    elif key == "one":                          # 1 finger  Fan speed 1 (slow)
        home.set_fan_speed(1)
        log_event("Gesture", "Fan Speed 1")

    elif key == "two":                          # 2 fingers  Fan speed 2 (medium)
        home.set_fan_speed(2)
        log_event("Gesture", "Fan Speed 2")

    elif key == "three":                        # 3 fingers  Fan speed 3 (fast)
        home.set_fan_speed(3)
        log_event("Gesture", "Fan Speed 3")

    elif key == "four":                         # 4 fingers  toggle AC
        with home._lock:
            home.ac_on = not s["ac"]
        state = "ON" if not s["ac"] else "OFF"
        log_event("Gesture", f"AC {state}")

# ──────────────────────────────────────────────────────────────────
#  SECTION 8  Voice Command Alias Table + Parser
#  Maps spoken phrases  functions that modify device state.
#  Longest-match: sort by phrase length (desc) so "turn on lights"
#  beats "lights on" when both appear in the recognised text.
# ──────────────────────────────────────────────────────────────────
def setattr_thread(obj, attr, val):
    """Thread-safe single-attribute setter using the object's own lock."""
    with obj._lock:
        setattr(obj, attr, val)         # Write attribute by name while holding the mutex

VOICE_ALIASES = {
    # ── Light ───────────────────────────────────────────────────
    "light on":            lambda: setattr_thread(home, "light_on", True),   #  Light ON
    "lights on":           lambda: setattr_thread(home, "light_on", True),   # plural form
    "turn on light":       lambda: setattr_thread(home, "light_on", True),   # verb form
    "turn on lights":      lambda: setattr_thread(home, "light_on", True),   # verb + plural
    "switch on light":     lambda: setattr_thread(home, "light_on", True),   # alternative verb
    "light off":           lambda: setattr_thread(home, "light_on", False),  #  Light OFF
    "lights off":          lambda: setattr_thread(home, "light_on", False),  # plural form
    "turn off light":      lambda: setattr_thread(home, "light_on", False),  # verb form
    "turn off lights":     lambda: setattr_thread(home, "light_on", False),  # verb + plural
    # ── Fan on/off ──────────────────────────────────────────────
    "fan on":              lambda: home.set_fan_speed(1),    # Start fan at speed 1
    "start fan":           lambda: home.set_fan_speed(1),    # Alternative phrase
    "turn on fan":         lambda: home.set_fan_speed(1),    # Verb form
    "fan off":             lambda: home.set_fan_speed(0),    # Stop fan
    "stop fan":            lambda: home.set_fan_speed(0),    # Alternative phrase
    "turn off fan":        lambda: home.set_fan_speed(0),    # Verb form
    # ── Fan speed ───────────────────────────────────────────────
    "fan speed one":       lambda: home.set_fan_speed(1),    # Slow (word)
    "fan speed 1":         lambda: home.set_fan_speed(1),    # Slow (digit)
    "fan speed two":       lambda: home.set_fan_speed(2),    # Medium (word)
    "fan speed 2":         lambda: home.set_fan_speed(2),    # Medium (digit)
    "fan speed three":     lambda: home.set_fan_speed(3),    # Fast (word)
    "fan speed 3":         lambda: home.set_fan_speed(3),    # Fast (digit)
    "fan high":            lambda: home.set_fan_speed(3),    # Shorthand: high = speed 3
    "fan low":             lambda: home.set_fan_speed(1),    # Shorthand: low  = speed 1
    "fan medium":          lambda: home.set_fan_speed(2),    # Shorthand: medium = speed 2
    # ── AC ──────────────────────────────────────────────────────
    "ac on":               lambda: setattr_thread(home, "ac_on", True),    #  AC ON
    "air on":              lambda: setattr_thread(home, "ac_on", True),    # Informal "air"
    "turn on ac":          lambda: setattr_thread(home, "ac_on", True),    # Verb form
    "ac off":              lambda: setattr_thread(home, "ac_on", False),   #  AC OFF
    "air off":             lambda: setattr_thread(home, "ac_on", False),   # Informal "air"
    "turn off ac":         lambda: setattr_thread(home, "ac_on", False),   # Verb form
    # ── TV ──────────────────────────────────────────────────────
    "tv on":               lambda: setattr_thread(home, "tv_on", True),    #  TV ON
    "television on":       lambda: setattr_thread(home, "tv_on", True),    # Full-word variant
    "turn on tv":          lambda: setattr_thread(home, "tv_on", True),    # Verb form
    "tv off":              lambda: setattr_thread(home, "tv_on", False),   #  TV OFF
    "television off":      lambda: setattr_thread(home, "tv_on", False),   # Full-word variant
    "turn off tv":         lambda: setattr_thread(home, "tv_on", False),   # Verb form
    # ── All devices ─────────────────────────────────────────────
    "all on":              home.all_on,               # Everything ON
    "everything on":       home.all_on,               # Alternative phrase
    "all off":             home.all_off,              # Everything OFF
    "everything off":      home.all_off,              # Alternative phrase
    "turn everything off": home.all_off,              # Verb form
    "turn everything on":  home.all_on,               # Verb form
    "goodnight":           home.all_off,              # Bedtime shortcut  all OFF
    "good night":          home.all_off,              # Two-word variant
}

def parse_voice_command(text: str) -> bool:
    """
    Match recognised speech against VOICE_ALIASES (longest phrase first).
    Returns True if a command was found and executed.
    """
    text = text.lower().strip()                                     # Normalise: lowercase + trim
    for phrase in sorted(VOICE_ALIASES.keys(), key=len, reverse=True):  # Longest first
        if phrase in text:                                          # Substring match in the text
            VOICE_ALIASES[phrase]()                                 # Execute the mapped function
            log_event("Voice", f'"{text}"  {phrase}')             # Log matched phrase
            return True                                             # Stop at first match
    log_event("Voice", f'Not understood: "{text}"')                # No alias matched
    return False

# ──────────────────────────────────────────────────────────────────
#  SECTION 9  Non-blocking Vosk Voice Listener Thread
#
#  How it works:
#    1. Press V  voice_active.set() wakes the daemon thread.
#    2. Thread reads raw PCM chunks from mic_stream (already open).
#    3. Each chunk is fed to KaldiRecognizer.AcceptWaveform().
#    4. AcceptWaveform() returns True when it has a FINAL result
#       (end-of-utterance detected by the Vosk VAD).
#    5. We parse the JSON result and call parse_voice_command().
#    6. A timeout stops listening if no speech starts within 5 s.
#
#  Why better than SpeechRecognition + Google?
#    • No internet required  runs entirely on-device.
#    • No API key, no quota, no cost.
#    • Lower latency  no network round-trip.
#    • Restricted vocabulary  far fewer false-positive matches.
#    • Camera never freezes  thread reads chunks asynchronously.
# ──────────────────────────────────────────────────────────────────
LISTEN_TIMEOUT = 5.0            # Maximum seconds to wait for speech before giving up
voice_active   = threading.Event()  # Semaphore: set() = start listening, clear() = done
voice_result   = {"text": "", "ts": 0}  # Shared dict: last recognised text + timestamp

def voice_worker():
    """
    Background daemon: waits for trigger  streams Vosk recognition  parses  repeats.
    Reads raw PCM from mic_stream that is already open  no blocking microphone open/close.
    """
    while True:                                         # Run forever (daemon exits with main)
        voice_active.wait()                             # Block here until main loop calls .set()
        voice_active.clear()                            # Clear flag immediately (one-shot cycle)

        log_event("Voice", "Listening (Vosk)")
        vosk_rec.Reset()                                # Clear any leftover audio from previous listen

        deadline = time.time() + LISTEN_TIMEOUT        # Latest time we'll keep listening
        recognised = False                              # Track whether we got a valid result

        try:
            while time.time() < deadline:               # Keep reading until timeout expires
                # Read one chunk of raw 16-bit PCM audio from the always-open mic stream
                data = mic_stream.read(CHUNK, exception_on_overflow=False)
                # Feed the chunk to Vosk; returns True when end-of-utterance is detected
                if vosk_rec.AcceptWaveform(data):
                    # Final result is ready  parse the JSON and extract the recognised text
                    result_json = json.loads(vosk_rec.Result())     # e.g. {"text": "light on"}
                    text = result_json.get("text", "").strip()      # Extract "text" field safely
                    if text:                                        # Non-empty recognition?
                        voice_result["text"] = text                 # Store for external access
                        voice_result["ts"]   = time.time()          # Record when recognised
                        parse_voice_command(text)                   # Match against alias table
                        recognised = True                           # Signal: we got something
                        break                                       # Stop reading after first result

            if not recognised:
                # Grab any partial result still in the buffer (partial can hold last words)
                partial_json = json.loads(vosk_rec.PartialResult()) # e.g. {"partial": "fan"}
                partial_text = partial_json.get("partial", "").strip()
                if partial_text:                                    # Something was partially heard
                    log_event("Voice", f"Partial: '{partial_text}'  retrying full match")
                    parse_voice_command(partial_text)               # Try to match the partial too
                else:
                    log_event("Voice", "Timeout  no speech detected")  # Nothing heard at all

        except OSError as e:
            log_event("Voice", f"Mic stream error: {e}")            # Stream closed unexpectedly

# Create and start the daemon thread  it blocks immediately at voice_active.wait()
voice_thread = threading.Thread(target=voice_worker, daemon=True)
voice_thread.start()                # Thread starts and immediately sleeps until triggered

# ──────────────────────────────────────────────────────────────────
#  SECTION 10  HUD Rendering Helpers
# ──────────────────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_DUPLEX    # Heavier font for panel headers
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX  # Lighter font for body text and small labels

# Colours in OpenCV BGR format (Blue-Green-Red, NOT RGB)
C_ON    = (80, 255, 120)    # Bright green   device is ON
C_OFF   = (60, 60, 220)     # Red            device is OFF
C_GOLD  = (30, 210, 255)    # Amber / gold   panel headers
C_CYAN  = (255, 220, 0)     # Cyan           voice prompt (idle)
C_PANEL = (15, 15, 15)      # Near-black     semi-transparent panel background
C_WHITE = (240, 240, 240)   # Near-white     legend text

def draw_panel(img, x, y, w, h, alpha=0.55):
    """Draw a semi-transparent dark rectangle using alpha blending."""
    overlay = img.copy()                                            # Copy current frame
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_PANEL, -1)   # Filled rect on copy
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)         # Blend copy into frame

def draw_device_row(img, x, y, label, is_on, detail=""):
    """Draw one device status row: symbol + label + state text."""
    indicator = "*" if is_on else "-"                               # * = ON,  - = OFF
    color     = C_ON if is_on else C_OFF                            # Green or red
    status    = detail if detail else ("ON" if is_on else "OFF")    # Custom or default text
    cv2.putText(img, f"{indicator} {label:<8} {status}", (x, y), FONT_SMALL, 0.62, color, 2)

def draw_hud(frame, state, fingers_list, listening):
    """Render all four HUD panels onto the camera frame."""
    H, W = frame.shape[:2]             # Frame height and width for relative positioning

    # Panel 1: Device status (top-left)
    draw_panel(frame, 10, 10, 250, 145)
    cv2.putText(frame, "SMART HOME", (20, 38), FONT, 0.7, C_GOLD, 2)   # Header
    fan_detail = f"SPD {state['fan_speed']}" if state["fan_on"] else "OFF"  # Fan speed label
    draw_device_row(frame, 22,  65, "Light",  state["light"],  "")
    draw_device_row(frame, 22,  90, "Fan",    state["fan_on"], fan_detail)
    draw_device_row(frame, 22, 115, "AC",     state["ac"],     "")
    draw_device_row(frame, 22, 140, "TV",     state["tv"],     "")

    # Panel 2: Gesture legend (top-right)
    legend = [
        ("0 fingers", "Light OFF"), ("1 finger ", "Fan Spd 1"), ("2 fingers", "Fan Spd 2"),
        ("3 fingers", "Fan Spd 3"), ("4 fingers", "AC Toggle"), ("5 fingers", "Light ON"),
        ("* Peace  ", "TV Toggle"), ("* Both  0", "All OFF"),   ("* Both  5", "All ON"),
    ]
    panel_w = 230
    panel_h = 30 + len(legend) * 23
    draw_panel(frame, W - panel_w - 10, 10, panel_w, panel_h)
    cv2.putText(frame, "GESTURES", (W - panel_w, 32), FONT, 0.6, C_GOLD, 2)
    for i, (g, desc) in enumerate(legend):
        cv2.putText(frame, f"{g}  {desc}", (W - panel_w, 55 + i * 23), FONT_SMALL, 0.5, C_WHITE, 1)

    # Panel 3: Finger count + voice prompt (bottom-left)
    badge = "|".join(str(f) for f in fingers_list) if fingers_list else "-"  # e.g. "3|2"
    cv2.putText(frame, f"Fingers: {badge}", (20, H - 60), FONT_SMALL, 0.9,
                C_ON if sum(fingers_list) > 0 else C_OFF, 2)
    v_color = (0, 180, 255) if listening else C_CYAN    # Orange while listening, cyan when idle
    # Show engine label so user knows it's offline
    v_label = "  LISTENING (Vosk)..." if listening else "  [V] Voice Command (Offline)"
    cv2.putText(frame, v_label, (20, H - 30), FONT_SMALL, 0.7, v_color, 2)

    # Panel 4: Command log (bottom-right, newest entry at bottom)
    log_entries = list(log)
    log_h = 12 + len(log_entries) * 20
    draw_panel(frame, W - 420, H - log_h - 30, 410, log_h)
    for i, entry in enumerate(reversed(log_entries)):
        fade = max(60, 240 - i * 30)           # Older entries fade from 240  60
        cv2.putText(frame, entry[-55:], (W - 415, H - 40 - i * 20),
                    FONT_SMALL, 0.42, (fade, fade, fade), 1)

    cv2.putText(frame, "[ESC] Quit", (W - 130, H - 10), FONT_SMALL, 0.5, C_GOLD, 1)

# ──────────────────────────────────────────────────────────────────
#  SECTION 11  Camera Initialisation
# ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)               # Open default system camera (index 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)    # Request 1280px wide frames (720p)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)     # Request 720px tall frames

log_event("System", "Started  press V for voice (Vosk offline), ESC to quit")

# ──────────────────────────────────────────────────────────────────
#  SECTION 12  Main Loop
#  Runs at ~30 fps. Each iteration:
#    1. Capture frame  flip (mirror)  convert BGRRGB
#    2. MediaPipe: detect hands  draw skeleton
#    3. Classify gestures  run debounce  fire device command
#    4. Draw HUD overlay  display window
#    5. Check keyboard: V = trigger voice, ESC = quit
# ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()             # Grab next camera frame; ret=False on failure
    if not ret:                         # Camera disconnected or stream ended
        break

    frame = cv2.flip(frame, 1)          # Horizontal mirror  feels natural (like looking in a mirror)

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # MediaPipe needs RGB; OpenCV gives BGR
    result = hands_det.process(rgb)     # Run hand landmark detection on the converted frame

    fingers_list = []                   # Reset: one finger count per detected hand this frame
    gesture_keys = []                   # Reset: gesture key strings for this frame

    # ── Hand Processing ──────────────────────────────────────
    if result.multi_hand_landmarks:                         # At least one hand was detected
        hand_data = list(zip(result.multi_hand_landmarks, result.multi_handedness))

        for lms, handedness in hand_data:                   # Iterate over each detected hand
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS, HAND_STYLE, CONN_STYLE)
            label = handedness.classification[0].label      # 'Left' or 'Right' from MediaPipe
            count = count_fingers(lms, label)               # Count extended fingers (0-5)
            fingers_list.append(count)                      # Record for HUD display

            # Map finger count / special gesture to a canonical string key
            if is_peace_sign(lms):          gesture_keys.append("peace")  # Peace before count==2
            elif count == 0:                gesture_keys.append("zero")
            elif count == 1:                gesture_keys.append("one")
            elif count == 2:                gesture_keys.append("two")
            elif count == 3:                gesture_keys.append("three")
            elif count == 4:                gesture_keys.append("four")
            elif count == 5:                gesture_keys.append("five")

        # ── Two-hand composite override ───────────────────────
        if len(fingers_list) == 2:
            a, b = sorted(fingers_list)             # Sort so 0+5 == 5+0 (order independent)
            if a == 0 and b == 0:   gesture_keys = ["two_all_off"]     # Both fists  ALL OFF
            elif a == 5 and b == 5: gesture_keys = ["two_all_on"]      # Both open  ALL ON

    # ── Debounce & Fire ──────────────────────────────────────
    composite_key = "_".join(sorted(gesture_keys)) if gesture_keys else "none"
    if gesture_keys and debounce_gesture(composite_key):    # Stable AND cooldown passed?
        primary = gesture_keys[0] if len(gesture_keys) == 1 else composite_key
        apply_gesture(primary, fingers_list)                # Execute the device command
    elif not gesture_keys:
        debounce_gesture("none")                            # Keep buffer honest/aligned in time

    # ── Draw HUD + Display ───────────────────────────────────
    state     = home.snapshot()                     # Thread-safe copy of device states
    listening = voice_active.is_set()               # True while voice daemon is active
    draw_hud(frame, state, fingers_list, listening)
    cv2.imshow("Smart Home HCI  Vosk Offline", frame)     # Display annotated frame

    # ── Keyboard Input ───────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF                     # Non-blocking 1ms key check
    if key == ord('v') or key == ord('V'):          # V key  trigger one voice listen cycle
        if not voice_active.is_set():               # Only start if not already listening
            voice_active.set()                      # Signal the background Vosk thread
    elif key == 27:                                 # ESC (ASCII 27)  exit main loop
        break

# ──────────────────────────────────────────────────────────────────
#  SECTION 13  Cleanup
# ──────────────────────────────────────────────────────────────────
mic_stream.stop_stream()    # Stop the PyAudio mic stream (no more audio data will be read)
mic_stream.close()          # Close and release the stream resource
pa.terminate()              # Terminate the PyAudio instance and free all audio system resources
cap.release()               # Release the camera hardware back to the OS
cv2.destroyAllWindows()     # Close all OpenCV display windows
print("Session ended.")     # Final terminal confirmation
