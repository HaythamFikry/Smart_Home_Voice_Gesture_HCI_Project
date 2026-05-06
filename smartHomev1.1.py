"""
==============================================
  SMART HOME CONTROLLER
  Control your home with HAND GESTURES + VOICE
==============================================

What this program does:
  - Camera watches your hand
  - Counts your fingers → controls devices
  - Press V → speak a command → controls devices

Devices you can control:
  - LIGHT   (on / off)
  - FAN     (off / slow / medium / fast)
  - AC      (on / off)
  - TV      (on / off)

Gestures:
  ✊  0 fingers = Light OFF
  ☝  1 finger  = Fan slow
  ✌  2 fingers = Fan medium
  🤟  3 fingers = Fan fast
  🖐  4 fingers = AC toggle
  ✋  5 fingers = Light ON

Voice examples:
  "light on"   "fan off"   "ac on"   "tv off"
  "all on"     "all off"   "goodnight"

HOW TO RUN:
  1. pip install opencv-python mediapipe pyaudio vosk
  2. Download a Vosk model from https://alphacephei.com/vosk/models
  3. Put the model folder next to this file, name it "model"
  4. Run: python smart_home.py
  5. Press V to speak, ESC to quit
"""

# ============================================================
# STEP 1 — Import the libraries we need
# ============================================================

import cv2           # cv2    = opens the camera and shows the video window
import mediapipe as mp  # mediapipe = finds your hand and fingers in the video
import pyaudio       # pyaudio   = listens to the microphone
import threading     # threading = lets voice run in the background (camera won't freeze)
import time          # time      = used to add a small delay between commands
import json          # json      = reads the text result from Vosk

from vosk import Model, KaldiRecognizer  # Vosk = offline speech-to-text (no internet needed)
from datetime import datetime            # datetime = gives us the current time (for the log)


# ============================================================
# STEP 2 — Load the Vosk voice model
# ============================================================

# This loads the voice recognition model from the "model" folder
# It takes a few seconds the first time — that is normal
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
print("Loading voice model... please wait")
voice_model = Model("ch9\Smart_Home_Voice_Gesture_HCI_Project\model")

# KaldiRecognizer listens to audio and converts it to text
# 16000 = the audio quality we need (16000 samples per second)
voice_recognizer = KaldiRecognizer(           # Create a streaming recogniser bound to the loaded model
    voice_model,                         # The acoustic model to use for inference
    16000,                              # Sample rate in Hz  must match PyAudio stream below
    VOCAB,                              # Restrict recognition to this JSON vocabulary list
)

# Open the microphone so we can record your voice
audio = pyaudio.PyAudio()
microphone = audio.open(
    format   = pyaudio.paInt16,  # audio format Vosk needs
    channels = 1,                # mono (one channel, not stereo)
    rate     = 16000,            # 16000 samples per second
    input    = True,             # we are RECORDING, not playing
    frames_per_buffer = 4096     # how much audio to read at one time
)
microphone.start_stream()  # start the microphone stream
print("Voice model loaded!")


# ============================================================
# STEP 3 — Set up hand detection (MediaPipe)
# ============================================================

# Get the hand detection tool from MediaPipe
mp_hands = mp.solutions.hands

# Create the hand detector
# max_num_hands=1 → we only look at ONE hand to keep things simple
hand_detector = mp_hands.Hands(
    max_num_hands          = 1,    # detect one hand
    min_detection_confidence = 0.8,  # 80% sure it's a hand before tracking it
    min_tracking_confidence  = 0.6   # 60% sure to keep tracking it
)

# This draws the hand skeleton (dots and lines) on the screen
hand_drawer = mp.solutions.drawing_utils


# ============================================================
# STEP 4 — Device states (True = ON, False = OFF)
# ============================================================

# These are simple True/False variables for each device
light_on  = False   # Light  starts OFF
fan_on    = False   # Fan    starts OFF
fan_speed = 0       # Fan speed: 0=off, 1=slow, 2=medium, 3=fast
ac_on     = False   # AC     starts OFF
tv_on     = False   # TV     starts OFF


# ============================================================
# STEP 5 — A simple log (shows the last 5 events on screen)
# ============================================================

# This list keeps track of the last 5 things that happened
event_log = []

def add_to_log(message):
    """Add a new event to the log (keeps only the last 5 events)."""
    global event_log
    current_time = datetime.now().strftime("%H:%M:%S")  # e.g. "14:32:07"
    event_log.append(f"[{current_time}]  {message}")   # add to the list
    event_log = event_log[-5:]                          # keep only the LAST 5
    print(f"[{current_time}]  {message}")               # also print to terminal


# ============================================================
# STEP 6 — Count how many fingers are up
# ============================================================

# These are the MediaPipe "landmark" numbers for each fingertip
#  4  = thumb tip
#  8  = index finger tip
# 12  = middle finger tip
# 16  = ring finger tip
# 20  = pinky tip
FINGERTIPS = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks):
    """
    Look at the hand landmarks and count how many fingers are raised.
    Returns a number from 0 (fist) to 5 (open hand).
    """
    fingers_up = 0  # start at zero

    # --- CHECK THE THUMB (left/right, not up/down) ---
    # Thumb tip (4) is to the LEFT of the joint below it (3) = thumb is open
    thumb_tip   = hand_landmarks.landmark[4]
    thumb_below = hand_landmarks.landmark[3]
    if thumb_tip.x < thumb_below.x:  # tip is further left = open
        fingers_up += 1

    # --- CHECK THE OTHER 4 FINGERS (up/down) ---
    # A finger is UP when its tip is HIGHER on the screen than the knuckle below it
    # In screen coordinates, a SMALLER y value means HIGHER on screen
    for tip_id in [8, 12, 16, 20]:        # index, middle, ring, pinky
        finger_tip    = hand_landmarks.landmark[tip_id]      # the very tip
        finger_knuckle = hand_landmarks.landmark[tip_id - 2]  # knuckle 2 joints below
        if finger_tip.y < finger_knuckle.y:  # tip is above knuckle = finger is UP
            fingers_up += 1

    return fingers_up  # return the total


# ============================================================
# STEP 7 — What to do when a gesture is detected
# ============================================================

# We use a small delay system so a gesture does not fire 30 times per second
# "hold_count" counts how many frames the same gesture has been held
hold_count    = 0       # how many frames the same gesture has been held
last_gesture  = -1      # what was the last gesture we saw
HOLD_NEEDED   = 10      # must hold the gesture for 10 frames (~0.3 seconds)
last_fire_time = 0      # when did we last run a command

def handle_gesture(fingers):
    """
    Decide which device to control based on the number of fingers shown.
    A gesture must be held for HOLD_NEEDED frames before it fires.
    """
    global hold_count, last_gesture, last_fire_time
    global light_on, fan_on, fan_speed, ac_on, tv_on

    # If the gesture changed, reset the hold counter
    if fingers != last_gesture:
        hold_count   = 0        # start counting from zero
        last_gesture = fingers  # remember the new gesture
        return                  # wait — do not fire yet

    # Count how many frames we have held this gesture
    hold_count += 1

    # Only fire when held long enough AND 1 second has passed since last command
    time_since_last = time.time() - last_fire_time
    if hold_count < HOLD_NEEDED or time_since_last < 1.0:
        return  # not ready yet

    # --- GESTURE ACTIONS ---
    # Each number of fingers controls a different device

    if fingers == 0:            # ✊  FIST = Light OFF
        light_on = False
        add_to_log("Gesture: Light OFF")

    elif fingers == 1:          # ☝  1 FINGER = Fan slow
        fan_on    = True
        fan_speed = 1
        add_to_log("Gesture: Fan speed 1 (slow)")

    elif fingers == 2:          # ✌  2 FINGERS = Fan medium
        fan_on    = True
        fan_speed = 2
        add_to_log("Gesture: Fan speed 2 (medium)")

    elif fingers == 3:          # 🤟  3 FINGERS = Fan fast
        fan_on    = True
        fan_speed = 3
        add_to_log("Gesture: Fan speed 3 (fast)")

    elif fingers == 4:          # 🖐  4 FINGERS = AC toggle
        ac_on = not ac_on       # flip ON→OFF or OFF→ON
        add_to_log(f"Gesture: AC {'ON' if ac_on else 'OFF'}")

    elif fingers == 5:          # ✋  OPEN HAND = Light ON
        light_on = True
        add_to_log("Gesture: Light ON")

    # Reset counters so this gesture does not keep firing
    hold_count     = 0
    last_fire_time = time.time()


# ============================================================
# STEP 8 — What to do when a voice command is heard
# ============================================================

def handle_voice(text):
    """
    Read the recognised speech text and control the right device.
    We just check if certain words are IN the sentence.
    Example: "please turn on the light" contains "light on" → Light ON
    """
    global light_on, fan_on, fan_speed, ac_on, tv_on

    text = text.lower()   # make everything lowercase so "Light ON" == "light on"

    # --- LIGHT COMMANDS ---
    if "light on" in text or "turn on light" in text or "lights on" in text:
        light_on = True
        add_to_log("Voice: Light ON")

    elif "light of" in text or "turn of light" in text or "lights of" in text:
        light_on = False
        add_to_log("Voice: Light OF")
    elif "light off" in text or "turn off light" in text or "lights off" in text:
        light_on = False
        add_to_log("Voice: Light OFF")

    # --- FAN COMMANDS ---
    elif "fan on" in text or "turn on fan" in text or "start fan" in text:
        fan_on    = True
        fan_speed = 1   # start at slow
        add_to_log("Voice: Fan ON (slow)")

    elif "fan of" in text or "turn of fan" in text or "stop fan" in text:
        fan_on    = False
        fan_speed = 0
        add_to_log("Voice: Fan OF")
    elif "fan off" in text or "turn off fan" in text or "stop fan" in text:
        fan_on    = False
        fan_speed = 0
        add_to_log("Voice: Fan OFF")

    elif "fan high" in text or "fan speed three" in text or "fan fast" in text:
        fan_on    = True
        fan_speed = 3
        add_to_log("Voice: Fan speed 3 (fast)")

    elif "fan medium" in text or "fan speed two" in text:
        fan_on    = True
        fan_speed = 2
        add_to_log("Voice: Fan speed 2 (medium)")

    elif "fan low" in text or "fan speed one" in text or "fan slow" in text:
        fan_on    = True
        fan_speed = 1
        add_to_log("Voice: Fan speed 1 (slow)")

    # --- AC COMMANDS ---
    elif "ac on" in text or "turn on ac" in text or "air on" in text:
        ac_on = True
        add_to_log("Voice: AC ON")

    elif "ac of" in text or "turn of ac" in text or "air of" in text:
        ac_on = False
        add_to_log("Voice: AC OF")
    elif "ac off" in text or "turn off ac" in text or "air off" in text:
        ac_on = False
        add_to_log("Voice: AC OFF")

    # --- TV COMMANDS ---
    elif "tv on" in text or "turn on tv" in text or "television on" in text:
        tv_on = True
        add_to_log("Voice: TV ON")

    elif "tv off" in text or "turn off tv" in text or "television off" in text:
        tv_on = False
        add_to_log("Voice: TV OFF")
    elif "tv of" in text or "turn of tv" in text or "television of" in text:
        tv_on = False
        add_to_log("Voice: TV OF")

    # --- ALL DEVICES ---
    elif "all on" in text or "everything on" in text:
        light_on  = True
        fan_on    = True
        fan_speed = 3
        ac_on     = True
        tv_on     = True
        add_to_log("Voice: ALL devices ON")

    elif "all of" in text or "everything off" in text or "goodnight" in text:
        light_on  = False
        fan_on    = False
        fan_speed = 0
        ac_on     = False
        tv_on     = False
        add_to_log("Voice: ALL devices OF")

    elif "all off" in text or "everything off" in text or "goodnight" in text:
        light_on  = False
        fan_on    = False
        fan_speed = 0
        ac_on     = False
        tv_on     = False
        add_to_log("Voice: ALL devices OFF")

    else:
        add_to_log(f'Voice: not understood → "{text}"')


# ============================================================
# STEP 9 — Voice listener (runs in the background)
# ============================================================

# This flag tells the background thread when to start listening
start_listening = threading.Event()

def voice_listener_thread():
    """
    This function runs in the background.
    It waits until the user presses V, then listens and converts speech to text.
    The camera keeps running because this is a separate thread.
    """
    while True:
        start_listening.wait()    # sleep here until the user presses V
        start_listening.clear()   # reset the flag so it waits again next time

        print("Listening... speak now!")
        voice_recognizer.Reset()  # clear any old audio data

        deadline = time.time() + 5   # listen for a maximum of 5 seconds
        got_result = False

        while time.time() < deadline:
            # read a small chunk of audio from the microphone
            audio_chunk = microphone.read(4096, exception_on_overflow=False)

            # give the chunk to Vosk — it returns True when it finishes a sentence
            if voice_recognizer.AcceptWaveform(audio_chunk):
                result      = json.loads(voice_recognizer.Result())  # get the result as a dictionary
                spoken_text = result.get("text", "")                 # get the "text" field
                if spoken_text:           # if we actually heard something
                    handle_voice(spoken_text)  # run the voice command
                    got_result = True
                    break                 # stop listening after one command

        if not got_result:
            add_to_log("Voice: nothing heard (5s timeout)")

# Create the background thread and start it
listener = threading.Thread(target=voice_listener_thread, daemon=True)
listener.start()   # start it — it will immediately wait at start_listening.wait()


# ============================================================
# STEP 10 — Draw the info panel on the video frame
# ============================================================

def draw_info(frame):
    """
    Draw the device status and event log onto the video frame.
    Green text = ON,  Red text = OFF
    """
    height, width = frame.shape[:2]   # get the size of the video frame

    # --- Draw a dark background box in the top-left corner ---
    # This makes the text easier to read on any background
    cv2.rectangle(frame,
                  (5, 5),          # top-left corner of the box
                  (260, 175),      # bottom-right corner of the box
                  (20, 20, 20),    # dark grey colour (BGR)
                  -1)              # -1 means FILLED (not just outline)
    cv2.rectangle(frame, (5, 5), (260, 175), (80, 80, 80), 1)  # thin border

    # --- Device status rows ---
    # Each row shows:  symbol + device name + ON or OFF
    # Green colour (0, 220, 80) = ON
    # Red   colour (50, 50, 200) = OFF

    # LIGHT
    light_color = (0, 220, 80) if light_on else (50, 50, 200)
    light_text  = "* Light   ON" if light_on else "- Light   OFF"
    cv2.putText(frame, light_text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, light_color, 2)

    # FAN  (shows speed if ON)
    fan_color = (0, 220, 80) if fan_on else (50, 50, 200)
    if fan_on:
        speed_names = {1: "slow", 2: "medium", 3: "fast"}
        fan_text = f"* Fan     {speed_names[fan_speed]}"
    else:
        fan_text = "- Fan     OFF"
    cv2.putText(frame, fan_text, (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fan_color, 2)

    # AC
    ac_color = (0, 220, 80) if ac_on else (50, 50, 200)
    ac_text   = "* AC      ON" if ac_on else "- AC      OFF"
    cv2.putText(frame, ac_text, (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ac_color, 2)

    # TV
    tv_color = (0, 220, 80) if tv_on else (50, 50, 200)
    tv_text   = "* TV      ON" if tv_on else "- TV      OFF"
    cv2.putText(frame, tv_text, (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, tv_color, 2)

    # --- Bottom label: press V for voice ---
    # Change colour to orange when listening, cyan when idle
    if start_listening.is_set():
        v_color = (0, 140, 255)    # orange = currently listening
        v_text  = "  LISTENING..."
    else:
        v_color = (255, 200, 0)    # cyan = idle
        v_text  = "  [V] Speak  |  [ESC] Quit"
    cv2.putText(frame, v_text, (15, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, v_color, 1)

    # --- Event log (bottom of screen) ---
    # Show the last 5 events at the bottom of the frame
    log_start_y = height - 20 - (len(event_log) * 22)  # where to start drawing
    for i, line in enumerate(event_log):
        y_pos = log_start_y + i * 22
        cv2.putText(frame, line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# ============================================================
# STEP 11 — Open the camera
# ============================================================

camera = cv2.VideoCapture(0)   # 0 = the default camera (built-in webcam)

# Try to use a higher resolution if the camera supports it
camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

add_to_log("System started! Press V to speak, ESC to quit.")


# ============================================================
# STEP 12 — Main loop (runs the whole program)
# ============================================================
#
#  Every frame (about 30 times per second), we:
#    1. Read a frame from the camera
#    2. Flip it so it looks like a mirror
#    3. Send it to MediaPipe to find the hand
#    4. Count the fingers and run the gesture
#    5. Draw the info panel on the frame
#    6. Show the frame in a window
#    7. Check if V or ESC was pressed

while True:

    # --- 1. Read a frame from the camera ---
    success, frame = camera.read()
    if not success:
        print("Camera error — could not read frame")
        break

    # --- 2. Flip the frame horizontally (mirror effect) ---
    # Without this, your right hand appears on the left side — confusing!
    frame = cv2.flip(frame, 1)

    # --- 3. Detect hands using MediaPipe ---
    # MediaPipe needs the image in RGB colour (camera gives us BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hand_detector.process(rgb_frame)   # run the hand detector

    # --- 4. If a hand was found, count fingers and run gesture ---
    if results.multi_hand_landmarks:   # True if at least one hand was detected
        # Get the first hand (we only track one)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw the hand skeleton (dots and connecting lines) on the frame
        hand_drawer.draw_landmarks(
            frame,                     # draw on this frame
            hand_landmarks,            # use these landmark positions
            mp_hands.HAND_CONNECTIONS  # connect the dots in the right order
        )

        # Count how many fingers are raised
        fingers = count_fingers(hand_landmarks)

        # Show the finger count on screen (top centre)
        cv2.putText(frame, f"Fingers: {fingers}", (width // 2 - 80, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Run the gesture command
        handle_gesture(fingers)

    else:
        # No hand visible — reset the hold counter
        hold_count   = 0
        last_gesture = -1

    # --- 5. Draw the info panel (device status + log) ---
    height, width = frame.shape[:2]   # get frame size (needed for putText above)
    draw_info(frame)

    # --- 6. Show the frame in a window ---
    cv2.imshow("Smart Home HCI Demo", frame)

    # --- 7. Check keyboard input ---
    key = cv2.waitKey(1) & 0xFF   # wait 1ms for a key press

    if key == ord('v') or key == ord('V'):
        # V was pressed → tell the background thread to start listening
        if not start_listening.is_set():  # only start if not already listening
            start_listening.set()

    elif key == 27:   # ESC key (number 27)
        print("ESC pressed — quitting...")
        break


# ============================================================
# STEP 13 — Clean up when the program ends
# ============================================================

microphone.stop_stream()   # stop the microphone
microphone.close()         # close the microphone stream
audio.terminate()          # release the audio system
camera.release()           # release the camera
cv2.destroyAllWindows()    # close the video window
print("Program ended. Goodbye!")