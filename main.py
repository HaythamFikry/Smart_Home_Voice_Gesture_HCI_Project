import cv2                          # OpenCV → camera + image processing
import mediapipe as mp             # MediaPipe → hand detection
import speech_recognition as sr    # SpeechRecognition → voice to text

# ---------- Hand Detection Setup ----------
mp_hands = mp.solutions.hands      # Access hand detection module from MediaPipe
hands = mp_hands.Hands()           # Create hand detection object
mp_draw = mp.solutions.drawing_utils  # Used to draw hand landmarks on screen

# ---------- Voice Setup ----------
recognizer = sr.Recognizer()       # Create speech recognizer object

# ---------- Camera Setup ----------
cap = cv2.VideoCapture(0)          # Open default webcam (0)

# ---------- Smart Home States ----------
light_on = False                   # Light initially OFF
fan_on = False                     # Fan initially OFF


# ---------- Function: Count Fingers ----------
def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]      # Landmark indexes for fingertips
    fingers = []                   # List to store each finger state (1 = up, 0 = down)

    # ---------- Thumb ----------
    # Compare thumb tip with previous joint (x-axis)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x:
        fingers.append(1)          # Thumb is open
    else:
        fingers.append(0)          # Thumb is closed

    # ---------- Other Fingers ----------
    # Check if fingertip is above its lower joint (y-axis)
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y:
            fingers.append(1)      # Finger is open
        else:
            fingers.append(0)      # Finger is closed

    return sum(fingers)            # Return total number of raised fingers


print("System Started... Press V for voice command")

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()        # Capture frame from camera
    frame = cv2.flip(frame, 1)     # Flip image horizontally (mirror effect)

    # Convert image to RGB (required for MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    result = hands.process(rgb)

    # ---------- Gesture Control ----------
    if result.multi_hand_landmarks:   # If any hand is detected
        for handLms in result.multi_hand_landmarks:

            # Draw hand landmarks (points + connections)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Count number of fingers
            fingers = count_fingers(handLms)

            # ---------- Control Logic ----------
            if fingers >= 4:
                light_on = True        # Open hand → Light ON

            elif fingers == 0:
                light_on = False       # Closed fist → Light OFF

            elif fingers == 2:
                fan_on = True          # Two fingers → Fan ON

            elif fingers == 1:
                fan_on = False         # One finger → Fan OFF

    # ---------- Display Information ----------
    # Show finger count on screen
    cv2.putText(frame, f"Fingers: {fingers if result.multi_hand_landmarks else 0}", 
                (20, 450),cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if fingers > 0  else (0,0,255), 2)
        
    # Show light status
    cv2.putText(frame, f"Light: {'ON' if light_on else 'OFF'}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if light_on else (0,0,255), 2)

    # Show fan status
    cv2.putText(frame, f"Fan: {'ON' if fan_on else 'OFF'}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if fan_on else (0,0,255), 2)

    # Show instruction
    cv2.putText(frame, "Press V for Voice Command",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,0), 2)

    # Show the camera window
    cv2.imshow("Smart Home HCI Demo", frame)

    # Wait for key press
    key = cv2.waitKey(1)

    # ---------- Voice Control ----------
    if key == ord('v'):   # If user presses 'V'
        try:
            with sr.Microphone() as source:   # Access microphone
                print("Listening...")

                # Capture audio for max 3 seconds
                audio = recognizer.listen(source, timeout=3)

                # Convert speech to text using Google API
                command = recognizer.recognize_google(audio).lower()
                print("Command:", command)

                # ---------- Voice Commands ----------
                if "light on" in command:
                    light_on = True

                elif "light off" in command:
                    light_on = False

                elif "fan on" in command:
                    fan_on = True

                elif "fan off" in command:
                    fan_on = False

        except:
            print("Voice not recognized")   # Handle errors

    # ---------- Exit Program ----------
    if key == 27:   # ESC key
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()