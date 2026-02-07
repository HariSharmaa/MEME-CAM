import cv2
import mediapipe as mp
import numpy as np
import imageio
from pathlib import Path

# venv\Scripts\activate.bat


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# =========================
# MEDIAPIPE SETUP
# =========================

# Face mesh module
mp_face_mesh = mp.solutions.face_mesh
# Hand Mesh Module
mp_hands = mp.solutions.hands

# Drawing helpers (dots, lines, styles)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create face mesh detector
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Create hand mesh detector
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# =========================
# OUTPUT FOLDER
# =========================

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


# =========================
# GIF PATHS
# =========================

tongue_gif = "assets/tongue.gif"
closed_eyes_gif = "assets/closed_eyes.gif"
smile_stare_gif="assets/smile-stare.gif"
monkey_thinking = 'assets/monkey-thinking.gif'
monkey_pointing = 'assets/monkey-pointing.gif' 
monkey_thumbsup =  'assets\monkey-thumbsup.gif'
oh_no = 'assets\oh_no.gif'


# =========================
# LOAD GIF FUNCTION
# =========================

def load_gif(path):
    """
    Reads a GIF file and converts it
    into OpenCV-compatible frames
    """
    try:
        gif_frames = imageio.mimread(path)

        # Convert RGB → BGR (OpenCV format)
        frames = [
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for frame in gif_frames
        ]

        return frames
    except Exception as e:
        print("Error loading GIF:", e)
        return None


# Load GIFs
tongue_frames = load_gif(tongue_gif)
eyes_frames = load_gif(closed_eyes_gif)
smile_frame = load_gif(smile_stare_gif)
monkey_thinking_frame = load_gif(monkey_thinking)
monkey_pointing_frame =  load_gif(monkey_pointing)
monkey_thumbsup_frame = load_gif(monkey_thumbsup)
oh_no_frame = load_gif(oh_no)

# =========================
# EYE ASPECT RATIO
# =========================

def eye_aspect_ratio(landmarks, eye_indices):
    """
    Measures how open the eye is
    Smaller value → closed
    """

    p1, p2, p3, p4, p5, p6 = [
        np.array([landmarks[i].x, landmarks[i].y])
        for i in eye_indices
    ]

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)

    return (v1 + v2) / (2.0 * h)


# =========================
# MOUTH ASPECT RATIO
# =========================

def mouth_aspect_ratio(landmarks):
    """
    Measures how open the mouth is
    Bigger value → open mouth
    """

    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])

    return np.linalg.norm(top - bottom) / np.linalg.norm(left - right)


# =========================
# SMILE RATIO
# =========================

def smile_ratio(landmarks):
    """
    Measures how much the person is smiling.
    Bigger value → bigger smile
    """

    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])

    width = np.linalg.norm(left - right)
    height = 1

    return width / height


# =========================
# Thinking Pointing Aspects
# =========================

def classify_gesture(hand_landmarks):

    y_thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    y_index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    y_middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    y_ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    y_pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

    y_middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

    is_thumb_up = y_thumb_tip < y_middle_pip
    are_fingers_down = (
        y_index_tip > y_middle_pip and
        y_middle_tip > y_middle_pip and
        y_ring_tip > y_middle_pip and
        y_pinky_tip > y_middle_pip
    )

    if is_thumb_up and are_fingers_down:
        return "THUMBS_UP"

    is_index_up = y_index_tip < y_middle_pip
    is_other_fingers_down = (
        y_middle_tip > y_middle_pip and
        y_ring_tip > y_middle_pip and
        y_pinky_tip > y_middle_pip
    )
    is_thumb_down = y_thumb_tip > y_middle_pip

    if is_index_up and is_other_fingers_down and is_thumb_down:
        return "POINTING"
    
    return "NEUTRAL"
    

def check_thinking_gesture(hand_landmarks, face_landmarks, frame_width, frame_height):

    if not hand_landmarks or not face_landmarks:
        return False

    # -------------------------------
    # INDEX NEAR MOUTH (THINKING)
    # -------------------------------
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    index_x = int(index_tip.x * frame_width)
    index_y = int(index_tip.y * frame_height)

    LipRightCorner = face_landmarks.landmark[61]
    LipRightCorner_x = int(LipRightCorner.x * frame_width)
    LipRightCorner_y = int(LipRightCorner.y * frame_height)

    distance = np.sqrt((index_x - LipRightCorner_x) ** 2 + (index_y - LipRightCorner_y) ** 2)
    MAX_DISTANCE = 50

    y_middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    y_middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    is_middle_finger_down = y_middle_tip > y_middle_pip

    if distance < MAX_DISTANCE and is_middle_finger_down:
        return "THINKING"
    
    return False

def check_hand_on_head(hand_landmarks, face_landmarks, frame_width, frame_height):

    if not hand_landmarks or not face_landmarks:
        return False

    palm = hand_landmarks.landmark[17]

    head_top = face_landmarks.landmark[10]

    palm_x = int(palm.x * frame_width)
    palm_y = int(palm.y * frame_height)

    head_x = int(head_top.x * frame_width)
    head_y = int(head_top.y * frame_height)

    distance = np.sqrt((palm_x - head_x) ** 2 + (palm_y - head_y) ** 2)

    HAND_ON_HEAD_DISTANCE = 80

    return distance < HAND_ON_HEAD_DISTANCE


# =========================
# THRESHOLDS
# =========================

EYE_AR_THRESH = 0.28
MOUTH_AR_THRESH = 0.70
SMILE_THRESH = 0.10


# =========================
# START WEBCAM
# =========================

cap = cv2.VideoCapture(0)

reaction_mode = None
reaction_index = 0

print("Press Q to quit")



while True:

    # Read webcam frame
    success, frame = cap.read()
    if not success:
        break

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb)

    # Detect hand landmarks
    hand_results = hands.process(rgb)

    # =========================
    # NEW: RESET ALL STATES EVERY FRAME
    # =========================
    eyes_closed = False
    tongue_out = False
    is_smiling = False

    is_monkey_thinking = False
    is_monkey_pointing = False
    is_monkey_thumbsup = False
    is_oh_no = False

    current_gesture = "NEUTRAL"
    reaction_mode = None
    prev_reaction_mode = None
    gesture = None
    hand_landmarks_data = None

    # =========================
    # IF FACE IS DETECTED
    # =========================

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # ---------- FACE CALCULATIONS ----------

        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [263, 387, 385, 362, 380, 373]

        left_EAR = eye_aspect_ratio(landmarks, left_eye_idx)
        right_EAR = eye_aspect_ratio(landmarks, right_eye_idx)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        mar = mouth_aspect_ratio(landmarks)
        smile_value = smile_ratio(landmarks)

        eyes_closed = avg_EAR < EYE_AR_THRESH
        tongue_out = mar > MOUTH_AR_THRESH
        is_smiling = smile_value > SMILE_THRESH

    # =========================
    # IF HAND IS DETECTED
    # =========================

    if hand_results.multi_hand_landmarks:

        hand_landmarks_data = hand_results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks_data,
            mp_hands.HAND_CONNECTIONS
        )

        
        # ---------- HAND CALCULATIONS ----------

        if hand_landmarks_data:

    # HIGHEST PRIORITY (needs face)
            if results.multi_face_landmarks and check_hand_on_head(
            hand_landmarks_data,
            face_landmarks,
            frame.shape[1],
            frame.shape[0]
            ):
                current_gesture = "oh_no"

    # SECOND PRIORITY (needs face)
            elif results.multi_face_landmarks and check_thinking_gesture(
                hand_landmarks_data,
                face_landmarks,
                frame.shape[1],
                frame.shape[0]
            ):
                current_gesture = "THINKING"

    # NORMAL HAND GESTURES (NO FACE REQUIRED)
            else:
                current_gesture = classify_gesture(hand_landmarks_data)

        is_monkey_thinking = current_gesture == "THINKING"
        is_monkey_pointing = current_gesture == "POINTING"
        is_monkey_thumbsup = current_gesture == "THUMBS_UP"
        is_oh_no = current_gesture == "oh_no"


        cv2.putText(
            frame,
            f"Hand Gesture: {current_gesture}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # =========================
    # NEW: REACTION SELECTION (FACE + HAND INDEPENDENT)
    # =========================

    # Reset reaction by default
    if reaction_mode != prev_reaction_mode:
        reaction_index = 0
        prev_reaction_mode = reaction_mode


    if eyes_closed:
        reaction_mode = "eyes"
        cv2.putText(frame, "hell nah", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    elif tongue_out:
        reaction_mode = "tongue"
        cv2.putText(frame, "freak of nature", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    elif is_smiling:
        reaction_mode = "smile"
        reaction_index = 0
        cv2.putText(frame, "Smiling :)", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        cv2.putText(frame, f"Smile Ratio: {smile_value:.2f}", (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
    elif is_monkey_thinking:
        reaction_mode = "monkey_thinking"
        cv2.putText(frame, "Monkey Thinking", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    elif is_monkey_pointing:
        reaction_mode = "monkey_pointing"
        cv2.putText(frame, "Pointing", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    elif is_monkey_thumbsup:
        reaction_mode = "monkey_thumbsup"
        cv2.putText(frame, "Thumbs up", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    elif is_oh_no:
        reaction_mode = "oh_no"
        cv2.putText(frame, "oh_no", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    else:
        cv2.putText(frame, "Normal", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    # =========================
    # SHOW WINDOWS
    # =========================

    frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    cv2.imshow("Meme Cam", frame_resized)

    # Always reset index when reaction changes
    if reaction_mode not in ["eyes", "tongue", "smile", "monkey_thinking", "monkey_pointing", "monkey_thumbsup", "oh_no" ]:
        reaction_index = 0



    if reaction_mode == "eyes" and eyes_frames:
        gif = eyes_frames[reaction_index % len(eyes_frames)]
        gif = cv2.resize(gif, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow("Reaction", gif)
        reaction_index += 1

    elif reaction_mode == "tongue" and tongue_frames:
        gif = tongue_frames[reaction_index % len(tongue_frames)]
        gif = cv2.resize(gif, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow("Reaction", gif)
        reaction_index += 1

    elif reaction_mode == "smile" and smile_frame:
        gif = smile_frame[reaction_index % len(smile_frame)]
        gif = cv2.resize(gif, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow("Reaction", gif)
        reaction_index += 1

    elif reaction_mode == "monkey_thinking" and monkey_thinking_frame:
        gif = monkey_thinking_frame[reaction_index % len(monkey_thinking_frame)]
        gif = cv2.resize(gif, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow("Reaction", gif)
        reaction_index += 1

    elif reaction_mode == "monkey_pointing" and monkey_pointing_frame:
        gif = monkey_pointing_frame[reaction_index % len(monkey_pointing_frame)]
        gif = cv2.resize(gif, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow("Reaction", gif)
        reaction_index += 1

    elif reaction_mode == "monkey_thumbsup" and monkey_thumbsup_frame:
        gif = monkey_thumbsup_frame[reaction_index % len(monkey_thumbsup_frame)]
        gif = cv2.resize(gif, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow("Reaction", gif)
        reaction_index += 1

    elif reaction_mode == "oh_no" and oh_no_frame:
        gif = oh_no_frame[reaction_index % len(oh_no_frame)]
        gif = cv2.resize(gif, (WINDOW_WIDTH, WINDOW_HEIGHT))
        cv2.imshow("Reaction", gif)
        reaction_index += 1

    else:
        blank = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        cv2.imshow("Reaction", blank)



    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing...")
        break

cap.release()
cv2.destroyAllWindows()
