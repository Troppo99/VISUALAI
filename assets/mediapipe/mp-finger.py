import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                finger_tips = [4, 8, 12, 16, 20]
                for tip in finger_tips:
                    x = int(hand_landmarks.landmark[tip].x * w)
                    y = int(hand_landmarks.landmark[tip].y * h)
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

        cv2.imshow("Deteksi Jari Tangan", frame)
        if cv2.waitKey(5) & 0xFF == ord("n"):
            break

cap.release()
cv2.destroyAllWindows()
