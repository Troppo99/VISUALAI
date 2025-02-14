import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyaudio
import wave

vertices = []
drag_idx = -1
max_vertices = 4


def mouse_cb(event, x, y, flags, _):
    global vertices, drag_idx
    th = 10
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(vertices):
            if abs(x - pt[0]) < th and abs(y - pt[1]) < th:
                drag_idx = i
                return
        if len(vertices) < max_vertices:
            vertices.append((x, y))
            drag_idx = len(vertices) - 1
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_idx != -1:
            vertices[drag_idx] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = -1


cap = cv2.VideoCapture(1)
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", mouse_cb)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    disp = frame.copy()
    for pt in vertices:
        cv2.circle(disp, pt, 5, (0, 255, 0), -1)
    if vertices:
        pts = np.array(vertices, np.int32)
        if len(vertices) == max_vertices:
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
        else:
            cv2.polylines(disp, [pts], False, (0, 255, 0), 2)
    cv2.putText(disp, "Press 'n' to confirm ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Select ROI", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("n") and len(vertices) == max_vertices:
        break
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Select ROI")

speaker_index = 4


def play_wav(wav_file, device_index):
    wf = wave.open(wav_file, "rb")
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pa.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True, output_device_index=device_index)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    pa.terminate()


music_should_loop = False
audio_thread = None


def audio_loop():
    global music_should_loop
    while music_should_loop:
        play_wav("assets/audio/berton.wav", speaker_index)


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.5)

detection_start_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(vertices, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
    disp = roi_frame.copy()
    cv2.polylines(disp, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
    rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for lm in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(disp, lm, mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
    face_detected = results.multi_face_landmarks is not None
    if face_detected:
        if detection_start_time is None:
            detection_start_time = time.time()
        elif time.time() - detection_start_time >= 1:
            music_should_loop = True
            if audio_thread is None or not audio_thread.is_alive():
                audio_thread = threading.Thread(target=audio_loop, daemon=True)
                audio_thread.start()
    else:
        detection_start_time = None
        music_should_loop = False
    cv2.imshow("Mediapipe Face Pose", disp)
    if cv2.waitKey(5) & 0xFF == ord("n"):
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()
