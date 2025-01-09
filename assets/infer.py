import os
import cv2
import cvzone
from ultralytics import YOLO
import time


def export_frame(frame):
    results = model(frame)
    boxes_info = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = model.names[int(box.cls[0])]
            if conf > 0.5:
                color = (0, 255, 0) if class_id == "O" else (0, 0, 255)
                boxes_info.append((x1, y1, x2, y2, conf, class_id, color))
    return frame, boxes_info


videos = [
    0,
    "rtsp://admin:oracle2015@172.16.0.43:554/Streaming/Channels/1",
    "videos/test/kon.mp4",
]
n = 0
cap = cv2.VideoCapture(videos[n])
model = YOLO("qc-project/models/best.pt")
model.overrides["verbose"] = False

while True:
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(videos[n])
        time.sleep(0.1)
        try:
            ret, frame = cap.read()
        except:
            continue

    frame_results, boxes_info = export_frame(frame)
    for x1, y1, x2, y2, conf, class_id, color in boxes_info:
        cvzone.cornerRect(frame_results, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
        cvzone.putTextRect(frame_results, f"This is {class_id}", (x1, y1 - 15), colorR=color)

    frame_show = cv2.resize(frame_results, (540, 360))
    cv2.imshow(f"Infer {videos[n]}", frame_show)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("n"):
        break
cap.release()
cv2.destroyAllWindows()
