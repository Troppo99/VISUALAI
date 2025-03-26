import cv2
from ultralytics import YOLO
import json


def camera_config():
    with open(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\conf\camera_config.json", "r") as f:
        config = json.load(f)
    ip = config["ROBOTICS"]["ip"]

    return ip


model = YOLO(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\resources\models\blower\weights\best.pt").to("cuda")
cap = cv2.VideoCapture(
    f"rtsp://admin:oracle2015@{camera_config()}:554/Streaming/Channels/1",
    # r"C:\xampp\htdocs\VISUALAI\website-django\inspection\static\videos\test2.mp4",
    # 0,
    # cv2.CAP_DSHOW,
)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not captured. Check your camera index or driver.")
        break
    frame = cv2.resize(frame, (1280, 1280))
    results = model(frame, imgsz=960)
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            cls_index = int(b.cls[0])
            cls_name = model.names[cls_index]
            if conf > 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, max(0, y2 + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Webcam Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord("n"):  # ESC
        break

cap.release()
cv2.destroyAllWindows()
