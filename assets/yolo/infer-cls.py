import cv2
from ultralytics import YOLO

# model = YOLO("resources/models/yolo11l-cls.pt")
model = YOLO(r"C:\xampp\htdocs\VISUALAI\website-django\inspection\static\resources\models\stitches\weights\best.pt")
cap = cv2.VideoCapture(r"C:\xampp\htdocs\VISUALAI\website-django\inspection\static\images\labeling\videos\stitches.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    if len(results) > 0:
        result = results[0]
        top1_index = result.probs.top1
        top1_conf = float(result.probs.top1conf)
        top1_label = result.names[top1_index]
        text_result = f"{top1_label} ({top1_conf:.2f})"
        cv2.putText(frame, text_result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("YOLO Classification - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
