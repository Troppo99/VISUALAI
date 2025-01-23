import cv2
from ultralytics import YOLO


def main():
    video_path = r"C:\xampp\htdocs\VISUALAI\website-django\static\videos\labeling\defect1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Tidak dapat membuka sumber video.")
        return

    model = YOLO(r"C:\xampp\htdocs\VISUALAI\website-django\static\resources\models\defect1l.pt")
    infer_size = (1280, 1280)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Selesai memutar video atau tidak dapat membaca frame.")
            break

        resized_frame = cv2.resize(frame, infer_size)
        results = model(resized_frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]

                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = f"{label} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(resized_frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(resized_frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("YOLO Inference", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
