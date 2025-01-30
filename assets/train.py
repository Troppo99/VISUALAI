from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\xampp\htdocs\VISUALAI\resources\models\defect2\weights\last.pt")
    model.train(
        data=r"C:\xampp\htdocs\VISUALAI\website-django\static\images\datasets\defect2.v1i.yolov11\data.yaml",
        epochs=100,
        imgsz=1280,
        project=r"resources\models",
        name="defect2",
        device="cuda",
        batch=8,
        resume=True,
        amp=True,
        # precision=16,
    )
