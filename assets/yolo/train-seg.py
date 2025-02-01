from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\xampp\htdocs\VISUALAI\resources\models\yolo11l-seg.pt")
    model.train(
        task="segment",
        data=r"C:\xampp\htdocs\VISUALAI\website-django\static\images\datasets\defect2.v1i.yolov11\data.yaml",
        epochs=100,
        imgsz=960,
        project=r"website-django\inspection\static\resources\models",
        name="defect-seg2",
        device="cuda",
        batch=8,
        resume=False,
        amp=True,
    )
