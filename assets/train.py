from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\xampp\htdocs\VISUALAI\website\static\resources\models\yolo11l.pt")
    model.train(
        data=r"C:\xampp\htdocs\VISUALAI\website\static\resources\datasets\blazing\data.yaml",
        epochs=100,
        imgsz=640,
        project=r"C:\xampp\htdocs\VISUALAI\website\static\resources\models",
        name="blazing",
        device="cuda",
        batch=16,
        resume=False,
        amp=True,
    )
