from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\xampp\htdocs\VISUALAI\resources\models\yolo11l.pt")
    model.train(
        task="detect",
        data=r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\images\datasets\blower\data.yaml",
        epochs=100,
        imgsz=640,
        project=r"website-django\five_s\static\resources\models",
        name="blower",
        device="cuda",
        batch=16,
        resume=False,
        amp=True,
    )
