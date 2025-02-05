from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\xampp\htdocs\VISUALAI\resources\models\yolo11l.pt")
    model.train(
        task="detect",
        data=r"C:\xampp\htdocs\VISUALAI\website-django\inspection\static\images\datasets\defect3-493\data.yaml",
        epochs=100,
        imgsz=960,
        project=r"website-django\inspection\static\resources\models",
        name="defect3-493",
        device="cuda",
        batch=8,
        resume=False,
        amp=True,
    )
