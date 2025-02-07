from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\xampp\htdocs\VISUALAI\resources\models\yolo11l-cls.pt")
    model.train(
        task="classify",
        data=r"C:\xampp\htdocs\VISUALAI\website-django\inspection\static\images\datasets\Stitches.v1i.folder",
        epochs=100,
        imgsz=960,
        project=r"website-django\inspection\static\resources\models",
        name="stitches2",
        device="cuda",
        batch=8,
        resume=False,
        amp=True,
    )
