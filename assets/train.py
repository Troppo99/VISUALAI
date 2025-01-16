from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"resources\models\defect12\weights\last.pt")
    model.train(
        data=r"resources\datasets\defect1\data.yaml",
        epochs=100,
        imgsz=1280,
        project=r"resources\models",
        name="defect12",
        device="cuda",
        batch=8,
        resume=True,
        amp=True,
        # precision=16,
    )
