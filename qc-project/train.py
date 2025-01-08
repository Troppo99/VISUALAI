from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\xampp\htdocs\VISUALAI\qc-project\models\OX\version1\weights\last.pt")
    model.train(
        data=r"C:\xampp\htdocs\VISUALAI\qc-project\datasets\cross-zero\data.yaml",
        epochs=100,
        imgsz=640,
        project="qc-project/models/OX",
        name="version1",
        device="cuda",
        batch=16,
        resume=True,
        amp=True,
    )
