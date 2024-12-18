from flask import Flask, render_template, Response
from src.BroomDetector import BroomDetector


app = Flask(__name__)

detector_args = {
    "confidence_threshold": 0.5,
    "camera_name": "OFFICE1",
    "video_source": "static/videos/bd_test3.mp4",
    "window_size": (320, 240),
}
broom_detector = BroomDetector(**detector_args)


@app.route("/video_feed")
def video_feed():
    return Response(broom_detector.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
