from flask import Flask, render_template, Response
from src.ContopDetector import ContopDetector
from src.Scheduling import Scheduling

app = Flask(__name__)

detector_args = {
    "contop_confidence_threshold": 0.5,
    "camera_name": "FREEMETAL1",
    "video_source": "static/videos/contop testing.mp4",
    "window_size": (320, 240),
}

# Inisialisasi BroomDetector
contop_detector = ContopDetector(**detector_args)

# Inisialisasi scheduling
scheduler = Scheduling(contop_detector, "OFFICE")  # ganti "SEWING" atau "OFFICE" sesuai kebutuhan


@app.route("/video_feed")
def video_feed():
    # generate_frames akan menghasilkan frame jika BroomDetector sedang dijalankan (stop_event tidak di-set)
    return Response(contop_detector.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # Aplikasi Flask berjalan, tetapi video hanya muncul sesuai jadwal.
    # Scheduling akan memanggil start_detection() atau stop_detection() yang mempengaruhi broom_detector.
    app.run(host="127.0.0.1", port=5000, debug=True)
