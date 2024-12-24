from flask import Flask, render_template, Response
from src.BroomDetector import BroomDetector
from src.Scheduling import Scheduling

app = Flask(__name__)

detector_args = {
    "camera_name": "OFFICE1",
    "video_source": "static/videos/bd_test3.mp4",
}

# Inisialisasi BroomDetector
broom_detector = BroomDetector(**detector_args)

# Inisialisasi scheduling
scheduler = Scheduling(broom_detector, "SEWING")  # ganti "SEWING" atau "OFFICE" sesuai kebutuhan


@app.route("/video_feed")
def video_feed():
    # generate_frames akan menghasilkan frame jika BroomDetector sedang dijalankan (stop_event tidak di-set)
    return Response(broom_detector.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # Aplikasi Flask berjalan, tetapi video hanya muncul sesuai jadwal.
    # Scheduling akan memanggil start_detection() atau stop_detection() yang mempengaruhi broom_detector.
    app.run(host="127.0.0.1", port=5000, debug=True)
