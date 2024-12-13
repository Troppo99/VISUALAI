# app.py
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)


def camera_feed():
    cap = cv2.VideoCapture("rtsp://admin:oracle2015@10.5.0.170:554/Streaming/Channels/1")

    while True:
        _, frame = cap.read()

        frame = cv2.imencode(".jpg", frame)[1].tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route("/video_feed")
def video_feed():
    return Response(camera_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # app.run(debug=True, host="0.0.0.0") # memunculkan ip lokal
    app.run(host="127.0.0.1", port=5000, debug=True)
