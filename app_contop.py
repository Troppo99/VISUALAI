from flask import Flask, render_template, Response
from src.ContopDetector import ContopDetector  # Import your ContopDetector
import threading
import atexit
import signal
import sys

app = Flask(__name__)

# Detector configuration arguments
detector_args = {
    "contop_confidence_threshold": 0.5,
    "camera_name": "FREEMETAL1",
    # "video_source": "static/videos/contop_test3.mp4",  # Update to your actual video source
    "window_size": (540, 360),
}

# Initialize ContopDetector
contop_detector = ContopDetector(**detector_args)

# Start the detector in a separate thread to avoid blocking
detector_thread = threading.Thread(target=contop_detector.start)
detector_thread.daemon = True  # Ensure thread exits when main program does
detector_thread.start()


def shutdown_detector(signum, frame):
    print("\nShutting down detector...")
    contop_detector.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_detector)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, shutdown_detector)  # Handle termination


@app.route("/video_feed")
def video_feed():
    return Response(
        contop_detector.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return render_template("index.html")


@atexit.register
def cleanup():
    print("Application is exiting. Stopping detector...")
    contop_detector.stop()


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        contop_detector.stop()
