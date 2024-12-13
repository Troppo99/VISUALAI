import time
from Scheduling import Scheduling


detector_args = {
    "confidence_threshold": 0.5,
    "camera_name": "OFFICE3",
    # "video_source": "static/videos/bd_test3.mp4",
    "window_size": (320, 240),
}

scheduler = Scheduling(detector_args)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated by user.")
    scheduler.shutdown()
