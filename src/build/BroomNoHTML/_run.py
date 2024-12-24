import time
from Scheduling import Scheduling

if __name__ == "__main__":
    detector_args = {
        "confidence_threshold": 0,
        "camera_name": "OFFICE1",
        # "video_source": "static/videos/bd_test3.mp4",
        "window_size": (320, 240),
    }

    scheduler = Scheduling(detector_args, "OFFICE")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
        scheduler.shutdown()
