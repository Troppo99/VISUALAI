import time
from Scheduling import Scheduling

if __name__ == "__main__":
    detector_args = {
        "confidence_threshold": 0.5,
        "camera_name": "OFFICE3",
        "video_source": "src/videos/bd_test3.mp4",
        "window_size": (320, 240),
    }

    scheduler = Scheduling(detector_args, "SEWING")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
        scheduler.shutdown()
