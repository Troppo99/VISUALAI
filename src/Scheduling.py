import schedule
import time
import threading
from BroomDetector import BroomDetector


class Scheduling:
    def __init__(self, detector_args):
        self.detector_args = detector_args
        self.detector = None
        self.scheduler_thread = threading.Thread(target=self.run_schedule)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

    def start_detection(self):
        if not self.detector:
            print("Starting BroomDetector...")
            self.detector = BroomDetector(**self.detector_args)
            detection_thread = threading.Thread(target=self.detector.main)
            detection_thread.daemon = True
            detection_thread.start()
        else:
            print("BroomDetector is already running.")

    def stop_detection(self):
        if self.detector:
            print("Stopping BroomDetector...")
            self.detector.stop_event.set()
            self.detector = None
        else:
            print("BroomDetector is not running.")

    def run_schedule(self):
        schedule.every().day.at("15:28").do(self.start_detection)
        schedule.every().day.at("15:29").do(self.stop_detection)

        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
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
        scheduler.stop_detection()
