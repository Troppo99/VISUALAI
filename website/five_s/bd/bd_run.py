import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone
from dist.bd_test import BroomDetector
import time


class Scheduling:
    def __init__(self, detector_args, broom_schedule_type):
        self.detector_args = detector_args
        self.broom_schedule_type = broom_schedule_type
        self.detector = None
        self.scheduler = BackgroundScheduler(
            timezone=timezone("Asia/Jakarta"),
            job_defaults={"misfire_grace_time": 180},
        )
        self.setup_schedule()
        self.scheduler.start()

    def start_detection(self):
        # playsound("output_gtts.mp3")
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

    def setup_schedule(self):
        if self.broom_schedule_type == "OFFICE":
            work_days = ["mon", "tue", "wed", "thu", "fri"]
            for day in work_days:
                # S1 : 06:00 - 08:30
                h1, m1, s1 = (6, 0, 0)
                h2, m2, s2 = (8, 30, 0)
                start_trigger = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                self.scheduler.add_job(self.start_detection, trigger=start_trigger, id=f"start_{day}", replace_existing=True)
                stop_trigger = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                self.scheduler.add_job(self.stop_detection, trigger=stop_trigger, id=f"stop_{day}", replace_existing=True)
        elif self.broom_schedule_type == "SEWING":
            work_days = ["mon", "tue", "wed", "thu", "fri"]
            for day in work_days:
                # S1 : 07:30 - 09:45
                # S2 : 09:45 - 12:50
                # S3 : 12:50 - 13:05
                h1, m1, s1 = (10, 12, 30)
                h2, m2, s2 = (10, 13, 30)
                h3, m3, s3 = (11, 20, 0)
                h4, m4, s4 = (12, 50, 0)
                h5, m5, s5 = (12, 45, 0)
                h6, m6, s6 = (13, 5, 0)
                s1_start = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                s1_stop = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                s2_start = CronTrigger(day_of_week=day, hour=h3, minute=m3, second=s3)
                s2_stop = CronTrigger(day_of_week=day, hour=h4, minute=m4, second=s4)
                s3_start = CronTrigger(day_of_week=day, hour=h5, minute=m5, second=s5)
                s3_stop = CronTrigger(day_of_week=day, hour=h6, minute=m6, second=s6)

                self.scheduler.add_job(self.start_detection, trigger=s1_start, id=f"s1_start_{day}", replace_existing=True)
                self.scheduler.add_job(self.stop_detection, trigger=s1_stop, id=f"s1_stop_{day}", replace_existing=True)

                self.scheduler.add_job(self.start_detection, trigger=s2_start, id=f"s2_start_{day}", replace_existing=True)
                self.scheduler.add_job(self.stop_detection, trigger=s2_stop, id=f"s2_stop_{day}", replace_existing=True)

                self.scheduler.add_job(self.start_detection, trigger=s3_start, id=f"s3_start_{day}", replace_existing=True)
                self.scheduler.add_job(self.stop_detection, trigger=s3_stop, id=f"s3_stop_{day}", replace_existing=True)

    def shutdown(self):
        print("Shutdown scheduler and BroomDetector if not running...")
        self.scheduler.shutdown(wait=False)
        self.stop_detection()


if __name__ == "__main__":
    detector_args = {
        "confidence_threshold": 0,
        "camera_name": "SEWINGBACK1",
        # "video_source": "static/videos/bd_test3.mp4",
        "window_size": (320, 240),
    }

    scheduler = Scheduling(detector_args, "SEWING")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
        scheduler.shutdown()
