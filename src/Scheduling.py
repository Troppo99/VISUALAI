import time
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone
from BroomDetector import BroomDetector


class Scheduling:
    def __init__(self, detector_args):
        self.detector_args = detector_args
        self.detector = None
        self.scheduler = BackgroundScheduler(timezone=timezone("Asia/Jakarta"))
        self.setup_schedule_office()
        self.scheduler.start()

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

    def setup_schedule_office(self):
        work_days = ["mon", "tue", "wed", "thu", "fri"]

        for day in work_days:
            start_trigger = CronTrigger(day_of_week=day, hour=6, minute=0, second=0)
            self.scheduler.add_job(self.start_detection, trigger=start_trigger, id=f"start_{day}", replace_existing=True)

            stop_trigger = CronTrigger(day_of_week=day, hour=8, minute=30, second=0)
            self.scheduler.add_job(self.stop_detection, trigger=stop_trigger, id=f"stop_{day}", replace_existing=True)

    def shutdown(self):
        print("Shutdown scheduler and BroomDetector if not running...")
        self.scheduler.shutdown(wait=False)
        self.stop_detection()
