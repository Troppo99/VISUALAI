import threading, os, sys
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone

sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")


class Scheduler:
    def __init__(self, detector_args, schedule_type):
        self.lock = threading.Lock()
        with self.lock:
            if schedule_type in ["bd_office", "bd_sewing"]:
                from bd.bd_core import BroomDetector as Detector
            elif schedule_type == "cd":
                from cd.cd_core import CarpalDetector as Detector
            elif schedule_type == "bcd":
                from bcd.bcd_core import BroCarpDetector as Detector
        self.Detector = Detector
        self.detector_args = detector_args
        self.schedule_type = schedule_type
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
            print("Starting Program...")
            self.detector = self.Detector(**self.detector_args)
            detection_thread = threading.Thread(target=self.detector.main)
            detection_thread.daemon = True
            detection_thread.start()
        else:
            print("Program is already running.")

    def stop_detection(self):
        if self.detector:
            print("Stopping Program...")
            self.detector.stop_event.set()
            self.detector = None
        else:
            print("Program is not running.")

    def setup_schedule(self):
        with self.lock:
            work_days = ["mon", "tue", "wed", "thu", "fri"]
            time_ranges = []

            if self.schedule_type == "bd_office":
                time_ranges = [((9, 57, 0), (9, 57, 10))]

            elif self.schedule_type == "bd_sewing":
                time_ranges = [
                    # S1 : 07:30 - 09:45
                    # S2 : 09:45 - 12:50
                    # S3 : 12:50 - 13:05
                    ((7, 30, 0), (9, 44, 0)),  # S1
                    ((9, 45, 0), (12, 49, 0)),  # S2
                    ((12, 50, 0), (13, 5, 0)),  # S3
                ]

            elif self.schedule_type == "cd":
                time_ranges = [((6, 0, 0), (8, 30, 0))]

            elif self.schedule_type == "bcd":
                work_days = ["sat"]
                time_ranges = [((6, 0, 0), (10, 0, 0))]

            for day in work_days:
                for idx, (start_time, stop_time) in enumerate(time_ranges, start=1):
                    h1, m1, s1 = start_time
                    h2, m2, s2 = stop_time
                    start_trigger = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                    job_id_start = f"{self.schedule_type}_start_{day}_{idx}"
                    self.scheduler.add_job(self.start_detection, trigger=start_trigger, id=job_id_start, replace_existing=True)
                    stop_trigger = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                    job_id_stop = f"{self.schedule_type}_stop_{day}_{idx}"
                    self.scheduler.add_job(self.stop_detection, trigger=stop_trigger, id=job_id_stop, replace_existing=True)

    def shutdown(self):
        print("Shutdown scheduler and Program if not running...")
        self.scheduler.shutdown(wait=False)
        self.stop_detection()
