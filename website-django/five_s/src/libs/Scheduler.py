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
            if self.schedule_type == "bd_office":
                work_days = ["mon", "tue", "wed", "thu", "fri"]
                for day in work_days:
                    # S1 : 06:00 - 08:30
                    h1, m1, s1 = (9, 57, 0)
                    h2, m2, s2 = (9, 57, 10)
                    start_trigger = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                    self.scheduler.add_job(self.start_detection, trigger=start_trigger, id=f"start_{day}", replace_existing=True)
                    stop_trigger = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                    self.scheduler.add_job(self.stop_detection, trigger=stop_trigger, id=f"stop_{day}", replace_existing=True)
            elif self.schedule_type == "bd_sewing":
                work_days = ["mon", "tue", "wed", "thu", "fri"]
                for day in work_days:
                    # S1 : 07:30 - 09:45
                    # S2 : 09:45 - 12:50
                    # S3 : 12:50 - 13:05
                    h1, m1, s1 = (14, 47, 50)
                    h2, m2, s2 = (14, 47, 55)
                    h3, m3, s3 = (14, 48, 0)
                    h4, m4, s4 = (14, 48, 10)
                    h5, m5, s5 = (14, 48, 40)
                    h6, m6, s6 = (14, 48, 50)
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
            elif self.schedule_type == "cd":
                work_days = ["mon", "tue", "wed", "thu", "fri"]
                for day in work_days:
                    # S1 : 06:00 - 08:30
                    h1, m1, s1 = (15, 5, 50)
                    h2, m2, s2 = (15, 6, 0)
                    start_trigger = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                    self.scheduler.add_job(self.start_detection, trigger=start_trigger, id=f"start_{day}", replace_existing=True)
                    stop_trigger = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                    self.scheduler.add_job(self.stop_detection, trigger=stop_trigger, id=f"stop_{day}", replace_existing=True)
            elif self.schedule_type == "bcd":
                work_days = ["mon", "tue", "wed", "thu", "fri"]
                for day in work_days:
                    # S1 : 06:00 - 08:30
                    h1, m1, s1 = (15, 19, 10)
                    h2, m2, s2 = (15, 19, 30)
                    start_trigger = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                    self.scheduler.add_job(self.start_detection, trigger=start_trigger, id=f"start_{day}", replace_existing=True)
                    stop_trigger = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                    self.scheduler.add_job(self.stop_detection, trigger=stop_trigger, id=f"stop_{day}", replace_existing=True)
            elif self.schedule_type == "ctd":
                pass
            elif self.schedule_type == "dd":
                pass
            elif self.schedule_type == "bcd":
                pass
            elif self.schedule_type == "bcd":
                pass

    def shutdown(self):
        print("Shutdown scheduler and Program if not running...")
        self.scheduler.shutdown(wait=False)
        self.stop_detection()
