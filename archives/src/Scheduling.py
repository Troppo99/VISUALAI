import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone
from datetime import datetime
import time

# pastikan import BroomDetector dan DataHandler jika diperlukan


class Scheduling:
    def __init__(self, detector_instance, broom_schedule_type):
        self.detector = detector_instance
        self.broom_schedule_type = broom_schedule_type
        self.scheduler = BackgroundScheduler(timezone=timezone("Asia/Jakarta"))
        self.setup_schedule()
        self.scheduler.start()

    def start_detection(self):
        if self.detector and not self.detector.stop_event.is_set() and self.detector.frame_thread and self.detector.frame_thread.is_alive():
            print("BroomDetector sudah berjalan.")
        elif self.detector:
            print("Memulai BroomDetector...")
            self.detector.start()
        else:
            print("Tidak ada instance BroomDetector yang tersedia.")

    def stop_detection(self):
        if self.detector and not self.detector.stop_event.is_set() and self.detector.frame_thread and self.detector.frame_thread.is_alive():
            print("Menghentikan BroomDetector...")
            self.detector.stop()
        elif self.detector:
            print("BroomDetector sudah dihentikan.")
        else:
            print("Tidak ada instance BroomDetector yang tersedia.")

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
                h1, m1, s1 = (8, 25, 0)
                h2, m2, s2 = (8, 25, 30)
                h3, m3, s3 = (8, 25, 50)
                h4, m4, s4 = (8, 26, 0)
                h5, m5, s5 = (8, 26, 10)
                h6, m6, s6 = (8, 26, 30)
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
