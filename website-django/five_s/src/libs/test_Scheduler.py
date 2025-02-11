import threading, sys
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone

sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")


class Scheduler:
    def __init__(self, detector_args, schedule_config, code):
        self.schedule_config = schedule_config
        self.lock = threading.Lock()
        self.detector_args = detector_args
        self.detector = None
        with self.lock:
            if code == "bd":
                from core.BroomDetector import BroomDetector as Detector
            elif code == "cd":
                from core.CarpalDetector import CarpalDetector as Detector
            elif code == "bcd":
                from core.BroCarpDetector import BroCarpDetector as Detector
            elif code == "cnd":
                from core.ConesDetector import ConesDetector as Detector

        self.Detector = Detector

        self.scheduler = BackgroundScheduler(
            timezone=timezone("Asia/Jakarta"),
            job_defaults={"misfire_grace_time": 180},
        )

        self.setup_schedule()  # pasang job
        self.scheduler.start()

    def start_detection(self):
        if not self.detector:
            print("Starting Program for camera:", self.detector_args.get("camera_name"))
            self.detector = self.Detector(**self.detector_args)
            detection_thread = threading.Thread(target=self.detector.main, daemon=True)
            detection_thread.start()
        else:
            print("Program is already running for camera:", self.detector_args.get("camera_name"))

    def stop_detection(self):
        if self.detector:
            print("Stopping Program for camera:", self.detector_args.get("camera_name"))
            self.detector.stop_event.set()
            self.detector = None
        else:
            print("Program is not running.")

    def setup_schedule(self):
        with self.lock:
            schedule_config = self.schedule_config
            work_days = schedule_config.get("work_days", [])
            time_ranges = schedule_config.get("time_ranges", [])

            if not work_days or not time_ranges:
                print("No valid schedule_config given. No scheduling will be done.")
                return

            # Daftarkan job Cron start-stop
            for day in work_days:
                for idx, (start_time, stop_time) in enumerate(time_ranges, start=1):
                    h1, m1, s1 = start_time
                    h2, m2, s2 = stop_time

                    start_trigger = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                    self.scheduler.add_job(self.start_detection, trigger=start_trigger, id=f"start_{day}_{idx}", replace_existing=True)

                    stop_trigger = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                    self.scheduler.add_job(self.stop_detection, trigger=stop_trigger, id=f"stop_{day}_{idx}", replace_existing=True)

    def shutdown(self):
        print("Shutdown scheduler and Program if running...")
        self.scheduler.shutdown(wait=False)
        self.stop_detection()
