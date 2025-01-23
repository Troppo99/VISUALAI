import threading, sys
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone

# Pastikan folder 'libs' dan 'core' punya __init__.py
# Sesuaikan path agar Python mengenali libs.
# (Boleh dibiarkan kalau sdh di-append oleh run-xxx.py)
# sys.path.append(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\src")
sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")


class Scheduler:
    def __init__(self, detector_args, schedule_config):
        self.schedule_config = schedule_config
        """
        detector_args: {
            "camera_name": str,
            "schedule_config": { "work_days": [...], "time_ranges": [...] },
            ...
        }
        """
        self.lock = threading.Lock()
        self.detector_args = detector_args
        self.detector = None

        # (OPSIONAL) Pilihan Detector berdasarkan param lain?
        # Contoh: if schedule_type in ["bd_office", ...], dsb.
        # Atau jika selalu BroCarpDetector, di sini:
        from core.BroCarpDetector import BroCarpDetector as Detector

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
