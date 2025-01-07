# bd_run.py
import os
import sys
import time
from multiprocessing import Process, Manager
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)

try:
    from bd_test import BroomDetector
except ImportError as e:
    print(f"Error importing BroomDetector: {e}")
    sys.exit(1)

processes = {}
stop_events = {}


def run_broom_detector(detector_args):
    # top-level function agar bisa di-pickle
    detector = BroomDetector(**detector_args)
    detector.main()


def start_detector(camera_name):
    if camera_name in processes and processes[camera_name].is_alive():
        print(f"{camera_name} is already running.")
        return

    event = stop_events[camera_name]
    detector_args = {"confidence_threshold": 0, "camera_name": camera_name, "video_source": None, "window_size": (320, 240), "stop_event": event}
    p = Process(target=run_broom_detector, args=(detector_args,))
    p.start()
    processes[camera_name] = p
    print(f"Started process for camera: {camera_name}")


def stop_detector(camera_name):
    p = processes.get(camera_name)
    if p and p.is_alive():
        print(f"Requesting {camera_name} to stop...")
        event = stop_events[camera_name]
        event.set()
        p.join(10)
        if p.is_alive():
            print(f"Forcing terminate {camera_name}.")
            p.terminate()
        print(f"Stopped process for camera: {camera_name}")
    else:
        print(f"No active process found for camera: {camera_name}")


def schedule_start_detector(camera_name):
    # Fungsi global, agar APScheduler bisa mempicklenya
    start_detector(camera_name)


def schedule_stop_detector(camera_name):
    stop_detector(camera_name)


def setup_schedule(camera_list):
    scheduler = BackgroundScheduler()
    for cam in camera_list:
        start_job = CronTrigger(hour=8, minute=31)
        stop_job = CronTrigger(hour=8, minute=32)

        scheduler.add_job(schedule_start_detector, trigger=start_job, args=[cam], id=f"start_{cam}", replace_existing=True)

        scheduler.add_job(schedule_stop_detector, trigger=stop_job, args=[cam], id=f"stop_{cam}", replace_existing=True)
    scheduler.start()
    return scheduler


if __name__ == "__main__":
    # Di Windows, wajib di bawah if __name__ == "__main__"
    manager = Manager()
    stop_events = manager.dict()

    camera_names = [
        "SEWING1",
        "SEWING2",
        "SEWING3",
        "SEWING4",
        "SEWING5",
        "SEWING6",
        "SEWING7",
        "SEWINGOFFICE",
    ]
    # Buat event untuk setiap kamera
    for cam in camera_names:
        stop_events[cam] = manager.Event()

    sched = setup_schedule(camera_names)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt. Stopping all...")
        for name in camera_names:
            stop_detector(name)
        sched.shutdown(wait=False)
        print("Scheduler shutdown.")
    finally:
        print("Main script ended.")
