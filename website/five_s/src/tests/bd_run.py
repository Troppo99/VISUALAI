import os
import sys
from multiprocessing import Process


def run_detector(camera_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, "..")

    sys.path.append(parent_dir)

    try:
        from bd_test import BroomDetector
    except ImportError as e:
        print(f"Error importing BroomDetector: {e}")
        return

    detector_args = {
        "confidence_threshold": 0,
        "camera_name": camera_name,
        # "video_source": r"C:\xampp\htdocs\VISUALAI\archives\static\videos\bd_test.mp4",
        "window_size": (320, 240),
    }

    detector = BroomDetector(**detector_args)
    detector.main()


if __name__ == "__main__":
    camera_names = [
        "SEWING1",
        "SEWING2",
        "SEWING3",
    ]
    processes = []

    for name in camera_names:
        p = Process(target=run_detector, args=(name,))
        p.start()
        processes.append(p)
        print(f"Started process for camera: {name}")

    for p in processes:
        p.join()
        print(f"Process {p.pid} has finished.")

    print("All detector processes have completed.")
