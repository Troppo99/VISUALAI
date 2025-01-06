import os
import sys


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, "..")
    sys.path.append(parent_dir)
    from bd_test import BroomDetector

    detector_args = {
        "confidence_threshold": 0,
        "camera_name": "SEWING1",
        # "video_source": r"C:\xampp\htdocs\VISUALAI\archives\static\videos\bd_test.mp4",
        "window_size": (320, 240),
    }

    detector = BroomDetector(**detector_args)
    detector.main()
