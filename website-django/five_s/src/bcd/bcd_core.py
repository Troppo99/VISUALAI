import cv2, cvzone, json, math, numpy as np, os, queue, threading, time, torch, sys
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)
from libs.DataHandler import DataHandler


class BroCarpDetector:
    def __init__(self, confidence_threshold=0.5, video_source=None, camera_name=None, window_size=(320, 240), stop_event=None):
        self.stop_event = stop_event
        if self.stop_event is None:
            self.stop_event = threading.Event()
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (640, 640)
        self.rois, self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.prev_frame_time = 0
        self.model_broom = YOLO(r"\\10.5.0.3\VISUALAI\website-django\static\resources\models\bd6l.pt").to("cuda")
        self.model_broom.overrides["verbose"] = False
        self.model_carpal = YOLO(r"\\10.5.0.3\VISUALAI\website-django\static\resources\models\yolo11l-pose.pt").to("cuda")
        self.model_carpal.overrides["verbose"] = False

        if len(self.rois) > 1:
            self.union_roi = unary_union(self.rois)
        elif len(self.rois) == 1:
            self.union_roi = self.rois[0]
        else:
            self.union_roi = None

        self.trail_map_polygon = Polygon()
        self.trail_map_mask = np.zeros((self.process_size[1], self.process_size[0], 3), dtype=np.uint8)

        self.last_detection_time = None
        self.trail_map_start_time = None
        self.start_run_time = time.time()
        self.capture_done = False
        self.pairs_human = [(0, 1), (0, 2), (1, 2), (2, 4), (1, 3), (4, 6), (3, 5), (5, 6), (6, 8), (8, 10), (5, 7), (7, 9), (6, 12), (12, 11), (11, 5), (12, 14), (14, 16), (11, 13), (13, 15)]

    def camera_config(self):
        with open(r"\\10.5.0.3\VISUALAI\website-django\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        scaled_rois = []
        rois_path = config[self.camera_name]["bcd_rois"]
        with open(rois_path, "r") as rois_file:
            original_rois = json.load(rois_file)
        for roi_group in original_rois:
            scaled_group = []
            for x, y in roi_group:
                scaled_x = int(x * (960 / 1280))
                scaled_y = int(y * (540 / 720))
                scaled_group.append((scaled_x, scaled_y))
            if len(scaled_group) >= 3:
                polygon = Polygon(scaled_group)
                if polygon.is_valid:
                    scaled_rois.append(polygon)
        return scaled_rois, ip

    def choose_video_source(self):
        if self.video_source is None:
            self.frame_queue = queue.Queue(maxsize=10)
            self.frame_thread = None
            self.video_fps = None
            self.is_local_video = False
            self.video_source = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"
        else:
            if os.path.isfile(self.video_source):
                self.is_local_video = True
                cap = cv2.VideoCapture(self.video_source)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25
                cap.release()
            else:
                self.is_local_video = False
                self.video_fps = None

    def capture_frame(self):
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                cap.release()
                time.sleep(5)
                continue
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    time.sleep(5)
                    break
                try:
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    pass
            cap.release()

    def export_frame(self):
        pass

    def process_frame(self):
        pass

    def main(self):
        print("anda masuk main")
        pass


if __name__ == "__main__":
    bcd = BroCarpDetector()
    bcd.main()
