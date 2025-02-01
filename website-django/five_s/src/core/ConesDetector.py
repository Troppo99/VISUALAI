import os, cv2, torch, cvzone, time, threading, queue, math, numpy as np, pymysql, json, sys
from shapely.geometry import Polygon
from ultralytics import YOLO
from datetime import datetime

sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")
from libs.DataHandler import DataHandler


class ConesDetector:
    def __init__(self, confidence_threshold=0.0, video_source=None, camera_name=None, window_size=(320, 240), stop_event=None, is_insert=False):
        self.stop_event = stop_event
        if self.stop_event is None:
            self.stop_event = threading.Event()
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (640, 640)
        self.rois, self.ip_camera = self.camera_config()
        # self.choose_video_source()
        self.prev_frame_time = 0
        self.model = YOLO("D:/NWR/run/kon/version2/weights/best.pt").to("cuda")
        self.model.overrides["verbose"] = False

        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = (960, 540)
        self.fps = 0
        self.borders, self.ip_camera = self.camera_config()

        self.is_local_file = False
        if video_source is not None:
            self.video_source = video_source
            if os.path.isfile(video_source):
                self.is_local_file = True
                cap = cv2.VideoCapture(self.video_source)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25
                cap.release()
                print(f"B`{self.camera_name} : Local video file detected. FPS: {self.video_fps}")
            else:
                self.is_local_file = False
                print(f"B`{self.camera_name} : RTSP stream detected. URL: {self.video_source}")
                self.video_fps = None
        else:
            self.video_source = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"
            self.video_fps = None
            self.is_local_file = False

        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None

        self.no_detection_duration = 3
        self.violation_start_time = None
        self.last_detected_time = None
        self.timestamp_start = None

    def camera_config(self):
        with open(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        scaled_rois = []
        rois_path = config[self.camera_name]["cnd_rois"]
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

    def draw_rois(self, frame):
        if not self.rois:
            return
        for roi in self.rois:
            if roi.geom_type != "Polygon":
                continue
            pts = np.array(roi.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

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

    def export_frame_segment(self, frame):
        with torch.no_grad():
            results = self.model.predict(frame, stream=True, imgsz=self.process_size[0], task="segmentation")
        boxes_info = []
        overlap_detected = False
        for result in results:
            if result.masks is None:
                continue
            for poly_xy in result.masks.xy:
                if len(poly_xy) < 3:
                    # Jika titik polygon kurang dari 3, tidak dapat membentuk polygon yang valid
                    continue
                polygon = Polygon(poly_xy)
                if polygon.is_empty or not polygon.is_valid:
                    # Lewati polygon kosong atau tidak valid
                    continue
                poly_area = polygon.area
                intersection_area_sum = 0.0
                for border in self.borders:
                    if polygon.intersects(border):
                        inter = polygon.intersection(border)
                        if not inter.is_empty:
                            intersection_area_sum += inter.area
                inside = False
                if intersection_area_sum > 0.5 * poly_area:
                    inside = True
                    overlap_detected = True

                # Pastikan polygon tidak kosong sebelum mengambil centroid
                if not polygon.is_empty:
                    c = polygon.centroid
                    boxes_info.append((poly_xy, inside, (c.x, c.y)))
        return boxes_info, overlap_detected

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, self.process_size)
        self.draw_rois(frame_resized)
        boxes_info, overlap_detected = self.export_frame_segment(frame_resized)

        any_inside = any(bi[1] for bi in boxes_info)
        if any_inside:
            self.last_detected_time = current_time
            if self.violation_start_time is None:
                self.violation_start_time = current_time
        else:
            if self.last_detected_time is not None:
                if (current_time - self.last_detected_time) > self.no_detection_duration:
                    self.violation_start_time = None
                    self.last_detected_time = None

        if overlap_detected and self.timestamp_start is None:
            self.timestamp_start = datetime.now()

        overlay = frame_resized.copy()

        for poly_xy, inside, (cx, cy) in boxes_info:
            pts = np.array(poly_xy, np.int32).reshape((-1, 1, 2))
            if inside:
                cv2.fillPoly(overlay, [pts], (0, 70, 255))
            else:
                cv2.fillPoly(overlay, [pts], (0, 255, 255))

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)

        for poly_xy, inside, (cx, cy) in boxes_info:
            if inside:
                if self.violation_start_time is not None:
                    elapsed = current_time - self.violation_start_time
                    hh = int(elapsed // 3600)
                    mm = int((elapsed % 3600) // 60)
                    ss = int(elapsed % 60)
                    timer_str = f"{hh:02}:{mm:02}:{ss:02}"
                    cvzone.putTextRect(frame_resized, timer_str, (int(cx), int(cy) - 40), scale=1, thickness=2, offset=5, colorR=(0, 70, 255), colorT=(255, 255, 255))
                cvzone.putTextRect(frame_resized, "Violation!", (int(cx), int(cy) - 10), scale=1, thickness=2, offset=5, colorR=(0, 70, 255), colorT=(255, 255, 255))
            else:
                cvzone.putTextRect(frame_resized, "Warning!", (int(cx), int(cy) - 10), scale=1, thickness=2, offset=5, colorR=(0, 255, 255), colorT=(0, 0, 0))

        return frame_resized, overlap_detected

    def check_conditions(self, percentage, overlap_detected, current_time, frame_resized):
        pass

    def main(self):
        process_every_n_frames = 2
        frame_count = 0
        window_name = f"BROOM : {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        if self.is_local_file:
            cap = cv2.VideoCapture(self.video_source)
            frame_delay = int(1000 / self.video_fps)
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print(f"B`{self.camera_name} : End of video file or cannot read the frame. Restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                self.prev_frame_time = current_time
                frame_resized, overlap_detected = self.process_frame(frame, current_time)
                percentage = 0
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
                self.check_conditions(percentage, overlap_detected, current_time, frame_resized)
                cv2.imshow(window_name, frame_resized)
                processing_time = (time.time() - start_time) * 1000
                adjusted_delay = max(int(frame_delay - processing_time), 1)
                key = cv2.waitKey(adjusted_delay) & 0xFF
                if key == ord("n"):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            self.frame_thread = threading.Thread(target=self.frame_capture)
            self.frame_thread.daemon = True
            self.frame_thread.start()
            while True:
                if self.stop_event.is_set():
                    break
                try:
                    frame = self.frame_queue.get(timeout=5)
                except queue.Empty:
                    continue
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                self.prev_frame_time = current_time
                frame_resized, overlap_detected = self.process_frame(frame, current_time)
                percentage = 0
                cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 75), scale=1, thickness=2, offset=5)
                self.check_conditions(percentage, overlap_detected, current_time, frame_resized)
                cv2.imshow(window_name, frame_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    self.stop_event.set()
                    break
            cv2.destroyAllWindows()
            self.frame_thread.join()


if __name__ == "__main__":
    cnd = ConesDetector(
        camera_name="CUTTING8",
        video_source=r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\videos\cones.mp4",
        is_insert=False,
    )
    cnd.main()
