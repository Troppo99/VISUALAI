import threading, time, numpy as np, json, cv2, os, queue, math, torch, cvzone
from ultralytics import YOLO
from shapely.geometry import Polygon


class SpreadingManual:
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
        self.model = YOLO(r"C:\xampp\htdocs\VISUALAI\website\static\resources\models\yolo11l.pt").to("cuda")
        self.model.overrides["verbose"] = False

        # if len(self.rois) > 1:
        #     self.union_roi = unary_union(self.rois)
        # elif len(self.rois) == 1:
        #     self.union_roi = self.rois[0]
        # else:
        # self.union_roi = None

        self.trail_map_polygon = Polygon()
        self.trail_map_mask = np.zeros((self.process_size[1], self.process_size[0], 3), dtype=np.uint8)

        self.last_detection_time = None
        self.trail_map_start_time = None
        self.start_run_time = time.time()
        self.capture_done = False

    def camera_config(self):
        with open(r"C:\xampp\htdocs\VISUALAI\website\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        scaled_rois = []
        rois_path = config[self.camera_name]["sm_rois"]
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

    # def draw_rois(self, frame):
    #     if not self.rois:
    #         return
    #     for roi in self.rois:
    #         if roi.geom_type != "Polygon":
    #             continue
    #         pts = np.array(roi.exterior.coords, np.int32)
    #         pts = pts.reshape((-1, 1, 2))
    #         cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

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

    def export_frame(self, frame):
        with torch.no_grad():
            results = self.model(frame, stream=True, imgsz=self.process_size[0])
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = self.model.names[int(box.cls[0])]
                if conf > self.confidence_threshold:
                    boxes.append((x1, y1, x2, y2, class_id))
        return boxes

    def check_overlap(self, x1, y1, x2, y2):
        main_box = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        overlap_results = []
        for roi in self.rois:
            intersection_area = main_box.intersection(roi).area
            union_area = main_box.union(roi).area
            iou = intersection_area / union_area if union_area != 0 else 0
            overlap_results.append(iou > 0)
        return overlap_results

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, self.process_size)
        boxes = self.export_frame(frame_resized)
        output_frame = frame_resized.copy()
        detected = False
        for box in boxes:
            x1, y1, x2, y2, class_id = box
            overlap_results = self.check_overlap(x1, y1, x2, y2)
            if any(overlap_results):
                cvzone.cornerRect(output_frame, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
                cvzone.putTextRect(output_frame, f"{class_id}", (x1, y1 - 15))
                detected = True
            # if detected:
            #     pass
        return output_frame

    def main(self):
        # state = ""
        skip_frames = 2
        frame_count = 0
        window_name = f"SM:{self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size)

        try:
            if self.video_fps is None:
                self.frame_queue = queue.Queue(maxsize=10)
                self.frame_thread = threading.Thread(target=self.capture_frame)
                self.frame_thread.daemon = True
                self.frame_thread.start()

                while not self.stop_event.is_set():
                    try:
                        frame = self.frame_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue
                    current_time = time.time()
                    time_diff = current_time - self.prev_frame_time
                    self.fps = 1 / time_diff if time_diff > 0 else 0
                    self.prev_frame_time = current_time
                    frame_resized = self.process_frame(frame)
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 90), scale=1, thickness=2, offset=5)
                    cv2.imshow(window_name, frame_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("n") or key == ord("N"):
                        print("Manual stop detected.")
                        self.stop_event.set()
                        break

                cv2.destroyAllWindows()
                if self.frame_thread.is_alive():
                    self.frame_thread.join()
            else:
                cap = cv2.VideoCapture(self.video_source)
                frame_delay = int(1000 / self.video_fps)
                while cap.isOpened() and not self.stop_event.is_set():
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        print("Video ended.")
                        break
                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue
                    current_time = time.time()
                    time_diff = current_time - self.prev_frame_time
                    self.fps = 1 / time_diff if time_diff > 0 else 0
                    self.prev_frame_time = current_time
                    frame_resized = self.process_frame(frame)
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 90), scale=1, thickness=2, offset=5)
                    cv2.imshow(window_name, frame_resized)
                    processing_time = (time.time() - start_time) * 1000
                    adjusted_delay = max(int(frame_delay - processing_time), 1)
                    key = cv2.waitKey(adjusted_delay) & 0xFF
                    if key == ord("n") or key == ord("N"):
                        print("Manual stop detected.")
                        self.stop_event.set()
                        break
                cap.release()
                cv2.destroyAllWindows()
        finally:
            pass


if __name__ == "__main__":
    sm = SpreadingManual(camera_name="CUTTING4", video_source=r"C:\xampp\htdocs\VISUALAI\website\static\videos\spreading_manual.mp4", window_size=(960, 540))
    sm.main()
