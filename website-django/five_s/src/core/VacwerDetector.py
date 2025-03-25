import os, cv2, torch, cvzone, time, threading, queue, math, numpy as np, json, sys
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from datetime import datetime


class VacwerDetector:
    def __init__(self, confidence_threshold=0.0, video_source=None, camera_name=None, window_size=(320, 240), stop_event=None):
        self.stop_event = stop_event if stop_event else threading.Event()
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (640, 640)

        self.rois, self.ip_camera = self.camera_config()
        self.prev_frame_time = 0
        self.model = YOLO(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\models\yolo11l.pt").to("cuda")
        self.model.overrides["verbose"] = False

        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = (960, 540)
        self.fps = 0
        self.borders, self.ip_camera = self.camera_config()

        self.is_local_file = False
        self.video_fps = None
        if video_source:
            if os.path.isfile(video_source):
                self.is_local_file = True
                cap = cv2.VideoCapture(self.video_source)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25
                cap.release()
        else:
            self.video_source = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"

        self.frame_queue = queue.Queue(maxsize=10)
        self.frame_thread = None

    def camera_config(self):
        with open(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        scaled_rois = []
        rois_path = config[self.camera_name]["vb_rois"]
        with open(rois_path, "r") as rois_file:
            original_rois = json.load(rois_file)
        for roi_group in original_rois:
            scaled_group = []
            for x, y in roi_group:
                sx = int(x * (960 / 1280))
                sy = int(y * (540 / 720))
                scaled_group.append((sx, sy))
            if len(scaled_group) >= 3:
                poly = Polygon(scaled_group)
                if poly.is_valid:
                    scaled_rois.append(poly)
        return scaled_rois, ip

    def draw_rois(self, frame):
        if not self.rois:
            return
        for roi in self.rois:
            if roi.geom_type != "Polygon":
                continue
            pts = np.array(roi.exterior.coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

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
            results = self.model.predict(frame, stream=True, imgsz=self.process_size[0])
        boxes_info = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                if conf < self.confidence_threshold and cls_id != "person":
                    continue
                label = self.model.names[cls_id] if cls_id in self.model.names else f"class_{cls_id}"
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                center = Point(cx, cy)
                inside_roi = any(center.within(roi) for roi in self.rois)
                if inside_roi:
                    boxes_info.append((x1, y1, x2, y2, label, conf))
        overlap_detected = len(boxes_info) > 0
        return boxes_info, overlap_detected

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, self.process_size)
        self.draw_rois(frame_resized)
        boxes_info, overlap_detected = self.export_frame(frame_resized)
        for x1, y1, x2, y2, label, conf in boxes_info:
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            txt = f"{label} {conf:.2f}"
            cvzone.putTextRect(frame_resized, txt, (int(x1), int(y1) - 10), scale=0.8, thickness=1, offset=5)
        return frame_resized, overlap_detected

    def main(self):
        skip_frames = 2
        frame_count = 0
        window_name = f"CND:{self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size)

        try:
            if self.video_fps is None:
                self.frame_thread = threading.Thread(target=self.capture_frame, daemon=True)
                self.frame_thread.start()
                while not self.stop_event.is_set():
                    try:
                        frame = self.frame_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue
                    now = time.time()
                    diff = now - self.prev_frame_time
                    self.fps = 1 / diff if diff > 0 else 0
                    self.prev_frame_time = now
                    frame_resized, _ = self.process_frame(frame, now)
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 30), 1, 2, offset=5)
                    cv2.imshow(window_name, frame_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("n"), ord("N")]:
                        self.stop_event.set()
                        break
                cv2.destroyAllWindows()
                if self.frame_thread.is_alive():
                    self.frame_thread.join()
            else:
                cap = cv2.VideoCapture(self.video_source)
                frame_delay = int(1000 / self.video_fps) if self.video_fps > 0 else 40
                while cap.isOpened() and not self.stop_event.is_set():
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue
                    now = time.time()
                    diff = now - self.prev_frame_time
                    self.fps = 1 / diff if diff > 0 else 0
                    self.prev_frame_time = now
                    frame_resized, _ = self.process_frame(frame, now)
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 30), 1, 2, offset=5)
                    cv2.imshow(window_name, frame_resized)
                    processing_time = (time.time() - start_time) * 1000
                    adjusted_delay = max(int(frame_delay - processing_time), 1)
                    key = cv2.waitKey(adjusted_delay) & 0xFF
                    if key in [ord("n"), ord("N")]:
                        self.stop_event.set()
                        break
                cap.release()
                cv2.destroyAllWindows()
        finally:
            print("CND is stopped.")


if __name__ == "__main__":
    cnd = VacwerDetector(camera_name="ROBOTICS")
    cnd.main()
