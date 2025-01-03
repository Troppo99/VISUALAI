import os, cv2, torch, cvzone, time, threading, queue, math, numpy as np, json
from ultralytics import YOLO
from shapely.geometry import Polygon


class ContopDetector:
    def __init__(self, confidence_threshold=0.5, video_source=None, camera_name=None, window_size=(320, 240)):
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (640, 640)
        self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.prev_frame_time = 0
        self.model = YOLO("website/static/resources/models/ctd2l.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.overrides["verbose"] = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def camera_config(self):
        with open(r"C:\xampp\htdocs\VISUALAI\website\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        return ip

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
            results = self.model(source=frame, stream=True, imgsz=self.process_size[0], task="segment")
        segments = []
        for result in results:
            if not result.boxes or not result.masks:
                continue

            for box, mask in zip(result.boxes, result.masks.xy):
                poly_xy = mask
                conf = box.conf[0]
                # class_id = self.model.names[int(box.cls[0])]
                if len(poly_xy) < 3:
                    continue
                polygon = Polygon(poly_xy)
                if polygon.is_empty or not polygon.is_valid:
                    continue
                if conf > self.confidence_threshold:
                    c = polygon.centroid
                    segments.append((poly_xy, (c.x, c.y)))

        return segments

    def process_frame(self, frame):
        frame_processed = cv2.resize(frame, self.process_size)
        segments = self.export_frame(frame_processed)
        overlay = frame_processed.copy()

        for poly_xy, (cx, cy) in segments:
            pts = np.array(poly_xy, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 165, 255))

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame_processed, 1 - alpha, 0, frame_processed)

        for poly_xy, (cx, cy) in segments:
            cvzone.putTextRect(frame_processed, "Warning!", (int(cx), int(cy) - 10), scale=0.5, thickness=1, offset=2, colorR=(0, 165, 255), colorT=(0, 0, 0))
        return frame_processed

    def main(self):
        skip_frames = 2
        frame_count = 0
        window_name = f"Container Top Detection : {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size)

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
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                self.prev_frame_time = current_time
                output_frame = self.process_frame(frame)
                cvzone.putTextRect(output_frame, f"FPS: {int(self.fps)}", (10, 90), scale=1, thickness=2, offset=5)
                cv2.imshow(window_name, output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("n"), ord("N")]:
                    print("Manual stop detected.")
                    self.stop_event.set()
                    break
            cv2.destroyAllWindows()
            if self.frame_thread.is_alive():
                self.frame_thread.join()
        else:
            cap = cv2.VideoCapture(self.video_source)
            frame_delay = max(int(1000 / self.video_fps), 1)
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
                output_frame = self.process_frame(frame)
                cvzone.putTextRect(output_frame, f"FPS: {int(self.fps)}", (10, 90), scale=1, thickness=2, offset=5)
                processing_time = (time.time() - start_time) * 1000
                adjusted_delay = max(frame_delay - int(processing_time), 1)
                cv2.imshow(window_name, output_frame)
                key = cv2.waitKey(adjusted_delay) & 0xFF
                if key in [ord("n"), ord("N")]:
                    print("Manual stop detected.")
                    self.stop_event.set()
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, "..")
    sys.path.append(parent_dir)
    from libs.DataHandler import DataHandler

    detector_args = {
        "confidence_threshold": 0,
        "camera_name": "FREEMETAL2",
        "video_source": r"website/static/videos/seiketsu/1230(1).mp4",
    }

    detector = ContopDetector(**detector_args)
    detector.main()
