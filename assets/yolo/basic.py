import cv2, json, os, math, queue, time, torch, threading, cvzone
from ultralytics import YOLO


class YoloInference:
    def __init__(self, confidence_threshold=0, video_source=None, camera_name=None, stop_event=None):
        self.stop_event = stop_event if stop_event else threading.Event()
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.process_size = (960, 960)
        self.window_size = (1280, 720)
        if camera_name not in [0, 1]:
            self.video = self.camera_config()
        else:
            self.video = camera_name
        self.choose_video_source()
        self.prev_frame_time = 0
        self.model_detect = YOLO(r"C:\xampp\htdocs\VISUALAI\website-django\inspection\static\resources\models\defect1-seg-960\weights\best.pt")
        # self.model_detect.overrides["verbose"] = False
        self.yolo_task = YoloTask(self.model_detect, self.confidence_threshold, self.process_size)

    def camera_config(self):
        with open(r"C:\xampp\htdocs\VISUALAI\website-django\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        return f"rtsp://admin:oracle2015@{ip}:554/Streaming/Channels/1"

    def choose_video_source(self):
        if self.video_source is None:
            self.frame_queue = queue.Queue(maxsize=10)
            self.frame_thread = None
            self.video_fps = None
            self.is_local_video = False
            self.video_source = self.video
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

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, self.process_size)
        boxes = self.yolo_task.export_frame_detect(frame_resized)
        output_frame = frame_resized.copy()
        overlay = output_frame.copy()
        for x1, y1, x2, y2, class_id in boxes:
            y1 -= 30
            y2 += 30
            x1 -= 10
            x2 += 10
            cvzone.cornerRect(overlay, (x1, y1, x2 - x1, y2 - y1), l=40, rt=0, t=5, colorC=(0, 0, 255))
            cvzone.putTextRect(overlay, f"{class_id}", (x1, y1), scale=3, thickness=3, offset=5)
        return cv2.addWeighted(overlay, 0.5, output_frame, 0.5, 0)

    def main(self):
        skip_frames = 2
        frame_count = 0
        window_name = f"DFD:{self.camera_name}"
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
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 90), scale=3, thickness=3, offset=5)
                    frame_show = cv2.resize(frame_resized, self.window_size)
                    cv2.imshow(window_name, frame_show)
                    if cv2.waitKey(1) & 0xFF in [ord("n"), ord("N")]:
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
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 90), scale=3, thickness=3, offset=5)

                    frame_show = cv2.resize(frame_resized, self.window_size)
                    cv2.imshow(window_name, frame_show)
                    processing_time = (time.time() - start_time) * 1000
                    adjusted_delay = max(int(frame_delay - processing_time), 1)
                    if cv2.waitKey(adjusted_delay) & 0xFF in [ord("n"), ord("N")]:
                        self.stop_event.set()
                        break
                cap.release()
                cv2.destroyAllWindows()
        finally:
            print("End of program")


class YoloTask:
    def __init__(self, model, confidence_threshold=0, process_size=(640, 640)):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.process_size = process_size

    def export_frame_detect(self, frame):
        with torch.no_grad():
            results = self.model(frame, stream=True, imgsz=self.process_size[0])
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = self.model.names[int(box.cls[0])]
                if conf > self.confidence_threshold:
                    boxes.append((x1, y1, x2, y2, class_id))
        return boxes

    def export_frame_segment(self, frame):
        pass

    def export_frame_classify(self, frame):
        pass

    def export_frame_pose(self, frame):
        pass

    def export_frame_obb(self, frame):
        pass


if __name__ == "__main__":
    yi = YoloInference(
        camera_name="ROBOTICS",
        video_source=r"C:\xampp\htdocs\VISUALAI\website-django\inspection\static\videos\test3.mp4",
    )
    yi.main()
