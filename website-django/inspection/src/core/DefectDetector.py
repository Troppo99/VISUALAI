import threading, json, os, queue, math, time, cv2, torch, cvzone
from ultralytics import YOLO


class ZoomIn:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.jx, self.jy = -1, -1
        self.zoom_rect = None
        self.is_zoomed = False
        self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT = 540, 360
        self.DISPLAY_ASPECT_RATIO = self.DISPLAY_WIDTH / self.DISPLAY_HEIGHT

    def mouse_callback(self, event, x, y, flags, param):
        if self.is_zoomed:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.jx, self.jy = x, y
            self.zoom_rect = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.jx, self.jy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.jx, self.jy = x, y
            x1, y1 = min(self.ix, self.jx), min(self.iy, self.jy)
            x2, y2 = max(self.ix, self.jx), max(self.iy, self.jy)
            selected_width = x2 - x1
            selected_height = y2 - y1
            selected_aspect = selected_width / selected_height if selected_height != 0 else 1

            if selected_aspect > self.DISPLAY_ASPECT_RATIO:
                new_width = selected_width
                new_height = int(new_width / self.DISPLAY_ASPECT_RATIO)
            else:
                new_height = selected_height
                new_width = int(new_height * self.DISPLAY_ASPECT_RATIO)

            x2 = x1 + new_width
            y2 = y1 + new_height
            x2 = min(x2, self.DISPLAY_WIDTH)
            y2 = min(y2, self.DISPLAY_HEIGHT)
            self.zoom_rect = (x1, y1, x2, y2)
            self.is_zoomed = True


class DefectDetector(ZoomIn):
    def __init__(self, video_source=None, camera_name=None, stop_event=None):
        super().__init__()  # Tidak melewatkan argumen
        self.stop_event = stop_event if stop_event else threading.Event()
        self.confidence_threshold = 0
        self.video_source = video_source
        self.camera_name = camera_name
        self.process_size = (1280, 1280)
        self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.prev_frame_time = 0
        self.model = YOLO(r"C:\xampp\htdocs\VISUALAI\website-django\static\resources\models\defect1l.pt")
        self.model.overrides["verbose"] = False

    def camera_config(self):
        with open(r"C:\xampp\htdocs\VISUALAI\website-django\static\resources\conf\camera_config.json", "r") as f:
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

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, self.process_size)
        boxes = self.export_frame(frame_resized)
        output_frame = frame_resized.copy()
        overlay = output_frame.copy()
        for box in boxes:
            x1, y1, x2, y2, class_id = box
            y1 = y1 - 30
            y2 = y2 + 30
            x1 = x1 - 10
            x2 = x2 + 10

            cvzone.cornerRect(overlay, (x1, y1, x2 - x1, y2 - y1), l=40, rt=0, t=5, colorC=(0, 0, 255))
            cvzone.putTextRect(overlay, f"{class_id}", (x1, y1), scale=3, thickness=3, offset=5)

        alpha = 0.5
        output_frame = cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0)
        return output_frame

    def main(self):
        state = ""
        skip_frames = 2
        frame_count = 0
        window_name = f"DFD:{self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        cv2.setMouseCallback(window_name, self.mouse_callback)

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

                    frame_show = cv2.resize(frame_resized, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
                    if self.is_zoomed and self.zoom_rect:
                        original_height, original_width = frame_resized.shape[:2]
                        scale_x = original_width / self.DISPLAY_WIDTH
                        scale_y = original_height / self.DISPLAY_HEIGHT

                        x1_disp, y1_disp, x2_disp, y2_disp = self.zoom_rect

                        orig_x1 = int(x1_disp * scale_x)
                        orig_y1 = int(y1_disp * scale_y)
                        orig_x2 = int(x2_disp * scale_x)
                        orig_y2 = int(y2_disp * scale_y)

                        orig_x1 = max(orig_x1, 0)
                        orig_y1 = max(orig_y1, 0)
                        orig_x2 = min(orig_x2, original_width)
                        orig_y2 = min(orig_y2, original_height)

                        roi = frame_resized[orig_y1:orig_y2, orig_x1:orig_x2]

                        if roi.size != 0:
                            roi_resized = cv2.resize(roi, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
                            frame_show = roi_resized

                    else:
                        if self.drawing:
                            selected_width = self.jx - self.ix
                            selected_height = self.jy - self.iy
                            selected_aspect = abs(selected_width / selected_height) if selected_height != 0 else 1

                            if selected_aspect > self.DISPLAY_ASPECT_RATIO:
                                new_width = abs(selected_width)
                                new_height = int(new_width / self.DISPLAY_ASPECT_RATIO)
                            else:
                                new_height = abs(selected_height)
                                new_width = int(new_height * self.DISPLAY_ASPECT_RATIO)

                            if self.jx < self.ix:
                                x1_draw = self.ix - new_width
                                x2_draw = self.ix
                            else:
                                x1_draw = self.ix
                                x2_draw = self.ix + new_width

                            if self.jy < self.iy:
                                y1_draw = self.iy - new_height
                                y2_draw = self.iy
                            else:
                                y1_draw = self.iy
                                y2_draw = self.iy + new_height

                            x1_draw = max(x1_draw, 0)
                            y1_draw = max(y1_draw, 0)
                            x2_draw = min(x2_draw, self.DISPLAY_WIDTH)
                            y2_draw = min(y2_draw, self.DISPLAY_HEIGHT)

                            cv2.rectangle(frame_show, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 0), 2)

                    cv2.imshow(window_name, frame_show)
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
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 90), scale=3, thickness=3, offset=5)

                    frame_show = cv2.resize(frame_resized, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
                    if self.is_zoomed and self.zoom_rect:
                        original_height, original_width = frame_resized.shape[:2]
                        scale_x = original_width / self.DISPLAY_WIDTH
                        scale_y = original_height / self.DISPLAY_HEIGHT

                        x1_disp, y1_disp, x2_disp, y2_disp = self.zoom_rect

                        orig_x1 = int(x1_disp * scale_x)
                        orig_y1 = int(y1_disp * scale_y)
                        orig_x2 = int(x2_disp * scale_x)
                        orig_y2 = int(y2_disp * scale_y)

                        orig_x1 = max(orig_x1, 0)
                        orig_y1 = max(orig_y1, 0)
                        orig_x2 = min(orig_x2, original_width)
                        orig_y2 = min(orig_y2, original_height)

                        roi = frame_resized[orig_y1:orig_y2, orig_x1:orig_x2]

                        if roi.size != 0:
                            roi_resized = cv2.resize(roi, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
                            frame_show = roi_resized

                    else:
                        if self.drawing:
                            selected_width = self.jx - self.ix
                            selected_height = self.jy - self.iy
                            selected_aspect = abs(selected_width / selected_height) if selected_height != 0 else 1

                            if selected_aspect > self.DISPLAY_ASPECT_RATIO:
                                new_width = abs(selected_width)
                                new_height = int(new_width / self.DISPLAY_ASPECT_RATIO)
                            else:
                                new_height = abs(selected_height)
                                new_width = int(new_height * self.DISPLAY_ASPECT_RATIO)

                            if self.jx < self.ix:
                                x1_draw = self.ix - new_width
                                x2_draw = self.ix
                            else:
                                x1_draw = self.ix
                                x2_draw = self.ix + new_width

                            if self.jy < self.iy:
                                y1_draw = self.iy - new_height
                                y2_draw = self.iy
                            else:
                                y1_draw = self.iy
                                y2_draw = self.iy + new_height

                            x1_draw = max(x1_draw, 0)
                            y1_draw = max(y1_draw, 0)
                            x2_draw = min(x2_draw, self.DISPLAY_WIDTH)
                            y2_draw = min(y2_draw, self.DISPLAY_HEIGHT)

                            cv2.rectangle(frame_show, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 0), 2)

                    cv2.imshow(window_name, frame_show)
                    processing_time = (time.time() - start_time) * 1000
                    adjusted_delay = max(int(frame_delay - processing_time), 1)
                    key = cv2.waitKey(adjusted_delay) & 0xFF
                    if key == ord("n") or key == ord("N"):
                        print("Manual stop detected.")
                        self.stop_event.set()
                        break
                    if key == ord("r") or key == ord("R"):
                        self.is_zoomed = False
                        self.zoom_rect = None
                cap.release()
                cv2.destroyAllWindows()
        finally:
            print("End of program")


if __name__ == "__main__":
    dfd = DefectDetector(
        camera_name="CUTTING1",
        video_source=r"C:\xampp\htdocs\VISUALAI\website-django\static\videos\labeling\defect1.mp4",
    )
    dfd.main()
