import json, cv2, numpy as np, time, queue, cvzone, threading
from shapely.geometry import Polygon


class SkipDetector:
    def __init__(self, camera_name, windwo_size=(960, 540), stop_event=None):
        self.video_source = 0
        self.camera_name = camera_name
        self.window_size = windwo_size
        self.stop_event = stop_event
        if self.stop_event is None:
            self.stop_event = threading.Event()
        self.rois = self.camera_config()
        self.prev_frame_time = 0

    def camera_config(self):
        with open(r"\\10.5.0.3\VISUALAI\website-django\inspection\static\resources\conf\camera_config_inspection.json", "r") as f:
            config = json.load(f)
        scaled_rois = []
        rois_path = config[self.camera_name]
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
        return scaled_rois

    def draw_rois(self, frame):
        if not self.rois:
            return
        for roi in self.rois:
            if roi.geom_type != "Polygon":
                continue
            pts = np.array(roi.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
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

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (960, 540))
        self.draw_rois(frame_resized)
        return frame_resized

    def main(self):
        skip_frames = 2
        frame_count = 0
        window_name = f"BD:{self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size)
        self.frame_queue = queue.Queue(maxsize=10)
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


if __name__ == "__main__":
    sd = SkipDetector(camera_name="Nana")
    sd.main()
