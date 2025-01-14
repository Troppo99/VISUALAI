import cv2, cvzone, json, math, numpy as np, os, queue, threading, time, torch, sys
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.append(parent_dir)
from libs.DataHandler import DataHandler


class CarpalDetector:
    def __init__(self, confidence_threshold=0.5, video_source=None, camera_name=None, window_size=(540, 360), stop_event=None):
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
        self.model = YOLO(r"C:\xampp\htdocs\VISUALAI\website-django\static\resources\models\yolo11l-pose.pt").to("cuda")
        self.model.overrides["verbose"] = False

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
        self.final_overlap = 0
        self.pairs_human = [(0, 1), (0, 2), (1, 2), (2, 4), (1, 3), (4, 6), (3, 5), (5, 6), (6, 8), (8, 10), (5, 7), (7, 9), (6, 12), (12, 11), (11, 5), (12, 14), (14, 16), (11, 13), (13, 15)]

    def camera_config(self):
        with open(r"C:\xampp\htdocs\VISUALAI\website-django\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        scaled_rois = []
        rois_path = config[self.camera_name]["cd_rois"]
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
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

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
            results = self.model(frame, stream=True, imgsz=self.process_size[0], task="pose")

        keypoint_positions = []
        for result in results:
            if result.keypoints is None:
                continue
            if not hasattr(result.keypoints, "conf") or result.keypoints.conf is None:
                continue

            kp_tensor = result.keypoints.xy.cpu().numpy()
            kp_conf = result.keypoints.conf.cpu().numpy()
            for kp_arr, kp_c in zip(kp_tensor, kp_conf):
                single_person_kps = []
                for (x, y), c in zip(kp_arr, kp_c):
                    if c >= self.confidence_threshold:
                        single_person_kps.append((int(x), int(y)))
                    else:
                        single_person_kps.append(None)
                keypoint_positions.append(single_person_kps)

        return keypoint_positions

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, self.process_size)
        self.draw_rois(frame_resized)
        keypoint_positions = self.export_frame(frame_resized)
        output_frame = frame_resized.copy()
        detected = False
        for kp_list in keypoint_positions:
            for idx in [9, 10]:
                if idx < len(kp_list) and kp_list[idx] is not None:
                    (x, y) = kp_list[idx]
                    kp_polygon = box(x - 10, y - 10, x + 10, y + 10)
                    if self.union_roi is not None:
                        current_area = kp_polygon.intersection(self.union_roi)
                    else:
                        current_area = kp_polygon

                    if not current_area.is_empty:
                        new_area = current_area.difference(self.trail_map_polygon)
                        if not new_area.is_empty:
                            self.trail_map_polygon = self.trail_map_polygon.union(new_area)
                            self.draw_polygon_on_mask(new_area, self.trail_map_mask)
                        detected = True

        overlap_percentage = 0
        if self.union_roi and not self.union_roi.is_empty:
            overlap_percentage = (self.trail_map_polygon.area / self.union_roi.area) * 100

        self.final_overlap = overlap_percentage
        current_time = time.time()
        if detected:
            self.last_detection_time = current_time
            if overlap_percentage >= 50 and self.trail_map_start_time is None:
                self.trail_map_start_time = current_time
        else:
            if self.last_detection_time is None:
                time_since_last_det = current_time - self.start_run_time
            else:
                time_since_last_det = current_time - self.last_detection_time

            if overlap_percentage < 10 and time_since_last_det > 10:
                self.reset_trail_map()
            elif overlap_percentage < 50 and time_since_last_det > 60:
                self.reset_trail_map()

        if overlap_percentage >= 50 and self.trail_map_start_time is not None:
            if (current_time - self.trail_map_start_time) > 60 and not self.capture_done:
                print("capture, save, and send (pose version)")
                self.capture_done = True

        alpha = 0.5
        cv2.addWeighted(output_frame, 1.0, self.trail_map_mask, 1 - alpha, 0, output_frame)
        cvzone.putTextRect(output_frame, f"Overlap: {overlap_percentage:.2f}%", (10, 30), scale=1, thickness=2, offset=5)
        return output_frame, overlap_percentage

    def reset_trail_map(self):
        print("Reset trail map.")
        self.trail_map_polygon = Polygon()
        self.trail_map_mask = np.zeros((self.process_size[1], self.process_size[0], 3), dtype=np.uint8)
        self.trail_map_start_time = None
        self.capture_done = False

    def draw_polygon_on_mask(self, polygon, mask, color=(0, 255, 0)):
        if polygon.is_empty:
            return
        if polygon.geom_type == "Polygon":
            polygons = [polygon]
        elif polygon.geom_type == "MultiPolygon":
            polygons = polygon.geoms
        else:
            polygons = []

        for poly in polygons:
            if not poly.is_empty:
                pts = np.array(poly.exterior.coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], color)

    def main(self):
        state = ""
        skip_frames = 2
        frame_count = 0
        window_name = f"CD: {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size)
        final_overlap = 0

        try:
            if self.video_fps is None:
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
                    frame_resized, final_overlap = self.process_frame(frame)
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
                    frame_resized, final_overlap = self.process_frame(frame)
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
            if final_overlap >= 50:
                state = "Mengelap kaca selesai"
            elif final_overlap >= 30:
                state = "Mengelap kaca tidak selesai"
            else:
                state = "Tidak Mengelap kaca"
            print(f"[{self.camera_name}] => {state}")

            if "frame_resized" in locals():
                DataHandler(task="-C").save_data(frame_resized, final_overlap, self.camera_name, insert=False)
            else:
                print("No frame to save.")


if __name__ == "__main__":
    detector = CarpalDetector(camera_name="ROBOTIC")
    detector.main()
