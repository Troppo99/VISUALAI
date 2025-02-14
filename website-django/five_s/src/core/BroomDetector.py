import cv2, cvzone, json, math, numpy as np, os, queue, threading, time, torch, sys
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO

sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")
from libs.DataHandler import DataHandler


class BroomDetector:
    def __init__(self, confidence_threshold=0.5, video_source=None, camera_name=None, window_size=(320, 240), stop_event=None, is_insert=False, which_method="trail_map"):
        self.stop_event = stop_event
        if self.stop_event is None:
            self.stop_event = threading.Event()
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (640, 640)
        self.which_method = which_method
        self.rois, self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.prev_frame_time = 0
        self.model = YOLO(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\models\bd6l.pt").to("cuda")
        self.model.overrides["verbose"] = False

        if self.which_method != "dot":
            if len(self.rois) > 1:
                self.union_roi = unary_union(self.rois)
            elif len(self.rois) == 1:
                self.union_roi = self.rois[0]
            else:
                self.union_roi = None
        else:
            self.union_roi = None

        self.trail_map_polygon = Polygon()
        self.trail_map_mask = np.zeros((self.process_size[1], self.process_size[0], 3), dtype=np.uint8)

        self.last_detection_time = None
        self.trail_map_start_time = None
        self.start_run_time = time.time()
        self.capture_done = False
        self.is_insert = is_insert

    def camera_config(self):
        with open(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        if self.which_method == "dot":
            rois_path = config[self.camera_name]["bd_dot"]
            with open(rois_path, "r") as dot_file:
                dot_data = json.load(dot_file)
            if dot_data and isinstance(dot_data[0], list) and isinstance(dot_data[0][0], list):
                dot_data = dot_data[0]
            scaled_dots = []
            for point in dot_data:
                scaled_x = int(float(point[0]) * (960 / 1280))
                scaled_y = int(float(point[1]) * (540 / 720))
                scaled_dots.append((scaled_x, scaled_y))
            self.dot_points = scaled_dots
            self.dot_status = [False] * len(scaled_dots)
            return [], ip
        else:
            scaled_rois = []
            rois_path = config[self.camera_name]["bd_rois"]
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

    def export_frame_detect(self, frame):
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
        if self.which_method == "dot":
            output_frame = frame_resized.copy()
            for idx, pt in enumerate(self.dot_points):
                color = (0, 255, 0) if self.dot_status[idx] else (0, 255, 255)
                cv2.circle(output_frame, pt, 5, color, -1)
            boxes = self.export_frame_detect(frame_resized)
            detected = False
            for box in boxes:
                x1, y1, x2, y2, class_id = box
                for idx, pt in enumerate(self.dot_points):
                    if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
                        self.dot_status[idx] = True
                cvzone.cornerRect(output_frame, (x1, y1, x2 - x1, y2 - y1), l=10, rt=0, t=2, colorC=(0, 255, 255))
                cvzone.putTextRect(output_frame, f"{class_id}", (x1, y1), scale=1, thickness=2, offset=5)
                detected = True
            touched = sum(1 for status in self.dot_status if status)
            total = len(self.dot_status)
            overlap_percentage = (touched / total * 100) if total > 0 else 0
            cvzone.putTextRect(output_frame, f"Percentage: {overlap_percentage:.2f}%", (10, 60), scale=1, thickness=2, offset=5)
            return output_frame, overlap_percentage
        else:
            self.draw_rois(frame_resized)
            boxes = self.export_frame_detect(frame_resized)
            output_frame = frame_resized.copy()
            detected = False
            for box in boxes:
                x1, y1, x2, y2, class_id = box
                overlap_results = self.check_overlap(x1, y1, x2, y2)
                if any(overlap_results):
                    detected = True
                    obj_box_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                    if self.union_roi is not None:
                        current_area = obj_box_polygon.intersection(self.union_roi)
                    else:
                        current_area = obj_box_polygon

                    if not current_area.is_empty:
                        new_area = current_area.difference(self.trail_map_polygon)

                        if not new_area.is_empty:
                            self.trail_map_polygon = self.trail_map_polygon.union(new_area)
                            self.draw_polygon_on_mask(new_area, self.trail_map_mask, color=(0, 255, 0))

                    cvzone.cornerRect(output_frame, (x1, y1, x2 - x1, y2 - y1), l=10, rt=0, t=2, colorC=(0, 255, 255))
                    cvzone.putTextRect(output_frame, f"{class_id} {overlap_results}", (x1, y1), scale=1, thickness=2, offset=5)

            overlap_percentage = 0
            if self.union_roi and not self.union_roi.is_empty:
                overlap_percentage = (self.trail_map_polygon.area / self.union_roi.area) * 100

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
                if current_time - self.trail_map_start_time > 60 and not self.capture_done:
                    print("capture, save, and send")
                    self.capture_done = True

            alpha = 0.5
            output_frame = cv2.addWeighted(output_frame, 1.0, self.trail_map_mask, alpha, 0)
            cvzone.putTextRect(output_frame, f"Percentage: {overlap_percentage:.2f}%", (10, 60), scale=1, thickness=2, offset=5)
            return output_frame, overlap_percentage

    def reset_trail_map(self):
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
        elif polygon.geom_type == "GeometryCollection":
            polygons = []
            for geom in polygon.geoms:
                if geom.geom_type in ["Polygon", "MultiPolygon"]:
                    if geom.geom_type == "Polygon":
                        polygons.append(geom)
                    elif geom.geom_type == "MultiPolygon":
                        polygons.extend(geom.geoms)
        else:
            return

        for poly in polygons:
            if poly.is_empty:
                continue
            pts = np.array(poly.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color)

    def check_overlap(self, x1, y1, x2, y2):
        main_box = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        overlap_results = []
        for roi in self.rois:
            other_polygon = Polygon(roi)
            intersection_area = main_box.intersection(other_polygon).area
            union_area = main_box.union(other_polygon).area
            iou = intersection_area / union_area if union_area != 0 else 0
            overlap_results.append(iou > 0)
        return overlap_results

    def main(self):
        state = ""
        skip_frames = 2
        frame_count = 0
        window_name = f"BD:{self.camera_name}"
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
                state = "Menyapu selesai"
            elif final_overlap >= 30:
                state = "Menyapu tidak selesai"
            else:
                state = "Tidak menyapu"
            print(f"{self.camera_name} => {state}")

            if "frame_resized" in locals():
                DataHandler(task="-B").save_data(frame_resized, final_overlap, self.camera_name, insert=self.is_insert)
            else:
                print("No frame to save.")


if __name__ == "__main__":
    bd = BroomDetector(camera_name="ROBOTICS", which_method="dot", is_insert=False)
    bd.main()
