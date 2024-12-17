from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from pytz import timezone
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO
import cv2
import cvzone
import json
import math
import numpy as np
import os
import pymysql
import queue
import threading
import time
import torch



class BroomDetector:
    def __init__(self, confidence_threshold=0.5, video_source=None, camera_name=None, window_size=(320, 240)):
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (960, 540)
        self.rois, self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.prev_frame_time = 0
        self.model = YOLO("src/models/broom6l.pt").to("cuda")
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
        self.stop_event = threading.Event()

    def camera_config(self):
        with open("src/data/bd_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        scaled_rois = []
        rois_path = config[self.camera_name]["rois"]
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
            self.video_source = self.video_source
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
                exit()

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
        self.draw_rois(frame_resized)
        boxes = self.export_frame(frame_resized)
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
        window_name = "Brooming Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size[0], self.window_size[1])
        final_overlap = 0

        try:
            if self.video_fps is None:
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
            print(state)
            if "frame_resized" in locals():
                DataHandler().save_data(frame_resized, final_overlap, self.camera_name, insert=True)
            else:
                print("No frame to save.")


class DataHandler:
    def __init__(self, host="localhost", user="root", password="robot123", database="visualai_db", table="cleaning_floor", port=3306):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.port = port
        self.connection = None
        self.cursor = None
        self.image_path = None

    def config_database(self):
        try:
            self.connection = pymysql.connect(host=self.host, user=self.user, password=self.password, database=self.database, port=self.port)
            self.cursor = self.connection.cursor()
        except pymysql.MySQLError as e:
            print(f"Database connection failed: {e}")
            raise

    def save_data(self, frame, percentage, camera_name, insert=True):
        try:
            cvzone.putTextRect(frame, f"Datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (10, 30), scale=1, thickness=2, offset=5)
            cvzone.putTextRect(frame, f"Camera: {camera_name}", (10, 90), scale=1, thickness=2, offset=5)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.image_path = f"static/images/brooming/{camera_name}_{timestamp_str}.jpg"
            os.makedirs(os.path.dirname(self.image_path), exist_ok=True)
            cv2.imwrite(self.image_path, frame)
            if insert:
                self.insert_data(percentage)
            print("Image saved and inserted successfully" if insert else "Image saved without inserting")
        except Exception as e:
            print(f"Failed to save image: {e}")
            raise

    def insert_data(self, percentage):
        try:
            self.config_database()
            if not self.image_path:
                raise ValueError("Image path is not set")

            with open(self.image_path, "rb") as file:
                binary_image = file.read()

            camera_name = os.path.basename(self.image_path).split("_")[0]
            query = f"""
            INSERT INTO {self.table} (camera_name, percentage, image)
            VALUES (%s, %s, %s)
            """
            self.cursor.execute(query, (camera_name, percentage, binary_image))
            self.connection.commit()
        except Exception as e:
            print(f"Error saving data: {e}")
        finally:
            if self.connection:
                self.connection.close()


class Scheduling:
    def __init__(self, detector_args, broom_schedule_type):
        self.detector_args = detector_args
        self.broom_schedule_type = broom_schedule_type
        self.detector = None
        self.scheduler = BackgroundScheduler(timezone=timezone("Asia/Jakarta"))
        self.setup_schedule()
        self.scheduler.start()

    def start_detection(self):
        if not self.detector:
            print("Starting BroomDetector...")
            self.detector = BroomDetector(**self.detector_args)
            detection_thread = threading.Thread(target=self.detector.main)
            detection_thread.daemon = True
            detection_thread.start()
        else:
            print("BroomDetector is already running.")

    def stop_detection(self):
        if self.detector:
            print("Stopping BroomDetector...")
            self.detector.stop_event.set()
            self.detector = None
        else:
            print("BroomDetector is not running.")

    def setup_schedule(self):
        if self.broom_schedule_type == "OFFICE":
            work_days = ["mon", "tue", "wed", "thu", "fri"]
            for day in work_days:
                # S1 : 06:00 - 08:30
                h1, m1, s1 = (6, 0, 0)
                h2, m2, s2 = (8, 30, 0)
                start_trigger = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                self.scheduler.add_job(self.start_detection, trigger=start_trigger, id=f"start_{day}", replace_existing=True)
                stop_trigger = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                self.scheduler.add_job(self.stop_detection, trigger=stop_trigger, id=f"stop_{day}", replace_existing=True)
        elif self.broom_schedule_type == "SEWING":
            work_days = ["mon", "tue", "wed", "thu", "fri"]
            for day in work_days:
                # S1 : 07:30 - 09:45
                # S2 : 09:45 - 12:50
                # S3 : 12:50 - 13:05
                h1, m1, s1 = (7, 30, 0)
                h2, m2, s2 = (9, 45, 0)
                h3, m3, s3 = (9, 45, 0)
                h4, m4, s4 = (12, 50, 0)
                h5, m5, s5 = (12, 50, 0)
                h6, m6, s6 = (13, 5, 0)
                s1_start = CronTrigger(day_of_week=day, hour=h1, minute=m1, second=s1)
                s1_stop = CronTrigger(day_of_week=day, hour=h2, minute=m2, second=s2)
                s2_start = CronTrigger(day_of_week=day, hour=h3, minute=m3, second=s3)
                s2_stop = CronTrigger(day_of_week=day, hour=h4, minute=m4, second=s4)
                s3_start = CronTrigger(day_of_week=day, hour=h5, minute=m5, second=s5)
                s3_stop = CronTrigger(day_of_week=day, hour=h6, minute=m6, second=s6)

                self.scheduler.add_job(self.start_detection, trigger=s1_start, id=f"s1_start_{day}", replace_existing=True)
                self.scheduler.add_job(self.stop_detection, trigger=s1_stop, id=f"s1_stop_{day}", replace_existing=True)

                self.scheduler.add_job(self.start_detection, trigger=s2_start, id=f"s2_start_{day}", replace_existing=True)
                self.scheduler.add_job(self.stop_detection, trigger=s2_stop, id=f"s2_stop_{day}", replace_existing=True)

                self.scheduler.add_job(self.start_detection, trigger=s3_start, id=f"s3_start_{day}", replace_existing=True)
                self.scheduler.add_job(self.stop_detection, trigger=s3_stop, id=f"s3_stop_{day}", replace_existing=True)

    def shutdown(self):
        print("Shutdown scheduler and BroomDetector if not running...")
        self.scheduler.shutdown(wait=False)
        self.stop_detection()

if __name__ == "__main__":
    detector_args = {
        "confidence_threshold": 0.5,
        "camera_name": "OFFICE3",
        "video_source": "src/videos/bd_test3.mp4",
        "window_size": (320, 240),
    }

    scheduler = Scheduling(detector_args, "SEWING")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
        scheduler.shutdown()
