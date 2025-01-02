import os
import cv2
from ultralytics import YOLO
import numpy as np
import time
import torch
import cvzone
import pymysql
from datetime import datetime
import threading
import queue
import math
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.geometry import JOIN_STYLE
import json


class CarpalDetector:
    def __init__(self, confidence_threshold=0.5, video_source=None, camera_name=None, window_size=(540, 360)):
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (640, 640)
        self.rois, self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.prev_frame_time = 0
        self.model = YOLO(r"C:\xampp\htdocs\VISUALAI\website\static\resources\models\yolo11l-pose.pt").to("cuda")
        self.model.overrides["verbose"] = False
        self.stop_event = threading.Event()

        self.start_time = None
        self.carpal_absence_timer_start = None
        self.first_green_time = None
        self.total_roi_area = sum(roi.area for roi in self.rois) if self.rois else 0
        self.union_polygon = None
        self.total_area = 0
        self.last_overlap_time = time.time()
        self.area_cleared = False
        self.start_no_overlap_time_high = None
        self.start_no_overlap_time_low = None
        self.detection_paused = False
        self.detection_resume_time = None
        self.detection_pause_duration = 10
        self.timestamp_start = None

    def camera_config(self):
        with open(r"C:\xampp\htdocs\VISUALAI\website\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
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
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

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

    def export_frame(self, results, color, pairs):
        points = []
        coords = []
        keypoint_positions = []
        confidence_threshold = self.confidence_threshold
        for result in results:
            keypoints_data = result.keypoints
            if keypoints_data is not None and keypoints_data.xy is not None and keypoints_data.conf is not None:
                if keypoints_data.shape[0] > 0:
                    keypoints_array = keypoints_data.xy.cpu().numpy()
                    keypoints_conf = keypoints_data.conf.cpu().numpy()
                    for keypoints_per_object, keypoints_conf_per_object in zip(keypoints_array, keypoints_conf):
                        keypoints_list = []
                        for kp, kp_conf in zip(keypoints_per_object, keypoints_conf_per_object):
                            if kp_conf >= confidence_threshold:
                                x, y = kp[0], kp[1]
                                keypoints_list.append((int(x), int(y)))
                            else:
                                keypoints_list.append(None)
                        if len(keypoints_list) > 9 and keypoints_list[7] and keypoints_list[9]:
                            kp7 = keypoints_list[7]
                            kp9 = keypoints_list[9]
                            vx = kp9[0] - kp7[0]
                            vy = kp9[1] - kp7[1]
                            norm = (vx**2 + vy**2) ** 0.5
                            if norm != 0:
                                vx /= norm
                                vy /= norm
                                extension_length = 20
                                x_new = int(kp9[0] + vx * extension_length)
                                y_new = int(kp9[1] + vy * extension_length)
                                keypoints_list[9] = (x_new, y_new)
                        if len(keypoints_list) > 10 and keypoints_list[8] and keypoints_list[10]:
                            kp8 = keypoints_list[8]
                            kp10 = keypoints_list[10]
                            vx = kp10[0] - kp8[0]
                            vy = kp10[1] - kp8[1]
                            norm = (vx**2 + vy**2) ** 0.5
                            if norm != 0:
                                vx /= norm
                                vy /= norm
                                extension_length = 20
                                x_new = int(kp10[0] + vx * extension_length)
                                y_new = int(kp10[1] + vy * extension_length)
                                keypoints_list[10] = (x_new, y_new)
                        keypoint_positions.append(keypoints_list)
                        for point in keypoints_list:
                            if point is not None:
                                points.append(point)
                        for i, j in pairs:
                            if i < len(keypoints_list) and j < len(keypoints_list):
                                if keypoints_list[i] is not None and keypoints_list[j] is not None:
                                    coords.append((keypoints_list[i], keypoints_list[j], color))
            else:
                continue
        return points, coords, keypoint_positions

    def keypoint_to_polygon(self, x, y, size=5):
        x1 = x - size
        y1 = y - size
        x2 = x + size
        y2 = y + size
        return box(x1, y1, x2, y2)

    def update_union_polygon(self, new_polygons):
        if new_polygons:
            if self.union_polygon is None:
                self.union_polygon = unary_union(new_polygons)
            else:
                self.union_polygon = unary_union([self.union_polygon] + new_polygons)
            self.union_polygon = self.union_polygon.simplify(tolerance=0.5, preserve_topology=True)
            if not self.union_polygon.is_valid:
                self.union_polygon = self.union_polygon.buffer(0, join_style=JOIN_STYLE.mitre)
            self.total_area = self.union_polygon.area

    def draw_segments(self, frame):
        overlay = frame.copy()
        if self.union_polygon is not None and not self.union_polygon.is_empty:
            if self.union_polygon.geom_type == "Polygon":
                coords = np.array(self.union_polygon.exterior.coords, np.int32)
                coords = coords.reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [coords], (0, 255, 0))
            elif self.union_polygon.geom_type == "MultiPolygon":
                for poly in self.union_polygon.geoms:
                    if poly.is_empty:
                        continue
                    coords = np.array(poly.exterior.coords, np.int32)
                    coords = coords.reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [coords], (0, 255, 0))
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def process_frame(self, frame, current_time, pairs_human):
        frame_resized = cv2.resize(frame, (self.process_size))
        if self.detection_paused:
            if current_time >= self.detection_resume_time:
                self.detection_paused = False
                print(f"C`{self.camera_name} : Resuming detection after {self.detection_pause_duration}-second pause.")
            else:
                self.draw_rois(frame_resized)
                return frame_resized
        with torch.no_grad():
            results = self.model(frame_resized, stream=True, imgsz=640)
        points, coords, keypoint_positions = self.export_frame(results, (0, 255, 0), pairs_human)
        new_polygons = []
        overlap_detected = False
        for keypoints_list in keypoint_positions:
            for idx in [9, 10]:
                if idx < len(keypoints_list):
                    kp = keypoints_list[idx]
                    if kp is not None:
                        kp_x, kp_y = kp
                        kp_polygon = self.keypoint_to_polygon(kp_x, kp_y, size=5)
                        for roi_polygon in self.rois:
                            if kp_polygon.intersects(roi_polygon):
                                intersection = kp_polygon.intersection(roi_polygon)
                                if not intersection.is_empty:
                                    overlap_detected = True
                                    new_polygons.append(intersection)
        if overlap_detected and self.timestamp_start is None:
            self.timestamp_start = datetime.now()
        self.update_union_polygon(new_polygons)
        percentage = (self.total_area / self.total_roi_area) * 100 if self.total_roi_area > 0 else 0
        self.draw_segments(frame_resized)
        self.draw_rois(frame_resized)
        if keypoint_positions:
            for x, y, color in coords:
                cv2.line(frame_resized, x, y, color, 2)
            for keypoints_list in keypoint_positions:
                for idx, point in enumerate(keypoints_list):
                    if point is not None:
                        if idx == 9 or idx == 10:
                            radius = 10
                        else:
                            radius = 3
                        cv2.circle(frame_resized, point, radius, (0, 255, 255), -1)
        cvzone.putTextRect(
            frame_resized,
            f"Percentage of Overlap: {percentage:.2f}%",
            (10, self.process_size[1] - 50),
            scale=1,
            thickness=2,
            offset=5,
        )
        cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, self.process_size[1] - 75), scale=1, thickness=2, offset=5)
        self.check_conditions(percentage, overlap_detected, current_time, frame_resized)
        return frame_resized

    def check_conditions(self, percentage, overlap_detected, current_time, frame_resized):
        if percentage >= 90:
            self.union_polygon = None
            self.total_area = 0
            print(f"C`{self.camera_name} : Percentage >= 90%")
            self.timestamp_start = None
            self.detection_paused = True
            self.detection_resume_time = current_time + self.detection_pause_duration
            self.start_no_overlap_time_high = None
            self.start_no_overlap_time_low = None
            return
        elif percentage >= 50:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 60:
                    self.union_polygon = None
                    self.total_area = 0
                    print(f"C`{self.camera_name} : Percentage >= 50%")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        elif percentage >= 5:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 30:
                    self.union_polygon = None
                    self.total_area = 0
                    print(f"C`{self.camera_name} : Percentage >= 5%")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        else:
            if not overlap_detected:
                if self.start_no_overlap_time_low is None:
                    self.start_no_overlap_time_low = current_time
                elif current_time - self.start_no_overlap_time_low >= 5:
                    self.union_polygon = None
                    self.total_area = 0
                    print(f"C`{self.camera_name} : Percentage < 5%")
                    self.start_no_overlap_time_low = None
            else:
                self.start_no_overlap_time_low = None
        if self.detection_paused:
            if current_time >= self.detection_resume_time:
                self.detection_paused = False
                print(f"C`{self.camera_name} : Resuming detection after {self.detection_pause_duration}-second pause.")

    def main(self):
        pairs_human = [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 4),
            (1, 3),
            (4, 6),
            (3, 5),
            (5, 6),
            (6, 8),
            (8, 10),
            (5, 7),
            (7, 9),
            (6, 12),
            (12, 11),
            (11, 5),
            (12, 14),
            (14, 16),
            (11, 13),
            (13, 15),
        ]
        process_every_n_frames = 2
        frame_count = 0
        window_name = f"CARPAL : {self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size)
        if self.video_fps is not None:
            cap = cv2.VideoCapture(self.video_source)
            frame_delay = int(1000 / self.video_fps)
            while cap.isOpened():
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print(f"C`{self.camera_name} : End of video file or cannot read the frame.")
                    break
                frame_count += 1
                if frame_count % process_every_n_frames != 0:
                    continue
                current_time = time.time()
                time_diff = current_time - self.prev_frame_time
                if time_diff > 0:
                    self.fps = 1 / time_diff
                else:
                    self.fps = 0
                self.prev_frame_time = current_time
                frame_resized = self.process_frame(frame, current_time, pairs_human)
                cv2.imshow(window_name, frame_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    break
                elif key == ord("s"):
                    self.show_text = not self.show_text
                processing_time = (time.time() - start_time) * 1000
                adjusted_delay = max(int(frame_delay - processing_time), 1)
            cap.release()
            cv2.destroyAllWindows()
        else:
            self.frame_thread = threading.Thread(target=self.capture_frame)
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
                if time_diff > 0:
                    self.fps = 1 / time_diff
                else:
                    self.fps = 0
                self.prev_frame_time = current_time
                frame_resized = self.process_frame(frame, current_time, pairs_human)
                cv2.imshow(window_name, frame_resized)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("n"):
                    self.stop_event.set()
                    break
                elif key == ord("s"):
                    self.show_text = not self.show_text
            cv2.destroyAllWindows()
            self.frame_thread.join()


if __name__ == "__main__":
    detector_args = {
        "camera_name": "SEWING1",
    }

    detector = CarpalDetector(**detector_args)
    detector.main()
