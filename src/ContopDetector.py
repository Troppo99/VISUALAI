import os
import cv2
from ultralytics import YOLO
import torch
import cvzone
import time
import threading
import queue
import math
import numpy as np
from shapely.geometry import Polygon
from datetime import datetime
import json


class ContopDetector:
    def __init__(self, contop_confidence_threshold=0.0, video_source=None, camera_name=None, window_size=(540, 360)):
        self.contop_confidence_threshold = contop_confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (960, 540)
        self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.model = YOLO("static/models/contop1l.pt").to("cuda")
        self.model.overrides["verbose"] = False

        # Initialize threading and queue
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.frame_thread = None

        # Initialize FPS tracking
        self.prev_frame_time = 0
        self.fps = 0

        # Initialize violation tracking
        self.violation_start_time = None
        self.lock = threading.Lock()
        self.last_output_frame = None
        self.last_violation_time = None

    def camera_config(self):
        with open("static/data/ctd_config.json", "r") as f:
            config = json.load(f)
        ip = config.get(self.camera_name)
        if not ip:
            raise ValueError(f"Camera name '{self.camera_name}' not found in configuration.")
        return ip

    def choose_video_source(self):
        if self.video_source is None:
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
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            cap.release()
            time.sleep(5)
            return
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(self.video_source)
                if not cap.isOpened():
                    time.sleep(5)
                    continue
                else:
                    continue
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass
        cap.release()

    def export_frame(self, frame):
        with torch.no_grad():
            results = self.model(frame, stream=True, imgsz=self.process_size[0], task="segment")
        segments = []
        for result in results:
            if result.masks is None:
                continue
            for mask in result.masks:
                poly_xy = mask.xy
                if len(poly_xy) < 3:
                    continue
                polygon = Polygon(poly_xy)
                if polygon.is_empty or not polygon.is_valid:
                    continue
                c = polygon.centroid
                segments.append((poly_xy, (c.x, c.y)))
        return segments

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, self.process_size)
        segments = self.export_frame(frame_resized)
        overlay = frame_resized.copy()

        violation_detected = False

        for poly_xy, (cx, cy) in segments:
            pts = np.array(poly_xy, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 70, 255))
            violation_detected = True

            if self.violation_start_time is None:
                self.violation_start_time = time.time()
                self.last_violation_time = self.violation_start_time
            else:
                self.last_violation_time = time.time()

            if self.violation_start_time is not None:
                elapsed = time.time() - self.violation_start_time
                hh = int(elapsed // 3600)
                mm = int((elapsed % 3600) // 60)
                ss = int(elapsed % 60)
                timer_str = f"{hh:02}:{mm:02}:{ss:02}"
                cvzone.putTextRect(frame_resized, timer_str, (int(cx), int(cy) - 40), scale=1, thickness=2, offset=5, colorR=(0, 70, 255), colorT=(255, 255, 255))
            cvzone.putTextRect(frame_resized, "Violation!", (int(cx), int(cy) - 10), scale=1, thickness=2, offset=5, colorR=(0, 70, 255), colorT=(255, 255, 255))

        if not violation_detected:
            if self.violation_start_time is not None:
                time_since_last_violation = time.time() - self.last_violation_time
                if time_since_last_violation > 5:  # Reset after 5 seconds of no violation
                    self.violation_start_time = None

        # Overlay the segments on the frame
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)

        # Update FPS
        current_time = time.time()
        time_diff = current_time - self.prev_frame_time
        self.fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_frame_time = current_time
        cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 30), scale=1, thickness=2, offset=5)

        with self.lock:
            self.last_output_frame = frame_resized
            # No overlap percentage in this context; can set to 0 or another relevant metric
            self.last_violation_time_record = self.violation_start_time

        return frame_resized

    def generate_frames(self):
        skip_frames = 2
        frame_count = 0
        while True:
            if self.video_fps is None:
                # Run frame capture in a separate thread
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
                    ret, buffer = cv2.imencode(".jpg", output_frame)
                    frame = buffer.tobytes()
                    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
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
                    output_frame = self.process_frame(frame)
                    ret, buffer = cv2.imencode(".jpg", output_frame)
                    frame = buffer.tobytes()
                    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                cap.release()

    def start(self):
        if self.stop_event.is_set():
            self.stop_event.clear()
        if not self.frame_thread or not self.frame_thread.is_alive():
            self.frame_thread = threading.Thread(target=self.capture_frame)
            self.frame_thread.daemon = True
            self.frame_thread.start()

    def stop(self):
        state = ""
        with self.lock:
            if self.violation_start_time is not None:
                elapsed = time.time() - self.violation_start_time
                if elapsed >= 60:
                    state = "Violation sustained for over 60 seconds."
                else:
                    state = f"Violation detected for {int(elapsed)} seconds."
                frame_to_save = self.last_output_frame
                violation_time = self.violation_start_time
            else:
                state = "No violation detected."
                frame_to_save = self.last_output_frame
                violation_time = None

        print(state)
        if frame_to_save is not None and violation_time is not None:
            try:
                # DataHandler().save_data(frame_to_save, violation_time, self.camera_name, insert=True)
                print("Image saved and violation time recorded successfully.")
            except Exception as e:
                print(f"Error saving data: {e}")
        else:
            print("No frame to save or no violation time recorded.")

        self.stop_event.set()
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join()
            print("Frame capture thread stopped.")
        self.frame_thread = None
        self.reset_violation()

    def reset_violation(self):
        with self.lock:
            self.violation_start_time = None
            self.last_violation_time = None
            self.last_output_frame = None
        print("Violation state has been reset.")
