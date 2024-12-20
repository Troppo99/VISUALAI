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
    def __init__(self, contop_confidence_threshold=0.5, video_source=None, camera_name=None, window_size=(320, 240)):
        self.contop_confidence_threshold = contop_confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (960, 540)
        self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.prev_frame_time = 0
        self.model = YOLO("static/models/contop1l.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.overrides["verbose"] = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.frame_thread = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def camera_config(self):
        with open("static/data/ctd_config.json", "r") as f:
            config = json.load(f)
        ip = config.get(self.camera_name)
        if not ip:
            raise ValueError(f"Camera name '{self.camera_name}' not found in configuration.")
        return ip

    def choose_video_source(self):
        if self.video_source is None:
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
                raise ValueError(f"Video source '{self.video_source}' is not a valid file.")

    def capture_frame(self):
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                cap.release()
                print("Failed to open video source. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame. Releasing and retrying...")
                    break
                try:
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    pass
            cap.release()
            time.sleep(1)

    def export_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = self.model.predict(source=frame_rgb, imgsz=self.process_size[0], conf=self.contop_confidence_threshold, task="segment")

        segments = []
        for result in results:
            for mask in result.masks.xy:
                poly_xy = mask
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

        for poly_xy, (cx, cy) in segments:
            pts = np.array(poly_xy, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 70, 255))
            cvzone.putTextRect(frame_resized, "Objek", (int(cx), int(cy) - 10), scale=1, thickness=2, offset=5, colorR=(0, 70, 255), colorT=(255, 255, 255))

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)

        current_time = time.time()
        time_diff = current_time - self.prev_frame_time
        self.fps = 1 / time_diff if time_diff > 0 else 0
        self.prev_frame_time = current_time
        cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 30), scale=1, thickness=2, offset=5, colorR=(0, 0, 0), colorT=(255, 255, 255))

        return frame_resized

    def main(self):
        """
        Main function to run detection and display results using OpenCV.
        """
        skip_frames = 2
        frame_count = 0
        window_name = "Container Top Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size[0], self.window_size[1])

        try:
            if self.video_fps is None:
                # RTSP Stream
                self.frame_thread = threading.Thread(target=self.capture_frame)
                self.frame_thread.daemon = True
                self.frame_thread.start()
                print("Frame capture thread started for RTSP stream.")

                while not self.stop_event.is_set():
                    try:
                        frame = self.frame_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue
                    frame_resized = self.process_frame(frame)
                    cv2.imshow(window_name, frame_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("n"), ord("N")]:
                        print("Manual stop detected.")
                        self.stop_event.set()
                        break

                if self.frame_thread.is_alive():
                    self.frame_thread.join()
            else:
                # Local Video File
                cap = cv2.VideoCapture(self.video_source)
                frame_delay = max(int(1000 / self.video_fps), 1)
                print("Processing local video file.")

                while cap.isOpened() and not self.stop_event.is_set():
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        print("Video ended.")
                        break
                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue
                    frame_resized = self.process_frame(frame)
                    cv2.imshow(window_name, frame_resized)
                    processing_time = (time.time() - start_time) * 1000
                    adjusted_delay = max(frame_delay - int(processing_time), 1)
                    key = cv2.waitKey(adjusted_delay) & 0xFF
                    if key in [ord("n"), ord("N")]:
                        print("Manual stop detected.")
                        self.stop_event.set()
                        break

                cap.release()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage:
    # For RTSP stream, set video_source=None and provide camera_name
    # For local video file, provide video_source and optionally camera_name
    ctd = ContopDetector(camera_name="FREEMETAL1", video_source=r"D:\NWR\videos\Bahan\C_171224.mp4")  # Set to None for RTSP
    ctd.main()
