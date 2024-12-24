import cv2
import json
import os
import queue
import math
import time
import threading
import cvzone


class TableDetector:
    def __init__(self, video_source=None, camera_name=None, window_size=(320, 240)):
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (640, 640)
        self.ip_camera = self.camera_config()
        self.choose_video_source()
        self.prev_frame_time = 0
        self.stop_event = threading.Event()

    def camera_config(self):
        with open("static/data/ctd_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        if not ip:
            raise ValueError(f"Camera name '{self.camera_name}' not found in configuration.")
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
                raise ValueError(f"Video source '{self.video_source}' is not a valid file.")

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
        return frame_resized

    def main(self):
        skip_frames = 2
        frame_count = 0
        window_name = "Container Top Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size[0], self.window_size[1])

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
                    frame_resized = self.process_frame(frame)
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 90), scale=1, thickness=2, offset=5)
                    cv2.imshow(window_name, frame_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("n"), ord("N")]:
                        self.stop_event.set()
                        break
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
                    frame_resized = self.process_frame(frame)
                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 90), scale=1, thickness=2, offset=5)
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
    td = TableDetector(
        video_source=None,
        camera_name="FREEMETAL1",
        window_size=(640, 360),
    )
    td.main()
