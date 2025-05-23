import threading, time, numpy as np, json, cv2, os, queue, math, torch, cvzone
from ultralytics import YOLO
from shapely.geometry import Polygon


class BlazingModel:
    def __init__(self, model_path, confidence_threshold=0.5, device="cuda"):
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path).to(device)
        self.model.overrides["verbose"] = False

    def detect(self, frame):
        with torch.no_grad():
            results = self.model(frame, stream=True, imgsz=640)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                class_id = self.model.names[int(box.cls[0])]
                if conf > self.confidence_threshold:
                    detections.append((x1, y1, x2, y2, class_id, conf))
        return detections


class SpreadingManual:
    def __init__(self, confidence_threshold=0.5, video_source=None, camera_name=None, window_size=(320, 240), stop_event=None):
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
        self.model = YOLO(r"C:\xampp\htdocs\VISUALAI\website\static\resources\models\yolo11l.pt").to("cuda")
        self.model.overrides["verbose"] = False

        # Inisialisasi model Blazing
        self.blazing_model = BlazingModel(r"C:\xampp\htdocs\VISUALAI\website\static\resources\models\blazing\weights\best.pt", confidence_threshold=0.5)

        self.trail_map_polygon = Polygon()
        self.trail_map_mask = np.zeros((self.process_size[1], self.process_size[0], 3), dtype=np.uint8)

        self.last_detection_time = None
        self.trail_map_start_time = None
        self.start_run_time = time.time()
        self.capture_done = False

        # Timer dan Checklist Variables
        self.checklist_start_time = time.time()
        self.checklist_duration = 60  # detik (1 menit)
        self.checklist_done = False
        self.bullmer_idle = True  # Asumsi awal: Tidak bergerak
        self.two_people_detected_summary = False  # Hasil akhir deteksi dua orang
        self.two_people_accumulated_time = 0  # Akumulasi durasi deteksi dua orang
        self.blazing_moving = False

    def camera_config(self):
        with open(r"C:\xampp\htdocs\VISUALAI\website\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        scaled_rois = []
        rois_path = config[self.camera_name]["sm_rois"]
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
            self.frame_thread = threading.Thread(target=self.capture_frame)
            self.frame_thread.daemon = True
            self.frame_thread.start()
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
                conf = box.conf[0].item()
                class_id = self.model.names[int(box.cls[0])]
                if conf > self.confidence_threshold:
                    boxes.append((x1, y1, x2, y2, class_id, conf))
        return boxes

    def check_overlap(self, bbox, roi):
        """
        Mengecek apakah bounding box overlap dengan ROI tertentu.
        bbox: Tuple (x1, y1, x2, y2)
        roi: Shapely Polygon
        """
        x1, y1, x2, y2 = bbox
        main_box = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        return main_box.intersects(roi)

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, self.process_size)
        # self.draw_rois(frame_resized)
        yolo_boxes = self.export_frame(frame_resized)
        output_frame = frame_resized.copy()

        # Dictionary untuk menyimpan deteksi per ROI
        detections_per_roi = {0: [], 1: []}  # Fokus hanya pada ROI 0 dan ROI 1

        for box in yolo_boxes:
            x1, y1, x2, y2, class_id, conf = box  # Pastikan unpack dengan benar
            for idx in [0, 1]:  # Hanya periksa ROI 0 dan ROI 1
                roi = self.rois[idx]
                if self.check_overlap((x1, y1, x2, y2), roi):
                    detections_per_roi[idx].append(box)
                    break  # Asumsi satu deteksi per ROI, jadi keluar dari loop ROIs

        # Pilih deteksi dengan confidence tertinggi untuk setiap ROI
        selected_detections = {}
        for idx in [0, 1]:
            if detections_per_roi[idx]:
                # Pilih deteksi dengan confidence tertinggi
                selected_box = max(detections_per_roi[idx], key=lambda x: x[5])  # x[5] adalah confidence
                selected_detections[idx] = selected_box

        # Tandai apakah dua orang terdeteksi
        self.current_two_people_detected = len(selected_detections) == 2

        # Gambar bounding box dan simpan titik deteksi
        detected_points = {}
        for idx, box in selected_detections.items():
            x1, y1, x2, y2, class_id, conf = box
            detected_points[idx] = (x1, y1)
            # Gambar bounding box dengan warna kuning
            cvzone.cornerRect(output_frame, (x1, y1, x2 - x1, y2 - y1), rt=0, l=15, t=2, colorC=(0, 255, 255))

        # Pendeteksian dengan BlazingModel
        blazing_detections = self.blazing_model.detect(frame_resized)
        for det in blazing_detections:
            x1, y1, x2, y2, class_id, conf = det
            # Gambar bounding box untuk Blazing
            cvzone.cornerRect(output_frame, (x1, y1, x2 - x1, y2 - y1), rt=0, l=8, t=2, colorC=(50, 0, 255))
            cvzone.putTextRect(output_frame, f"{class_id}: {conf:.2f}", (x1, y1 - 10), scale=0.5, thickness=1, offset=1)
            # Cek apakah Blazing overlap dengan ROI ketiga (indeks 2)
            if not self.blazing_moving and self.check_overlap((x1, y1, x2, y2), self.rois[2]):
                self.blazing_moving = True

        # Menggambar garis penghubung jika dua orang terdeteksi
        if self.current_two_people_detected and 0 in detected_points and 1 in detected_points:
            pt1, pt2 = detected_points[0], detected_points[1]
            cv2.line(output_frame, pt1, pt2, (255, 0, 0), 2)  # Garis biru

        # Update akumulasi deteksi dua orang
        current_time = time.time()
        frame_duration = 1 / self.fps if hasattr(self, "fps") and self.fps > 0 else 0.04  # Default 25 FPS ~ 0.04 detik

        if self.current_two_people_detected:
            self.two_people_accumulated_time += frame_duration
            # Batasi akumulasi hingga 10 detik
            self.two_people_accumulated_time = min(self.two_people_accumulated_time, 10)
        else:
            pass

        accumulated_minutes = int(self.two_people_accumulated_time) // 60
        accumulated_seconds = int(self.two_people_accumulated_time) % 60
        accumulated_timer_text = f"v: {accumulated_minutes:02}:{accumulated_seconds:02}"
        checklist_text = f"Apakah terdapat dua orang : {'Ya' if self.two_people_detected_summary else 'Tidak'} {accumulated_timer_text}\n" f"Apakah bullmer diam : {'Ya'}\n" f"Apakah blazing bergerak : {'Ya' if self.blazing_moving else 'Tidak'}"
        # Menampilkan Checklist di Frame

        # Tentukan ukuran dan posisi overlay
        overlay = output_frame.copy()
        x, y, w, h = 10, frame_resized.shape[0] - 100, 300, 90  # Posisi dan ukuran rectangle (ditambah tinggi untuk akumulasi)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Hitam solid
        alpha = 0.5  # Transparansi 50%
        cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)

        # Menampilkan teks checklist di atas overlay
        for i, line in enumerate(checklist_text.split("\n")):
            cv2.putText(output_frame, line, (x + 10, y + 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Menampilkan Countdown Timer di Frame
        elapsed_time = current_time - self.checklist_start_time
        remaining_time = max(int(self.checklist_duration - elapsed_time), 0)
        minutes = remaining_time // 60
        seconds = remaining_time % 60
        timer_text = f"Timer: {minutes:02}:{seconds:02}"

        # Tentukan posisi timer (pojok kanan atas)
        timer_x, timer_y = frame_resized.shape[1] - 150, 30
        # Membuat overlay untuk timer
        overlay_timer = output_frame.copy()
        cv2.rectangle(overlay_timer, (timer_x - 10, timer_y - 20), (timer_x + 140, timer_y + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay_timer, alpha, output_frame, 1 - alpha, 0, output_frame)
        # Menampilkan teks timer
        cv2.putText(output_frame, timer_text, (timer_x, timer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Timer dan Simpulan
        if not self.checklist_done and elapsed_time >= self.checklist_duration:
            # Tentukan simpulan
            if self.two_people_accumulated_time >= 10 and self.bullmer_idle and self.blazing_moving:
                self.two_people_detected_summary = True
            else:
                self.two_people_detected_summary = False

            # Cetak simpulan ke console
            conclusion = ""
            if self.two_people_detected_summary == True:
                conclusion = "SPREADING MANUAL"
            else:
                conclusion = "Deteksi Tidak Dapat Disimpulkan"
            print(conclusion)

            # Reset checklist variables untuk siklus berikutnya
            self.checklist_start_time = current_time
            self.checklist_done = False
            self.two_people_accumulated_time = 0
            self.two_people_detected_summary = False
            # Bullmer tetap idle untuk siklus berikutnya
            self.bullmer_idle = True
            self.blazing_moving = False

        return output_frame

    def main(self):
        skip_frames = 2
        frame_count = 0
        window_name = f"SM:{self.camera_name}"
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
                    self.fps = 1 / time_diff if time_diff > 0 else 25  # Default FPS jika time_diff nol
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
                    self.fps = 1 / time_diff if time_diff > 0 else 25  # Default FPS jika time_diff nol
                    self.prev_frame_time = current_time
                    frame_resized = self.process_frame(frame)
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
            pass


if __name__ == "__main__":
    sm = SpreadingManual(
        camera_name="CUTTING4",
        video_source=r"C:\xampp\htdocs\VISUALAI\website\static\videos\spreading_manual.mp4",
        # window_size=(960, 540),
    )
    sm.main()
