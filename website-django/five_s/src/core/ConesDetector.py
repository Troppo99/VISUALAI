import os, cv2, torch, cvzone, time, threading, queue, math, numpy as np, pymysql, json, sys
from shapely.geometry import Polygon
from ultralytics import YOLO
from datetime import datetime

sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")
from libs.DataHandler import DataHandler


class ConesDetector:
    def __init__(self, confidence_threshold=0.0, video_source=None, camera_name=None, window_size=(320, 240), stop_event=None, is_insert=False):
        self.stop_event = stop_event
        if self.stop_event is None:
            self.stop_event = threading.Event()
        self.confidence_threshold = confidence_threshold
        self.video_source = video_source
        self.camera_name = camera_name
        self.window_size = window_size
        self.process_size = (640, 640)
        self.rois, self.ip_camera = self.camera_config()

        self.prev_frame_time = 0
        self.model = YOLO(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\models\kon\version2\weights\best.pt").to("cuda")
        self.model.overrides["verbose"] = False

        self.window_width, self.window_height = window_size
        self.new_width, self.new_height = (960, 540)
        self.fps = 0
        self.borders, self.ip_camera = self.camera_config()

        # (*) Kita hapus/abaikan logika 'is_local_file' untuk ringkas (opsional)
        self.is_local_file = False
        self.video_fps = None
        if video_source is not None:
            if os.path.isfile(video_source):
                self.is_local_file = True
                cap = cv2.VideoCapture(self.video_source)
                self.video_fps = cap.get(cv2.CAP_PROP_FPS)
                if not self.video_fps or math.isnan(self.video_fps):
                    self.video_fps = 25
                cap.release()

        else:
            # Default RTSP
            self.video_source = f"rtsp://admin:oracle2015@{self.ip_camera}:554/Streaming/Channels/1"

        # (*) PARAM TOLERANSI
        # Jika object hilang >10 detik => reset durasi
        self.no_detection_duration = 10
        # Jika akumulasi durasi >=15 detik => kirim “Warning”
        self.warning_threshold = 15
        # Jika akumulasi durasi >=10 menit (600 detik) => “Violation”
        self.violation_threshold = 600

        # (*) Variabel akumulasi
        self.accumulated_duration = 0.0
        self.last_detect_time = None  # Kapan terakhir kali object terdeteksi di ROI

        # (*) Flag sudah mengirim data
        self.warning_sent = False
        self.violation_sent = False

        # Tracking status (bisa dipakai untuk keperluan tampilan)
        self.current_state = None
        self.violation_row_id = None
        self.insert_done_for_today = False
        self.last_update_date = datetime.now().date()

        self.frame_queue = queue.Queue(maxsize=10)
        self.frame_thread = None

    def camera_config(self):
        with open(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        ip = config[self.camera_name]["ip"]
        scaled_rois = []
        rois_path = config[self.camera_name]["cnd_rois"]
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

    def export_frame_segment(self, frame):
        with torch.no_grad():
            results = self.model.predict(frame, stream=True, imgsz=self.process_size[0], task="segmentation")
        boxes_info = []
        overlap_detected = False
        for result in results:
            if result.masks is None:
                continue
            for poly_xy in result.masks.xy:
                if len(poly_xy) < 3:
                    continue
                polygon = Polygon(poly_xy)
                if polygon.is_empty or not polygon.is_valid:
                    continue
                poly_area = polygon.area
                intersection_area_sum = 0.0
                for border in self.borders:
                    if polygon.intersects(border):
                        inter = polygon.intersection(border)
                        if not inter.is_empty:
                            intersection_area_sum += inter.area
                inside = False
                if intersection_area_sum > 0.5 * poly_area:
                    inside = True
                    overlap_detected = True

                if not polygon.is_empty:
                    c = polygon.centroid
                    boxes_info.append((poly_xy, inside, (c.x, c.y)))
        return boxes_info, overlap_detected

    def process_frame(self, frame, current_time):
        frame_resized = cv2.resize(frame, self.process_size)
        self.draw_rois(frame_resized)
        boxes_info, overlap_detected = self.export_frame_segment(frame_resized)

        # (*) Apakah ada object di ROI?
        any_inside = any(bi[1] for bi in boxes_info)

        if any_inside:
            # Hitung selisih terhadap frame sebelumnya
            if self.last_detect_time is not None:
                # Berapa lama sejak frame sebelumnya?
                dt = current_time - self.prev_time_for_acc
                self.accumulated_duration += dt
            else:
                # Pertama kali terdeteksi (setelah lama hilang atau awal)
                # Mulai dari 0 atau lanjutan? Tergantung jika gap < no_detection_duration
                # tapi kita tangani di else di bawah
                pass

            # Update last_detect_time
            self.last_detect_time = current_time
            self.prev_time_for_acc = current_time
        else:
            # Tidak ada object
            if self.last_detect_time is not None:
                gap = current_time - self.last_detect_time
                if gap > self.no_detection_duration:
                    # reset akumulasi
                    self.accumulated_duration = 0
                    self.last_detect_time = None
                    self.warning_sent = False
                    self.violation_sent = False

        # (*) Cek threshold
        #  - 15 detik => kirim warning (sekali saja)
        #  - 600 detik => violation
        if self.accumulated_duration >= self.warning_threshold and not self.warning_sent:
            # Kirim data "Warning"
            self.warning_sent = True
            self._handle_violation_state("Terjadi Warning 1-10 Menit", frame_resized, self.accumulated_duration)

        if self.accumulated_duration >= self.violation_threshold and not self.violation_sent:
            self.violation_sent = True
            self._handle_violation_state("Terjadi Violation >10 Menit", frame_resized, self.accumulated_duration)

        # Gambar poligon segmen + warna
        overlay = frame_resized.copy()
        for poly_xy, inside, (cx, cy) in boxes_info:
            pts = np.array(poly_xy, np.int32).reshape((-1, 1, 2))
            color_fill = (0, 70, 255) if inside else (0, 255, 255)
            cv2.fillPoly(overlay, [pts], color_fill)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame_resized, 1 - alpha, 0, frame_resized)

        # Tambahkan teks durasi & status
        if any_inside and self.accumulated_duration > 0:
            # Tampilkan akumulasi ke frame
            hh = int(self.accumulated_duration // 3600)
            mm = int((self.accumulated_duration % 3600) // 60)
            ss = int(self.accumulated_duration % 60)
            dur_str = f"{hh:02}:{mm:02}:{ss:02}"
            # Tulis di pojok
            cvzone.putTextRect(frame_resized, f"Duration: {dur_str}", (10, 40), scale=1, thickness=2, colorR=(0, 70, 255))
            # Jika sudah > 10 menit => “Violation”
            # Kalau > 15 detik => “Warning”
            if self.accumulated_duration < self.violation_threshold:
                cvzone.putTextRect(frame_resized, "Warning!" if self.accumulated_duration >= self.warning_threshold else "Inside", (10, 70), scale=1, thickness=2, colorR=(0, 70, 255))
            else:
                cvzone.putTextRect(frame_resized, "Violation!", (10, 70), scale=1, thickness=2, colorR=(0, 0, 255))

        return frame_resized, overlap_detected

    # Di bawah, fungsi-fungsi DB sama, hanya logika panggilannya berbeda
    def _handle_violation_state(self, state_str, frame, elapsed):
        """
        Tangani pembuatan/update record di DB.
        state_str = "Terjadi Warning 1-10 Menit" atau "Terjadi Violation >10 Menit"
        """
        if self.current_state != state_str:
            self.current_state = state_str
            today = datetime.now().date()
            if today != self.last_update_date:
                self.insert_done_for_today = False
                self.violation_row_id = None
                self.last_update_date = today

            table_name = "violation"
            if table_name == "violation":
                if not self.insert_done_for_today:
                    # Insert baru
                    self._check_or_create_violation_row(frame, state_str)
                else:
                    # Update baris
                    self._update_violation_row(frame, state_str)

    def _check_or_create_violation_row(self, frame, state_str):
        today_str = datetime.now().strftime("%Y-%m-%d")
        try:
            conn = pymysql.connect(host="10.5.0.2", user="robot", password="robot123", database="visualai_db", port=3307)
            with conn.cursor() as cursor:
                check_sql = """
                    SELECT id, state
                    FROM violation
                    WHERE camera_name=%s
                      AND DATE(created_at) = %s
                    ORDER BY id DESC
                    LIMIT 1
                """
                cursor.execute(check_sql, (self.camera_name, today_str))
                row = cursor.fetchone()
                if row:
                    self.violation_row_id = row[0]
                    self.insert_done_for_today = True
                    self._update_violation_row(frame, state_str)
                else:
                    data_handler = DataHandler(table="violation", task="-CN")
                    args = state_str
                    data_handler.save_data(frame, args, self.camera_name, insert=True)
                    get_id_sql = """
                        SELECT id
                        FROM violation
                        WHERE camera_name=%s
                          AND DATE(created_at)=%s
                        ORDER BY id DESC
                        LIMIT 1
                    """
                    cursor.execute(get_id_sql, (self.camera_name, today_str))
                    new_row = cursor.fetchone()
                    if new_row:
                        self.violation_row_id = new_row[0]
                        self.insert_done_for_today = True
            conn.close()
        except Exception as e:
            print(f"Error check_or_create_violation_row: {e}")

    def _update_violation_row(self, frame, state_str):
        if not self.violation_row_id:
            return
        data_handler = DataHandler(table="violation", task="(violation)")
        data_handler.save_data(frame, state_str, self.camera_name, insert=False)
        try:
            with open(data_handler.image_path, "rb") as f:
                binary_image = f.read()
            conn = pymysql.connect(host="10.5.0.2", user="robot", password="robot123", database="visualai_db", port=3307)
            with conn.cursor() as cursor:
                sql = """
                    UPDATE violation
                    SET state=%s,
                        image=%s
                    WHERE id=%s
                """
                cursor.execute(sql, (state_str, binary_image, self.violation_row_id))
                conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating violation row: {e}")

    def main(self):
        skip_frames = 2
        frame_count = 0
        window_name = f"CND:{self.camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_size)

        # (*) Waktu untuk perhitungan akumulasi antar frame
        self.prev_time_for_acc = time.time()

        try:
            if self.video_fps is None:
                # Streaming
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

                    frame_resized, final_overlap = self.process_frame(frame, current_time)

                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 100), scale=1, thickness=2, offset=5)
                    cv2.imshow(window_name, frame_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("n"), ord("N")]:
                        print("Manual stop detected.")
                        self.stop_event.set()
                        break

                cv2.destroyAllWindows()
                if self.frame_thread.is_alive():
                    self.frame_thread.join()

            else:
                # Video File
                cap = cv2.VideoCapture(self.video_source)
                frame_delay = int(1000 / self.video_fps) if self.video_fps > 0 else 40

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

                    frame_resized, final_overlap = self.process_frame(frame, current_time)

                    cvzone.putTextRect(frame_resized, f"FPS: {int(self.fps)}", (10, 100), scale=1, thickness=2, offset=5)
                    cv2.imshow(window_name, frame_resized)

                    processing_time = (time.time() - start_time) * 1000
                    adjusted_delay = max(int(frame_delay - processing_time), 1)
                    key = cv2.waitKey(adjusted_delay) & 0xFF
                    if key in [ord("n"), ord("N")]:
                        print("Manual stop detected.")
                        self.stop_event.set()
                        break

                cap.release()
                cv2.destroyAllWindows()

        finally:
            print("CND is stopped.")


if __name__ == "__main__":
    cnd = ConesDetector(
        camera_name="CUTTING4",
        # video_source=r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\videos\cones.mp4",
        is_insert=False,
    )
    cnd.main()
