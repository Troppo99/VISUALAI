import json, cv2, numpy as np, time, queue, cvzone, threading
from shapely.geometry import Polygon, Point
from sklearn.cluster import KMeans


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

    def group_bboxes_by_row(self, bboxes, overlap_thresh=10):
        rows = []
        for bb in bboxes:
            l, t, r, b = bb
            placed = False
            for row in rows:
                l0, t0, r0, b0 = row[0]
                if not (b < t0 - overlap_thresh or t > b0 + overlap_thresh):
                    row.append(bb)
                    placed = True
                    break
            if not placed:
                rows.append([bb])
        return rows

    def detect_line_breaks_bbox_horizontal(self, frame, mask, gap_threshold=20):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        bboxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 10:
                bboxes.append((x, y, x + w, y + h))
        if len(bboxes) <= 1:
            return frame, False, False, 0

        rows = self.group_bboxes_by_row(bboxes, overlap_thresh=10)
        line_breaks = False
        line_breaks_in_roi = False
        circle_in_roi_count = 0
        for row in rows:
            row.sort(key=lambda b: b[0])
            for i in range(len(row) - 1):
                lA, tA, rA, bA = row[i]
                lB, tB, rB, bB = row[i + 1]
                distance = lB - rA
                if distance > gap_threshold:
                    line_breaks = True
                    mx_gap = rA + distance // 2
                    my_gap = (tA + bA) // 2

                    circle_center = Point(mx_gap, my_gap)
                    circle_shape = circle_center.buffer(10)
                    in_roi = any(roi.intersects(circle_shape) for roi in self.rois)
                    circle_color = (0, 0, 255) if in_roi else (128, 128, 200)
                    cv2.circle(frame, (mx_gap, my_gap), 10, circle_color, 3)

                    if in_roi:
                        line_breaks_in_roi = True
                        circle_in_roi_count += 1

        return frame, line_breaks, line_breaks_in_roi, circle_in_roi_count

    def process_frame(self, frame):
        frame_960 = cv2.resize(frame, (960, 540))
        k_val = 5
        if k_val % 2 == 0:
            k_val += 1
        if k_val <= 0:
            k_val = 1
        blurred = cv2.GaussianBlur(frame_960, (k_val, k_val), 5)
        h, w = blurred.shape[:2]
        data = blurred.reshape(-1, 3).astype(np.float32)
        km = KMeans(n_clusters=2, random_state=42).fit(data)
        labels = km.labels_.reshape(h, w)
        centers = km.cluster_centers_
        brightness = []
        for c in centers:
            b, g, r = c
            gray_approx = 0.114 * b + 0.587 * g + 0.299 * r
            brightness.append(gray_approx)
        darkest_cluster = np.argmin(brightness)
        mask = np.where(labels == darkest_cluster, 0, 255).astype(np.uint8)

        result_frame, is_broken, line_breaks_in_roi, circle_in_roi_count = self.detect_line_breaks_bbox_horizontal(frame_960, mask, gap_threshold=30)
        if is_broken:
            cvzone.putTextRect(result_frame, "Frame : Skip Detected", (10, 40), 1, 2, offset=5, colorR=(0, 255, 255), colorT=(0, 0, 0))
        else:
            cvzone.putTextRect(result_frame, "Frame : Good", (10, 40), 1, 2, offset=5, colorR=(0, 255, 0), colorT=(0, 0, 0))

        if line_breaks_in_roi:
            cvzone.putTextRect(result_frame, f"ROI : Skip Detected ({circle_in_roi_count})", (10, 65), 1, 2, offset=5, colorR=(0, 255, 255), colorT=(0, 0, 0))
        else:
            cvzone.putTextRect(result_frame, "ROI : Good", (10, 65), 1, 2, offset=5, colorR=(0, 255, 0), colorT=(0, 0, 0))
        self.draw_rois(result_frame)
        return result_frame

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
            fps = 1 / time_diff if time_diff > 0 else 0
            self.prev_frame_time = current_time

            frame_processed = self.process_frame(frame)
            cvzone.putTextRect(frame_processed, f"FPS: {int(fps)}", (10, 90), 1, 2, offset=5)
            cv2.imshow(window_name, frame_processed)
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
