import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import threading
import queue
import time
import os
import json
import sys


class DifferenceDetector:
    def __init__(self, video_source="rtsp", rois=None, reference_filename=None, ip_camera=None):
        self.target_width = 960
        self.target_height = 540
        self.video_source = video_source
        original_rois = rois
        self.reference_folder = "D:/NWR/sources/AlFaruq/media/"
        self.reference_filename = reference_filename
        self.reference_path = os.path.join(self.reference_folder, self.reference_filename)
        self.reference_img = cv2.imread(self.reference_path)
        if self.reference_img is None:
            raise ValueError(f"Tidak dapat membaca gambar referensi dari {self.reference_path}")
        if self.video_source == "rtsp":
            self.cap = cv2.VideoCapture(f"rtsp://admin:oracle2015@{ip_camera}:554/Streaming/Channels/1")
        else:
            self.cap = cv2.VideoCapture("C:/path/to/local/video.mp4")
        if not self.cap.isOpened():
            raise ValueError("Tidak dapat membuka sumber video.")

        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Tidak dapat membaca frame dari sumber video.")

        self.frame_height, self.frame_width = frame.shape[:2]
        self.scale_x = self.target_width / self.frame_width
        self.scale_y = self.target_height / self.frame_height
        self.reference_img = cv2.resize(self.reference_img, (self.target_width, self.target_height))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.reference_display = self.reference_img.copy()
        self.rois = [[(int(x * self.scale_x), int(y * self.scale_y)) for (x, y) in roi] for roi in original_rois]
        for roi in self.rois:
            cv2.polylines(self.reference_display, [np.array(roi, dtype=np.int32)], True, (0, 255, 0), 2)

        self.precomputed_masks = []
        self.bounding_boxes = []
        self.cropped_polygons = []
        for roi in self.rois:
            mask = self.create_polygon_mask(self.reference_img.shape[:2], roi)
            x, y, w, h = cv2.boundingRect(np.array(roi, dtype=np.int32))
            self.bounding_boxes.append((x, y, w, h))
            cropped_polygon = [(pt[0] - x, pt[1] - y) for pt in roi]
            self.cropped_polygons.append(cropped_polygon)
            cropped_mask = self.create_polygon_mask((h, w), cropped_polygon)
            self.precomputed_masks.append(cropped_mask)

        self.capture_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.frame_counter = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_original_frame = None

        # Sensitivity default
        self.sensitivity_threshold = 50

        # Timer ROI
        self.num_rois = len(self.rois)
        self.roi_active = [False] * self.num_rois
        self.roi_pause = [False] * self.num_rois
        self.roi_timer_start = [None] * self.num_rois
        self.roi_pause_start = [None] * self.num_rois
        self.roi_timer_offset = [0] * self.num_rois

        # Tambahan: flag untuk show_reference
        self.show_reference = False

    def create_polygon_mask(self, image_shape, polygon):
        mask = np.zeros(image_shape, dtype=np.uint8)
        pts = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        return mask

    def align_images(self, reference_roi, target_roi, max_features=500, good_match_percent=0.15):
        try:
            gray_ref = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)
            gray_target = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(max_features)
            kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
            kp_tgt, desc_tgt = orb.detectAndCompute(gray_target, None)
            if desc_ref is None or desc_tgt is None:
                return target_roi
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
            matches = matcher.match(desc_ref, desc_tgt, None)
            if len(matches) == 0:
                return target_roi
            matches = sorted(matches, key=lambda x: x.distance)
            num_good_matches = int(len(matches) * good_match_percent)
            matches = matches[:num_good_matches]
            if len(matches) < 4:
                return target_roi
            pts_ref = np.zeros((len(matches), 2), dtype=np.float32)
            pts_tgt = np.zeros((len(matches), 2), dtype=np.float32)
            for i, m in enumerate(matches):
                pts_ref[i, :] = kp_ref[m.queryIdx].pt
                pts_tgt[i, :] = kp_tgt[m.trainIdx].pt
            h, mask = cv2.findHomography(pts_tgt, pts_ref, cv2.RANSAC)
            if h is None:
                return target_roi
            hh, ww, cc = reference_roi.shape
            aligned = cv2.warpPerspective(target_roi, h, (ww, hh))
            return aligned
        except:
            return target_roi

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("Tidak dapat membaca frame dari video.")
                self.stop_event.set()
                break
            self.frame_counter += 1
            # Skip setiap frame ganjil untuk menurunkan beban
            if self.frame_counter % 2 != 0:
                continue

            f_resized = cv2.resize(frame, (self.target_width, self.target_height))
            try:
                self.capture_queue.put(f_resized, timeout=1)
            except queue.Full:
                print("Frame queue penuh. Melewati frame.")
                continue

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.capture_queue.get(timeout=1)
            except queue.Empty:
                continue

            output = frame.copy()
            current_time = time.time()
            with self.lock:
                for idx, roi in enumerate(self.rois):
                    x, y, w, h = self.bounding_boxes[idx]
                    x2 = min(x + w, self.target_width)
                    y2 = min(y + h, self.target_height)
                    if x2 <= x or y2 <= y:
                        self.handle_roi_timer(idx, False, current_time)
                        continue

                    mask = self.precomputed_masks[idx]
                    mask_clipped = mask[0 : (y2 - y), 0 : (x2 - x)]
                    ref_sub = self.reference_img[y:y2, x:x2]
                    tgt_sub = frame[y:y2, x:x2]
                    if ref_sub.size == 0 or tgt_sub.size == 0 or mask_clipped.size == 0:
                        self.handle_roi_timer(idx, False, current_time)
                        continue

                    ref_crop = cv2.bitwise_and(ref_sub, ref_sub, mask=mask_clipped)
                    tgt_crop = cv2.bitwise_and(tgt_sub, tgt_sub, mask=mask_clipped)

                    aligned_tgt = self.align_images(ref_crop, tgt_crop)
                    if aligned_tgt is None:
                        self.handle_roi_timer(idx, False, current_time)
                        continue

                    try:
                        g_ref = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)
                        g_tgt = cv2.cvtColor(aligned_tgt, cv2.COLOR_BGR2GRAY)
                        score, diff = ssim(g_ref, g_tgt, full=True)
                        diff = (diff * 255).astype("uint8")
                        thresh = cv2.threshold(diff, self.sensitivity_threshold, 255, cv2.THRESH_BINARY_INV)[1]
                        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        thresh = cv2.dilate(thresh, kern, iterations=2)
                        thresh = cv2.erode(thresh, kern, iterations=1)
                        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        found = False
                        for c in cnts:
                            if cv2.contourArea(c) > 100:
                                xx, yy, ww, hh = cv2.boundingRect(c)
                                cv2.rectangle(output, (x + xx, y + yy), (x + xx + ww, y + yy + hh), (0, 0, 255), 2)
                                found = True
                        self.handle_roi_timer(idx, found, current_time)
                        timer_str = self.get_roi_timer_str(idx, current_time)
                        tx, ty = self.place_text_in_roi(output, timer_str, x, y, w, h)
                        cv2.putText(output, timer_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    except Exception as e:
                        print(f"Error ROI {idx}: {e}")
                        self.handle_roi_timer(idx, False, current_time)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                print(f"FPS: {fps:.2f}")

            with self.lock:
                self.latest_frame = output.copy()
                self.latest_original_frame = frame.copy()

            cv2.putText(output, f"Threshold: {self.sensitivity_threshold}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            for roi in self.rois:
                if len(roi) >= 2:
                    cv2.polylines(output, [np.array(roi, np.int32)], True, (0, 255, 0), 2)

            if self.show_reference:
                combo = cv2.hconcat([output, self.reference_display])
                cv2.imshow("DifferenceDetector", combo)
            else:
                cv2.imshow("DifferenceDetector", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("n"):
                self.stop_event.set()
            if key == ord("r"):
                self.show_reference = not self.show_reference
            if key == ord("c"):
                self.calibrate_reference()
            if key == ord("="):
                self.sensitivity_threshold = min(100, self.sensitivity_threshold + 1)
            if key == ord("-"):
                self.sensitivity_threshold = max(0, self.sensitivity_threshold - 1)

    def handle_roi_timer(self, idx, detection_found, now):
        if detection_found:
            if not self.roi_active[idx]:
                if self.roi_pause[idx]:
                    if now - self.roi_pause_start[idx] < 5:
                        paused_dur = now - self.roi_pause_start[idx]
                        self.roi_timer_offset[idx] += paused_dur
                    else:
                        self.roi_timer_offset[idx] = 0
                    self.roi_pause[idx] = False
                    self.roi_pause_start[idx] = None
                    self.roi_active[idx] = True
                    if self.roi_timer_start[idx] is None:
                        self.roi_timer_start[idx] = now
                else:
                    self.roi_active[idx] = True
                    self.roi_timer_offset[idx] = 0
                    self.roi_timer_start[idx] = now
        else:
            if self.roi_active[idx]:
                self.roi_active[idx] = False
                self.roi_pause[idx] = True
                self.roi_pause_start[idx] = now
            else:
                if self.roi_pause[idx]:
                    if now - self.roi_pause_start[idx] >= 5:
                        self.roi_timer_offset[idx] = 0
                        self.roi_timer_start[idx] = None
                        self.roi_pause[idx] = False
                        self.roi_pause_start[idx] = None

    def get_roi_timer_str(self, idx, now):
        if self.roi_active[idx]:
            elapsed = (now - self.roi_timer_start[idx]) + self.roi_timer_offset[idx]
        else:
            if self.roi_pause[idx]:
                elapsed = self.roi_timer_offset[idx]
            else:
                elapsed = 0
        hh = int(elapsed // 3600)
        mm = int((elapsed % 3600) // 60)
        ss = int(elapsed % 60)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def calibrate_reference(self):
        with self.lock:
            if self.latest_original_frame is not None:
                frame = self.latest_original_frame.copy()
            else:
                print("Tidak ada frame untuk kalibrasi.")
                return
        ts = time.strftime("%Y%m%d-%H%M%S")
        new_ref_name = f"room_{ts}.jpg"
        new_ref_path = os.path.join(self.reference_folder, new_ref_name)
        cv2.imwrite(new_ref_path, frame)
        print(f"Referensi baru disimpan: {new_ref_path}")
        self.update_reference_image(frame)

    def update_reference_image(self, new_ref):
        try:
            with self.lock:
                resized = cv2.resize(new_ref, (self.target_width, self.target_height))
                self.reference_img = resized
                self.reference_display = self.reference_img.copy()
                for roi in self.rois:
                    cv2.polylines(self.reference_display, [np.array(roi, np.int32)], True, (0, 255, 0), 2)
                self.precomputed_masks = []
                self.bounding_boxes = []
                self.cropped_polygons = []
                for roi in self.rois:
                    mask = self.create_polygon_mask(self.reference_img.shape[:2], roi)
                    x, y, w, h = cv2.boundingRect(np.array(roi, np.int32))
                    self.bounding_boxes.append((x, y, w, h))
                    cropped_polygon = [(pt[0] - x, pt[1] - y) for pt in roi]
                    self.cropped_polygons.append(cropped_polygon)
                    cropped_mask = self.create_polygon_mask((h, w), cropped_polygon)
                    self.precomputed_masks.append(cropped_mask)
            print("Gambar referensi diperbarui.")
        except Exception as e:
            print(f"Error update_reference_image: {e}")

    def place_text_in_roi(self, output, text, x, y, w, h):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        candidates = [(x + 5, y + 20), (x + w - text_width - 5, y + 20), (x + 5, y + h - 5), (x + w - text_width - 5, y + h - 5)]
        for tx, ty in candidates:
            if tx < 0:
                continue
            if ty - text_height < 0:
                continue
            if (tx + text_width) > self.target_width:
                continue
            if ty > self.target_height:
                continue
            return (tx, ty)
        tx = max(0, min(x + 5, self.target_width - text_width - 5))
        ty = max(text_height + 5, min(y + 20, self.target_height - 5))
        return (tx, ty)

    def camera_config():
        with open(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\resources\conf\camera_config.json", "r") as f:
            config = json.load(f)
        camera_name = "CUTTING8"
        reference_filename = config[camera_name]["dd_reference"]
        ip_camera = config[camera_name]["ip"]
        dd_rois_path = config[camera_name]["dd_rois"]
        with open(dd_rois_path, "r") as f_rois:
            rois = json.load(f_rois)
        return reference_filename, ip_camera, rois


if __name__ == "__main__":
    reference_filename, ip_camera, rois = DifferenceDetector.camera_config()
    try:
        dd = DifferenceDetector(video_source="rtsp", rois=rois, reference_filename=reference_filename, ip_camera=ip_camera)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    t_capture = threading.Thread(target=dd.capture_frames)
    t_process = threading.Thread(target=dd.process_frames)
    t_capture.start()
    t_process.start()
    t_process.join()
    dd.stop_event.set()
    t_capture.join()
    dd.cap.release()
    cv2.destroyAllWindows()
