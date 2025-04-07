import cv2
import json
import math
import time
import numpy as np
from shapely.geometry import Polygon, box
from ultralytics import YOLO


def get_camera_config(camera_name):
    with open(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\conf\camera_config.json", "r") as f:
        config = json.load(f)
    ip = config[camera_name]["ip"]
    vb_path = config[camera_name]["vb_rois"]
    with open(vb_path, "r") as f:
        rois_data = json.load(f)
    return rois_data, ip


def scale_roi(roi_points, orig_w, orig_h, new_w, new_h):
    sw, sh = new_w / orig_w, new_h / orig_h
    out = []
    for polygon in roi_points:
        sp = []
        for x, y in polygon:
            sp.append((int(x * sw), int(y * sh)))
        out.append(sp)
    return out


def bbox_polygon(x1, y1, x2, y2):
    return box(x1, y1, x2, y2)


def roi_polygon(coords):
    return Polygon(coords)


def is_overlapped_with_any_roi(x1, y1, x2, y2, rois):
    bp = bbox_polygon(x1, y1, x2, y2)
    for roi_coords in rois:
        if bp.intersects(roi_polygon(roi_coords)):
            return True
    return False


def dist_pts(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


camera_name = "ROBOTICS"
roi_list, ip = get_camera_config(camera_name)
model = YOLO(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\resources\models\vaccum4\weights\best.pt").to("cuda")

# cap = cv2.VideoCapture(f"rtsp://admin:oracle2015@{ip}:554/Streaming/Channels/1")
# cap = cv2.VideoCapture(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\videos\labeling\blower.mp4")
cap = cv2.VideoCapture(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\videos\labeling\vaccum.mp4")

target_infer_size = (1280, 1280)
original_roi_size = (1280, 720)
movement_threshold = 25
hysteresis_time = 5.0

prev_center = None
active_time = 0.0
total_time = 0.0
last_time = time.time()
realtime_status = "Not used"
last_movement_time = 0.0

# Variabel untuk freeze
last_good_frame = None
frozen_mode = False  # Apakah sedang menampilkan frame beku

while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    try:
        # Kalau sempat error di line-line di bawah, akan lompat ke except dan freeze
        frame = cv2.resize(raw_frame, target_infer_size)
        scaled_rois = scale_roi(roi_list, original_roi_size[0], original_roi_size[1], target_infer_size[0], target_infer_size[1])

        results = model(frame, imgsz=target_infer_size[0])

        highest_conf_box = None
        highest_conf = 0

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])
                cls_name = model.names[int(b.cls[0])]
                if conf > highest_conf:
                    if is_overlapped_with_any_roi(x1, y1, x2, y2, scaled_rois):
                        highest_conf = conf
                        highest_conf_box = (x1, y1, x2, y2)

        current_time = time.time()
        delta_time = current_time - last_time

        if highest_conf_box is not None:
            x1, y1, x2, y2 = highest_conf_box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            total_time += delta_time

            if prev_center is not None:
                movement = dist_pts((cx, cy), prev_center)
                if movement > movement_threshold:
                    last_movement_time = current_time
                    if realtime_status != "Active Working":
                        realtime_status = "Active Working"
                    active_time += delta_time
                else:
                    time_since_move = current_time - last_movement_time
                    if time_since_move <= hysteresis_time:
                        realtime_status = "Active Working"
                        active_time += delta_time
                    else:
                        realtime_status = "Not used"
            else:
                realtime_status = "Not used"

            percentage = (active_time / total_time) * 100 if total_time > 0 else 0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, f"{realtime_status}", (x1, max(0, y1 - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"% Active : {percentage:.1f}%", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            prev_center = (cx, cy)
        else:
            prev_center = None

        # Gambar poligon ROI
        for poly in scaled_rois:
            arr = np.array(poly, dtype=np.int32)
            cv2.polylines(frame, [arr], True, (0, 0, 255), 2)

        last_time = current_time

        # Berhasil memproses frame => update last_good_frame, matikan frozen mode jika aktif
        last_good_frame = frame.copy()
        if frozen_mode:
            frozen_mode = False

    except Exception as e:
        # Jika ada error => freeze tampilan
        if last_good_frame is not None:
            # Kita tetap menampilkan last_good_frame
            frame = last_good_frame
            frozen_mode = True
        else:
            # Kalau belum ada last_good_frame, ya kita break atau skip
            break

    cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Inference", 1280, 720)

    # Tampilkan frame (bisa frame normal atau last_good_frame jika beku)
    cv2.imshow("Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
