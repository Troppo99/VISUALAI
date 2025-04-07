import cv2
import json
import math
import time
import numpy as np
from shapely.geometry import Polygon, box, LineString
from ultralytics import YOLO


def get_camera_config(camera_name):
    with open(r"\\10.5.0.3\VISUALAI\website-django\five_s\static\resources\conf\camera_config.json", "r") as f:
        config = json.load(f)
    ip = config[camera_name]["ip"]
    rois_path = config[camera_name]["rois"]
    line_path = config[camera_name]["line"]

    with open(rois_path, "r") as f:
        rois_data = json.load(f)
    with open(line_path, "r") as f:
        line_data = json.load(f)
    return rois_data, line_data, ip


def scale_roi(roi_points, orig_w, orig_h, new_w, new_h):
    sw, sh = new_w / orig_w, new_h / orig_h
    out = []
    for polygon in roi_points:
        sp = []
        for x, y in polygon:
            sp.append((int(x * sw), int(y * sh)))
        out.append(sp)
    return out


def scale_line(line_points, orig_w, orig_h, new_w, new_h):
    sw, sh = new_w / orig_w, new_h / orig_h
    scaled = []
    for (x1, y1), (x2, y2) in line_points:
        scaled.append([(int(x1 * sw), int(y1 * sh)), (int(x2 * sw), int(y2 * sh))])
    return scaled


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


camera_name = "OFFICE5"
roi_list, line_list, ip = get_camera_config(camera_name)
model = YOLO(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\resources\models\yolo11l.pt").to("cuda")

# cap = cv2.VideoCapture(f"rtsp://admin:oracle2015@{ip}:554/Streaming/Channels/1")
cap = cv2.VideoCapture(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\videos\labeling\chairneat2.mp4")

size = 1280
target_infer_size = (size, size)
original_roi_size = (1280, 720)

scaled_lines = scale_line(line_list, original_roi_size[0], original_roi_size[1], target_infer_size[0], target_infer_size[1])

while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(raw_frame, target_infer_size)

    scaled_rois = scale_roi(roi_list, original_roi_size[0], original_roi_size[1], target_infer_size[0], target_infer_size[1])

    results = model(frame, imgsz=target_infer_size[0])
    chair_boxes = []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            cls_name = model.names[int(b.cls[0])]
            if conf > 0 and cls_name == "chair":
                if is_overlapped_with_any_roi(x1, y1, x2, y2, scaled_rois):
                    chair_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


    # Gambar ROI (poligon)
    # for poly in scaled_rois:
    #     arr = np.array(poly, dtype=np.int32)
    #     cv2.polylines(frame, [arr], True, (0, 0, 255), 2)

    # Gambar garis dan cek interseksi
    for (lx1, ly1), (lx2, ly2) in scaled_lines:
        line_geo = LineString([(lx1, ly1), (lx2, ly2)])
        color = (0, 255, 0)
        for bx1, by1, bx2, by2 in chair_boxes:
            if bbox_polygon(bx1, by1, bx2, by2).intersects(line_geo):
                color = (0, 0, 255)
                break
        cv2.line(frame, (lx1, ly1), (lx2, ly2), color, 2)

    cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Inference", 1280, 720)
    cv2.imshow("Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
