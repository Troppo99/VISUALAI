import os
import cv2
import cvzone
from ultralytics import YOLO
import time
import torch

# Global variables for mouse callback
drawing = False  # True jika mouse sedang ditekan
ix, iy = -1, -1  # Koordinat awal seleksi
jx, jy = -1, -1  # Koordinat akhir seleksi
zoom_rect = None  # Koordinat area yang di-zoom
is_zoomed = False  # Status apakah sedang dalam mode zoom

# Ukuran jendela tetap
DISPLAY_WIDTH, DISPLAY_HEIGHT = 540, 360
DISPLAY_ASPECT_RATIO = DISPLAY_WIDTH / DISPLAY_HEIGHT


def mouse_callback(event, x, y, flags, param):
    global ix, iy, jx, jy, drawing, zoom_rect, is_zoomed

    if is_zoomed:
        # Tidak melakukan apapun jika sedang dalam mode zoom
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        jx, jy = x, y
        zoom_rect = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            jx, jy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        jx, jy = x, y

        # Hitung koordinat seleksi awal dan akhir
        x1, y1 = min(ix, jx), min(iy, jy)
        x2, y2 = max(ix, jx), max(iy, jy)

        # Sesuaikan seleksi agar mempertahankan aspek rasio
        selected_width = x2 - x1
        selected_height = y2 - y1
        selected_aspect = selected_width / selected_height if selected_height != 0 else 1

        if selected_aspect > DISPLAY_ASPECT_RATIO:
            # Lebar adalah dimensi pembatas
            new_width = selected_width
            new_height = int(new_width / DISPLAY_ASPECT_RATIO)
        else:
            # Tinggi adalah dimensi pembatas
            new_height = selected_height
            new_width = int(new_height * DISPLAY_ASPECT_RATIO)

        # Recalculate x2 dan y2 berdasarkan dimensi baru
        x2 = x1 + new_width
        y2 = y1 + new_height

        # Pastikan koordinat tidak melebihi batas frame
        x2 = min(x2, DISPLAY_WIDTH)
        y2 = min(y2, DISPLAY_HEIGHT)

        # Simpan koordinat area yang di-zoom
        zoom_rect = (x1, y1, x2, y2)
        is_zoomed = True  # Aktifkan mode zoom


def export_frame(frame, model):
    # Resize frame ke 1280x1280 sebelum inferensi
    resized_frame = cv2.resize(frame, (1280, 1280))

    with torch.no_grad():
        results = model(resized_frame, stream=True, imgsz=(1280, 1280))  # Menggunakan tuple untuk ukuran
    # results = model(frame)
    boxes_info = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = model.names[int(box.cls[0])]
            if conf > 0.5:  # Anda bisa menyesuaikan threshold ini
                color = (0, 255, 0) if class_id == "O" else (0, 0, 255)
                boxes_info.append((x1, y1, x2, y2, conf, class_id, color))
    return resized_frame, boxes_info  # Kembalikan frame yang telah di-resize


def main():
    global zoom_rect, is_zoomed

    videos = [
        r"C:\xampp\htdocs\VISUALAI\qc-project\videos\labeling\defect\defect1.mp4",
        "rtsp://admin:oracle2015@172.16.0.162:554/Streaming/Channels/1",
        "videos/test/kon.mp4",
    ]
    n = 0  # Pilih video ke-2 (indeks 1)
    cap = cv2.VideoCapture(videos[n])

    model = YOLO(r"C:\xampp\htdocs\VISUALAI\qc-project\models\defect1l.pt")
    model.overrides["verbose"] = False

    # Buat jendela bernama dan tetapkan callback mouse
    window_name = f"Infer {os.path.basename(videos[n])}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(videos[n])
            time.sleep(0.1)
            try:
                ret, frame = cap.read()
            except:
                continue

        frame_results, boxes_info = export_frame(frame, model)

        for x1, y1, x2, y2, conf, class_id, color in boxes_info:
            cvzone.cornerRect(frame_results, (x1, y1, x2 - x1, y2 - y1), rt=0, colorC=(0, 255, 255))
            cvzone.putTextRect(frame_results, f"{class_id} {conf:.2f}", (x1, y1 - 15), colorR=color)

        # Resize frame untuk tampilan
        frame_show = cv2.resize(frame_results, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        if is_zoomed and zoom_rect:
            # Hitung faktor skala antara frame asli dan frame tampilan
            original_height, original_width = frame_results.shape[:2]
            scale_x = original_width / DISPLAY_WIDTH
            scale_y = original_height / DISPLAY_HEIGHT

            x1_disp, y1_disp, x2_disp, y2_disp = zoom_rect

            # Peta koordinat tampilan kembali ke koordinat asli
            orig_x1 = int(x1_disp * scale_x)
            orig_y1 = int(y1_disp * scale_y)
            orig_x2 = int(x2_disp * scale_x)
            orig_y2 = int(y2_disp * scale_y)

            # Pastikan koordinat berada dalam batas frame
            orig_x1 = max(orig_x1, 0)
            orig_y1 = max(orig_y1, 0)
            orig_x2 = min(orig_x2, original_width)
            orig_y2 = min(orig_y2, original_height)

            # Ekstrak ROI dari frame asli
            roi = frame_results[orig_y1:orig_y2, orig_x1:orig_x2]

            if roi.size != 0:
                # Resize ROI untuk mengisi jendela tetap tanpa mengubah aspek rasio
                roi_resized = cv2.resize(roi, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
                frame_show = roi_resized  # Ganti frame tampil dengan ROI yang di-zoom

        else:
            # Jika tidak dalam mode zoom, dan sedang menggambar, tampilkan kotak seleksi
            if drawing:
                # Hitung lebar dan tinggi seleksi
                selected_width = jx - ix
                selected_height = jy - iy
                selected_aspect = abs(selected_width / selected_height) if selected_height != 0 else 1

                if selected_aspect > DISPLAY_ASPECT_RATIO:
                    # Lebar adalah dimensi pembatas
                    new_width = abs(selected_width)
                    new_height = int(new_width / DISPLAY_ASPECT_RATIO)
                else:
                    # Tinggi adalah dimensi pembatas
                    new_height = abs(selected_height)
                    new_width = int(new_height * DISPLAY_ASPECT_RATIO)

                # Tentukan arah drag untuk menentukan posisi kotak
                if jx < ix:
                    x1_draw = ix - new_width
                    x2_draw = ix
                else:
                    x1_draw = ix
                    x2_draw = ix + new_width

                if jy < iy:
                    y1_draw = iy - new_height
                    y2_draw = iy
                else:
                    y1_draw = iy
                    y2_draw = iy + new_height

                # Pastikan kotak seleksi berada dalam batas frame
                x1_draw = max(x1_draw, 0)
                y1_draw = max(y1_draw, 0)
                x2_draw = min(x2_draw, DISPLAY_WIDTH)
                y2_draw = min(y2_draw, DISPLAY_HEIGHT)

                # Gambar kotak seleksi
                cv2.rectangle(frame_show, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 0), 2)

        cv2.imshow(window_name, frame_show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            break
        elif key == ord("r"):
            # Reset zoom
            is_zoomed = False
            zoom_rect = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
