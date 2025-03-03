import cv2, torch, numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("rtsp://admin:oracle2015@172.20.0.24:554/Streaming/Channels/1")
# cap = cv2.VideoCapture(r"C:\Users\Troppo\Downloads\depth_test.mp4")
frame_skip = 2
frame_count = 0
depth_map = None

roi_x, roi_y, roi_w, roi_h = 0,0, 1280, 960

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    original_frame = frame.copy()

    # Proses inferensi setiap 'frame_skip' frame
    if frame_count % frame_skip == 0:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()

    # Gambar ROI pada frame asli agar terlihat
    cv2.rectangle(original_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    original_frame = cv2.resize(original_frame, (960, 540))
    cv2.imshow("Original", original_frame)

    if depth_map is not None:
        h, w = depth_map.shape
        # Pastikan ROI berada dalam batas frame
        if roi_x + roi_w > w or roi_y + roi_h > h:
            print("ROI di luar batas frame, sesuaikan koordinatnya.")
        else:
            roi_depth = depth_map[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            # Pastikan ROI tidak kosong
            if roi_depth.size == 0:
                print("ROI kosong, periksa koordinat ROI.")
            else:
                depth_norm = cv2.normalize(roi_depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_norm = np.uint8(depth_norm)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                depth_color = cv2.resize(depth_color, (960, 540))
                cv2.imshow("Depth ROI", depth_color)

    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
