import cv2, torch, numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.to(device)
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform

# cap = cv2.VideoCapture("rtsp://admin:oracle2015@10.5.5.1:554/Streaming/Channels/1")
cap = cv2.VideoCapture("rtsp://admin:oracle2015@172.20.0.13:554/Streaming/Channels/1")
frame_skip = 2
frame_count = 0
depth_map = None
# polygon_pts = np.array([[928, 958], [1098, 852], [1209, 912], [1128, 987], [1146, 1000], [1077, 1059]], dtype=np.int32)
polygon_pts = np.array([[0, 0], [1275, 0], [1275, 715], [0, 715]], dtype=np.int32)

plt.ion()
distances = []
fig = plt.figure()

buffer_size = 5
distance_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    original_frame = frame.copy()

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

    cv2.polylines(original_frame, [polygon_pts], True, (0, 255, 0), 2)
    cv2.imshow("Original", cv2.resize(original_frame, (960, 540)))

    if depth_map is not None:
        h, w = depth_map.shape
        if np.all(polygon_pts[:, 0] < w) and np.all(polygon_pts[:, 1] < h):
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            mask = np.zeros_like(depth_color, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_pts], (255, 255, 255))
            masked_depth = cv2.bitwise_and(depth_color, mask)
            masked_depth_resized = cv2.resize(masked_depth, (960, 540))
            cv2.imshow("Depth with Polygon Mask", masked_depth_resized)

            mask_2d = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_2d, [polygon_pts], 255)
            roi_vals = depth_map[mask_2d == 255]
            if roi_vals.size:
                distance_est = roi_vals.min()

                distance_buffer.append(distance_est)
                if len(distance_buffer) > buffer_size:
                    distance_buffer.pop(0)
                smoothed_dist = np.mean(distance_buffer)
                print(f"{smoothed_dist:.2f}")

                # print(f"{distance_est:.2f}")
                distances.append(smoothed_dist)
                plt.clf()
                plt.plot(distances)
                plt.draw()
                plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
cv2.destroyAllWindows()
