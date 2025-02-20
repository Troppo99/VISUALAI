import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalize(img_org):
    img_tmp = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_tmp = cv2.equalizeHist(img_tmp)
    count = np.array([[0, 0], [img_org.shape[1] - 1, 0], [img_org.shape[1] - 1, 340], [0, 340]])
    img_tmp = cv2.fillPoly(img_tmp, pts=[count], color=(0))
    return img_tmp


def yellow(img_org):
    img_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
    lower_color = np.array([15, 100, 50], np.uint8)
    upper_color = np.array([40, 200, 200], np.uint8)
    img_mask = cv2.inRange(img_hsv, lower_color, upper_color)
    img_tmp = cv2.bitwise_and(img_hsv, img_hsv, mask=img_mask)
    img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
    ret, img_tmp = cv2.threshold(img_tmp, 10, 256, cv2.THRESH_BINARY)
    count = np.array([[0, 0], [img_org.shape[1] - 1, 0], [img_org.shape[1] - 1, 340], [0, 340]])
    img_tmp = cv2.fillPoly(img_tmp, pts=[count], color=(0))
    return img_tmp


cap = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Grid Gambar")
ax.axis("off")

im_display = ax.imshow(np.zeros((480, 1280, 3), dtype=np.uint8))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame tidak terbaca")
        break

    img_equal = equalize(frame)
    img_yellow = yellow(frame)
    img_or = cv2.bitwise_or(img_equal, img_yellow)
    img_blur = cv2.blur(img_or, ksize=(5, 5))

    img_blur_color = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)
    grid = np.hstack((frame, img_blur_color))
    grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

    im_display.set_data(grid_rgb)
    fig.canvas.draw()
    fig.canvas.flush_events()

    if cv2.waitKey(1) & 0xFF == ord("n"):
        break

cap.release()
plt.ioff()
plt.show()
exit()