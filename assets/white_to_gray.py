import cv2, numpy as np


def white_to_gray(img_path, out_path):
    img = cv2.imread(img_path)
    tol = 20
    lb = np.array([255 - tol, 255 - tol, 255 - tol])
    ub = np.array([255, 255, 255])
    mask = cv2.inRange(img, lb, ub).astype(bool)
    img[mask] = [128, 128, 128]
    cv2.imwrite(out_path, img)


white_to_gray(r"output.jpg", "output2.jpg")
