import cv2
import numpy as np


def main(input_file, output_file=None, contour_file=None):
    cap = cv2.VideoCapture(input_file)
    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    size = (1280, 720)
    if output_file is not None:
        output_writer = cv2.VideoWriter(output_file, fmt, 30, size)
    if contour_file is not None:
        contour_writer = cv2.VideoWriter(contour_file, fmt, 30, size)
    while cap.isOpened():
        ret, img_org = cap.read()
        if not ret:
            break
        img_tmp = equalize(img_org)
        img_yellow = yellow(img_org)
        img_tmp = cv2.bitwise_or(img_tmp, img_yellow)
        img_tmp = cv2.blur(img_tmp, ksize=(5, 5))
        img_for_draw = cv2.Canny(img_tmp, 200, 255, apertureSize=3)
        road_lines = road(img_for_draw)
        img_for_draw = cv2.cvtColor(img_for_draw, cv2.COLOR_GRAY2BGR)
        road_lines_resized = cv2.resize(road_lines, (img_org.shape[1], img_org.shape[0]))
        if len(road_lines_resized.shape) == 2:
            road_lines_resized = cv2.cvtColor(road_lines_resized, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_or(img_org, road_lines_resized)
        cv2.imshow("img", img)
        if output_file is not None:
            output_writer.write(img)
        if contour_file is not None:
            contour_writer.write(img_for_draw)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if output_file is not None:
        output_writer.release()
    if contour_file is not None:
        contour_writer.release()
    cv2.destroyAllWindows()


def equalize(img_org):
    img_tmp = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_tmp = cv2.equalizeHist(img_tmp)
    count = np.array([[0, 0], [1279, 0], [1279, 340], [0, 340]])
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
    count = np.array([[0, 0], [1279, 0], [1279, 340], [0, 340]])
    img_tmp = cv2.fillPoly(img_tmp, pts=[count], color=(0))
    return img_tmp


def road(img_for_draw):
    lines = cv2.HoughLinesP(img_for_draw, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=100)
    img_lines = np.zeros_like(img_for_draw)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_lines


if __name__ == "__main__":
    input_file = r"C:\xampp\htdocs\VISUALAI\website-django\inspection\src\core\img\road.mp4"
    output_file = "output_video.mp4"
    contour_file = "contour_video.mp4"
    main(input_file, output_file, contour_file)
