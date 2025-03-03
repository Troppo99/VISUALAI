import cv2
import math
import os
import json


def camera_config():
    with open(r"\\10.5.0.3\VISUALAI\website-django\static\resources\conf\camera_config.json", "r") as f:
        config = json.load(f)
    ip = config["GM2CUTTING2"]["ip"]
    return ip


file_name = "coordinates"
video_path = f"rtsp://admin:oracle2015@{camera_config()}:554/Streaming/Channels/1"

ORIGINAL_WIDTH = 960
ORIGINAL_HEIGHT = 540
NEW_WIDTH = 640
NEW_HEIGHT = 640

SCALE_X = NEW_WIDTH / ORIGINAL_WIDTH
SCALE_Y = NEW_HEIGHT / ORIGINAL_HEIGHT

chains = []
dragging = False
preview_point = None
magnet_threshold = 10
display_width = 1280
display_height = 720


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_nearest_point(preview_point):
    nearest_point = None
    min_distance = float("inf")
    for chain in chains:
        for point in chain:
            dist = distance(preview_point, point)
            if dist < magnet_threshold and dist < min_distance:
                nearest_point = point
                min_distance = dist
    return nearest_point


def create_keypoint(event, x, y, flags, param):
    global chains, frame, dragging, preview_point
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        preview_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            preview_point = (x, y)
            nearest_point = find_nearest_point(preview_point)
            if nearest_point is not None:
                preview_point = nearest_point
            frame_copy = frame.copy()
            draw_chains(frame_copy)
            if len(chains) > 0 and len(chains[-1]) > 0:
                cv2.line(frame_copy, chains[-1][-1], preview_point, (255, 0, 0), 2)
            cv2.circle(frame_copy, preview_point, 5, (0, 255, 0), -1)
            cv2.imshow("Video", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        if dragging:
            nearest_point = find_nearest_point((x, y))
            if nearest_point is not None:
                x, y = nearest_point
            dragging = False
            if len(chains) == 0 or len(chains[-1]) == 0:
                chains.append([(x, y)])
            else:
                chains[-1].append((x, y))
                cv2.line(frame, chains[-1][-2], chains[-1][-1], (255, 0, 0), 2)
            cv2.circle(frame, chains[-1][-1], 5, (0, 255, 0), -1)
            cv2.imshow("Video", frame)


def draw_chains(img):
    for chain in chains:
        if len(chain) > 0:
            for i in range(1, len(chain)):
                cv2.line(img, chain[i - 1], chain[i], (255, 0, 0), 2)
            for point in chain:
                cv2.circle(img, point, 5, (0, 255, 0), -1)


def undo_last_point():
    global chains
    if len(chains) > 0:
        if len(chains[-1]) > 0:
            chains[-1].pop()
        if len(chains[-1]) == 0:
            chains.pop()


def print_chains():
    print("\nChains and Points:")
    non_empty_chain_count = 0
    for idx, chain in enumerate(chains):
        if len(chain) > 0:
            non_empty_chain_count += 1
            print(f"Chain {non_empty_chain_count}:")
            for i, point in enumerate(chain):
                print(f"  Point {i + 1}: ({point[0]}, {point[1]})")


def scale_coordinate(coord):
    x, y = coord
    new_x = round(x * SCALE_X)
    new_y = round(y * SCALE_Y)
    return [new_x, new_y]


def print_borders():
    borders = [[[p[0], p[1]] for p in chain] for chain in chains if len(chain) > 0]
    print(f"Original Borders = {borders}")
    scaled_borders = [[scale_coordinate(p) for p in chain] for chain in chains if len(chain) > 0]
    print(f"Scaled Borders = {scaled_borders}")


# Fungsi untuk memeriksa frame tidak hanya blur/gelap, tapi juga tidak corrupt
def is_frame_good(frame):
    # Pastikan frame tidak kosong dan memiliki ukuran yang wajar
    if frame is None or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    if w < 100 or h < 100:  # ukuran tidak masuk akal
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = cv2.mean(gray)[0]
    # Threshold bisa disesuaikan; ini hanya contoh
    if variance < 100:
        return False
    if brightness < 50:
        return False
    return True


def main():
    global chains, frame, dragging, preview_point
    is_image = False
    is_video = False

    if video_path.startswith("rtsp://"):
        is_video = True
    elif os.path.isfile(video_path):
        ext = os.path.splitext(video_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            is_image = True
        else:
            is_video = True
    else:
        print("Error: Path is not RTSP and not a valid file.")
        exit()

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", create_keypoint)

    if is_image:
        frame = cv2.imread(video_path)
        if frame is None:
            print("Error: could not read image.")
            exit()
        frame = cv2.resize(frame, (display_width, display_height))
        while True:
            frame_copy = frame.copy()
            draw_chains(frame_copy)
            if dragging and preview_point is not None:
                if len(chains) > 0 and len(chains[-1]) > 0:
                    cv2.line(frame_copy, chains[-1][-1], preview_point, (255, 0, 0), 2)
                cv2.circle(frame_copy, preview_point, 5, (0, 255, 0), -1)
            cv2.imshow("Video", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("n"), ord("N")]:
                print_borders()
                break
            elif key == 13:
                if len(chains) > 0 and len(chains[-1]) > 0:
                    print_chains()
                    chains.append([])
            elif key == ord("a"):
                chains = []
            elif key == ord("f"):
                undo_last_point()
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            exit()

        print("Mencari frame yang bagus dari CCTV...")
        good_frame = None
        while True:
            ret, frame_temp = cap.read()
            # Periksa apakah frame valid (ret True, tidak kosong, ukuran wajar)
            if not ret or frame_temp is None or frame_temp.size == 0:
                continue
            # Jika ukuran frame jauh berbeda dari yang diharapkan, skip frame corrupt
            h, w = frame_temp.shape[:2]
            if w < 100 or h < 100:
                continue
            if is_frame_good(frame_temp):
                good_frame = frame_temp
                break
        frame = cv2.resize(good_frame, (display_width, display_height))
        print("Frame yang bagus telah ditemukan, siap membuat koordinat.")
        while True:
            frame_copy = frame.copy()
            draw_chains(frame_copy)
            if dragging and preview_point is not None:
                if len(chains) > 0 and len(chains[-1]) > 0:
                    cv2.line(frame_copy, chains[-1][-1], preview_point, (255, 0, 0), 2)
                cv2.circle(frame_copy, preview_point, 5, (0, 255, 0), -1)
            cv2.imshow("Video", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("n"), ord("N")]:
                print_borders()
                break
            elif key == 13:
                if len(chains) > 0 and len(chains[-1]) > 0:
                    print_chains()
                    chains.append([])
            elif key == ord("a"):
                chains = []
            elif key == ord("f"):
                undo_last_point()
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
