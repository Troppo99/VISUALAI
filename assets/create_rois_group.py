import cv2
import math
import os
import json


def camera_config(camera_name):
    """
    Retrieves the IP address for the given camera name from the configuration file.
    """
    config_path = r"C:\xampp\htdocs\VISUALAI\website-django\static\resources\conf\camera_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    ip = config.get(camera_name, {}).get("ip")
    if not ip:
        raise ValueError(f"IP address for camera '{camera_name}' not found in configuration.")
    return ip


# List of camera names to process
CAMERA_LIST = ["SEWING1", "SEWING2", "SEWING3", "SEWING4", "SEWING5", "SEWING6", "SEWING7", "SEWINGBACK1", "SEWINGBACK2", "SEWINGOFFICE"]

# Directory where JSON files will be saved
OUTPUT_DIRECTORY = r"C:\xampp\htdocs\VISUALAI\coconf"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Resolusi asli dan baru
ORIGINAL_WIDTH = 960
ORIGINAL_HEIGHT = 540
NEW_WIDTH = 640
NEW_HEIGHT = 640

# Faktor skala
SCALE_X = NEW_WIDTH / ORIGINAL_WIDTH  # ≈ 0.6667
SCALE_Y = NEW_HEIGHT / ORIGINAL_HEIGHT  # ≈ 1.1852

# Initialize variables for storing keypoints
chains = []  # List to store all chains of keypoints
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
    """
    Mengubah ukuran koordinat berdasarkan faktor skala.
    """
    x, y = coord
    new_x = round(x * SCALE_X)
    new_y = round(y * SCALE_Y)
    return [new_x, new_y]


def print_borders(camera_name):
    borders = [[[p[0], p[1]] for p in chain] for chain in chains if len(chain) > 0]
    print(f"Original Borders for {camera_name} = {borders}")

    # Skalakan koordinat
    scaled_borders = [[scale_coordinate(p) for p in chain] for chain in chains if len(chain) > 0]
    print(f"Scaled Borders for {camera_name} = {scaled_borders}")

    # Simpan ke file JSON
    output_file = os.path.join(OUTPUT_DIRECTORY, f"{camera_name}.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(scaled_borders, f, indent=2)
        print(f"Koordinat yang diskalakan telah disimpan di: {output_file}")
    except Exception as e:
        print(f"Error menyimpan file JSON untuk {camera_name}: {e}")


def main():
    global chains  # Ensure chains is recognized as global
    """
    Memproses input video atau gambar, memungkinkan pengguna untuk memilih ROI,
    menskalakan koordinat, dan menyimpan hasilnya ke file JSON untuk setiap kamera dalam CAMERA_LIST.
    """
    for camera_name in CAMERA_LIST:
        print(f"\nProcessing camera: {camera_name}")
        try:
            camera_ip = camera_config(camera_name)
        except ValueError as ve:
            print(ve)
            continue  # Skip to next camera if IP not found

        video_path = f"rtsp://admin:oracle2015@{camera_ip}:554/Streaming/Channels/1"
        # You can adjust the username, password, and channel as needed

        # Reset chains for the new camera
        chains = []

        # Check if RTSP stream is valid
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video stream for camera '{camera_name}'. Skipping to next.")
            continue

        cv2.namedWindow("Video")
        cv2.setMouseCallback("Video", create_keypoint)

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from camera '{camera_name}'. Exiting stream.")
                break

            frame = cv2.resize(frame, (display_width, display_height))
            draw_chains(frame)

            if dragging and preview_point is not None:
                frame_copy = frame.copy()
                if len(chains) > 0 and len(chains[-1]) > 0:
                    cv2.line(frame_copy, chains[-1][-1], preview_point, (255, 0, 0), 2)
                cv2.circle(frame_copy, preview_point, 5, (0, 255, 0), -1)
                cv2.imshow("Video", frame_copy)
            else:
                cv2.imshow("Video", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("n"), ord("N")]:
                if len(chains) == 0:
                    print("No coordinates defined for this camera.")
                else:
                    print_borders(camera_name)
                break  # Move to next camera
            elif key == 13:  # Enter key
                if len(chains) > 0 and len(chains[-1]) > 0:
                    print_chains()
                    chains.append([])
            elif key == ord("a"):
                chains = []
                print("All chains cleared.")
            elif key == ord("f"):
                undo_last_point()
                print("Last point undone.")

        cap.release()
        cv2.destroyAllWindows()

    print("\nAll cameras have been processed.")


if __name__ == "__main__":
    main()
