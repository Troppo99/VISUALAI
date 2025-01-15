import cv2
import os


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


print(40 * "- ")
video_path = r"C:\xampp\htdocs\VISUALAI\qc-project\videos\labeling\defect\defect1.mp4"

cap = cv2.VideoCapture(video_path)

base_output_dir = fr"C:\xampp\htdocs\VISUALAI\qc-project\images\labeling\defect1"
make_dir(base_output_dir)

total_frames = 200
folders_count = 2
frames_per_folder = total_frames // folders_count

frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
interval = total_video_frames // total_frames

current_frame = 0
saved_frames = 0
folder_index = 1

while saved_frames < total_frames:
    ret, frame = cap.read()

    if not ret:
        break

    if current_frame % interval == 0:
        print(f"Proses folder {folder_index:02}...")
        folder_name = f"{folder_index:02}"
        folder_path = os.path.join(base_output_dir, folder_name)
        make_dir(folder_path)

        frame_filename = os.path.join(folder_path, f"frame_{saved_frames + 1}.jpg")
        cv2.imwrite(frame_filename, frame)

        saved_frames += 1

        if saved_frames % frames_per_folder == 0:
            folder_index += 1

    current_frame += 1

cap.release()
print("Proses selesai!")
