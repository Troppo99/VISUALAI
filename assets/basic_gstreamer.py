# .draft.py
import subprocess
import cv2
import numpy as np
import logging
import os
import sys
import threading

# Konfigurasikan logging
logging.basicConfig(level=logging.DEBUG)

# Definisikan path lengkap ke gst-launch-1.0.exe
gst_launch_path = r"D:\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"  # Gantilah sesuai dengan path Anda

# Periksa apakah gst-launch-1.0.exe ada
if not os.path.isfile(gst_launch_path):
    logging.error(f"Executable GStreamer tidak ditemukan di path: {gst_launch_path}")
    sys.exit(1)

# Definisikan pipeline GStreamer
gst_pipeline = [gst_launch_path, "-v", "rtspsrc", "location=rtsp://admin:oracle2015@10.5.0.7:554/Streaming/Channels/1", "latency=0", "!", "rtph265depay", "!", "h265parse", "!", "avdec_h265", "!", "videoconvert", "!", "jpegenc", "!", "multipartmux", "!", "filesink", "location=-"]  # Mengarah ke stdout


def read_stderr(pipe):
    while True:
        line = pipe.readline()
        if not line:
            break
        logging.error(f"GStreamer stderr: {line.decode().strip()}")


# Jalankan pipeline sebagai subprocess
try:
    process = subprocess.Popen(gst_pipeline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    logging.debug("Pipeline GStreamer berjalan.")

    # Mulai thread untuk membaca stderr
    stderr_thread = threading.Thread(target=read_stderr, args=(process.stderr,))
    stderr_thread.start()

    while True:
        # Baca buffer dari stdout
        data = process.stdout.read(1024)
        if not data:
            break

        # Cari batas frame multipart
        frames = data.split(b"\xff\xd8")  # JPEG SOI marker
        for i in range(1, len(frames)):
            frame_data = b"\xff\xd8" + frames[i]
            # Decode frame menggunakan OpenCV
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except FileNotFoundError as e:
    logging.error(f"Executable tidak ditemukan: {e}")
except Exception as e:
    logging.error(f"Error saat menjalankan GStreamer: {e}")
    # Baca stderr
    stderr = process.stderr.read().decode()
    logging.error(f"Stderr dari GStreamer: {stderr}")
finally:
    process.kill()
    cv2.destroyAllWindows()
