import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi modul Selfie Segmentation Mediapipe
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Pilih model: 0 untuk model ringan, 1 untuk model dengan akurasi lebih tinggi
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    cap = cv2.VideoCapture(0)  # Buka kamera (webcam)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame secara horizontal untuk tampilan mirror
        frame = cv2.flip(frame, 1)
        # Konversi BGR ke RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses frame untuk segmentation
        results = selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask

        # Buat kondisi threshold untuk memilih area foreground
        condition = mask > 0.5  # Jika nilai lebih dari 0.5, anggap itu foreground

        # Tentukan background baru, misalnya warna abu-abu
        bg_color = (192, 192, 192)  # BGR
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = bg_color

        # Gunakan np.where untuk mengganti background:
        # Jika condition True, gunakan frame asli, jika False, gunakan bg_image.
        output_image = np.where(condition[..., np.newaxis], frame, bg_image)

        cv2.imshow("Background Removal", output_image)
        if cv2.waitKey(1) & 0xFF == ord("n"):  # Tekan ESC untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()
