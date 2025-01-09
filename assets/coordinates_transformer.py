import os
import json

# Direktori tempat file JSON berada
DIRECTORY = r"C:\xampp\htdocs\VISUALAI\coords"

# Resolusi asli dan baru
ORIGINAL_WIDTH = 960
ORIGINAL_HEIGHT = 540
NEW_WIDTH = 640
NEW_HEIGHT = 640

# Faktor skala
SCALE_X = NEW_WIDTH / ORIGINAL_WIDTH  # ≈ 0.6667
SCALE_Y = NEW_HEIGHT / ORIGINAL_HEIGHT  # ≈ 1.1852


def scale_coordinate(coord):
    """
    Mengubah ukuran koordinat berdasarkan faktor skala.
    """
    x, y = coord
    new_x = round(x * SCALE_X)
    new_y = round(y * SCALE_Y)
    return [new_x, new_y]


def process_json_file(file_path):
    """
    Membaca file JSON, mengubah ukuran koordinat, dan menyimpan kembali.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Asumsikan struktur data adalah list dari list dari list koordinat
        # Contoh: [[[x1, y1], [x2, y2], ...]]
        # Modifikasi sesuai dengan struktur sebenarnya jika berbeda
        if isinstance(data, list):
            for i, sublist in enumerate(data):
                if isinstance(sublist, list):
                    for j, coord in enumerate(sublist):
                        if isinstance(coord, list) and len(coord) == 2:
                            data[i][j] = scale_coordinate(coord)

        # Simpan kembali ke file yang sama atau file baru
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Berhasil memproses: {file_path}")

    except Exception as e:
        print(f"Error memproses {file_path}: {e}")


def main():
    """
    Memproses semua file JSON di direktori yang ditentukan.
    """
    for filename in os.listdir(DIRECTORY):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(DIRECTORY, filename)
            process_json_file(file_path)


if __name__ == "__main__":
    main()
