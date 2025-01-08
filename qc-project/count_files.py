import os


def hitung_file(folder_images, folder_labels, ekstensi_gambar=None):
    """
    Menghitung jumlah file gambar dan label dalam folder yang diberikan.

    Parameters:
    - folder_images (str): Path ke folder images.
    - folder_labels (str): Path ke folder labels.
    - ekstensi_gambar (list, optional): Daftar ekstensi file gambar yang diakui.
                                        Jika None, akan menggunakan ['.jpg', '.jpeg', '.png', '.bmp', '.gif'].

    Returns:
    - tuple: (jumlah_gambar, jumlah_label)
    """
    if ekstensi_gambar is None:
        ekstensi_gambar = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    # Hitung jumlah file gambar
    try:
        files_gambar = os.listdir(folder_images)
        jumlah_gambar = sum(1 for file in files_gambar if os.path.splitext(file)[1].lower() in ekstensi_gambar)
    except FileNotFoundError:
        print(f"Folder gambar tidak ditemukan: {folder_images}")
        jumlah_gambar = 0

    # Hitung jumlah file label (.txt)
    try:
        files_label = os.listdir(folder_labels)
        jumlah_label = sum(1 for file in files_label if os.path.splitext(file)[1].lower() == ".txt")
    except FileNotFoundError:
        print(f"Folder label tidak ditemukan: {folder_labels}")
        jumlah_label = 0

    return jumlah_gambar, jumlah_label


def main():
    # Tentukan path ke folder X
    folder_X = r"C:\xampp\htdocs\VISUALAI\qc-project\datasets\cross-zero\OX"  # Ganti dengan path yang sesuai jika diperlukan

    # Path ke folder train dan valid
    train_images = os.path.join(folder_X, "train", "images")
    train_labels = os.path.join(folder_X, "train", "labels")
    valid_images = os.path.join(folder_X, "valid", "images")
    valid_labels = os.path.join(folder_X, "valid", "labels")

    # Hitung untuk folder train
    jumlah_gambar_train, jumlah_label_train = hitung_file(train_images, train_labels)
    print(f"File gambar pada folder train ada {jumlah_gambar_train} dan file labelnya ada {jumlah_label_train}.")

    # Hitung untuk folder valid
    jumlah_gambar_valid, jumlah_label_valid = hitung_file(valid_images, valid_labels)
    print(f"File gambar pada folder valid ada {jumlah_gambar_valid} dan file labelnya ada {jumlah_label_valid}.")


if __name__ == "__main__":
    main()
