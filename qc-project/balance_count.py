import os
import shutil
import random


def hitung_file(folder_images, folder_labels, ekstensi_gambar=None):
    """
    Menghitung jumlah file gambar dan label dalam folder yang diberikan.

    Parameters:
    - folder_images (str): Path ke folder images.
    - folder_labels (str): Path ke folder labels.
    - ekstensi_gambar (list, optional): Daftar ekstensi file gambar yang diakui.
                                        Jika None, akan menggunakan ['.jpg', '.jpeg', '.png', '.bmp', '.gif'].

    Returns:
    - set: Set nama file (tanpa ekstensi) yang memiliki pasangan gambar dan label.
    """
    if ekstensi_gambar is None:
        ekstensi_gambar = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    # Dapatkan set nama file gambar tanpa ekstensi
    try:
        files_gambar = os.listdir(folder_images)
        set_gambar = set(os.path.splitext(file)[0] for file in files_gambar if os.path.splitext(file)[1].lower() in ekstensi_gambar)
    except FileNotFoundError:
        print(f"Folder gambar tidak ditemukan: {folder_images}")
        set_gambar = set()

    # Dapatkan set nama file label tanpa ekstensi
    try:
        files_label = os.listdir(folder_labels)
        set_label = set(os.path.splitext(file)[0] for file in files_label if os.path.splitext(file)[1].lower() == ".txt")
    except FileNotFoundError:
        print(f"Folder label tidak ditemukan: {folder_labels}")
        set_label = set()

    # Hanya pertimbangkan file yang memiliki pasangan gambar dan label
    set_tersedia = set_gambar & set_label
    return set_tersedia


def hitung_persentase(jumlah_valid, jumlah_train, total):
    """
    Menghitung persentase valid dan train.

    Parameters:
    - jumlah_valid (int): Jumlah file di valid.
    - jumlah_train (int): Jumlah file di train.
    - total (int): Total file.

    Returns:
    - tuple: (persentase_valid, persentase_train)
    """
    persentase_valid = (jumlah_valid / total) * 100 if total > 0 else 0
    persentase_train = (jumlah_train / total) * 100 if total > 0 else 0
    return persentase_valid, persentase_train


def balance_dataset(folder_X, target_valid_percent=30):
    """
    Menyeimbangkan dataset sehingga folder valid berisi target_valid_percent% dari total dataset.

    Parameters:
    - folder_X (str): Path ke folder X.
    - target_valid_percent (float): Persentase target untuk folder valid.
    """
    # Path ke train dan valid
    train_images = os.path.join(folder_X, "train", "images")
    train_labels = os.path.join(folder_X, "train", "labels")
    valid_images = os.path.join(folder_X, "valid", "images")
    valid_labels = os.path.join(folder_X, "valid", "labels")

    # Hitung file yang tersedia
    set_train = hitung_file(train_images, train_labels)
    set_valid = hitung_file(valid_images, valid_labels)

    jumlah_train = len(set_train)
    jumlah_valid = len(set_valid)
    total = jumlah_train + jumlah_valid

    if total == 0:
        print("Tidak ada file yang ditemukan dalam dataset.")
        return

    persentase_valid, persentase_train = hitung_persentase(jumlah_valid, jumlah_train, total)

    print(f"Jumlah file train: {jumlah_train}")
    print(f"Jumlah file valid: {jumlah_valid}")
    print(f"Total file: {total}")
    print(f"Persentase train: {persentase_train:.2f}%")
    print(f"Persentase valid: {persentase_valid:.2f}%")

    # Hitung jumlah yang diinginkan untuk valid
    target_valid = int((target_valid_percent / 100) * total)

    if jumlah_valid == target_valid:
        print("Dataset sudah seimbang.")
        return
    elif jumlah_valid > target_valid:
        # Perlu memindahkan (jumlah_valid - target_valid) file dari valid ke train
        jumlah_pindah = jumlah_valid - target_valid
        action = "memindahkan"
        source_set = set_valid
        dest_set = set_train
        source_images = valid_images
        source_labels = valid_labels
        dest_images = train_images
        dest_labels = train_labels
    else:
        # Perlu memindahkan (target_valid - jumlah_valid) file dari train ke valid
        jumlah_pindah = target_valid - jumlah_valid
        action = "memindahkan"
        source_set = set_train
        dest_set = set_valid
        source_images = train_images
        source_labels = train_labels
        dest_images = valid_images
        dest_labels = valid_labels

    if jumlah_pindah == 0:
        print("Tidak perlu melakukan penyesuaian.")
        return

    # Acak urutan file untuk pemindahan
    source_list = list(source_set)
    random.shuffle(source_list)

    # Tentukan berapa banyak file yang dapat dipindahkan
    jumlah_pindah = min(jumlah_pindah, len(source_list))

    if jumlah_pindah == 0:
        print("Tidak ada file yang dapat dipindahkan.")
        return

    print(f"Jumlah file yang akan dipindahkan: {jumlah_pindah}")

    for i in range(jumlah_pindah):
        filename = source_list[i]
        # File gambar
        file_gambar_src = find_file_with_any_extension(os.path.join(source_images), filename)
        # File label
        file_label_src = os.path.join(source_labels, f"{filename}.txt")

        if not file_gambar_src or not os.path.exists(file_label_src):
            print(f"Skipping {filename}: Gambar atau label tidak ditemukan.")
            continue

        # Tentukan tujuan
        file_gambar_dst = os.path.join(dest_images, os.path.basename(file_gambar_src))
        file_label_dst = os.path.join(dest_labels, f"{filename}.txt")

        # Pindahkan file gambar
        shutil.move(file_gambar_src, file_gambar_dst)
        # Pindahkan file label
        shutil.move(file_label_src, file_label_dst)

        # Update sets
        source_set.remove(filename)
        dest_set.add(filename)

        print(f"Berhasil memindahkan: {filename}")

    # Recount setelah pemindahan
    jumlah_train = len(set_train)
    jumlah_valid = len(set_valid)
    total = jumlah_train + jumlah_valid
    persentase_valid, persentase_train = hitung_persentase(jumlah_valid, jumlah_train, total)

    print("\nSetelah penyesuaian:")
    print(f"Jumlah file train: {jumlah_train}")
    print(f"Jumlah file valid: {jumlah_valid}")
    print(f"Total file: {total}")
    print(f"Persentase train: {persentase_train:.2f}%")
    print(f"Persentase valid: {persentase_valid:.2f}%")


def find_file_with_any_extension(folder, filename_without_ext, ekstensi_gambar=None):
    """
    Mencari file gambar dengan nama tertentu tanpa memperhatikan ekstensi.

    Parameters:
    - folder (str): Path ke folder tempat mencari file gambar.
    - filename_without_ext (str): Nama file tanpa ekstensi.
    - ekstensi_gambar (list, optional): Daftar ekstensi file gambar yang diakui.
                                        Jika None, akan menggunakan ['.jpg', '.jpeg', '.png', '.bmp', '.gif'].

    Returns:
    - str atau None: Path ke file gambar jika ditemukan, else None.
    """
    if ekstensi_gambar is None:
        ekstensi_gambar = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    for ext in ekstensi_gambar:
        file_path = os.path.join(folder, filename_without_ext + ext)
        if os.path.exists(file_path):
            return file_path
    return None


def main():
    # Tentukan path ke folder X
    folder_X = r"C:\xampp\htdocs\VISUALAI\qc-project\datasets\cross-zero\OX"  # Ganti dengan path yang sesuai jika diperlukan

    # Pastikan folder X ada
    if not os.path.isdir(folder_X):
        print(f"Folder X tidak ditemukan: {folder_X}")
        return

    balance_dataset(folder_X, target_valid_percent=30)


if __name__ == "__main__":
    main()
