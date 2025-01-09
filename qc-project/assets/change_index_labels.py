import os


def ubah_indeks_label(folder_labels, indeks_baru=1, indeks_lama=0):
    if not os.path.isdir(folder_labels):
        print(f"Folder {folder_labels} tidak ditemukan.")
        return

    for filename in os.listdir(folder_labels):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_labels, filename)

            with open(file_path, "r") as file:
                lines = file.readlines()

            lines_baru = []
            perubahan = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        indeks = int(parts[0])
                        if indeks == indeks_lama:
                            parts[0] = str(indeks_baru)
                            perubahan = True
                        lines_baru.append(" ".join(parts) + "\n")
                    except ValueError:
                        lines_baru.append(line)
                else:
                    lines_baru.append(line)

            if perubahan:
                with open(file_path, "w") as file:
                    file.writelines(lines_baru)
                print(f"Diubah: {file_path}")
            else:
                print(f"Tidak ada perubahan pada: {file_path}")


def main():
    folder_X = r"C:\xampp\htdocs\VISUALAI\qc-project\datasets\cross-zero\X - Copy"  # Ganti dengan path yang sesuai jika diperlukan
    train_labels = os.path.join(folder_X, "train", "labels")
    valid_labels = os.path.join(folder_X, "valid", "labels")

    print("Mengubah indeks label di folder train/labels...")
    ubah_indeks_label(train_labels)

    print("\nMengubah indeks label di folder valid/labels...")
    ubah_indeks_label(valid_labels)


if __name__ == "__main__":
    main()
