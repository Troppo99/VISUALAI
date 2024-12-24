import os


def is_import_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("import ") or stripped.startswith("from ")


def update_merged_file(directory, output_file, custom_order=None):
    # Ambil semua file .py, kecuali file output
    files = [f for f in os.listdir(directory) if f.endswith(".py") and f != output_file]

    # Urutkan file sesuai custom_order jika ada
    if custom_order:
        order_map = {name: i for i, name in enumerate(custom_order)}
        files.sort(key=lambda x: order_map.get(x, len(files)))
    else:
        files.sort()

    exclude_imports = {"from BroomDetector import BroomDetector", "from DataHandler import DataHandler", "from Scheduling import Scheduling"}

    all_imports = set()
    code_lines = []

    # Baca setiap file, kumpulkan import dan kode lainnya
    for fname in files:
        fpath = os.path.join(directory, fname)
        with open(fpath, "r") as infile:
            for line in infile:
                # Cek apakah baris adalah import
                if is_import_line(line):
                    line_stripped = line.strip()
                    if line_stripped not in exclude_imports:
                        all_imports.add(line_stripped)
                else:
                    code_lines.append(line)

    # Tulis file output:
    # - Pertama tulis semua import unik
    # - Kemudian tulis sisa kode
    with open(os.path.join(directory, output_file), "w") as outfile:
        # Tulis import terlebih dahulu
        # Gunakan sorted agar konsisten urutannya
        for imp_line in sorted(all_imports):
            outfile.write(imp_line + "\n")

        outfile.write("\n")

        # Tulis baris kode lainnya
        for line in code_lines:
            outfile.write(line)


# Contoh penggunaan
update_merged_file("src", "build/_all.py", custom_order=["BroomDetector.py", "DataHandler.py", "Scheduling.py"])
