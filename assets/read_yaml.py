import os, glob, yaml

main_dir = r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\images\datasets\gisco_merged"

for dataset in os.listdir(main_dir):
    dataset_path = os.path.join(main_dir, dataset)
    if os.path.isdir(dataset_path):
        yaml_files = glob.glob(os.path.join(dataset_path, "*.yaml"))
        if not yaml_files:
            print(dataset, ": File YAML tidak ditemukan")
            continue
        with open(yaml_files[0], "r") as f:
            data = yaml.safe_load(f)
        nc_val = data.get("nc", "Key 'nc' tidak ditemukan")
        names_val = data.get("names", "Key 'names' tidak ditemukan")
        print(f"Dataset: {dataset}")
        print(f"Nilai 'nc': {nc_val}")
        print(f"Nilai 'names': {names_val}")

        label_dirs = []
        for sub in ["train", "valid"]:
            sub_labels = os.path.join(dataset_path, sub, "labels")
            if os.path.isdir(sub_labels):
                label_dirs.append(sub_labels)
        if not label_dirs:
            print("Folder 'labels' tidak ditemukan")
        else:
            counts = {}
            for label_dir in label_dirs:
                label_files = glob.glob(os.path.join(label_dir, "*.txt"))
                for label_file in label_files:
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                idx = parts[0]
                                counts[idx] = counts.get(idx, 0) + 1
            if counts:
                print("Jumlah objek per kelas:")
                for cls, count in counts.items():
                    if isinstance(names_val, list):
                        try:
                            class_name = names_val[int(cls)]
                        except (IndexError, ValueError):
                            class_name = f"Kelas {cls}"
                    elif isinstance(names_val, dict):
                        class_name = names_val.get(cls, f"Kelas {cls}")
                    else:
                        class_name = f"Kelas {cls}"
                    print(f"{class_name} : {count} label")
            else:
                print("Tidak ada file label .txt ditemukan")
        print("-" * 40)
