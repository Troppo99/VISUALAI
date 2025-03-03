import os, glob, yaml, shutil


def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    unique = filename
    counter = 1
    while os.path.exists(os.path.join(directory, unique)):
        unique = f"{base}_{counter}{ext}"
        counter += 1
    return unique


src_dir = r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\images\datasets\GISCO"
combined_dir = r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\images\datasets\combined"
train_target = os.path.join(combined_dir, "train")
valid_target = os.path.join(combined_dir, "valid")
os.makedirs(train_target, exist_ok=True)
os.makedirs(valid_target, exist_ok=True)

dataset_mapping = {}
all_classes = set()
datasets = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
for ds in datasets:
    ds_path = os.path.join(src_dir, ds)
    yaml_files = glob.glob(os.path.join(ds_path, "*.yaml"))
    if not yaml_files:
        print(f"{ds}: YAML tidak ditemukan")
        continue
    with open(yaml_files[0], "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if not names:
        print(f"{ds}: Key 'names' tidak ditemukan")
        continue
    dataset_mapping[ds] = names
    all_classes.update(names)

sorted_classes = sorted(all_classes)
combined_mapping = {cls: i for i, cls in enumerate(sorted_classes)}
print("Mapping kelas gabungan:", combined_mapping)

for ds in datasets:
    ds_path = os.path.join(src_dir, ds)
    if ds not in dataset_mapping:
        print(f"{ds}: tidak ada mapping kelas")
        continue
    names = dataset_mapping[ds]
    for split in ["train", "valid"]:
        split_path = os.path.join(ds_path, split)
        if not os.path.isdir(split_path):
            print(f"{ds}: Folder {split} tidak ditemukan")
            continue
        labels_dir = os.path.join(split_path, "labels")
        images_dir = os.path.join(split_path, "images")
        if not os.path.isdir(labels_dir):
            print(f"{ds}: Folder labels di {split} tidak ditemukan")
            continue
        if not os.path.isdir(images_dir):
            print(f"{ds}: Folder images di {split} tidak ditemukan")
            continue
        label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
        if not label_files:
            print(f"{ds}: Tidak ada file label di folder {labels_dir}")
            continue
        for lab in label_files:
            new_lines = []
            with open(lab, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        orig_idx = int(parts[0])
                    except Exception as e:
                        print(f"Error di {lab}: {e}")
                        continue
                    if orig_idx < 0 or orig_idx >= len(names):
                        print(f"Indeks {orig_idx} tidak valid untuk {ds} pada {lab}")
                        continue
                    cls_name = names[orig_idx]
                    new_idx = combined_mapping.get(cls_name, orig_idx)
                    parts[0] = str(new_idx)
                    new_lines.append(" ".join(parts))
            target_split_dir = train_target if split == "train" else valid_target
            base_lab = os.path.basename(lab)
            new_lab_name = get_unique_filename(target_split_dir, f"{ds}_{base_lab}")
            target_lab_path = os.path.join(target_split_dir, new_lab_name)
            with open(target_lab_path, "w") as f:
                f.write("\n".join(new_lines))
            print(f"Label disalin ke: {target_lab_path}")

            base_img = os.path.splitext(os.path.basename(lab))[0]
            image_found = False
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = os.path.join(images_dir, base_img + ext)
                if os.path.exists(img_file):
                    new_img_name = get_unique_filename(target_split_dir, f"{ds}_{base_img}{ext}")
                    target_img_path = os.path.join(target_split_dir, new_img_name)
                    shutil.copy2(img_file, target_img_path)
                    print(f"Image disalin ke: {target_img_path}")
                    image_found = True
                    break
            if not image_found:
                print(f"Tidak menemukan gambar untuk label {lab}")
