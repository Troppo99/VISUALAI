import os, glob, yaml

src_dir = r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\images\datasets\GISCO"
combined_dir = r"C:\xampp\htdocs\VISUALAI\website-django\five_s\static\images\datasets\gisco_merged\combined"
train_images_dir = os.path.join(combined_dir, "train", "images")
valid_images_dir = os.path.join(combined_dir, "valid", "images")

all_classes = set()
datasets = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
for ds in datasets:
    ds_path = os.path.join(src_dir, ds)
    yf = glob.glob(os.path.join(ds_path, "*.yaml"))
    if not yf:
        continue
    with open(yf[0], "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    all_classes.update(names)

sorted_classes = sorted(all_classes)

data_yaml = {
    "train": train_images_dir,
    "val": valid_images_dir,
    "names": sorted_classes
}

output_path = os.path.join(combined_dir, "data.yaml")
with open(output_path, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print("data.yaml berhasil dibuat di:", output_path)
