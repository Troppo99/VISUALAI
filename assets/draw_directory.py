import os


def print_directory_tree(root_dir, indent=""):
    items = sorted(os.listdir(root_dir))
    for index, item in enumerate(items):
        item_path = os.path.join(root_dir, item)
        is_last = index == len(items) - 1
        if os.path.isdir(item_path):
            print(indent + ("└── " if is_last else "├── ") + item + "/")
            new_indent = indent + ("    " if is_last else "│   ")
            print_directory_tree(item_path, new_indent)
        else:
            print(indent + ("└── " if is_last else "├── ") + item)


def main():
    root_folder = r"C:\xampp\htdocs\VISUALAI\website"
    if not os.path.exists(root_folder):
        print(f"Folder sumber '{root_folder}' tidak ditemukan.")
        return

    folder_name = os.path.basename(os.path.normpath(root_folder))
    print(folder_name + "/")
    print_directory_tree(root_folder)


if __name__ == "__main__":
    main()
