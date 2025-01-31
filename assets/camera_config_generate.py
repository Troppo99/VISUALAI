import json
import os

# Step 1: Define the list of cameras with their corresponding IP addresses
camera_list = [
    {"name": "BUFFER1", "ip": "172.16.0.25"},
    {"name": "CUTTING1", "ip": "172.16.0.43"},
    {"name": "CUTTING2", "ip": "172.16.0.41"},
    {"name": "CUTTING3", "ip": "172.16.0.40"},
    {"name": "CUTTING4", "ip": "172.16.0.145"},
    {"name": "CUTTING5", "ip": "172.16.0.113"},
    {"name": "CUTTING8", "ip": "172.16.0.137"},
    {"name": "CUTTING9", "ip": "172.16.0.142"},
    {"name": "CUTTING10", "ip": "172.16.0.144"},
    {"name": "EKSPEDISI1", "ip": "172.16.0.153"},
    {"name": "EKSPEDISI2", "ip": "172.16.0.16"},
    {"name": "FOLDING1", "ip": "172.16.0.148"},
    {"name": "FOLDING2", "ip": "172.16.0.141"},
    {"name": "FOLDING3", "ip": "172.16.0.20"},
    {"name": "FREEMETAL1", "ip": "172.16.0.18"},
    {"name": "FREEMETAL2", "ip": "172.16.0.140"},
    {"name": "GUDANGACC1", "ip": "172.16.0.151"},
    {"name": "GUDANGACC2", "ip": "172.16.0.152"},
    {"name": "GUDANGACC3", "ip": "172.16.0.154"},
    {"name": "GUDANGACC4", "ip": "172.16.0.37"},
    {"name": "GUDANGKAIN1", "ip": "172.16.0.158"},
    {"name": "GUDANGKAIN2", "ip": "172.16.0.156"},
    {"name": "GUDANGKAIN3", "ip": "172.16.0.157"},
    {"name": "GUDANGKAIN5", "ip": "172.16.0.161"},
    {"name": "HALAMANBELAKANG1", "ip": "172.16.0.33"},
    {"name": "HALAMANBELAKANG2", "ip": "172.16.0.35"},
    {"name": "HALAMANDEPAN1", "ip": "172.16.0.150"},
    {"name": "INNERBOX1", "ip": "172.16.0.139"},
    {"name": "JALURCUTTING1", "ip": "172.16.0.38"},
    {"name": "JALURCUTTING2", "ip": "172.16.0.39"},
    {"name": "KANTIN1", "ip": "172.16.0.36"},
    {"name": "KANTIN2", "ip": "172.16.0.108"},
    {"name": "LINEMANUAL10", "ip": "172.16.0.31"},
    {"name": "LINEMANUAL14", "ip": "172.16.0.11"},
    {"name": "LINEMANUAL15", "ip": "172.16.0.19"},
    {"name": "METALDET1", "ip": "172.16.0.14"},
    {"name": "OFFICE1", "ip": "172.16.0.34"},
    {"name": "OFFICE2", "ip": "172.16.0.100"},
    {"name": "OFFICE3", "ip": "172.16.0.42"},
    {"name": "SEWING1", "ip": "172.16.0.116"},
    {"name": "SEWING2", "ip": "172.16.0.27"},
    {"name": "SEWING3", "ip": "172.16.0.147"},
    {"name": "SEWING4", "ip": "172.16.0.143"},
    {"name": "SEWING5", "ip": "172.16.0.26"},
    {"name": "SEWING6", "ip": "172.16.0.146"},
    {"name": "SEWING7", "ip": "172.16.0.12"},
    {"name": "SEWINGBACK1", "ip": "172.16.0.138"},
    {"name": "SEWINGBACK2", "ip": "172.16.0.17"},
    {"name": "SEWINGOFFICE", "ip": "172.16.0.32"},
    {"name": "ROBOTICS", "ip": "10.5.5.1"},
]

# Step 2: Define the base paths
BASE_CONF_PATH = "//10.5.0.3/VISUALAI/website-django/static/resources/conf"
BASE_IMAGE_PATH = "//10.5.0.3/VISUALAI/website-django/static/images/dd_reference"

# Step 3: Define the ROI categories and their corresponding subdirectories
roi_categories = {"bd_rois": "bd_rois", "cd_rois": "cd_rois", "bcd_rois": "bcd_rois", "dd_rois": "dd_rois", "sm_rois": "sm_rois"}

# Step 4: Generate the configuration for each camera
camera_config = {}

for camera in camera_list:
    camera_name = camera["name"]
    camera_ip = camera["ip"]

    # Initialize the camera entry with the IP
    camera_entry = {"ip": camera_ip}

    # Add ROI paths
    for roi_key, sub_dir in roi_categories.items():
        camera_entry[roi_key] = f"{BASE_CONF_PATH}/{sub_dir}/{camera_name}.json"

    # Add dd_reference path
    camera_entry["dd_reference"] = f"{BASE_IMAGE_PATH}/{camera_name}.jpg"

    # Add the camera entry to the main config
    camera_config[camera_name] = camera_entry

# Step 5: Define the output file path
output_file_path = "camera_config.json"  # You can change this path as needed

# Optional: If you want to save it to a specific directory, uncomment and set the path
# OUTPUT_DIRECTORY = r"C:\path\to\your\desired\directory"
# os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
# output_file_path = os.path.join(OUTPUT_DIRECTORY, "camera_config.json")

# Step 6: Save the configuration to a JSON file
try:
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(camera_config, f, indent=2)
    print(f"Configuration JSON has been successfully saved to '{output_file_path}'.")
except Exception as e:
    print(f"An error occurred while saving the JSON file: {e}")
