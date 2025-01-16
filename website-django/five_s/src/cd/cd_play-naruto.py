import subprocess

camera_list = [
    "HALAMANDEPAN1",
    "EKSPEDISI1",
    # "OFFICE1",
    # "OFFICE2",
    # "OFFICE3",
]

processes = []
for cam in camera_list:
    p = subprocess.Popen(["C:\\xampp\\htdocs\\VISUALAI\\.venv\\Scripts\\python.exe", f"C:\\xampp\\htdocs\\VISUALAI\\website-django\\five_s\\src\\cd\\run-{cam}.py"])
    processes.append(p)

for p in processes:
    p.wait()
