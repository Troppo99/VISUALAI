import subprocess
from pathlib import Path
import os

camera_list = ["ROBOTIC"]

processes = []
cwd = Path.cwd()
script_dir = Path(__file__).resolve().parent

template = """from bcd_sch import Scheduling
import time


detector_args = {{
    "camera_name": "{camera}",
    "window_size": (320, 240)
}}
scheduler = Scheduling(detector_args, "OFFICE")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated by user.")
    scheduler.shutdown()
"""

for cam in camera_list:
    python_executable = cwd / ".venv" / "Scripts" / "python.exe"
    script_path = script_dir / f"run-{cam}.py"

    if not python_executable.exists():
        print(f"Python executable not found: {python_executable}")
        continue

    if not script_path.exists():
        print(f"Script not found: {script_path}. Membuat file baru...")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(template.format(camera=cam))
            print(f"File {script_path} berhasil dibuat.")
        except Exception as e:
            print(f"Gagal membuat file {script_path}: {e}")
            continue

    try:
        p = subprocess.Popen(
            [
                str(python_executable),
                str(script_path),
            ],
            cwd=script_dir,
        )
        processes.append(p)
        print(f"Menjalankan {script_path} dengan PID {p.pid}.")
    except Exception as e:
        print(f"Gagal menjalankan {script_path}: {e}")

for p in processes:
    try:
        p.wait()
        print(f"Proses dengan PID {p.pid} telah selesai.")
    except Exception as e:
        print(f"Error saat menunggu proses dengan PID {p.pid}: {e}")
