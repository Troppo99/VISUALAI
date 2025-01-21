import subprocess, socket
from pathlib import Path

all_camera = [
    "HALAMANDEPAN1",
    "EKSPEDISI1",
    "GUDANGACC1",
    "GUDANGACC2",
    "FOLDING1",
    "FOLDING2",
    "FOLDING3",
    "METALDET1",
    "FREEMETAL1",
    "FREEMETAL2",
    "CUTTING3",
    "CUTTING2",
]

pcs = [
    "PC-100",
    "PC-101",
    "PC-102",
    "PC-8",
    # "TroppoLungo",
]


def distribute_cameras(all_cameras, pcs, special_pc="PC-8"):
    total_cameras = len(all_cameras)
    total_pcs = len(pcs)

    base = total_cameras // total_pcs
    remainder = total_cameras % total_pcs

    camera_distribution = {pc: base for pc in pcs}

    non_special_pcs = [pc for pc in pcs if pc != special_pc]

    for i in range(remainder):
        pc = non_special_pcs[i % len(non_special_pcs)]
        camera_distribution[pc] += 1

    return camera_distribution


camera_distribution = distribute_cameras(all_camera, pcs, special_pc="PC-8")

nama_pc = socket.gethostname()

if nama_pc in camera_distribution:
    start_idx = 0
    assigned_cameras = {}
    for pc in pcs:
        count = camera_distribution[pc]
        assigned_cameras[pc] = all_camera[start_idx : start_idx + count]
        start_idx += count
    camera_list = assigned_cameras.get(nama_pc, [])
    print(f"PC '{nama_pc}' menerima kamera: {camera_list}")
else:
    print(f"PC '{nama_pc}' tidak dikenali. Tidak ada kamera yang dialokasikan.")
    camera_list = []

processes = []
cwd = Path.cwd()
script_dir = Path(__file__).resolve().parent

template = """import time, os, sys


sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")
from libs.Scheduler import Scheduler


detector_args = {{
    "camera_name": "{camera}",
    "window_size": (320, 240),
    "is_insert": False,
}}
scheduler = Scheduler(detector_args, "bcd")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated by user.")
    scheduler.shutdown()
"""

pc_name = socket.gethostname()

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
                f.write(template.format(camera=cam, pc_name=pc_name))
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
