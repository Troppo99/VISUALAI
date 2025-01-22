import subprocess, socket
from pathlib import Path

# --- Daftar kamera & PC ---
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
    "TroppoLungo",
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


# Lakukan pembagian kamera
camera_distribution = distribute_cameras(all_camera, pcs, special_pc="PC-8")
nama_pc = socket.gethostname()

# ------------------------------------------------
#  Jadwal per kamera
#  "DEFAULT" jika kamera tidak ada di list ini.
# ------------------------------------------------
camera_schedules = {
    # Jadwal untuk CUTTING3
    "CUTTING3": {
        "work_days": ["mon", "tue", "wed", "thu", "fri"],
        "time_ranges": [
            ((7, 30, 0), (9, 44, 0)),  # S1
            ((9, 45, 0), (12, 49, 0)),  # S2
            ((12, 50, 0), (13, 5, 0)),  # S3
        ],
    },
    # Jadwal untuk CUTTING2
    "CUTTING2": {
        "work_days": ["mon", "tue", "wed", "thu", "fri"],
        "time_ranges": [
            ((7, 30, 0), (9, 44, 0)),  # S1
            ((9, 45, 0), (12, 49, 0)),  # S2
            ((12, 50, 0), (13, 5, 0)),  # S3
        ],
    },
    # Jadwal default (untuk kamera lain yang tidak disebut di atas)
    "DEFAULT": {
        "work_days": ["mon", "tue", "wed", "thu", "fri"],
        "time_ranges": [
            ((10, 44, 0), (10, 44, 10)),
        ],
    },
}

# Periksa apakah PC ini dikenal
if nama_pc in camera_distribution:
    start_idx = 0
    assigned_cameras = {}
    for pc in pcs:
        count = camera_distribution[pc]
        assigned_cameras[pc] = all_camera[start_idx : start_idx + count]
        start_idx += count

    # Daftar kamera untuk PC ini
    camera_list = assigned_cameras.get(nama_pc, [])
    print(f"PC '{nama_pc}' menerima kamera: {camera_list}")
else:
    print(f"PC '{nama_pc}' tidak dikenali. Tidak ada kamera yang dialokasikan.")
    camera_list = []

processes = []
cwd = Path.cwd()
script_dir = Path(__file__).resolve().parent

# -----------------------------------------------------
# Template run-{camera}.py
# Perhatikan double braces {{ }}
# Pastikan path sys.path.append(...) mengarah ke folder 'src'
# -----------------------------------------------------
template = r"""import time, sys

# Pastikan path ke folder 'src'
sys.path.append(r"C:\xampp\htdocs\VISUALAI\website-django\five_s\src")
# Jika Anda yakin UNC path berfungsi, boleh gunakan:
# sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")

from libs.Scheduler import Scheduler

camera_schedules = {{
    "CUTTING3": {{
        "work_days": ["mon", "tue", "wed", "thu", "fri"], 
        "time_ranges": [
            ((11, 12, 0), (11, 12, 10)),
            ((11, 12, 15), (11, 12, 20)),
            ((11, 12, 25), (11, 12, 40)),
        ],
    }},
    "CUTTING2": {{
        "work_days": ["mon", "tue", "wed", "thu", "fri"], 
        "time_ranges": [
            ((11, 12, 0), (11, 12, 10)),
            ((11, 12, 15), (11, 12, 20)),
            ((11, 12, 25), (11, 12, 40)),
        ],
    }},
    "DEFAULT": {{
        "work_days": ["mon", "tue", "wed", "thu", "fri"], 
        "time_ranges": [
            ((10, 59, 0), (10, 59, 10))
        ],
    }}
}}

camera_name = "{camera}"

# Ambil jadwal dari dict di atas
schedule_config = camera_schedules.get(camera_name, camera_schedules["DEFAULT"])

detector_args = {{
    "camera_name": camera_name,
    # "schedule_config": schedule_config,
    "window_size": (320, 240),
    "is_insert": False,
}}
scheduler = Scheduler(detector_args, schedule_config)

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

    # Buat file run-xxx.py jika belum ada
    if not script_path.exists():
        print(f"Script not found: {script_path}. Membuat file baru...")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                # format() hanya dipakai utk {camera}
                f.write(template.format(camera=cam))
            print(f"File {script_path} berhasil dibuat.")
        except Exception as e:
            print(f"Gagal membuat file {script_path}: {e}")
            continue

    # Jalankan script
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

# Tunggu sampai semua proses selesai (blocking)
for p in processes:
    try:
        p.wait()
        print(f"Proses dengan PID {p.pid} telah selesai.")
    except Exception as e:
        print(f"Error saat menunggu proses dengan PID {p.pid}: {e}")
