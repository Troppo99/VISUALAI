import subprocess
import socket
from pathlib import Path


class CameraPlay:
    def __init__(self):
        self.all_camera = [
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

        self.pcs = [
            "PC-100",
            "PC-101",
            "PC-102",
            "PC-8",
            "TroppoLungo",
        ]

        self.camera_distribution = {}
        self.nama_pc = socket.gethostname()
        self.camera_list = []
        self.processes = []
        self.cwd = Path.cwd()
        self.script_dir = Path(__file__).resolve().parent
        self.template = r"""import time, sys

sys.path.append(r"\\10.5.0.3\VISUALAI\website-django\five_s\src")
from libs.Scheduler import Scheduler

camera_schedules = {{
    "CUTTING3": {{
        "work_days": ["mon", "tue", "wed", "thu", "fri"], 
        "time_ranges": [
            ((8, 34, 0), (8, 34, 10)),
            ((8, 34, 15), (8, 34, 20)),
            ((8, 34, 25), (8, 34, 40)),
        ],
    }},
    "CUTTING2": {{
        "work_days": ["mon", "tue", "wed", "thu", "fri"], 
        "time_ranges": [
            ((8, 5, 0), (8, 5, 10)),
            ((8, 5, 15), (8, 5, 20)),
            ((8, 5, 25), (8, 5, 40)),
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

schedule_config = camera_schedules.get(camera_name, camera_schedules["DEFAULT"])

detector_args = {{
    "camera_name": camera_name,
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

    def distribute_cameras(self, all_cameras, pcs, special_pc="PC-8"):
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

    def assign_cameras_to_pc(self):
        if self.nama_pc not in self.camera_distribution:
            print(f"PC '{self.nama_pc}' tidak dikenali. Tidak ada kamera yang dialokasikan.")
            self.camera_list = []
            return

        start_idx = 0
        assigned_cameras = {}
        for pc in self.pcs:
            count = self.camera_distribution[pc]
            assigned_cameras[pc] = self.all_camera[start_idx : start_idx + count]
            start_idx += count

        self.camera_list = assigned_cameras.get(self.nama_pc, [])
        print(f"PC '{self.nama_pc}' menerima kamera: {self.camera_list}")

    def create_and_run_scripts(self):
        for cam in self.camera_list:
            python_executable = self.cwd / ".venv" / "Scripts" / "python.exe"
            script_path = self.script_dir / f"run-{cam}.py"

            if not python_executable.exists():
                print(f"Python executable not found: {python_executable}")
                continue

            if not script_path.exists():
                print(f"Script not found: {script_path}. Membuat file baru...")
                try:
                    with open(script_path, "w", encoding="utf-8") as f:
                        f.write(self.template.format(camera=cam))
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
                    cwd=self.script_dir,
                )
                self.processes.append(p)
                print(f"Menjalankan {script_path} dengan PID {p.pid}.")
            except Exception as e:
                print(f"Gagal menjalankan {script_path}: {e}")

    def wait_for_processes(self):
        for p in self.processes:
            try:
                p.wait()
                print(f"Proses dengan PID {p.pid} telah selesai.")
            except Exception as e:
                print(f"Error saat menunggu proses dengan PID {p.pid}: {e}")

    def run(self):
        self.camera_distribution = self.distribute_cameras(self.all_camera, self.pcs, special_pc="PC-8")
        self.assign_cameras_to_pc()
        self.create_and_run_scripts()
        self.wait_for_processes()


if __name__ == "__main__":
    app = CameraPlay()
    app.run()
