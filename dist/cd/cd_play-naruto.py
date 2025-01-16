import subprocess
from pathlib import Path

camera_list = [
    "HALAMANDEPAN1",
    "EKSPEDISI1",
    # "OFFICE1",
    # "OFFICE2",
    # "OFFICE3",
]

processes = []
cwd = Path.cwd()
script_dir = Path(__file__).resolve().parent

for cam in camera_list:
    python_executable = cwd / ".venv" / "Scripts" / "python.exe"
    script_path = script_dir / f"run-{cam}.py"

    if not python_executable.exists():
        print(f"Python executable not found: {python_executable}")
        continue

    if not script_path.exists():
        print(f"Script not found: {script_path}")
        continue

    p = subprocess.Popen(
        [
            str(python_executable),
            str(script_path),
        ],
    )
    processes.append(p)

for p in processes:
    p.wait()
