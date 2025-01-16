import subprocess
from pathlib import Path

camera_list = [
    # "BUFFER1",
    # "CUTTING3",
    # "EKSPEDISI2",
    # "FOLDING2",
    # "FOLDING3",
    # "FREEMETAL1",
    # "FREEMETAL2",
    # "GUDANGACC1",
    # "GUDANGACC2",
    # "GUDANGACC3",
    # "GUDANGACC4",
    # "INNERBOX1",
    # "KANTIN1",
    # "LINEMANUAL10",
    # "LINEMANUAL14",
    # "LINEMANUAL15",
    # "METALDET1",
    "SEWING1",
    "SEWING2",
    "SEWING3",
    "SEWING4",
    "SEWING5",
    "SEWING6",
    "SEWING7",
    # "SEWINGBACK1",
    # "SEWINGBACK2",
    # "SEWINGOFFICE",
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
