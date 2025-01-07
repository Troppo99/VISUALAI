import subprocess

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
for cam in camera_list:
    p = subprocess.Popen([".venv\\Scripts\\python.exe", f"tests\\bd_run-{cam}.py"])
    processes.append(p)

for p in processes:
    p.wait()
