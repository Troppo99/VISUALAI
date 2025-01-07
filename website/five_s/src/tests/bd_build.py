camera_list = ["BUFFER1", "CUTTING3", "EKSPEDISI2", "FOLDING2", "FOLDING3", "FREEMETAL1", "FREEMETAL2", "GUDANGACC1", "GUDANGACC2", "GUDANGACC3", "GUDANGACC4", "INNERBOX1", "KANTIN1", "LINEMANUAL10", "LINEMANUAL14", "LINEMANUAL15", "METALDET1", "SEWING1", "SEWING2", "SEWING3", "SEWING4", "SEWING5", "SEWING6", "SEWING7", "SEWINGBACK1", "SEWINGBACK2", "SEWINGOFFICE"]

template = """from bd_sch import Scheduling
import time


detector_args = {{
    "confidence_threshold": 0,
    "camera_name": "{camera}",
    "window_size": (320, 240)
}}
scheduler = Scheduling(detector_args, "SEWING")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Program terminated by user.")
    scheduler.shutdown()
"""

for camera in camera_list:
    with open(f"website/five_s/src/tests/bd_run-{camera}.py", "w") as f:
        f.write(template.format(camera=camera))
