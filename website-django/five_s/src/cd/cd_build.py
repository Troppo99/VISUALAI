camera_list = [
    "HALAMANDEPAN1",
    "EKSPEDISI1",
    # "OFFICE1",
    # "OFFICE2",
    # "OFFICE3",
]

template = """from cd_sch import Scheduling
import time


detector_args = {{
    "confidence_threshold": 0,
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

for camera in camera_list:
    with open(f"website-django/five_s/src/cd/run-{camera}.py", "w") as f:
        f.write(template.format(camera=camera))
