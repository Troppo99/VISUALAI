import subprocess

processes = []
for i in range(1, 5):
    p = subprocess.Popen(["python", f"build\\subprocess\\run{i}.py"])
    processes.append(p)

for p in processes:
    p.wait()
