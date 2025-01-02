# dengan cara "python -m build.code.main"
# from build.libs.myLib import Hello

# hello = Hello()
# hello

# dengan cara "python build/code/main.py"
import os, sys

# Menambahkan parent folder ke sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
print("Current dir:", current_dir)
parent_dir = os.path.join(current_dir, "..")
print("Parent dir:", parent_dir)
print("Before : ", sys.path)
sys.path.append(parent_dir) # ini yang paling krusial
print("After : ", sys.path)

# Setelah sys.path dimodifikasi, baru import
from libs.myLib import Hello

hello = Hello()
hello
