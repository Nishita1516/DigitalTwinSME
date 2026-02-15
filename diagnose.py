import os
import sys

try:
    with open("diagnose_output.txt", "w") as f:
        f.write("Python is working.\n")
        f.write(f"Executable: {sys.executable}\n")
        f.write(f"CWD: {os.getcwd()}\n")
        
    print("Diagnosis successful.")
except Exception as e:
    print(f"Diagnosis failed: {e}")
