import sys
print("Starting imports...", flush=True)

import os
print("Imported os", flush=True)
import pandas as pd
print("Imported pandas", flush=True)
import numpy as np
print("Imported numpy", flush=True)
import joblib
print("Imported joblib", flush=True)
from sklearn.metrics import mean_squared_error
print("Imported sklearn.metrics", flush=True)
from sklearn.ensemble import RandomForestRegressor
print("Imported sklearn.ensemble", flush=True)

# Project imports
sys.path.insert(0, r"c:\WORKFILES\dissertation\Digital Twin")
print("Added path", flush=True)

try:
    from DIGITAL_TWIN.data_loader import load_fd001
    print("Imported data_loader", flush=True)
except Exception as e:
    print(f"Failed data_loader: {e}", flush=True)

try:
    from DIGITAL_TWIN.preprocessing import add_rul
    print("Imported preprocessing", flush=True)
except Exception as e:
    print(f"Failed preprocessing: {e}", flush=True)

try:
    from MODELS.predict_rul import load_model, predict_rul
    print("Imported predict_rul", flush=True)
except Exception as e:
    print(f"Failed predict_rul: {e}", flush=True)

print("All imports done.", flush=True)
