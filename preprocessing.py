# Step 2 RUL + normalization
# RUL generation, normalization

import pandas as pd
from sklearn.preprocessing import StandardScaler

def add_rul(fd001):
    max_cycles = fd001.groupby("engine_id")["cycle"].max()
    fd001 = fd001.merge(max_cycles, on="engine_id", suffixes=("", "_max"))
    fd001["RUL"] = fd001["cycle_max"] - fd001["cycle"]
    fd001.drop(columns=["cycle_max"], inplace=True)
    return fd001

def normalize_sensors(fd001):
    sensor_cols = [c for c in fd001.columns if c.startswith("s")]
    scaler = StandardScaler()
    fd001[sensor_cols] = scaler.fit_transform(fd001[sensor_cols])
    return fd001, scaler
