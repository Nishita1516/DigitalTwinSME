# STEP 3 – Synthetic Digital Twin generation
# Synthetic Digital twin dataset generation
# Synthetic twin generation (noise, drift, bias)


import numpy as np
import pandas as pd

# Gaussian Noise Twin (Measurement Error)
def add_gaussian_noise(fd001, sensor_cols, noise_level=0.01):
    noisy = fd001.copy()
    for col in sensor_cols:
        sigma = noise_level * fd001[col].std()
        noisy[col] += np.random.normal(0, sigma, size=len(fd001))
    return noisy

# Sensor Drift Twin (Aging Effect)
def add_sensor_drift(fd001, sensor_cols, drift_rate=0.0001):
    drifted = fd001.copy()
    for col in sensor_cols:
        drifted[col] += drift_rate * drifted["cycle"]
    return drifted

# Environmental Bias Twin (Operating Shift)
def add_bias(fd001, sensor_cols, bias_factor=0.02):
    biased = fd001.copy()
    for col in sensor_cols:
        bias = bias_factor * fd001[col].mean()
        biased[col] += bias
    return biased

# Clone Engine-by-Engine (CRITICAL)
def generate_digital_twins(fd001):
    sensor_cols = [c for c in fd001.columns if c.startswith("s")]
    synthetic_data = []

    for engine_id, engine_fd001 in fd001.groupby("engine_id"):

        # REAL
        real = engine_fd001.copy()
        real["twin_id"] = engine_id
        real["data_type"] = "real"
        real["twin_type"] = "none"
        synthetic_data.append(real)

        # NOISE TWIN
        noise = add_gaussian_noise(engine_fd001, sensor_cols)
        noise["twin_id"] = f"{engine_id}_noise"
        noise["data_type"] = "synthetic"
        noise["twin_type"] = "noise"
        synthetic_data.append(noise)

        # DRIFT TWIN
        drift = add_sensor_drift(engine_fd001, sensor_cols)
        drift["twin_id"] = f"{engine_id}_drift"
        drift["data_type"] = "synthetic"
        drift["twin_type"] = "drift"
        synthetic_data.append(drift)

        # BIAS TWIN
        bias = add_bias(engine_fd001, sensor_cols)
        bias["twin_id"] = f"{engine_id}_bias"
        bias["data_type"] = "synthetic"
        bias["twin_type"] = "bias"
        synthetic_data.append(bias)

    # Combine All Twins
    return pd.concat(synthetic_data, ignore_index=True)
