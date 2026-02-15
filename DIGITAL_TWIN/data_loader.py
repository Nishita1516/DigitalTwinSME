import os

import pandas as pd


# STEP 1: Load & Clean Data
# Load FD001, clean columns, drop sensors
def load_fd001(path: str):
    columns = (
        ["engine_id", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )

    # Extra blank columns in C-MAPSS files
    # NASA files contain extra spaces, which create empty columns. hence we use sep=r"\s+"
    #
    # “Advanced datasets such as N-C-MAPSS provide higher-fidelity simulations;
    # however, due to their increased dimensionality and preprocessing complexity,
    # the classical C-MAPSS FD001 dataset was selected to ensure reproducibility
    # and focused model evaluation.”
    fd001 = pd.read_csv(path, sep=r"\s+", header=None)
    fd001.columns = columns
    fd001["dataset"] = "FD001"

    # Remove Non-Informative Sensors (standard in literature)
    drop_sensors = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
    fd001.drop(columns=drop_sensors, inplace=True)

    return fd001


def load_synthetic_data() -> pd.DataFrame:
    """
    Load the pre-generated synthetic Digital Twin dataset.
    Looks for the CSV in the project root alongside this package.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "DATA", "FD001_Synthetic_Digital_Twin_Dataset.csv")
    df = pd.read_csv(csv_path)
    return df

