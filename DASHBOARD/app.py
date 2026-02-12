# ======================================
# Imports
# ======================================
import os
import sys

import pandas as pd
import torch
import streamlit as st
import numpy as np

# Ensure project root is on Python path so package imports work when
# running `streamlit run DASHBOARD/app.py`
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from DASHBOARD.ui import load_css, page_title, metric_card, section_header

from DIGITAL_TWIN.data_loader import load_synthetic_data
from DIGITAL_TWIN.twin_data_prep import create_sliding_windows
from DIGITAL_TWIN.sequences import create_sequences
from DIGITAL_TWIN.config import SEQ_LEN

from MODELS.predict_rul import load_model, predict_rul
from MODELS.shap_explainer import get_shap_explainer, explain_prediction


# ======================================
# Streamlit Config
# ======================================
st.set_page_config(
    page_title="Digital Twin Dashboard",
    layout="wide"
)

load_css()
page_title()


# ======================================
# Load Data
# ======================================
test_df = load_synthetic_data()

sensor_cols = [col for col in test_df.columns if col.startswith("s")]
input_size = len(sensor_cols)


# ======================================
# Load Model (cached)
# ======================================
@st.cache_resource
def initialize_model_and_explainer():
    model = load_model(input_size)
    
    # Create sequences for background SHAP data

    X_all, _ = create_sliding_windows(
        test_df,
        window_size=SEQ_LEN,              # MUST match training
        feature_cols=sensor_cols,
        target_col="RUL"
    )
    background_data = X_all[:100]

    explainer = get_shap_explainer(model, background_data)

    return model, explainer


model, explainer = initialize_model_and_explainer()


# ======================================
# Helper Function
# ======================================
def prepare_dashboard_payload(engine_id, predicted_rul, importance, feature_names):

    # Ensure importance is a 1D array aligned with feature names
    importance = np.array(importance)

    # Reduce any extra dimensions by averaging over non-feature axes
    while importance.ndim > 1:
        importance = importance.mean(axis=0)

    # Align lengths to avoid shape mismatches
    min_len = min(len(feature_names), len(importance))
    sensors = feature_names[:min_len]
    importance = importance[:min_len]

    df = pd.DataFrame(
        {
            "sensor": sensors,
            "importance": importance,
        }
    ).sort_values(by="importance", ascending=False)

    return {
        "engine_id": engine_id,
        "predicted_rul": round(float(predicted_rul), 2),
        "top_contributing_sensors": df.head(5).to_dict(orient="records")
    }


# ======================================
# Real Inference Loop
# ======================================
engine_ids = test_df["engine_id"].unique()[:3]

dashboard_data = []

for engine_id in engine_ids:

    engine_data = test_df[test_df["engine_id"] == engine_id]

    X_seq, _ = create_sequences(
        engine_data,
        seq_length=SEQ_LEN,
        feature_cols=sensor_cols,
        target_col="RUL"
    )

    if len(X_seq) == 0:
        continue

    last_sequence = X_seq[-1]

    # 1️⃣ Predict RUL
    predicted_rul = predict_rul(model, last_sequence)

    # 2️⃣ SHAP Explanation
    shap_values = explain_prediction(explainer, last_sequence)

    # DeepExplainer returns list for regression models
    shap_array = shap_values[0][0]  # shape: (seq_len, features)

    # Aggregate importance across time dimension
    importance = np.abs(shap_array).mean(axis=0)

    dashboard_data.append(
        prepare_dashboard_payload(
            engine_id,
            predicted_rul,
            importance,
            sensor_cols
        )
    )


dashboard_df = pd.DataFrame([{
    "engine_id": d["engine_id"],
    "predicted_rul": d["predicted_rul"],
    "top_sensors": ", ".join(
        [f"{s['sensor']} ({round(s['importance'], 3)})"
         for s in d['top_contributing_sensors']]
    )
} for d in dashboard_data])


# ======================================
# UI
# ======================================
section_header("Engine Overview")

cols = st.columns(len(dashboard_data))

for col, item in zip(cols, dashboard_data):
    with col:
        metric_card(
            label=f"Engine {item['engine_id']} RUL",
            value=item['predicted_rul']
        )

section_header("Predicted RUL Chart")
st.bar_chart(dashboard_df.set_index('engine_id')['predicted_rul'])

section_header("Top Contributing Sensors")

for item in dashboard_data:
    st.markdown(f"### Engine {item['engine_id']}")
    st.table(pd.DataFrame(item['top_contributing_sensors']))