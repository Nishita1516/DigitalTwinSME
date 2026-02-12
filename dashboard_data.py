# ======================================
# Imports
# ======================================
import os
import sys

# Add project root directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import pandas as pd
import torch
import streamlit as st
import numpy as np

from ui import load_css, page_title, metric_card, section_header

from data_loader import load_synthetic_data  
from sequences import create_sequences

from step5_model_xai_dashboard.predict_rul import load_model, predict_rul
from step5_model_xai_dashboard.shap_explainer import (
    get_shap_explainer,
    explain_prediction
)


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
        window_size=30,              # MUST match training
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

    df = pd.DataFrame({
        "sensor": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

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

    X_seq, _ = create_sequences(engine_data, sensor_cols)

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
