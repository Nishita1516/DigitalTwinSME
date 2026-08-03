import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import torch
from sklearn.ensemble import RandomForestRegressor

# Ensure project root is on Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from DIGITAL_TWIN.data_loader import load_fd001
from DIGITAL_TWIN.preprocessing import add_rul, normalize_sensors
from DIGITAL_TWIN.twin_data_prep import split_engines, create_sliding_windows
from DIGITAL_TWIN.config import SENSOR_COLS, SEQ_LEN
from MODELS.predict_rul import load_model, predict_rul
from MODELS.shap_explainer import get_shap_explainer, explain_prediction

# Page Config
st.set_page_config(page_title="Engineer Dashboard - Digital Twin", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .main { background: #ffffff; padding: 2rem; border-radius: 10px; }
    .stMetric { background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #e9ecef; }
    [data-testid="stMetricValue"] { color: #1a1f2e !important; }
    [data-testid="stMetricLabel"] { color: #6c757d !important; }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Engineer Technical Dashboard")
st.subheader("Model Performance, Comparison & Sensor Deep-Dive")

@st.cache_resource
def load_and_prep_data():
    data_path = os.path.join(ROOT_DIR, "DATA", "Sensor Data", "NASA C-MAPSS 1 Turbofan Engine Degradation Dataset", "train_FD001.txt")
    fd001 = load_fd001(data_path)
    fd001 = add_rul(fd001)
    fd001, scaler = normalize_sensors(fd001)
    train_df, _, test_df = split_engines(fd001)
    
    # Load LSTM
    lstm_model = load_model(len(SENSOR_COLS))
    
    # Train a quick RF baseline if not saved (for demo purposes)
    X_train_w, y_train = create_sliding_windows(train_df, SEQ_LEN, SENSOR_COLS, "RUL")
    X_train_rf = np.mean(X_train_w, axis=1) # Aggregated features
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_rf, y_train)
    
    # Setup SHAP Background
    background_data = X_train_w[:100]
    explainer = get_shap_explainer(lstm_model, background_data)
    
    return test_df, lstm_model, rf_model, explainer

test_df, lstm_model, rf_model, explainer = load_and_prep_data()

# --- Sidebar ---
st.sidebar.header("Control Panel")
selected_engine = st.sidebar.selectbox("Select Engine ID", test_df["engine_id"].unique())

# --- Top Metrics ---
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("LSTM Goal (RMSE)", "16.79")
with c2:
    st.metric("RF Baseline (RMSE)", "50.56")
with c3:
    st.metric("Dataset", "NASA FD001")

# --- Engine Data Prep ---
engine_data = test_df[test_df["engine_id"] == selected_engine]
X_w, y_true = create_sliding_windows(engine_data, SEQ_LEN, SENSOR_COLS, "RUL")

# --- Predictions ---
lstm_preds = [predict_rul(lstm_model, seq) for seq in X_w]
X_rf = np.mean(X_w, axis=1)
rf_preds = rf_model.predict(X_rf)

# --- Model Comparison Plot ---
st.divider()
st.header("📈 Model Prediction Comparison")
st.write(f"Comparing Remaining Useful Life (RUL) for Engine **#{selected_engine}**")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_true, label="Ground Truth", color="#39FF14", linewidth=2.5) # Neon Green
ax.plot(lstm_preds, label="LSTM (Advanced)", color="#FF007F", linestyle="--", linewidth=2) # Neon Pink
ax.plot(rf_preds, label="Random Forest (Baseline)", color="#00F5FF", linestyle=":", linewidth=2) # Neon Cyan
ax.set_xlabel("Cycle")
ax.set_ylabel("RUL")
ax.legend()
st.pyplot(fig)

# --- Sensor Analysis ---
st.divider()
st.header("🌡️ Detailed Sensor Telemetry")
st.write("Reviewing original sensor trends to identify anomalies.")

sensors_to_plot = st.multiselect("Select Sensors to View", SENSOR_COLS, default=SENSOR_COLS[:4])
if sensors_to_plot:
    # Define bright colors matching the selected sensors
    bright_palette = ['#FF007F', '#00F5FF', '#39FF14', '#FFCC00', '#CC00FF', '#FF5722']
    selected_colors = [bright_palette[i % len(bright_palette)] for i in range(len(sensors_to_plot))]
    st.line_chart(engine_data.set_index("cycle")[sensors_to_plot], color=selected_colors)

# --- SHAP Interpretation ---
st.divider()
st.header("🧠 Root Cause Analysis (SHAP)")
st.write("Technical breakdown of which sensors impacted the *latest* LSTM prediction.")

if len(X_w) > 0:
    last_seq = X_w[-1]
    sample_tensor = torch.tensor(np.expand_dims(last_seq, axis=0), dtype=torch.float32)
    shap_vals = explainer.shap_values(sample_tensor, check_additivity=False)
    
    # Process SHAP for plotting
    # Make the reduction robust across different SHAP versions/output shapes
    try:
        shap_vals_arr = np.array(shap_vals)
        # We want to collapse to (features,) by averaging over all other dims
        # The last dimension is always the features
        if shap_vals_arr.shape[-1] == len(SENSOR_COLS):
            importance = np.abs(shap_vals_arr).reshape(-1, len(SENSOR_COLS)).mean(axis=0)
        else:
            # If shape doesn't match, try to find where the 14 is
            for dim in range(shap_vals_arr.ndim):
                if shap_vals_arr.shape[dim] == len(SENSOR_COLS):
                    importance = np.abs(shap_vals_arr).mean(axis=tuple(i for i in range(shap_vals_arr.ndim) if i != dim))
                    break
            else:
                importance = np.zeros(len(SENSOR_COLS))
    except:
        importance = np.zeros(len(SENSOR_COLS))

    # Safety check to prevent ValueError
    if len(importance) != len(SENSOR_COLS):
        importance = np.zeros(len(SENSOR_COLS))

    indices = np.argsort(importance)[::-1]
    all_impacted_sensors = [SENSOR_COLS[i] for i in indices]
    
    # 🔍 Interactive Feature Selection for RCA
    st.write("🔍 Select sensors to compare their relative impacts on the prediction.")
    top_sensor = all_impacted_sensors[0]
    
    selected_for_rca = st.multiselect(
        "Select Sensors to Compare",
        options=SENSOR_COLS, 
        default=all_impacted_sensors[:5],
        help="The most impacted sensor is always included to provide context.",
        key="rca_multiselect"
    )
    
    # Always ensure the top sensor is in the final plotted list
    plot_sensors = list(set(selected_for_rca) | {top_sensor})
    
    shap_df = pd.DataFrame({
        "Sensor": SENSOR_COLS,
        "Feature Importance": importance
    })
    
    # Filter and display
    filtered_shap_df = shap_df[shap_df["Sensor"].isin(plot_sensors)].sort_values(by="Feature Importance", ascending=False)
    st.bar_chart(filtered_shap_df.set_index("Sensor"), color='#00F5FF')
    
    st.caption("Note: Sensors 1, 5, 6, 10, 16, 18, and 19 are excluded as they provide non-informative constant values per NASA dataset standards.")
else:
    st.info("Not enough cycles for SHAP analysis.")

st.sidebar.info("This is the technical 'Engineer View'. It runs independently from the main User Interface.")
