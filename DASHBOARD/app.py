# ======================================
# Imports
# ======================================
import os
import sys

import pandas as pd
import torch
import streamlit as st
import numpy as np
import joblib

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
from MODELS.nlp_baseline import MaintenanceLogClassifier 


# ======================================
# Streamlit Config - Professional Dark Theme
# ======================================
st.set_page_config(
    page_title="Engine Predictive Maintenance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Dark Theme CSS
st.markdown("""
<style>
    /* Dark theme background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    
    /* Dashboard Title */
    .dashboard-title {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .dashboard-subtitle {
        color: #a0a0a0;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    
    /* Summary metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #242938 100%);
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 4px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        height: 100%;
    }
    
    .metric-card-green { border-left-color: #00d97e; }
    .metric-card-orange { border-left-color: #f7b924; }
    .metric-card-red { border-left-color: #e63757; }
    .metric-card-gray { border-left-color: #6c757d; }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .metric-value-green { color: #00d97e; }
    .metric-value-orange { color: #f7b924; }
    .metric-value-red { color: #e63757; }
    .metric-value-white { color: #ffffff; }
    
    .metric-subtitle {
        color: #6c757d;
        font-size: 0.8rem;
    }
    
    /* Panel styling - Clean look without shaded background */
    .dashboard-panel {
        background: transparent;
        border-radius: 0;
        padding: 0.5rem 0;
        margin-bottom: 0 rem;
        border: none;
    }
    
    .panel-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d3748;
    }
    
    /* Table styling - Transparent and clean */
    .stDataFrame, .stTable {
        background-color: transparent !important;
    }
    
    /* Remove white background from tables */
    table {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    
    th {
        background-color: #242938 !important;
        color: #ffffff !important;
    }
    
    td {
        background-color: transparent !important;
    }
    
    /* Compact spacing - Remove blue gaps */
    h1, h2, h3 {
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Remove default Streamlit spacing that causes blue gaps */
    .main .block-container {
        max-width: 100%;
        padding-top: 1.5rem;
    }
    
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
    
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #242938 !important;
        border-radius: 4px;
        font-size: 0.85rem !important;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-healthy {
        background-color: rgba(0, 217, 126, 0.15);
        color: #00d97e;
    }
    
    .status-warning {
        background-color: rgba(247, 185, 36, 0.15);
        color: #f7b924;
    }
    
    .status-critical {
        background-color: rgba(230, 55, 87, 0.15);
        color: #e63757;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown("""
<div class="dashboard-title">🏭 Engine Predictive Maintenance Dashboard</div>
<div class="dashboard-subtitle">Using Machine Learning to classify engine status based on trained profiles</div>
""", unsafe_allow_html=True)


# ======================================
# Load Data
# ======================================
test_df = load_synthetic_data()
log_df = pd.read_csv(os.path.join(ROOT_DIR, "DATA", "maintenance_logs.csv"))

sensor_cols = [col for col in test_df.columns if col.startswith("s")]
input_size = len(sensor_cols)


# ======================================
# Load Model (cached)
# ======================================
@st.cache_resource
def initialize_model_and_explainer():
    model = load_model(input_size)
    
    # Create sequences for background SHAP data
    
    # Load NLP Model
    nlp_model_path = os.path.join(ROOT_DIR, "MODELS", "nlp_baseline.pkl")
    if os.path.exists(nlp_model_path):
        nlp_model = joblib.load(nlp_model_path)
    else:
        nlp_model = None

    X_all, _ = create_sliding_windows(
        test_df,
        window_size=SEQ_LEN,              # MUST match training
        feature_cols=sensor_cols,
        target_col="RUL"
    )
    background_data = X_all[:100]

    explainer = get_shap_explainer(model, background_data)

    return model, explainer, nlp_model


model, explainer, nlp_model = initialize_model_and_explainer()


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
# Helper Function for Status Indicators
# ======================================
def get_status_indicator(rul_value):
    """
    Returns status indicator (emoji + color) based on RUL threshold.
    🔴 CRITICAL: RUL < 30 days
    ⚠️ WARNING: 30 <= RUL < 100 days
    ✅ HEALTHY: RUL >= 100 days
    """
    if rul_value < 30:
        return "🚨 CRITICAL", "#e63757"  # Red
    elif rul_value < 100:
        return "⚠️ WARNING", "#f7b924"  # Orange
    else:
        return "✅ HEALTHY", "#00d97e"  # Green

# ======================================
# PROFESSIONAL DASHBOARD UI
# ======================================

# Calculate summary statistics
total_engines = len(dashboard_data)
healthy_count = sum(1 for item in dashboard_data if item['predicted_rul'] >= 100)
warning_count = sum(1 for item in dashboard_data if 30 <= item['predicted_rul'] < 100)
critical_count = sum(1 for item in dashboard_data if item['predicted_rul'] < 30)

# ===========================================
# TOP ROW: Summary Metric Cards
# ===========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card metric-card-gray">
        <div class="metric-label">⚙️ Total Engines In Operations</div>
        <div class="metric-value metric-value-white">{total_engines}</div>
        <div class="metric-subtitle">Engines</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card metric-card-green">
        <div class="metric-label">✅ Healthy Engines</div>
        <div class="metric-value metric-value-green">{healthy_count}</div>
        <div class="metric-subtitle">Optimal Condition</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card metric-card-orange">
        <div class="metric-label">⚠️ Warning Engines</div>
        <div class="metric-value metric-value-orange">{warning_count}</div>
        <div class="metric-subtitle">Monitor Changes</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card metric-card-red">
        <div class="metric-label">🚨 Critical Engines</div>
        <div class="metric-value metric-value-red">{critical_count}</div>
        <div class="metric-subtitle">Requires Immediate Maintenance</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===========================================
# MIDDLE ROW: Engine Details in Grid Layout
# ===========================================

# Left Column: Machine Health Status Table
left_panel, right_panel = st.columns([1.3, 1])

with left_panel:
    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔋 Machine Health Status - Sorted by RUL</div>', unsafe_allow_html=True)
    
    # Create detailed table
    health_data = []
    for item in sorted(dashboard_data, key=lambda x: x['predicted_rul']):
        rul_val = item['predicted_rul']
        
        if rul_val < 30:
            status_html = '<span class="status-badge status-critical">CRITICAL</span>'
            action = "Schedule Maintenance NOW"
        elif rul_val < 100:
            status_html = '<span class="status-badge status-warning">WARNING</span>'
            action = "Plan Maintenance Soon"
        else:
            status_html = '<span class="status-badge status-healthy">HEALTHY</span>'
            action = "Continue Normal Operation"
        
        health_data.append({
            "Engine": f"Engine {item['engine_id']}",
            "RUL (Days)": f"{rul_val:.0f}",
            "Status": status_html,
            "Action": action
        })
    
    health_df = pd.DataFrame(health_data)
    
    # Display HTML table for better styling
    table_html = health_df.to_html(escape=False, index=False)
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with right_panel:
    st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔍 Top Failure Risk Sensors</div>', unsafe_allow_html=True)
    
    # Consolidate all sensors into one table
    all_sensors_data = []
    for item in dashboard_data:
        rul_value = item['predicted_rul']
        
        if rul_value < 30:
            status = "🚨 CRITICAL"
        elif rul_value < 100:
            status = "⚠️ WARNING"
        else:
            status = "✅ HEALTHY"
        
        # Get top 3 sensors for this engine
        top_sensors = item['top_contributing_sensors'][:3]
        for sensor in top_sensors:
            all_sensors_data.append({
                "Engine": f"Engine {item['engine_id']}",
                "Status": status,
                "Sensor": sensor.get('sensor', 'N/A'),
                "Risk Impact": f"{sensor.get('importance', 0):.3f}"
            })
    
    # Display as single consolidated table - Use st.table for clean static look
    sensors_table_df = pd.DataFrame(all_sensors_data)
    st.table(sensors_table_df)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================
# BOTTOM ROW: Maintenance Logs
# ===========================================
st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">📝 Maintenance Logs - Latest Entries</div>', unsafe_allow_html=True)

# Get logs
relevant_logs = log_df[log_df["engine_id"].isin([d["engine_id"] for d in dashboard_data])].copy()

if not relevant_logs.empty and nlp_model is not None:
    predictions = nlp_model.predict(relevant_logs["log_text"])
    relevant_logs["Predicted Fault"] = predictions
    
    probs = nlp_model.predict_proba(relevant_logs["log_text"])
    confidence = np.max(probs, axis=1)
    relevant_logs["Confidence"] = (confidence * 100).round(1).astype(str) + "%"
    
    relevant_logs = relevant_logs.sort_values("timestamp", ascending=False)
    
    # Columns for each engine
    log_cols = st.columns(len(dashboard_data))
    
    for idx, (col, item) in enumerate(zip(log_cols, dashboard_data)):
        with col:
            engine_id = item['engine_id']
            rul_value = item['predicted_rul']
            engine_logs = relevant_logs[relevant_logs["engine_id"] == engine_id].copy()
            
            if not engine_logs.empty:
                latest_log = engine_logs.iloc[0]
                
                if rul_value < 30:
                    badge_class = "status-critical"
                    emoji = "🚨"
                elif rul_value < 100:
                    badge_class = "status-warning"
                    emoji = "⚠️"
                else:
                    badge_class = "status-healthy"
                    emoji = "✅"
                
                st.markdown(f'<span class="status-badge {badge_class}">{emoji} Engine {engine_id}</span>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background: #242938; padding: 10px; border-radius: 6px; margin-top: 8px; font-size: 0.8rem;'>
                    <strong style='color: #ffffff;'>Latest Entry:</strong><br/>
                    <span style='color: #a0a0a0;'>{latest_log['log_text'][:55]}...</span><br/>
                    <span style='color: #f7b924;'><strong>Fault:</strong> {latest_log['Predicted Fault']}</span>
                    <span style='color: #00d97e; margin-left: 10px;'><strong>Conf:</strong> {latest_log['Confidence']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"📋 View All {len(engine_logs)} Logs", expanded=False):
                    display_cols = ["timestamp", "log_text", "Predicted Fault", "Confidence"]
                    all_logs_df = engine_logs[display_cols]  # Show ALL logs, not just 10
                    
                    st.dataframe(
                        all_logs_df,
                        hide_index=True,
                        use_container_width=True  # No fixed height = no scrolling
                    )
            else:
                st.info(f"No logs")

elif nlp_model is None:
    st.warning("NLP Model not found.")
else:
    st.info("No logs available.")

st.markdown('</div>', unsafe_allow_html=True)