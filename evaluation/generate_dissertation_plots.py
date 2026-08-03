import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import shap

# Setup paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from DIGITAL_TWIN.data_loader import load_fd001
from DIGITAL_TWIN.preprocessing import add_rul, normalize_sensors
from DIGITAL_TWIN.twin_data_prep import split_engines, create_sliding_windows, scale_windows
from MODELS.lstm_model import LSTMModel
from MODELS.predict_rul import load_model, predict_rul
from MODELS.nlp_baseline import MaintenanceLogClassifier

EVAL_DIR = os.path.join(ROOT_DIR, "EVALUATION")
DATA_DIR = os.path.join(ROOT_DIR, "DATA")
SENSOR_DATA_DIR = os.path.join(DATA_DIR, "Sensor Data", "NASA C-MAPSS 1 Turbofan Engine Degradation Dataset")

# --- 1. DATA PREP ---
print("Loading C-MAPSS Data for evaluation plots...")
fd001 = load_fd001(os.path.join(SENSOR_DATA_DIR, "train_FD001.txt"))
fd001 = add_rul(fd001)
fd001, scaler = normalize_sensors(fd001)

train_fd001, val_fd001, test_fd001 = split_engines(fd001)

FEATURE_COLS = [c for c in fd001.columns if c.startswith("s")]
WINDOW_SIZE = 30
TARGET = "RUL"

X_train, y_train = create_sliding_windows(train_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)
X_val, y_val = create_sliding_windows(val_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)
X_test, y_test = create_sliding_windows(test_fd001, WINDOW_SIZE, FEATURE_COLS, TARGET)

# Note: Using small subsets for extremely fast plot generation to avoid memory bottlenecks
X_train, X_val, X_test = scale_windows(X_train, X_val, X_test)

# --- 2. TRAIN & LOSS CURVE ---
print("Training quick LSTM for Loss Curves...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=len(FEATURE_COLS)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

epochs = 15 # Shorter epoch for plot generation
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    batch_losses = []
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        outs = model(X_b).squeeze()
        loss = criterion(outs, y_b)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))
    
    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            outs = model(X_b).squeeze()
            loss = criterion(outs, y_b)
            val_batch_losses.append(loss.item())
    val_losses.append(np.mean(val_batch_losses))

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss (MSE)', color='#00F5FF', linewidth=2) # Bright electric cyan
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss (MSE)', color='#FF007F', linewidth=2) # Bright neon pink/magenta
plt.title('LSTM Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(EVAL_DIR, "lstm_loss_curve.png"))
plt.close()
print("Saved Loss Curve: lstm_loss_curve.png")

# --- 3. TEST EVALUATION (SCATTER) ---
model.eval()
y_pred = []
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=64)
with torch.no_grad():
    for X_b, _ in test_loader:
        X_b = X_b.to(device)
        outs = model(X_b).squeeze()
        y_pred.extend(outs.cpu().numpy())

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6, s=10, color='#00F5FF') # Bright electric cyan
plt.plot([0, 150], [0, 150], color='#FF007F', linestyle='--', lw=2, label='Perfect Prediction') # Bright neon pink
plt.xlabel('Actual RUL (Cycles)')
plt.ylabel('Predicted RUL (Cycles)')
plt.title('Actual vs Predicted RUL (Test Dataset)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "prediction_scatter.png"))
plt.close()
print("Saved Scatter Plot: prediction_scatter.png")

# --- 4. CONTINUOUS CYCLE PLOT FOR A SINGLE ENGINE ---
eng1 = test_fd001[test_fd001['engine_id'] == test_fd001['engine_id'].unique()[0]].copy()
X_eng1, y_eng1 = create_sliding_windows(eng1, WINDOW_SIZE, FEATURE_COLS, TARGET)
# Must scale
X_eng1_scaled, _, _ = scale_windows(X_eng1, X_eng1, X_eng1) # hacky reuse just to scale
# actually scale_windows does standard logic, wait let's use global normalization since that's already applied via normalize_sensors
# wait scale_windows also normalizes each window. Yes, we just call scale_windows
X_eng1, _, _ = scale_windows(X_eng1, X_eng1, X_eng1)

X_eng1 = torch.tensor(X_eng1, dtype=torch.float32).to(device)
with torch.no_grad():
    preds_eng1 = model(X_eng1).squeeze().cpu().numpy()

plt.figure(figsize=(8,4))
cycles = eng1['cycle'].values[WINDOW_SIZE:] # starts after first window
plt.plot(cycles, y_eng1, label='Ground Truth RUL', lw=2, color='#00F5FF', linestyle='dashed') # Bright electric cyan
plt.plot(cycles, preds_eng1, label='Predicted RUL', lw=2, color='#39FF14') # Neon green
plt.xlabel('Operational Cycles')
plt.ylabel('Remaining Useful Life (RUL)')
plt.title(f'Run-to-Failure Trajectory (Engine ID {eng1["engine_id"].iloc[0]})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(EVAL_DIR, "single_engine_rul_trajectory.png"))
plt.close()
print("Saved Trajectory Plot: single_engine_rul_trajectory.png")

# --- 5. SHAP ---
print("Generating SHAP Explanations...")
model.cpu()
try:
    bg_data = torch.tensor(X_train[:50], dtype=torch.float32)
    explainer = shap.DeepExplainer(model, bg_data)
    sample_data = torch.tensor(X_test[:50], dtype=torch.float32)
    shap_values = explainer.shap_values(sample_data, check_additivity=False)
    
    # Check if shap_values is a list (binary classification behavior vs regression)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    shap_values_2d = np.mean(shap_values, axis=1) # average over temporal sequences
    sample_data_2d = np.mean(sample_data.numpy(), axis=1)
    
    plt.figure()
    shap.summary_plot(shap_values_2d, sample_data_2d, feature_names=FEATURE_COLS, show=False)
    plt.title("SHAP Feature Importance (Digital Twin LSTM)")
    plt.savefig(os.path.join(EVAL_DIR, "shap_summary_plot.png"), bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values_2d, sample_data_2d, plot_type="bar", feature_names=FEATURE_COLS, show=False)
    plt.savefig(os.path.join(EVAL_DIR, "shap_bar_feature_importance.png"), bbox_inches='tight')
    plt.close()
    print("Saved SHAP Plots: shap_summary_plot.png")
except Exception as e:
    print("Failed to generate SHAP plots due to an exception:", e)


# --- 6. NLP ---
print("Evaluating NLP Models on Synthetic Logs...")
try:
    logs_df = pd.read_csv(os.path.join(DATA_DIR, "maintenance_logs.csv"))
    texts = logs_df['log_text']
    labels = logs_df['fault_type']

    nlp_model = MaintenanceLogClassifier()
    nlp_model.train(texts, labels)
    preds = nlp_model.predict(texts)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='cool', xticklabels=nlp_model.pipeline.classes_, yticklabels=nlp_model.pipeline.classes_) # Vibrant cool colormap (cyan-magenta)
    plt.title('NLP Confusion Matrix (Maintenance Text)')
    plt.ylabel('Actual Event/Fault')
    plt.xlabel('Predicted Event/Fault')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "nlp_confusion_matrix.png"))
    plt.close()

    # Feature importance
    clf = nlp_model.pipeline.named_steps['clf']
    tfidf = nlp_model.pipeline.named_steps['tfidf']
    feature_names = tfidf.get_feature_names_out()

    avg_importance = np.mean(np.abs(clf.coef_), axis=0) if len(clf.coef_.shape) > 1 else np.abs(clf.coef_[0])
    top_idx = np.argsort(avg_importance)[-15:]
    top_features = feature_names[top_idx]
    top_scores = avg_importance[top_idx]

    plt.figure(figsize=(8,6))
    plt.barh(top_features, top_scores, color='#00F5FF') # Bright electric cyan
    plt.xlabel('Absolute Importance Weight')
    plt.title('Top 15 Predictive Terms in Maintenance Logs (TF-IDF)')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "nlp_top_terms.png"))
    plt.close()
    print("Saved NLP Plots: nlp_top_terms.png, nlp_confusion_matrix.png")
except Exception as e:
    print("Failed to generate NLP plots:", e)

print("All tasks complete!")
