import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from DIGITAL_TWIN.data_loader import load_fd001

# 1. Load data
data_path = r'c:\WORKFILES\dissertation\Digital Twin\DATA\Sensor Data\NASA C-MAPSS 1 Turbofan Engine Degradation Dataset\train_FD001.txt'
df = load_fd001(data_path)

# 2. Pick an engine (Engine 1)
df_eng = df[df['engine_id'] == 1].copy()
cycles = df_eng['cycle']

# Calculate simple rolling mean to smooth the plots (window = 5)
for col in df_eng.columns:
    if col.startswith('s'):
        df_eng[col] = df_eng[col].rolling(window=5, min_periods=1).mean()

# 3. Define categories
categories = {
    'Temperatures': ['s2', 's3', 's4'],
    'Pressures': ['s7', 's11'],
    'Speeds': ['s8', 's9', 's13', 's14'],
    'Flows_and_Ratios': ['s12', 's15', 's17', 's20', 's21']
}

out_dir = r'c:\WORKFILES\dissertation\Digital Twin\EVALUATION'

# Vibrant/bright colors for statistical lines
bright_colors = ['#FF007F', '#00F5FF', '#39FF14', '#FFCC00', '#CC00FF', '#FF5722']

for cat, sensors in categories.items():
    plt.figure(figsize=(8, 4))
    for idx, s in enumerate(sensors):
        color = bright_colors[idx % len(bright_colors)]
        plt.plot(cycles, df_eng[s], label=f'Sensor {s}', linewidth=2, color=color)
    
    plt.title(f'{cat} Sensors Degradation Trend (Engine 1)')
    plt.xlabel('Operational Cycles')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'category_{cat}.png'))
    plt.close()

print("Category plots generated successfully!")
