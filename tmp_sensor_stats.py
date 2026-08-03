import sys, os
import pandas as pd
from DIGITAL_TWIN.data_loader import load_fd001
from DIGITAL_TWIN.preprocessing import add_rul
from DIGITAL_TWIN.config import SENSOR_COLS

df = load_fd001(r'c:\WORKFILES\dissertation\Digital Twin\DATA\NASA C-MAPSS 1 Turbofan Engine Degradation Dataset\train_FD001.txt')
df = add_rul(df)

corr_with_rul = df[SENSOR_COLS].apply(lambda x: x.corr(df['RUL']))
variance = df[SENSOR_COLS].var()

results = pd.DataFrame({'Sensor': SENSOR_COLS, 'Correlation with RUL': corr_with_rul, 'Variance': variance})
print(results.to_markdown())
