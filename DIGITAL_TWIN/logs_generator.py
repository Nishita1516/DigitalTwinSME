import pandas as pd
import numpy as np
import random
import os

# Define fault types and corresponding templates
FAULT_TEMPLATES = {
    "No Fault": [
        "Routine check completed, no issues found.",
        "System operating within normal parameters.",
        "Scheduled maintenance: oil level checked.",
        "Visual inspection passed.",
        "Sensors calibrated, readings normal.",
        "Operator log: shift ended, machine running smoothly.",
        "Cooling system check: OK.",
    ],
    "Bearing Issue": [
        "High vibration detected in main bearing.",
        "Audible noise reported from bearing housing.",
        "Bearing temperature exceeding threshold.",
        "Lubrication low, potential bearing wear.",
        "Vibration analysis indicates inner race fault.",
        "Abnormal sound from shaft, suspect bearing damage."
    ],
    "Fan/Compressor Issue": [
        "Compressor stall detected.",
        "Fan blade vibration high.",
        "Airflow obstruction in intake.",
        "Compressor efficiency dropped below 85%.",
        "Fan balance check failed.",
        "Surge detected in low pressure compressor."
    ],
    "Overheating": [
        "Temperature sensor T2 reading critical high.",
        "Exhaust gas temperature warning.",
        "Coolant pressure drop, potential leak.",
        "Overheating warning in high pressure turbine.",
        "Thermal runaway risk, shutdown initiated for cooling.",
        "Heat exchanger efficiency low."
    ]
}

PRIORITY_MAP = {
    "No Fault": 0,
    "Bearing Issue": 2,
    "Fan/Compressor Issue": 2,
    "Overheating": 3
}

def generate_synthetic_logs(num_engines=100, logs_per_engine=50):
    """
    Generate a synthetic maintenance log dataset.
    
    Args:
        num_engines: Number of engines (should match C-MAPSS FD001)
        logs_per_engine: Average number of logs per engine
    """
    
    data = []
    
    for engine_id in range(1, num_engines + 1):
        # Simulate engine life (random cycles between 100 and 300)
        max_cycles = np.random.randint(150, 300)
        
        # Determine number of logs for this engine
        n_logs = int(np.random.normal(logs_per_engine, 10))
        n_logs = max(5, n_logs) # At least 5 logs
        
        # Generate random cycle times for logs
        log_cycles = np.sort(np.random.choice(range(1, max_cycles), n_logs, replace=False))
        
        for cycle in log_cycles:
            # Determine fault probability based on life stage
            # Higher probability of fault as cycle increases
            progress = cycle / max_cycles
            
            if progress < 0.6:
                fault_prob = 0.1
            elif progress < 0.8:
                fault_prob = 0.3
            else:
                fault_prob = 0.7
                
            if np.random.random() < fault_prob:
                # Pick a fault type (weighted)
                fault_type = np.random.choice(
                    ["Bearing Issue", "Fan/Compressor Issue", "Overheating"], 
                    p=[0.4, 0.3, 0.3]
                )
            else:
                fault_type = "No Fault"
            
            # Select template
            template = np.random.choice(FAULT_TEMPLATES[fault_type])
            
            # Add some randomness to text
            if np.random.random() < 0.3:
                template = template.lower()
            
            # Timestamp (dummy, just informative)
            # Assuming 1 cycle ~ 1 hour for simplicity, starting from a base date
            timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=int(cycle) + (engine_id * 100))
            
            data.append({
                "engine_id": engine_id,
                "cycle": cycle,
                "timestamp": timestamp,
                "log_text": template,
                "fault_type": fault_type,
                "severity": PRIORITY_MAP[fault_type]
            })
            
    df = pd.DataFrame(data)
    return df

def main():
    print("Generating synthetic maintenance logs...")
    df = generate_synthetic_logs()
    
    # Save to DATA folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(root_dir, "DATA", "maintenance_logs.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} logs.")
    print(f"Saved to: {output_path}")
    print("\nSample Data:")
    print(df.head())
    print("\nClass Distribution:")
    print(df["fault_type"].value_counts())

if __name__ == "__main__":
    main()
