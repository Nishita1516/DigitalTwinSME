import base64
import zlib
import urllib.request
import os

diagrams = {
    "dfd_level_0": '''%%{init: {"theme": "default", "themeVariables": {"background": "#ffffff"}, "flowchart": {"curve": "linear"}}}%%
flowchart LR
    E1[ Industrial Equipment / Sensor Data Source ]
    E2[ Maintenance Engineers ]
    E3[ Dashboard End User / Factory Operator ]
    S(( Predictive Maintenance Digital Twin System ))
    E1 -->| Raw Sensor Readings & Operating Conditions | S
    E2 -->| Raw Maintenance Shift Reports | S
    S -->| Predicted RUL & Warning Alerts | E3
    S -->| Root Cause Analysis & SHAP Explanations | E3
    S -->| Categorized Fault Logs | E3''',
    
    "dfd_level_1": '''%%{init: {"theme": "default", "themeVariables": {"background": "#ffffff"}, "flowchart": {"curve": "linear"}}}%%
flowchart TD
    E1[ Industrial Equipment / Sensor Data Source ]
    E2[ Maintenance Engineers ]
    E3[ Dashboard End User ]
    D1[( System Data Cache )]
    P1(( 1.0 Data Generation & Simulation ))
    P2(( 2.0 Sensor-Based RUL Prediction Pipeline ))
    P3(( 3.0 NLP-Based Log Classification Pipeline ))
    P4(( 4.0 Interactive Dashboard Visualization ))
    E1 -->| Base C-MAPSS Sensor Data | P1
    E2 -->| Seed Textual Reports | P1
    P1 -->| Injected Noise, Drift, & Bias | D1
    D1 -->| Simulated Twin Sensor Data | P2
    D1 -->| Simulated Textual Logs | P3
    P2 -->| Forecasted RUL & SHAP Feature Importances | P4
    P3 -->| Fault Categories & Confidence Scores | P4
    P4 -->| Fleet Overview & Status Tables | E3
    P4 -->| Intelligent Logs & RCA Specifics | E3''',

    "dfd_level_2": '''%%{init: {"theme": "default", "themeVariables": {"background": "#ffffff"}, "flowchart": {"curve": "linear"}}}%%
flowchart TD
    D1[( Simulated Twin Sensor Data )]
    D2[( Simulated Maintenance Logs )]
    D3[( Predictions & Explanations DB )]
    subgraph 2.0 Sensor-Based ML Pipeline
        P2_1(( 2.1 Time-Series Sequence Chunking ))
        P2_2(( 2.2 LSTM-Based RUL Prediction ))
        P2_3(( 2.3 SHAP Explainability ))
        P2_4(( 2.4 Classical Model Baselines ))
        D1 -->| Raw 21-Channel Data | P2_1
        P2_1 -->| 30-Cycle Rolling Windows | P2_2
        P2_1 -->| Aggregated Statistical Windows | P2_4
        P2_2 -->| Numerical Prediction output | D3
        P2_2 -->| Model Weights & Outputs | P2_3
        P2_3 -->| Top 3 Problematic Sensors | D3
        P2_4 -->| Baseline Benchmarks | D3
    end
    subgraph 3.0 NLP-Based Log Pipeline
        P3_1(( 3.1 Text Preprocessing & TF-IDF Vectorization ))
        P3_2(( 3.2 Naive Bayes Categorization ))
        D2 -->| Raw Textual Logs | P3_1
        P3_1 -->| TF-IDF Feature Matrices | P3_2
        P3_2 -->| Fault Class: Bearing, Fan, Overheating, No Fault | D3
    end''',

    "cloud_architecture": '''%%{init: {"theme": "default", "themeVariables": {"background": "#ffffff"}, "flowchart": {"curve": "linear"}}}%%
flowchart TD
    A[" Industrial Equipment (Sensors / IoT Devices) "] --> B[" IoT Data Ingestion (AWS IoT / Azure IoT Hub) "]
    B --> C[" Cloud Storage (S3 / Blob Storage) "]
    C --> D[" Data Processing & Features (Lambda / Cloud Functions) "]
    D --> E[" ML Model (RUL - LSTM) (API Deployment) "]
    D --> F[" NLP Model (Log Class.) (API Deployment) "]
    E --> G[" Explainability (SHAP) "]
    F --> G
    G --> H[" Dashboard (Streamlit App) (Cloud Hosted Interface) "]
    H --> I[" End Users (SMEs) Managers / Engineers "]''',

    "project_flowchart": '''%%{init: {"theme": "default", "themeVariables": {"background": "#ffffff"}, "flowchart": {"curve": "step"}}}%%
flowchart TD
    Start([ Start System ]) 
    subgraph Data Acquisition & Simulation
        Ingest[ Ingest NASA C-MAPSS FD001 Dataset ]
        GenDigitalTwin[ digital_twin.py: Inject Gaussian Noise, Sensor Drift, Environmental Bias ]
        GenLogs[ logs_generator.py: Synthesize Textual Shift Reports ]
    end
    Start --> Ingest
    Ingest --> GenDigitalTwin
    Ingest --> GenLogs
    GenDigitalTwin --> SensorSplit{ Process Sensor Data }
    GenLogs --> NLPSplit{ Process Text Logs }
    subgraph Sensor ML Pipeline
        Windowing[ Slice data into 30-cycle Rolling Windows ]
        SplitClassical[ Calculate Aggregations: Mean, Variance, Min/Max ]
        LSTMModel[ Pass Sequences to PyTorch LSTM Network ]
        ClassicalModel[ Train/Run Random Forest & Gradient Boosting ]
        CalcRUL[ Output: Remaining Useful Life - RUL ]
        Compare[ Provide Baseline Comparison ]
        RunSHAP[ shap_explainer.py: Calculate SHAP Values ]
        ExtractRootCause[ Extract Top 3 Anomalous Sensors ]
    end
    SensorSplit --> Windowing
    Windowing --> LSTMModel
    Windowing --> SplitClassical
    SplitClassical --> ClassicalModel
    ClassicalModel --> Compare
    LSTMModel --> CalcRUL
    CalcRUL --> RunSHAP
    RunSHAP --> ExtractRootCause
    subgraph NLP Log Pipeline
        TFIDF[ nlp_baseline.py: Apply TF-IDF Vectorizer ]
        NaiveBayes[ Run Multinomial Naive Bayes Classifier ]
        AssignLabel[ Assign Label: Bearing Issue, Fan/Compressor Issue, Overheating, No Fault ]
    end
    NLPSplit --> TFIDF
    TFIDF --> NaiveBayes
    NaiveBayes --> AssignLabel
    subgraph Interactive Streamlit Dashboard
        RenderUI[ app.py & ui.py: Initialize Dark-Themed UI ]
        Metrics[ Render Fleet Overview Metric Cards ]
        Table[ Render Predictive Status Table sorted by RUL ]
        RCAView[ Display SHAP Root Cause Analysis ]
        LogsView[ Display Intelligent Logs with Confidence Scores ]
    end
    ExtractRootCause --> RenderUI
    AssignLabel --> RenderUI
    Compare --> RenderUI
    RenderUI --> Metrics
    RenderUI --> Table
    RenderUI --> RCAView
    RenderUI --> LogsView
    Metrics --> End([ End User Monitoring ])
    Table --> End
    RCAView --> End
    LogsView --> End'''
}

def generate_pngs():
    out_dir = r"c:\WORKFILES\dissertation\Digital Twin\ARCHITECTURE\diagrams_png"
    os.makedirs(out_dir, exist_ok=True)
    
    for name, code in diagrams.items():
        compressed = zlib.compress(code.encode('utf-8'), 9)
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
        url = f"https://kroki.io/mermaid/png/{encoded}"
        target_path = os.path.join(out_dir, f"{name}.png")
        print(f"Downloading {name} to {target_path}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Successfully generated {name}.png")
        except Exception as e:
            print(f"Error generating {name}.png: {e}")

if __name__ == "__main__":
    generate_pngs()
