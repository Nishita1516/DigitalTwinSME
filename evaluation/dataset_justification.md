# Dataset Documentation and Justification

## Digital Twin Platform for Predictive Maintenance

This document provides a detailed justification for all datasets used in this research project, including their sources, characteristics, purposes, and rationale for selection.

---

## 1. NASA C-MAPSS Dataset (Sensor Data)

### 1.1 Dataset Overview

**Full Name:** Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)  
**Subset Used:** FD001 (First of Four Datasets)  
**Source:** NASA Ames Prognostics Data Repository  
**Type:** Multivariate time-series sensor data  
**Domain:** Turbofan engine degradation simulation  

### 1.2 Dataset Characteristics

**File:** `train_FD001.txt`  
**Size:** 20,631 data points across 100 turbofan engines  
**Features:** 26 columns total
- Engine ID (unit number)
- Time (operational cycles)
- 3 Operational settings
- 21 Sensor measurements (temperature, pressure, vibration, speed, etc.)

**Example Sensors:**
- Sensor 2: Total temperature at fan inlet (°R)
- Sensor 3: Total temperature at LPC outlet (°R)
- Sensor 4: Total temperature at HPC outlet (°R)
- Sensor 7: Total pressure at HPC outlet (psia)
- Sensor 11: Physical fan speed (rpm)
- Sensor 12: Physical core speed (rpm)
- Sensor 13: Engine pressure ratio (P50/P2)
- Sensor 15: Bypass ratio
- Sensor 17: Bleed enthalpy
- Sensor 20: HPC coolant bleed (lbm/s)
- Sensor 21: HPT coolant bleed (lbm/s)

### 1.3 Purpose in Project

**Primary Uses:**
1. **RUL Prediction Training Data**
   - Train LSTM deep learning model for Remaining Useful Life prediction
   - Train classical ML baselines (Random Forest, Gradient Boosting)
   - Validate model performance on test split

2. **Feature Engineering**
   - Normalize sensor readings
   - Create sliding window sequences for temporal pattern learning
   - Calculate statistical features (mean, std, min, max)

3. **SHAP Explainability Analysis**
   - Identify which sensors contribute most to failure predictions
   - Provide human-readable explanations for predictions
   - Build trust with factory operators

4. **Dashboard Visualization**
   - Display real-time RUL predictions for multiple engines
   - Show top contributing sensors
   - Demonstrate Digital Twin concept with realistic data

### 1.4 Justification for Selection

**Why NASA C-MAPSS?**

1. **Academic Standard Benchmark**
   - Widely used in predictive maintenance research
   - Peer-reviewed and validated by NASA
   - Enables comparison with existing literature
   - Provides credibility to dissertation results

2. **Real-World Complexity**
   - Simulates actual turbofan engine behavior
   - Contains realistic degradation patterns
   - Includes 21 different sensor types (mirrors industrial IoT setups)
   - Models run-to-failure scenarios

3. **Suitable for SME Context**
   - Although from aerospace domain, the concepts (vibration, temperature, pressure monitoring) apply to:
     - Manufacturing presses
     - CNC machines
     - Pumps and compressors
     - Conveyor systems
   - Sensor-based monitoring is domain-agnostic

4. **Publicly Available & Free**
   - Aligns with "low-cost" thesis for SMEs
   - No proprietary licensing fees
   - Reproducible research (others can validate findings)

5. **Appropriate Scale**
   - 100 engines = manageable for prototype
   - Sufficient data for deep learning (20,000+ samples)
   - Not too large (can run on standard laptops, SME-friendly)

6. **Labeled Ground Truth**
   - Clear RUL labels for supervised learning
   - Known failure points for validation
   - Enables quantitative performance metrics (RMSE, MAE)

### 1.5 Limitations & Mitigation

**Limitations:**
- Aviation domain, not manufacturing
- Simulated data, not from actual sensors
- Single failure mode (high-pressure compressor degradation)

**Mitigation:**
- Clearly state in dissertation: "Dataset chosen as a proxy for industrial machine degradation"
- Emphasize transferability of ML techniques across domains
- Focus on methodology (applicable to any sensor data)
- Acknowledge in "Future Work": Need for real manufacturing data validation

---

## 2. Synthetic Maintenance Logs (NLP Data)

### 2.1 Dataset Overview

**File:** `DATA/maintenance_logs.csv`  
**Generation Method:** Programmatically created using `DIGITAL_TWIN/logs_generator.py`  
**Type:** Textual maintenance records  
**Domain:** Industrial equipment maintenance logs  

### 2.2 Dataset Characteristics

**Size:** ~5,000 log entries  
**Engines Covered:** 100 engines (matching C-MAPSS)  
**Time Range:** Simulated over engine lifetime (0 to max RUL)

**Columns:**
- `engine_id`: Engine identifier (1-100)
- `timestamp`: Date and time of log entry
- `log_text`: Natural language description of maintenance activity
- `fault_type`: Ground truth label for NLP classification
- `severity`: Priority level (LOW, MEDIUM, HIGH)

**Fault Types (4 categories):**
1. **No Fault** (~70% of logs)  
   - Examples: "Scheduled maintenance completed", "Routine inspection passed"
2. **Bearing Issue** (~10%)  
   - Examples: "Abnormal vibration in shaft", "Bearing temperature high"
3. **Overheating** (~10%)  
   - Examples: "Temperature spike during operation", "Cooling system malfunction"
4. **Fan/Compressor Issue** (~10%)  
   - Examples: "Fan imbalance detected", "Compressor blade damage"

### 2.3 Generation Methodology

**Template-Based Approach:**
```python
# Example templates
NO_FAULT_TEMPLATES = [
    "Scheduled maintenance: oil level checked",
    "Routine inspection: all systems operational",
    "Preventive maintenance completed successfully"
]

BEARING_TEMPLATES = [
    "Abnormal vibration detected in shaft bearing",
    "High bearing temperature ({temp}°C) recorded",
    "Unusual noise from main bearing, inspection required"
]
```

**Parameterization:**
- Templates filled with realistic values (temperatures, pressures)
- Timestamp distribution based on engine lifecycle
- Severity assigned based on fault type and RUL proximity

**Realism Enhancements:**
- Typos and variations added to simulate human writing
- Mix of technical and colloquial language
- Temporal patterns (more logs near failure)

### 2.4 Purpose in Project

**Primary Uses:**
1. **NLP Model Training**
   - Train TF-IDF + Logistic Regression classifier
   - Learn to categorize logs into fault types
   - Achieve text classification for maintenance prediction

2. **Dashboard Integration**
   - Display classified logs per engine
   - Show predicted fault types with confidence scores
   - Demonstrate NLP capability in Digital Twin

3. **Proof of Concept**
   - Show feasibility of combining sensor data (ML) + text data (NLP)
   - Illustrate multi-modal approach to predictive maintenance
   - Demonstrate value of unstructured data analysis

### 2.5 Justification for Synthetic Data

**Why Generate Synthetic Logs Instead of Using Real Logs?**

1. **Availability Problem**
   - **Real industrial logs are proprietary** (companies don't share for confidentiality)
   - Public maintenance log datasets **do not exist**
   - Pairing logs with C-MAPSS sensor data would be impossible

2. **Controlled Experimentation**
   - **Known ground truth** enables accurate NLP evaluation
   - Can create balanced dataset (avoid class imbalance)
   - Ensures alignment with C-MAPSS engine IDs

3. **Acceptable for Proof-of-Concept**
   - Dissertation goal: Demonstrate **methodology**, not production deployment
   - Synthetic data proves the **technical feasibility** of NLP integration
   - Shows system architecture works end-to-end

4. **Realistic Patterns**
   - Based on actual maintenance terminology
   - Mirrors real-world log structures (timestamp + text + severity)
   - Sufficient complexity for TF-IDF feature extraction

5. **Transparency & Reproducibility**
   - Generation script is open-source (`logs_generator.py`)
   - Other researchers can reproduce or modify
   - No proprietary data issues

### 2.6 Limitations & Mitigation

**Limitations:**
- **Not real-world data** (template-based, lacks true linguistic diversity)
- **Perfect labels** (100% accuracy may not generalize)
- **No spelling errors or ambiguity** (cleaner than real logs)

**Mitigation in Dissertation:**
1. **Explicit Acknowledgment**
   - Section 3.2: "Synthetic Log Generation"
   - Clearly state limitations in Chapter 6 (Discussion)

2. **Realistic Expectations Setting**
   - State: "Real-world accuracy expected to be 85-95%, not 100%"
   - Explain: "Synthetic data demonstrates feasibility; real deployment requires domain-specific retraining"

3. **Future Work Recommendations**
   - Suggest validation with actual factory logs
   - Propose collaboration with manufacturing partners
   - Recommend transfer learning from synthetic → real data

4. **Focus on Methodology**
   - Emphasize: "The TF-IDF + Logistic Regression pipeline is proven to work on text classification"
   - Highlight: "Same approach applies to real logs with minimal changes"

---

## 3. Data Processing Pipeline

### 3.1 Sensor Data Processing

**Preprocessing Steps:**
1. **RUL Calculation**  
   - Compute Remaining Useful Life for each engine based on max cycle count
   - Formula: `RUL = max_cycle - current_cycle`

2. **Normalization**  
   - Min-max scaling of sensor values to [0, 1] range
   - Prevents sensor magnitude bias in ML models

3. **Sliding Window Creation**  
   - Sequence length: 50 cycles
   - Captures temporal patterns for LSTM
   - Example: Cycles 1-50, 2-51, 3-52, etc.

4. **Train-Test Split**  
   - 80% training, 20% testing (standard practice)
   - Ensures model generalization evaluation

**Code Reference:** `DIGITAL_TWIN/preprocessing.py`, `DIGITAL_TWIN/twin_data_prep.py`

### 3.2 Log Data Processing

**Preprocessing Steps:**
1. **Text Cleaning**  
   - Lowercase conversion
   - Removal of special characters (retained in TF-IDF)

2. **TF-IDF Vectorization**  
   - Max features: 1000
   - Stopword removal: English stopwords
   - Captures word importance across documents

3. **Train-Test Split**  
   - 80% training, 20% testing (matching sensor data)
   - Random seed: 42 (reproducibility)

**Code Reference:** `MODELS/nlp_baseline.py`

---

## 4. Dataset Alignment with Dissertation Objectives

### 4.1 Objective 1: Design Digital Twin for SMEs

**Requirement:** Cost-effective, open-source data  
**Satisfaction:**
- ✅ NASA C-MAPSS: Free, public dataset
- ✅ Synthetic logs: Generated in-house, no cost
- ✅ No proprietary data dependencies

### 4.2 Objective 2: Implement ML for Predictive Maintenance

**Requirement:** Time-series sensor data with RUL labels  
**Satisfaction:**
- ✅ C-MAPSS has 21 sensors × 20,631 time points
- ✅ Clear RUL ground truth for supervised learning
- ✅ Enables RMSE/MAE evaluation

### 4.3 Objective 3: Apply NLP to Maintenance Logs

**Requirement:** Textual maintenance records with fault labels  
**Satisfaction:**
- ✅ Synthetic logs have 5,000 text entries
- ✅ 4 fault categories with ground truth
- ✅ Enables accuracy/F1 evaluation

### 4.4 Objective 4: Provide Explainable Insights

**Requirement:** Data suitable for SHAP analysis  
**Satisfaction:**
- ✅ C-MAPSS sensor data compatible with SHAP
- ✅ Feature importance can be calculated
- ✅ Human-readable sensor names available

### 4.5 Objective 5: Evaluate with Metrics

**Requirement:** Labeled data for quantitative evaluation  
**Satisfaction:**
- ✅ ML: RMSE, MAE on test set
- ✅ NLP: Accuracy, Precision, Recall, F1-score
- ✅ Both datasets have ground truth

---

## 5. Ethical & Legal Considerations

### 5.1 Data Licensing

**NASA C-MAPSS:**
- **License:** Public domain (U.S. government work)
- **Citation:** Properly cited in dissertation references
- **Usage:** Permitted for academic research

**Synthetic Logs:**
- **License:** Self-generated, full rights
- **Source code:** Available in project repository
- **Reproducibility:** Scripts provided for transparency

### 5.2 Privacy & Confidentiality

**No Personal Data:**
- NASA C-MAPSS: Simulated engine data (no human subjects)
- Synthetic logs: Fictional, no real company information
- No GDPR or privacy concerns

### 5.3 Research Ethics

**Transparency:**
- ✅ Data sources clearly documented
- ✅ Synthetic nature of logs explicitly stated
- ✅ Limitations acknowledged in dissertation

**Reproducibility:**
- ✅ All preprocessing code available
- ✅ Random seeds fixed for consistency
- ✅ Dataset download links provided

---

## 6. Comparison with Alternative Datasets

### 6.1 Why Not Other Public Datasets?

| Dataset | Reason for Exclusion |
|---------|---------------------|
| **PRONOSTIA Bearing Dataset** | Only single sensor (vibration), lacks multi-sensor richness |
| **PHM 2012 Dataset** | Similar to C-MAPSS but less widely adopted (harder to compare) |
| **FEMTO Bearing** | Limited to bearings only, not full machine health |
| **Kaggle Manufacturing Logs** | No paired sensor data, inconsistent labeling |

**Conclusion:** C-MAPSS offers the best balance of:
- Multi-sensor coverage
- Academic credibility
- Data quality
- Community adoption

### 6.2 Why Not Real Factory Data?

**Challenges:**
- Requires industry partnerships (time-consuming)
- Data access restrictions (NDAs, IP concerns)
- Inconsistent data quality (missing values, sensor drift)
- No ground truth RUL (machines rarely run to failure)

**For a Master's Dissertation:**
- Public benchmark data is **acceptable and preferred**
- Focus should be on methodology, not data collection
- Real data validation can be future work (PhD/industry collaboration)

---

## 7. Dataset Summary Table

| Aspect | NASA C-MAPSS (Sensor Data) | Synthetic Logs (NLP Data) |
|--------|----------------------------|---------------------------|
| **Source** | NASA Ames Research | Generated in-house |
| **Type** | Time-series multivariate | Textual records |
| **Size** | 20,631 × 26 features | 5,000 log entries |
| **Labels** | RUL (continuous) | Fault type (categorical) |
| **Domain** | Turbofan engines | General industrial maintenance |
| **Cost** | Free (public) | Free (self-generated) |
| **Use Case** | ML model training (LSTM, RF) | NLP model training (TF-IDF) |
| **Ground Truth** | Yes (labeled RUL) | Yes (labeled fault types) |
| **Evaluation Metric** | RMSE, MAE | Accuracy, F1-Score |
| **Limitation** | Simulated (not real sensors) | Template-based (not real logs) |
| **Mitigation** | Standard benchmark, methodology focus | Transparent generation, future validation |

---

## 8. Data Availability Statement (For Dissertation)

**Suggested Text for Chapter 3 (Methodology):**

> **Data Availability:**  
> This research utilizes two datasets: (1) the NASA C-MAPSS FD001 turbofan engine degradation dataset, publicly available from the NASA Ames Prognostics Data Repository [cite], and (2) a synthetically generated maintenance log dataset created specifically for this study. The synthetic logs were generated using a template-based approach to simulate realistic maintenance records, as paired public datasets of sensor data and maintenance logs do not exist. The generation script (`logs_generator.py`) is available in the project repository to ensure reproducibility. While synthetic data has limitations in linguistic diversity compared to real-world logs, it provides a controlled environment to demonstrate the technical feasibility of integrating NLP with sensor-based predictive maintenance. All preprocessing code, trained models, and evaluation scripts are provided for full reproducibility.

---

## 9. Dissertation Chapter Recommendations

### Where to Include This Information:

**Chapter 3: Methodology**
- Section 3.1: Dataset Description
  - Subsection 3.1.1: Sensor Data (NASA C-MAPSS)
  - Subsection 3.1.2: Maintenance Logs (Synthetic)
- Section 3.2: Data Preprocessing
  - Subsection 3.2.1: Sensor Data Processing
  - Subsection 3.2.2: Log Data Processing

**Chapter 6: Discussion**
- Section 6.3: Limitations
  - Acknowledge synthetic log limitations
  - Discuss generalization to real-world scenarios

**Chapter 7: Conclusion & Future Work**
- Section 7.2: Future Work
  - Recommend validation with real factory data
  - Suggest industry collaboration

---

## 10. Final Justification Summary

**The datasets chosen for this project are:**

1. **Scientifically Sound**
   - NASA C-MAPSS is a peer-reviewed benchmark
   - Synthetic logs follow established text generation practices

2. **Aligned with Objectives**
   - Multi-sensor data enables ML (LSTM, RF, GB)
   - Textual data enables NLP (TF-IDF classification)
   - Both support explainability (SHAP)

3. **Appropriate for SME Context**
   - Free and open-source (no licensing costs)
   - Manageable size (runs on standard hardware)
   - Demonstrates low-cost solution

4. **Academically Rigorous**
   - Transparent limitations acknowledged
   - Reproducible methodology
   - Suitable for Master's-level dissertation

5. **Ethically Compliant**
   - No privacy issues (simulated data)
   - Properly cited sources
   - Open methodology

**These datasets enable a complete proof-of-concept Digital Twin system that integrates ML and NLP for predictive maintenance, fulfilling all dissertation requirements.**
