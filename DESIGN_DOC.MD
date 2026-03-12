# System Design & Architecture Document
**Predictive Alerting for Cloud Metrics**

## Overview
This document serves as an extended architectural overview of the Predictive Alerting project. It details the decision-making process, tool selection, data strategy, and outlines how this proof-of-concept can be scaled into a production-ready alerting system.

---

## 1. Tech Stack & Tools
The project deliberately relies on a robust, industry-standard data science stack to ensure reproducibility and stability:
* **Python 3:** The core language for data processing and modeling.
* **Pandas & NumPy:** Chosen for efficient time-series manipulation, rolling window calculations, and tabular data management.
* **Scikit-learn:** Selected as the primary ML framework. For tabular time-series features, tree-based models (Random Forest) and linear baselines (Logistic Regression) are highly effective, interpretability-friendly, and faster to train than deep learning approaches.
* **Jupyter Notebook:** Acts as the Presentation Layer (`solution.ipynb`), combining code execution, narrative text, and visual artifacts into a cohesive story.
* **Mermaid.js:** Used for documentation as code (diagrams), ensuring that architectural schemas live alongside the codebase and are version-controlled without relying on external image files.

---

## 2. Overall High-Level Architecture
Below is the macroscopic view of the project's lifecycle, from raw data generation to final business insights.


```mermaid
graph TD
    %% Global Styles with forced black text for professional look
    classDef start fill:#f5f5f5,stroke:#333,stroke-width:2px,color:#000;
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;

    A((Project Start)):::start --> B[Step 1: Simulation]:::process
    B -->|Generates raw CSV telemetry| C[Step 2: Engineering]:::process
    C -->|Calculates sliding windows| D[Step 3: Training]:::process
    D -->|Saves serialized models| E[Step 4: Evaluation]:::process
    E -->|Exports visual artifacts| F((Final Insights)):::output

    %% Descriptive Comments (Author's technical notes)
    %% Step 1: Mimics cloud infrastructure behavior and incident patterns based on statistical distributions.
    %% Step 2: Transforms raw time-series into a supervised ML format (X, y) via windowing.
    %% Step 3: Fits a Random Forest while strictly avoiding chronological data leakage (shuffle=False).
    %% Step 4: Quantifies performance using PR AUC and Risk Score timelines for business validation.
```
*(Author's Original Diagram: High-level overview of the end-to-end data lifecycle)*

Stage Overview & Commentary:
This diagram illustrates the logical flow of information. The project begins with Simulation, creating a baseline of "ground truth". Engineering then distills this raw data into features. Training builds the predictive engine, and Evaluation proves its business value. This linear progression ensures that any change in the simulation is automatically reflected in the final evaluation metrics.

---

## 3. Project Details: The ML Pipeline Deep-Dive
*(Author's Original Diagram: Detailed interaction between modules and data layers)*

```mermaid
flowchart TD
    %% Colors and Styling for high readability
    classDef script fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef test fill:#fffde7,stroke:#fbc02d,stroke-width:2px,color:#000;

    subgraph Data_Processing [Stage A: Data Engineering]
        S1[generate_data.py]:::script -->|Raw CSV Data| DA1[(synthetic_metrics.csv)]:::data
        DA1 -->|Time-series stream| S2[build_windows.py]:::script
        S2 -->|30-min Rolling Window| S2_F[Feature Extraction]:::script
        S2_F -.->|Logic Verification| T1[tests/test_logic.py]:::test
    end

   subgraph Training_Logic [Stage B: Strict Validation]
        S2_F -->|Vectors & Targets| DA2[(features.csv)]:::data
        DA2 -->|Chronological Input| S3[train_model.py]:::script
        S3 -->|Split without Shuffling| TSet{70/30 Split}:::model
        TSet -->|Past Samples| TrSet[(Train Set)]:::data
        TSet -->|Future Samples| TeSet[(Test Set)]:::data
    end

   subgraph Modeling_Output [Stage C: Predictive Insights]
        TrSet -->|Balanced Learning| RF[Random Forest]:::model
        RF -->|Pickle Serialization| MO1[random_forest.pkl]:::model
        MO1 -->|Inference on Test Data| S4[evaluate.py]:::script
        TeSet --> S4
        S4 -->|Risk vs Incident Plot| M1{{risk_scores.png}}:::data
    end
```

### Technical Commentary on Pipeline Stages:

#### A. Data Processing & Logic
In this stage, raw telemetry is transformed into structured features. 
* **Key Innovation:** Instead of feeding raw noise, we use a 30-minute rolling window to calculate "momentum" (Trend) and "volatility" (Std). 
* **Reliability:** Verified by `test_logic.py` to prevent "garbage-in, garbage-out" scenarios.

#### B. Strict Training Logic (Anti-Leakage)
* **The Problem:** Random shuffling would allow the model to learn from "future" data.
* **The Solution:** A strict chronological split. The model is trained on the past and tested on the unseen future.

#### C. Modeling & Insight Generation
* **Model Selection:** Random Forest was chosen for its ability to handle non-linear spikes and outliers.
* **Business Value:** The output `risk_scores.png` proves the model crosses the Alerting Threshold (0.5) before the actual incident occurs, providing critical lead time.

**How it's generated:** We use statistical distributions (Normal for CPU/Memory, Lognormal for Latency, Exponential for Error Rates) and inject programmatic spikes and trends to simulate failures.

## 4. Engineering Decisions

*(The technical choices detailed below are visually summarized in the **Pipeline Execution Flow** diagram at the end of this section).*

### 4.1 Data Strategy: Why Synthetic Data?
High-quality, labeled datasets of severe cloud infrastructure outages are rarely open-sourced due to strict NDAs. By synthesizing the data, we achieve:
1. **Total Control:** We can programmatically define the exact lifecycle of an incident (normal state -> degradation -> active incident -> recovery).
2. **Realistic Imbalance:** We explicitly engineered a realistic class imbalance (~95% normal, ~5% incidents) to properly test the model's alerting capabilities without artificial balancing.

### 4.2 Feature Aggregation
Instead of feeding 30 raw minutes (120 dimensions) into the model, we compute statistical summaries (mean, std, min, max, diff). This mitigates the "curse of dimensionality" and helps tree-based models find split points more effectively by capturing momentum and volatility.

### 4.3 Strict Time-Based Split
We explicitly avoid random shuffling. In time-series tasks, shuffling causes **Data Leakage** (the model accidentally learns from future events to predict the past). Our 70/30 split is strictly chronological.

### 4.4 PR AUC over Accuracy
Because the dataset is highly imbalanced, Accuracy is a misleading metric (always predicting "0" would yield 95% accuracy). We prioritize **Precision-Recall AUC** to measure how well the model handles the minority class (incidents) while minimizing "alert fatigue."

### 4.5 Pipeline Execution Flow

```mermaid
flowchart TD
    %% Detailed pipeline styles
    classDef script fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef core fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000;
    %% New Note Style (Yellow Sticky Note)
    classDef note fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000,stroke-dasharray: 5 5;

    subgraph Data Processing
        N1[<b>Data Processing Note:</b><br>Starts with generate_data.py 15,000 mins telemetry.<br>build_windows.py processes this raw stream,<br>applying a 30-min lookback for rolling statistics,<br>separating inputs and labels.]:::note
        
        S1[generate_data.py]:::script -->|15,000 mins| D1[(synthetic_metrics.csv)]:::data
        D1 --> S2[build_windows.py]:::script
        S2 -->|Lookback W=30 mins| D2[(features.csv)]:::data
        S2 -->|Horizon H=10 mins| D3[(target.csv)]:::data
        
        N1 -.-> S1
    end

    subgraph Strict Training Logic
        N2[<b>Strict Training Logic Note:</b><br>train_model.py applies the critical shuffle=False<br>time-based split. First 70% becomes Train Set,<br>remaining 30% is strictly isolated as Test Set.]:::note

        D2 --> S3[train_model.py]:::script
        D3 --> S3
        S3 -->|CRITICAL: shuffle=False| Split{Time-Based Split}:::core
        Split -->|First 70%| Train[(Train Set)]:::data
        Split -->|Last 30%| Test[(Test Set)]:::data
        
        N2 -.-> S3
    end

    subgraph Modeling & Output
        N3[<b>Modeling & Output Note:</b><br>Random Forest learns patterns from Train Set.<br>evaluate.py tests model against unseen Test Set,<br>generating visual proofs PR Curve & Risk Timeline.]:::note

        Train --> M1[Random Forest]:::script
        Test --> S4[evaluate.py]:::script
        M1 --> S4
        S4 --> V1((PR Curve)):::data
        S4 --> V2((Risk Timeline)):::data
        
        N3 -.-> M1
    end
```
(Author's Original Diagram: Step-by-step execution flow of the src/ directory with embedded logic notes)

## 5. Production Readiness & Future Work
While this proof-of-concept successfully predicts synthetic incidents, deploying it to a real-world Kubernetes cluster requires addressing several system complexities.

```mermaid
flowchart TD
    %% Production styles
    classDef prod fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,color:#000;
    classDef issue fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#000;

    subgraph Future Production Architecture
        A1[Live Telemetry\nPrometheus / Datadog]:::prod --> B1{Stream Processing\nKafka / Flink}:::prod
        B1 --> C1[Dynamic Feature Extraction]:::prod
        
        %% Issue 1: Missing Data
        I1[Challenge: Dropped Packets]:::issue -.->|Needs Imputation| C1
        
        C1 --> D1[Sequence Models\nLSTM / Transformers]:::prod
        
        %% Issue 2: Fixed Windows
        I2[Challenge: Variable Incident Speeds]:::issue -.->|Move away from fixed W=30| D1
        
        D1 --> E1[Alert Router\nPagerDuty]:::prod
        
        %% Issue 3: Concept Drift
        I3[Challenge: Concept Drift\nNew deployments change metrics]:::issue -.->|Requires Online Learning| D1
        
        %% Issue 4: Distributed Tracing
        I4[Challenge: Multi-node Causality]:::issue -.->|Requires Graph Networks| A1
    end
```

**Production Upgrades to Consider:**
1. **Handling Concept Drift:** Cloud metrics are non-stationary. A new software deployment might increase the baseline CPU usage permanently. The system needs "Online Learning" (continuous retraining) and drift detection (e.g., Evidently AI) to avoid false positives over time.
2. **Dynamic Horizons:** A memory leak might take days to crash a system, while a traffic spike takes seconds. Fixed windows ($W=30$, $H=10$) should be replaced by multi-scale windows (5m, 1h, 24h) or RNNs/Transformers that dynamically learn context length.
3. **Multivariate Causality:** In a microservices architecture, a CPU spike on Node A might be caused by a database lock on Node B. Future iterations should incorporate Distributed Tracing (OpenTelemetry) and graph-based features.

---

## 6. References & Acknowledgment
The architecture, feature engineering methodology, and evaluation strategies in this project were synthesized based on industry best practices:
* **Google Site Reliability Engineering (SRE) Handbook:** For definitions of monitoring, alerting fatigue, and incident management.
* **Scikit-learn Documentation:** Specifically guidelines on handling Imbalanced Datasets (`class_weight='balanced'`) and Cross-Validation for Time Series.
* **Feature Engineering for Time Series:** Standard sliding window techniques used in classic anomaly detection.
* *Note: The structural synthesis, code generation assistance, and markdown formatting of this project were accelerated using Google's Gemini AI, specifically instructed to adhere to industry-standard data science and software engineering patterns.*

