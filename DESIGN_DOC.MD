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
flowchart LR
    %% High-level architecture styles
    classDef phase fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000;
    classDef output fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;

    A[1. Data Generation]:::phase --> B[2. Feature Engineering]:::phase
    B --> C[3. Model Training]:::phase
    C --> D[4. Evaluation & Insights]:::phase

    D --> E((Actionable Alerts & Visuals)):::output

    %% Comments
    %% Phase 1 simulates the cloud environment
    %% Phase 2 compresses time into logical features
    %% Phase 3 applies Machine Learning
    %% Phase 4 validates business value
```

---

## 3. Data Strategy: Synthetic Metrics
**Why Synthetic Data?**
High-quality, labeled datasets of severe cloud infrastructure outages are rarely open-sourced due to strict NDAs and security policies. By synthesizing the data, we achieve:
1. **Total Control:** We can programmatically define the exact lifecycle of an incident (normal state -> degradation -> active incident -> recovery).
2. **Realistic Imbalance:** We explicitly engineered a realistic class imbalance (~95% normal, ~5% incidents) to properly test the model's alerting capabilities without artificial balancing.

**How it's generated:** We use statistical distributions (Normal for CPU/Memory, Lognormal for Latency, Exponential for Error Rates) and inject programmatic spikes and trends to simulate failures.

---

## 4. Project Details: The ML Pipeline
This section details the inner workings of the `src/` directory.

```mermaid
flowchart TD
    %% Detailed pipeline styles
    classDef script fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef core fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000;

    subgraph Data Processing
        S1[generate_data.py]:::script -->|15,000 mins| D1[(synthetic_metrics.csv)]:::data
        D1 --> S2[build_windows.py]:::script
        S2 -->|Lookback W=30 mins| D2[(features.csv)]:::data
        S2 -->|Horizon H=10 mins| D3[(target.csv)]:::data
    end

    subgraph Strict Training Logic
        D2 --> S3[train_model.py]:::script
        D3 --> S3
        S3 -->|CRITICAL: shuffle=False| Split{Time-Based Split}:::core
        Split -->|First 70%| Train[(Train Set)]:::data
        Split -->|Next 15%| Val[(Validation Set)]:::data
        Split -->|Last 15%| Test[(Test Set)]:::data
    end

    subgraph Modeling & Output
        Train --> M1[Random Forest]:::script
        Test --> S4[evaluate.py]:::script
        M1 --> S4
        S4 --> V1((PR Curve)):::data
        S4 --> V2((Risk Timeline)):::data
    end
```

### Key Design Decisions:
* **Feature Aggregation:** Instead of feeding 30 raw minutes (120 dimensions) into the model, we compute statistical summaries (mean, std, min, max, diff). This mitigates the curse of dimensionality and helps tree-based models find split points easier.
* **Strict Time-Based Split:** We explicitly set `shuffle=False` in `train_test_split`. Randomly shuffling time-series data causes *Data Leakage* (the model learns from future events to predict past events).
* **PR AUC over Accuracy:** Because the dataset is highly imbalanced, Accuracy is misleading (always predicting "0" yields 95% accuracy). We use Precision-Recall AUC to measure how well the model handles the minority class (incidents) without generating alert fatigue.

---

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

