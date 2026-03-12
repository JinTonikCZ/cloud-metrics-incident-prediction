# Predictive Alerting for Cloud Metrics

A professional Machine Learning pipeline designed to predict infrastructure incidents before they occur. This project demonstrates end-to-end engineering: from synthetic data generation to automated model evaluation, unit testing, and containerization.

---

## 🚀 Problem Formulation
This task is formulated as a **binary classification problem over time-series data**:

* **Goal:** Predict whether an incident will occur within the next **H** steps based on the previous **W** steps of system behavior.
* **Input (Features):** Previous `W = 30` time steps (look-back window).
* **Prediction Horizon (Target):** Next `H = 10` time steps.
* **Output:** * `1` — an incident **will occur** within the next H steps.
  * `0` — no incident will occur.
* **Time Scale:** One time step represents **1 minute**.

---

## 🏗 Architecture Flow
This diagram illustrates the high-level project lifecycle and technical branching.

```mermaid
flowchart TD
    %% Global Styles with forced black text for readability
    classDef start fill:#f5f5f5,stroke:#333,stroke-width:2px,color:#000;
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef logic fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;

    Start[Project Start]:::start --> Docker[Setup Docker Environment]:::process
    Docker -->|Ensures Reproducibility| Gen[Step 1 Data Generation]:::process
    
    Gen -->|Simulates telemetry| RawData[(synthetic_metrics.csv)]:::output
    
    RawData --> Engineering[Step 2 Feature Engineering]:::process
    Engineering -->|Extracts Statistics| Windows[Apply Sliding Window]:::process
    
    Windows --> Branch{Data Validation}:::logic
    Branch -->|Passed| Split[Step 3 Time-Based Split]:::process
    Branch -->|Failed| Tests[Run Unit Tests]:::process
    
    Tests -->|Tests Passed| Split
    
    Split -->|No Shuffling| Training[Step 4 Model Training]:::process
    Training -->|Saves Pipeline| ModelFile[random_forest_pipeline.pkl]:::output
    
    ModelFile --> Eval[Step 5 Evaluation]:::process
    Eval -->|Generates Metrics| Plots[Visual Reports]:::output
    
    Plots --> Notebook[Final Analysis solution.ipynb]:::process
    Notebook --> End[System Ready]:::start
```

---

## 🔬 The Microscopic View
Detailed module interaction and data transformation pipeline.

```mermaid
flowchart TD
    %% Colors and Styles with forced black text
    classDef docker fill:#fce4ec,stroke:#d81b60,stroke-width:2px,color:#000;
    classDef script fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef model fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef test fill:#fffde7,stroke:#fbc02d,stroke-width:2px,color:#000;

    subgraph Container_Layer
        D1[Dockerfile]:::docker -->|Builds Env| Img[Python 3.9 Image]:::docker
        D2[docker-compose.yml]:::docker -->|Mounts Volumes| Img
    end

    subgraph Data_Engineering_Layer
        S1[generate_data.py]:::script -->|Saves Raw Data| DA1[(synthetic_metrics.csv)]:::data
        
        DA1 --> S2[build_windows.py]:::script
        S2 -->|Extract Features| S2_F[Calculates Statistics]:::script
        
        S2_F --> DA2[(features.csv)]:::data
        S2_F --> DA3[(target.csv)]:::data
        
        S2_F -.->|Logic Check| T1[tests/test_logic.py]:::test
    end

    subgraph ML_Training_Layer
        DA2 --> S3[train_model.py]:::script
        DA3 --> S3
        
        S3 -->|Chronological Split| TSet[(Train and Test Sets)]:::data
        TSet -->|Fits Model| RF[Random Forest]:::model
        
        RF -->|Serializes| MO1[random_forest_pipeline.pkl]:::model
    end

    subgraph Monitoring_Layer
        MO1 --> S4[evaluate.py]:::script
        S4 -->|Metric| M1[PR AUC Score]:::model
        S4 -->|Saves Plots| M2[(figures/risk_scores.png)]:::data
    end
```

## 🚀 Quick Start (Docker)
Run the entire pipeline with one command:

```bash
docker-compose up --build
```

## 🧪 Quality Assurance
* **Unit Testing:** `python -m unittest discover tests`
* **CI/CD:** Automated testing via **GitHub Actions** on every push.
* **Type Hinting:** Fully implemented for robust maintenance.
* **Reproducibility:** Guaranteed by Docker containerization.
