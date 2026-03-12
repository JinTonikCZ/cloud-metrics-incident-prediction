# Predictive Alerting for Cloud Metrics

A professional Machine Learning pipeline designed to predict infrastructure incidents before they occur. This project demonstrates end-to-end engineering: from synthetic data generation to automated model evaluation, unit testing, and containerization.

---

## 🚀 Problem Formulation
This task is formulated as a **binary classification problem over time-series data**:
* **Input:** previous `W = 30` time steps.
* **Prediction horizon:** next `H = 10` time steps.
* **Output:**
  * `1` — an incident will occur within the next H steps.
  * `0` — no incident will occur within the next H steps.
* **Time Step:** One time step represents **1 minute**.
* **Logic:** The model uses the previous **30 minutes** of monitoring data to predict whether an incident will happen during the next **10 minutes**.

---

## 🛠 Project Structure
```text
predictive-alerting-cloud-metrics/
├─ src/                        # Core Python modules with Type Hinting
│  ├─ generate_data.py         # Synthetic telemetry generation
│  ├─ build_windows.py         # Feature engineering (sliding windows)
│  ├─ train_model.py           # Model training (Time-based split)
│  └─ evaluate.py              # Performance metrics & plotting
├─ notebooks/                  # Interactive analysis
│  └─ solution.ipynb           # Final project walkthrough
├─ tests/                      # Quality Assurance
│  └─ test_logic.py            # Unit tests for feature extraction
├─ data/                       # Raw and processed datasets
├─ models/                     # Serialized model pipelines (.pkl)
├─ figures/                    # PR Curves and Risk Timelines
├─ DESIGN_DOC.MD               # Detailed architecture and design decisions
├─ Dockerfile                  # Container definition
├─ docker-compose.yml          # Multi-container orchestration
├─ requirements.txt            # Project dependencies
└─ README.md                   # Project overview
```


---

## 🚀 Quick Start (Docker)
The easiest way to run the entire pipeline (data generation -> feature engineering -> training -> evaluation) is using Docker Compose. This ensures a consistent environment:

```bash
docker-compose up --build
```


---

## 📊 Key Engineering Features
* **Early Warning System:** Successfully predicts incidents 10 minutes ahead using a 30-minute sliding window of telemetry.
* **Engineering Excellence:**
    * Full **Type Hinting** for code clarity and maintainability.
    * Automated **CI via GitHub Actions** (runs tests on every push).
    * **Unit Testing** to ensure mathematical correctness of features.
    * **Containerization** via Docker for seamless deployment.
* **Robust ML Pipeline:** Utilizes a Random Forest model with `class_weight='balanced'` to handle highly imbalanced cloud datasets.

## 🧪 Running Tests Manually
If you are not using Docker, you can verify the logic using:
```bash
python -m unittest discover tests
```


---

## Architecture Flow
```mermaid
flowchart TD
    %% Define styles for different node types
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef data fill:#fff3e0,stroke:#f57c00,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef artifact fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;

    subgraph Step 1: Data Generation
        A[src/generate_data.py]:::process -->|Simulates metrics| B[(data/synthetic_metrics.csv)]:::data
    end

    subgraph Step 2: Feature Engineering
        B --> C[src/build_windows.py]:::process
        C -->|Sliding Window| D[(data/features.csv)]:::data
    end

    subgraph Step 3: Model Training
        D --> F[src/train_model.py]:::process
        F -->|Time-Based Split| G[models/random_forest_pipeline.pkl]:::artifact
    end

    subgraph Step 4: Evaluation
        G --> J[src/evaluate.py]:::process
        J -->|Visual Reports| L{{figures/risk_scores.png}}:::artifact
    end
```
