import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve

def main():
    """
    Evaluates the trained model on the unseen Test set and generates visualization artifacts.
    """
    print("Loading test data and the trained Random Forest model...")
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')['target']
    rf_model = joblib.load('../models/random_forest.pkl')

    os.makedirs('../figures', exist_ok=True)

    # ==========================================
    # PREDICTION
    # ==========================================
    # y_pred: Binary outputs (0 or 1) based on default 0.5 threshold
    # y_prob: Continuous risk scores (0.0 to 1.0) indicating probability of an incident
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    # ==========================================
    # METRICS CALCULATION
    # ==========================================
    # Precision: Out of all alerts fired, how many were ACTUAL incidents? (Low precision = Alert Fatigue)
    # Recall: Out of all real incidents, how many did we catch beforehand? (Low recall = Missed Outages)
    # PR AUC (Precision-Recall Area Under Curve): The most important metric for imbalanced datasets.
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\n--- Test Set Performance ---")
    print(f"Precision: {precision:.3f} (When it alerts, it is correct {precision*100:.1f}% of the time)")
    print(f"Recall:    {recall:.3f} (It successfully caught {recall*100:.1f}% of upcoming incidents)")
    print(f"F1-score:  {f1:.3f}")
    print(f"ROC AUC:   {roc_auc:.3f}")
    print(f"PR AUC:    {pr_auc:.3f} <- Best metric for imbalanced alerting")
    print("----------------------------\n")

    # ==========================================
    # VISUALIZATION 1: Precision-Recall Curve
    # ==========================================
    # Shows the trade-off between missing incidents and spamming engineers with false alerts.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'Random Forest (PR AUC = {pr_auc:.3f})', color='blue')
    plt.xlabel('Recall (Catching Incidents)')
    plt.ylabel('Precision (Avoiding False Alarms)')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../figures/pr_curve.png', dpi=300)
    plt.close()

    # ==========================================
    # VISUALIZATION 2: Predictive Risk Score Timeline
    # ==========================================
    # Plots the model's "Anxiety Level" (Probability) over time against actual incident zones.
    plt.figure(figsize=(14, 5))
    subset_size = min(400, len(y_test)) # Plot 400 minutes for readability
    
    plt.plot(y_prob[:subset_size], label='Predicted Risk Score (0 to 1)', color='darkorange', lw=2)
    plt.plot(y_test.values[:subset_size], label='Actual Danger Zone (Ground Truth)', color='red', linestyle='--', alpha=0.7)
    plt.axhline(0.5, color='gray', linestyle=':', label='Alert Firing Threshold (0.5)')
    
    plt.xlabel('Time Step (Minutes)')
    plt.ylabel('Probability')
    plt.title('Alerting Timeline: Predicted Risk vs Actual Incidents')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('../figures/risk_scores.png', dpi=300)
    plt.close()

    # ==========================================
    # VISUALIZATION 3: Raw Metric Degradation
    # ==========================================
    # Shows how a specific raw metric (e.g., CPU) degrades leading up to a failure.
    df_raw = pd.read_csv('../data/synthetic_metrics.csv')
    incident_indices = df_raw[df_raw['is_incident'] == 1].index
    
    if len(incident_indices) > 0:
        # Find the first incident in the whole dataset and plot 100 mins before and after
        first_incident = incident_indices[0]
        df_slice = df_raw.iloc[max(0, first_incident - 100) : first_incident + 100].reset_index(drop=True)

        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(df_slice['cpu_usage'], label='CPU Usage (%)', color='steelblue', lw=2)
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_xlabel('Time Step (Minutes)')
        
        # Shade the actual incident period in red
        incidents = df_slice['is_incident'].values
        for i in range(len(incidents)):
            if incidents[i] == 1:
                ax1.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.3, lw=0)

        plt.title('Example of Raw Metric Degradation Ending in an Incident')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85))
        plt.tight_layout()
        plt.savefig('../figures/metrics_incidents.png', dpi=300)
        plt.close()

    print("All plots generated successfully in the '../figures' folder.")

if __name__ == "__main__":
    main()