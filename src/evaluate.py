import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve, auc, classification_report

def evaluate_performance() -> None:
    """
    Evaluates the model using Precision-Recall analysis and visualizes risk scores.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, 'models', 'random_forest_pipeline.pkl')
    X_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'y_test.csv')).values.ravel()

    model = joblib.load(model_path)
    
    # Get probability scores for the positive class (incident)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, (y_probs > 0.5).astype(int)))

    # 1. Precision-Recall Curve (Best for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Random Forest (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall (Catching Incidents)')
    plt.ylabel('Precision (Avoiding False Alarms)')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    
    fig_dir = os.path.join(BASE_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, 'pr_curve.png'))
    
    # 2. Risk Score Visualization (First 500 steps of test set)
    plt.figure(figsize=(15, 5))
    plt.plot(y_probs[:500], label='Predicted Risk (Probability)', color='orange')
    plt.fill_between(range(500), 0, y_test[:500], color='red', alpha=0.2, label='Actual Incident Zone')
    plt.axhline(y=0.5, color='black', linestyle='--', label='Alert Threshold')
    plt.title('Predictive Alerting Timeline: Risk Score vs Actual Ground Truth')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, 'risk_scores.png'))

    print(f"Evaluation complete. PR AUC: {pr_auc:.3f}")
    print(f"Visualizations saved to {fig_dir}/")

if __name__ == "__main__":
    evaluate_performance()