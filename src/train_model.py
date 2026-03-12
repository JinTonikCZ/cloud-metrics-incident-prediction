import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_alerting_models(features_path: str, target_path: str) -> None:
    """
    Trains and persists predictive models using a chronological split.
    """
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).values.ravel()

    # CRITICAL: We use shuffle=False to respect the temporal nature of the data.
    # Training on the past, validating on the future.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)}")

    # Define a Pipeline: Scaling + Model
    # Random Forest is our primary model due to its handling of non-linear trends
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    rf_pipeline.fit(X_train, y_train)
    
    # Save the model and data for evaluation
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, 'models')
    data_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(rf_pipeline, os.path.join(model_dir, 'random_forest_pipeline.pkl'))
    
    # Save test sets for the evaluation script
    X_test.to_csv(os.path.join(data_dir, 'X_test.csv'), index=False)
    pd.Series(y_test).to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)
    
    print("Models and test datasets saved successfully.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_alerting_models(
        os.path.join(BASE_DIR, 'data', 'features.csv'),
        os.path.join(BASE_DIR, 'data', 'target.csv')
    )