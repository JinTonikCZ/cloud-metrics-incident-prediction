import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
    """
    Splits the data safely over time, scales features, and trains ML models.
    """
    print("Loading extracted features...")
    X = pd.read_csv('../data/features.csv')
    y = pd.read_csv('../data/target.csv')['target']

    # ==========================================
    # DATA SPLITTING (CRITICAL STEP)
    # ==========================================
    # In Time-Series, we CANNOT use random cross-validation or random shuffling.
    # If we randomize, the model learns from "future" data to predict the "past" (Data Leakage).
    # Therefore, we strictly set shuffle=False. We train on the past, test on the future.
    # We aim for roughly: 70% Train, 15% Validation, 15% Test.
    
    # 1. Split off the last 15% as the final Test Set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    # 2. Split the remaining 85% into Train and Validation. 
    # (0.1765 of 85% equals ~15% of the total dataset)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, shuffle=False)

    print(f"Data split sequentially: Train ({len(X_train)}), Val ({len(X_val)}), Test ({len(X_test)})")

    # ==========================================
    # FEATURE SCALING
    # ==========================================
    # Linear models (like Logistic Regression) are sensitive to feature scales.
    # CPU is 0-100, latency can be thousands, error rate is 0.0-1.0. 
    # We standardize them to mean=0, std=1.
    scaler = StandardScaler()
    
    # IMPORTANT: We ONLY fit the scaler on the Training data to prevent data leakage.
    # Then we apply it to Validation and Test data.
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # ==========================================
    # MODEL TRAINING
    # ==========================================
    # Our data is highly imbalanced (~95% normal, ~5% incidents).
    # We use class_weight='balanced' so the model pays more attention to the minority class (incidents),
    # penalizing mistakes on incidents much heavier than false alarms on normal data.

    print("Training Baseline Model: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)

    print("Training Primary Model: Random Forest...")
    # Tree-based models handle non-linear relationships well (e.g., when CPU AND Latency spike together).
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, y_train) # Note: Random Forest doesn't actually require scaled features, but we use raw X_train.

    # ==========================================
    # SAVING ARTIFACTS
    # ==========================================
    # Save the trained models and the test datasets so the evaluation script can use them.
    os.makedirs('../models', exist_ok=True)
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(lr_model, '../models/logistic_regression.pkl')
    joblib.dump(rf_model, '../models/random_forest.pkl')

    # Save splits
    X_test.to_csv('../data/X_test.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)
    print("Models trained and artifacts saved to disk.")

if __name__ == "__main__":
    main()