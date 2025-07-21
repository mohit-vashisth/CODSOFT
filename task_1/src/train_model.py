import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load processed dataset
df = pd.read_csv("../dataset/titanic_processed.csv", index_col=0)

# Step 2: Split features and target
X = df.drop(columns=["survived"])
y = df["survived"]

# Step 3: One-hot encode if needed
X = pd.get_dummies(X, drop_first=True)

# Step 4: Train-test split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Logistic Regression Model
lr = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)
lr.fit(X_train_scaled, y_train)

# Step 7: Random Forest Model (Unscaled)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 8: Evaluation (Silent Mode: Save Results)
lr_metrics = {
    "accuracy": np.round(lr.score(X_test_scaled, y_test), 4),
    "roc_auc": roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]),
    "cv_score": np.round(cross_val_score(lr, X, y, cv=5, scoring="accuracy").mean(), 4),
}

rf_metrics = {
    "accuracy": np.round(rf.score(X_test, y_test), 4),
    "roc_auc": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
    "cv_score": np.round(cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean(), 4),
}

# Optional: Save metrics to file
metrics_df = pd.DataFrame(
    [lr_metrics, rf_metrics], index=["LogisticRegression", "RandomForest"]
)
metrics_df.to_csv("../models/metrics.csv")

# Step 9: Save the best model and preprocessing tools
joblib.dump(rf, "../models/rf_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(X.columns.tolist(), "../models/feature_columns.pkl")
