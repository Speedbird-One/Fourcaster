import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# 1. DATA PREP
# ---------------------------------------------------------
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Reserve 15% for the "Production/Streaming" demo later
X_dev, X_prod, y_dev, y_prod = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Save Production Data
pd.concat([X_prod, y_prod], axis=1).to_csv("production_simulation.csv", index=False)

# 2. 5-FOLD CROSS-VERIFICATION SETUP
# ---------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

print(f"Starting 5-Fold Cross-Verification on {len(X_dev)} records...")


# Decision Logic (Same as before)
def cost_aware_decision(row):
    score, amount = row["Risk_Score"], row["Amount"]
    FRICTION_COST = 20.0
    if score > 0.95:
        return "REJECT"
    if (score * amount) < FRICTION_COST:
        return "ACCEPT"
    if score > 0.30 and amount > 50:
        return "REVIEW"
    return "REJECT"


# 3. THE CROSS-VALIDATION LOOP
# ---------------------------------------------------------
for fold, (train_index, test_index) in enumerate(skf.split(X_dev, y_dev), 1):
    # Split
    X_train, X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    y_train, y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]

    # Scale (Required for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)

    # Train XGBoost
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=100, scale_pos_weight=ratio, eval_metric="aucpr", random_state=42
    )
    xgb.fit(X_train, y_train)

    # Ensemble Scoring (80/20)
    p_lr = lr.predict_proba(X_test_scaled)[:, 1]
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    scores = (0.8 * p_xgb) + (0.2 * p_lr)

    # Decisions & Metrics
    res = pd.DataFrame(
        {"Actual": y_test, "Risk_Score": scores, "Amount": X_test["Amount"].values}
    )
    res["Decision"] = res.apply(cost_aware_decision, axis=1)

    y_pred = res["Decision"].map({"ACCEPT": 0, "REVIEW": 1, "REJECT": 1})

    m = {
        "Fold": fold,
        "AUPRC": average_precision_score(y_test, scores),
        "F1": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }
    fold_metrics.append(m)
    print(
        f"Fold {fold} - AUPRC: {m['AUPRC']:.4f}, F1: {m['F1']:.4f}, Recall: {m['Recall']:.4f}"
    )

# 4. FINAL RESULTS & SAVING
# ---------------------------------------------------------
metrics_df = pd.DataFrame(fold_metrics)
print("\n=== FINAL AVERAGE PERFORMANCE ===")
print(metrics_df.drop("Fold", axis=1).mean())

# Save the Scaler and Models from the final fold
joblib.dump(scaler, "./models/scaler.joblib")
joblib.dump(lr, "./models/logistic_regression_model.joblib")
joblib.dump(xgb, "./models/xgboost_model.joblib")

print(
    "\nSuccess! Models saved: 'scaler.joblib', 'logistic_regression_model.joblib', 'xgboost_model.joblib'"
)
