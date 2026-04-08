"""
Fraud Detection Pipeline — Ensemble (XGBoost + Logistic Regression)

Workflow:
  1. Load & split data (dev 85% / production holdout 15%)
  2. 5-fold stratified cross-validation with per-fold metrics
  3. Threshold optimisation on each validation fold (maximise F1)
  4. Final retrain on full dev set → save artefacts
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "data_path": "creditcard.csv",
    "production_csv": "production_simulation.csv",
    "model_dir": "./models",
    "prod_split": 0.15,
    "n_folds": 5,
    "random_state": 42,

    "ensemble_weights": (0.8, 0.2),

    "hard_reject_threshold": 0.95,
    "review_score_threshold": 0.30,
    "review_amount_threshold": 50.0,
    "friction_cost": 20.0,

    "xgb_n_estimators": 100,
    "lr_max_iter": 1000,
}


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------
@dataclass
class FoldMetrics:
    fold: int
    auprc: float
    f1: float
    precision: float
    recall: float
    best_threshold: float
    n_fraud_detected: int
    n_fraud_total: int

    def display(self) -> str:
        return (
            f"Fold {self.fold} | AUPRC={self.auprc:.4f}  "
            f"F1={self.f1:.4f}  Recall={self.recall:.4f}  "
            f"Threshold={self.best_threshold:.3f}  "
            f"Fraud caught: {self.n_fraud_detected}/{self.n_fraud_total}"
        )


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def make_model_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(path: str):
    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError(f"'Class' column not found in {path}")

    return df.drop("Class", axis=1), df["Class"]


def cost_aware_decision(score, amount, cfg):

    if score > cfg["hard_reject_threshold"]:
        return "REJECT"

    if (score * amount) < cfg["friction_cost"]:
        return "ACCEPT"

    if (
        score > cfg["review_score_threshold"]
        and amount > cfg["review_amount_threshold"]
    ):
        return "REVIEW"

    return "ACCEPT"


def apply_decisions(scores, amounts, cfg):

    return np.array(
        [cost_aware_decision(s, a, cfg) for s, a in zip(scores, amounts)]
    )


def decisions_to_binary(decisions):

    return np.where(decisions == "ACCEPT", 0, 1)


def find_best_threshold(y_true, scores):

    precisions, recalls, thresholds = precision_recall_curve(
        y_true,
        scores,
    )

    f1s = np.where(
        (precisions + recalls) == 0,
        0.0,
        2 * precisions * recalls / (precisions + recalls),
    )

    best_idx = np.argmax(f1s[:-1])

    return float(thresholds[best_idx])


def build_ensemble(
    X_train,
    y_train,
    X_train_scaled,
    cfg,
):

    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=cfg["xgb_n_estimators"],
        scale_pos_weight=ratio,
        eval_metric="aucpr",
        random_state=cfg["random_state"],
        verbosity=0,
    )

    xgb.fit(X_train, y_train)

    lr = LogisticRegression(
        max_iter=cfg["lr_max_iter"],
        class_weight="balanced",
        random_state=cfg["random_state"],
    )

    lr.fit(X_train_scaled, y_train)

    return xgb, lr


def ensemble_scores(
    xgb,
    lr,
    X,
    X_scaled,
    weights,
):

    w_xgb, w_lr = weights

    p_xgb = xgb.predict_proba(X)[:, 1]
    p_lr = lr.predict_proba(X_scaled)[:, 1]

    return (w_xgb * p_xgb) + (w_lr * p_lr)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline(cfg):

    make_model_dir(cfg["model_dir"])

    X, y = load_data(cfg["data_path"])

    X_dev, X_prod, y_dev, y_prod = train_test_split(
        X,
        y,
        test_size=cfg["prod_split"],
        random_state=cfg["random_state"],
        stratify=y,
    )

    pd.concat([X_prod, y_prod], axis=1).to_csv(
        cfg["production_csv"],
        index=False,
    )

    print(
        f"Production holdout saved → {cfg['production_csv']} "
        f"({len(X_prod)} rows)"
    )

    skf = StratifiedKFold(
        n_splits=cfg["n_folds"],
        shuffle=True,
        random_state=cfg["random_state"],
    )

    fold_results = []

    print(
        f"\nStarting {cfg['n_folds']}-Fold CV "
        f"on {len(X_dev)} dev records …\n"
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_dev, y_dev),
        1,
    ):

        X_train = X_dev.iloc[train_idx]
        X_val = X_dev.iloc[val_idx]

        y_train = y_dev.iloc[train_idx]
        y_val = y_dev.iloc[val_idx]

        scaler = StandardScaler()

        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        xgb_model, lr_model = build_ensemble(
            X_train,
            y_train,
            X_train_sc,
            cfg,
        )

        scores = ensemble_scores(
            xgb_model,
            lr_model,
            X_val,
            X_val_sc,
            cfg["ensemble_weights"],
        )

        # FIXED (.values removed)
        best_thr = find_best_threshold(
            y_val,
            scores,
        )

        decisions = apply_decisions(
            scores,
            X_val["Amount"].values,
            cfg,
        )

        y_pred = decisions_to_binary(decisions)

        auprc = average_precision_score(
            y_val,
            scores,
        )

        metrics = FoldMetrics(
            fold=fold,
            auprc=auprc,
            f1=f1_score(
                y_val,
                y_pred,
                zero_division=0,
            ),
            precision=precision_score(
                y_val,
                y_pred,
                zero_division=0,
            ),
            recall=recall_score(
                y_val,
                y_pred,
                zero_division=0,
            ),
            best_threshold=best_thr,

            # MAIN BUG FIX
            n_fraud_detected=int(
                ((y_pred == 1) & (y_val == 1)).sum()
            ),

            n_fraud_total=int(
                y_val.sum()
            ),
        )

        fold_results.append(metrics)

        print(metrics.display())

    metrics_df = pd.DataFrame(
        [asdict(m) for m in fold_results]
    )

    print("\n=== AVERAGE ACROSS FOLDS ===")

    print(
        metrics_df[
            ["auprc", "f1", "precision", "recall", "best_threshold"]
        ]
        .mean()
        .to_string()
    )

    print(
        "\nRetraining on full dev set for production artefacts …"
    )

    final_scaler = StandardScaler()

    X_dev_sc = final_scaler.fit_transform(X_dev)

    final_xgb, final_lr = build_ensemble(
        X_dev,
        y_dev,
        X_dev_sc,
        cfg,
    )

    joblib.dump(
        final_scaler,
        f"{cfg['model_dir']}/scaler.joblib",
    )

    joblib.dump(
        final_lr,
        f"{cfg['model_dir']}/logistic_regression_model.joblib",
    )

    joblib.dump(
        final_xgb,
        f"{cfg['model_dir']}/xgboost_model.joblib",
    )

    print(
        f"Artefacts saved to '{cfg['model_dir']}': "
        "scaler, logistic_regression_model, xgboost_model"
    )


if __name__ == "__main__":
    run_pipeline(CONFIG)