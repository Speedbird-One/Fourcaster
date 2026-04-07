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
# CONFIG — all tunable knobs in one place
# ---------------------------------------------------------------------------
CONFIG = {
    "data_path": "creditcard.csv",
    "production_csv": "production_simulation.csv",
    "model_dir": "./models",
    "prod_split": 0.15,
    "n_folds": 5,
    "random_state": 42,
    # Ensemble weights (XGB, LR) — must sum to 1.0
    "ensemble_weights": (0.8, 0.2),
    # Cost-aware decision thresholds
    "hard_reject_threshold": 0.95,   # always reject above this
    "review_score_threshold": 0.30,  # flag for review above this …
    "review_amount_threshold": 50.0, # … only when amount exceeds this
    "friction_cost": 20.0,           # break-even cost of manual review
    # XGBoost
    "xgb_n_estimators": 100,
    # Logistic Regression
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
            f"Fold {self.fold} | AUPRC={self.auprc:.4f}  F1={self.f1:.4f}"
            f"  Recall={self.recall:.4f}  Threshold={self.best_threshold:.3f}"
            f"  Fraud caught: {self.n_fraud_detected}/{self.n_fraud_total}"
        )


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def make_model_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError(f"'Class' column not found in {path}")
    return df.drop("Class", axis=1), df["Class"]


def cost_aware_decision(score: float, amount: float, cfg: dict) -> str:
    """
    Three-way decision for a single transaction.

    Priority order (highest to lowest):
      1. Hard reject:  score > hard_reject_threshold
      2. Auto-accept:  expected loss (score × amount) < friction_cost
      3. Manual review: score > review_score_threshold AND amount > review_amount_threshold
      4. Accept (default)
    
    Bug fix: original code had an unreachable REJECT at the end — any
    transaction that passed the ACCEPT test (score*amount < FRICTION_COST)
    was silently accepted, but if it failed that test AND failed the REVIEW
    test it fell into REJECT.  The logic below makes every branch explicit.
    """
    if score > cfg["hard_reject_threshold"]:
        return "REJECT"
    if (score * amount) < cfg["friction_cost"]:
        return "ACCEPT"
    if score > cfg["review_score_threshold"] and amount > cfg["review_amount_threshold"]:
        return "REVIEW"
    return "ACCEPT"          # low-risk, low-amount → accept


def apply_decisions(
    scores: np.ndarray, amounts: np.ndarray, cfg: dict
) -> np.ndarray:
    return np.array(
        [cost_aware_decision(s, a, cfg) for s, a in zip(scores, amounts)]
    )


def decisions_to_binary(decisions: np.ndarray) -> np.ndarray:
    """REVIEW and REJECT both count as 'flagged' (positive class)."""
    return np.where(decisions == "ACCEPT", 0, 1)


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return the score threshold that maximises F1 on the given split."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    # Avoid division by zero
    f1s = np.where(
        (precisions + recalls) == 0,
        0.0,
        2 * precisions * recalls / (precisions + recalls),
    )
    best_idx = np.argmax(f1s[:-1])   # last element has no corresponding threshold
    return float(thresholds[best_idx])


def build_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_train_scaled: np.ndarray,
    cfg: dict,
) -> tuple[XGBClassifier, LogisticRegression]:
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
    xgb: XGBClassifier,
    lr: LogisticRegression,
    X: pd.DataFrame,
    X_scaled: np.ndarray,
    weights: tuple[float, float],
) -> np.ndarray:
    w_xgb, w_lr = weights
    p_xgb = xgb.predict_proba(X)[:, 1]
    p_lr = lr.predict_proba(X_scaled)[:, 1]
    return (w_xgb * p_xgb) + (w_lr * p_lr)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline(cfg: dict) -> None:
    make_model_dir(cfg["model_dir"])

    # 1. Load & split
    X, y = load_data(cfg["data_path"])
    X_dev, X_prod, y_dev, y_prod = train_test_split(
        X, y,
        test_size=cfg["prod_split"],
        random_state=cfg["random_state"],
        stratify=y,
    )
    pd.concat([X_prod, y_prod], axis=1).to_csv(cfg["production_csv"], index=False)
    print(f"Production holdout saved → {cfg['production_csv']}  ({len(X_prod)} rows)")

    # 2. Cross-validation
    skf = StratifiedKFold(
        n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["random_state"]
    )
    fold_results: list[FoldMetrics] = []
    print(f"\nStarting {cfg['n_folds']}-Fold CV on {len(X_dev)} dev records …\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), 1):
        X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

        # Scale (scaler fitted on train split only — no leakage)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        xgb_model, lr_model = build_ensemble(X_train, y_train, X_train_sc, cfg)

        scores = ensemble_scores(
            xgb_model, lr_model, X_val, X_val_sc, cfg["ensemble_weights"]
        )

        # Tune threshold on this fold's validation split
        best_thr = find_best_threshold(y_val.values, scores)

        decisions = apply_decisions(scores, X_val["Amount"].values, cfg)
        y_pred = decisions_to_binary(decisions)

        auprc = average_precision_score(y_val, scores)
        metrics = FoldMetrics(
            fold=fold,
            auprc=auprc,
            f1=f1_score(y_val, y_pred, zero_division=0),
            precision=precision_score(y_val, y_pred, zero_division=0),
            recall=recall_score(y_val, y_pred, zero_division=0),
            best_threshold=best_thr,
            n_fraud_detected=int((y_pred == 1) & (y_val.values == 1)).sum(),
            n_fraud_total=int(y_val.sum()),
        )
        fold_results.append(metrics)
        print(metrics.display())

    # 3. Summary
    metrics_df = pd.DataFrame([asdict(m) for m in fold_results])
    print("\n=== AVERAGE ACROSS FOLDS ===")
    print(
        metrics_df[["auprc", "f1", "precision", "recall", "best_threshold"]]
        .mean()
        .to_string()
    )

    # 4. Final retrain on full dev set → production artefacts
    print("\nRetraining on full dev set for production artefacts …")
    final_scaler = StandardScaler()
    X_dev_sc = final_scaler.fit_transform(X_dev)
    final_xgb, final_lr = build_ensemble(X_dev, y_dev, X_dev_sc, cfg)

    joblib.dump(final_scaler,  f"{cfg['model_dir']}/scaler.joblib")
    joblib.dump(final_lr,      f"{cfg['model_dir']}/logistic_regression_model.joblib")
    joblib.dump(final_xgb,     f"{cfg['model_dir']}/xgboost_model.joblib")
    print(
        f"Artefacts saved to '{cfg['model_dir']}': "
        "scaler, logistic_regression_model, xgboost_model"
    )


if __name__ == "__main__":
    run_pipeline(CONFIG)
