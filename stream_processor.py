import pandas as pd
import numpy as np
import joblib
import time
import argparse
import os
import warnings
from datetime import datetime

# Silence the 'UserWarning' regarding feature names in StandardScaler
warnings.filterwarnings("ignore", category=UserWarning)

# 1. CLI CONFIGURATION
# ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Fraud Detection Production Stream Simulator"
)
parser.add_argument(
    "--capacity", type=int, default=5, help="Max manual reviews allowed per window"
)
args = parser.parse_args()

K_LIMIT = args.capacity
WINDOW_SIZE = 120
FRICTION_COST = 20.0

# 2. LOAD MODELS AND ASSETS
# ---------------------------------------------------------
print("--- Initializing Fraud Decision Engine ---")
try:
    scaler = joblib.load("./models/scaler.joblib")
    lr_model = joblib.load("./models/logistic_regression_model.joblib")
    xgb_model = joblib.load("./models/xgboost_model.joblib")
    prod_data = pd.read_csv("creditcard.csv")
    print("Assets loaded successfully.\n")
except FileNotFoundError as e:
    print(f"Error: Missing files. Ensure you ran the training script first.")
    exit()

# 3. LOGGING SETUP
# ---------------------------------------------------------
LOG_FILE_CSV = "active_learning_log.csv"
REVIEW_LOG_TXT = "manual_review_queue.txt"

# Initialize CSV for Active Learning
if not os.path.isfile(LOG_FILE_CSV):
    headers = list(prod_data.columns) + ["Risk_Score", "Decision", "Timestamp"]
    pd.DataFrame(columns=headers).to_csv(LOG_FILE_CSV, index=False)

# Initialize/Clear the Text File for the current session
with open(REVIEW_LOG_TXT, "w") as f:
    f.write(f"--- MANUAL REVIEW QUEUE | Session Started: {datetime.now()} ---\n")
    f.write(
        f"{'Timestamp':<10} | {'Trans ID':<10} | {'Amount':<10} | {'Risk Score':<10}\n"
    )
    f.write("-" * 55 + "\n")

# 4. STREAMING EXECUTION
# ---------------------------------------------------------
print(f"Starting Stream Simulation...")
print(f"Manual Review Queue logging to: {REVIEW_LOG_TXT}")
print("-" * 100)

review_counter = 0
transaction_count = 0

try:
    for index, row in prod_data.iterrows():
        # Reset the window counter periodically
        if transaction_count % WINDOW_SIZE == 0 and transaction_count > 0:
            review_counter = 0
            print(f"\n[SYSTEM] Window reset. Capacity: 0/{K_LIMIT}\n")

        features_df = pd.DataFrame([row.drop("Class")])
        amount = row["Amount"]

        # Scoring
        features_scaled = scaler.transform(features_df)
        p_lr = lr_model.predict_proba(features_scaled)[:, 1][0]
        p_xgb = xgb_model.predict_proba(features_df)[:, 1][0]
        risk_score = (0.8 * p_xgb) + (0.2 * p_lr)
        expected_fraud_loss = risk_score * amount

        # 5. DYNAMIC THROTTLING & DECISION ENGINE
        decision = "ACCEPT"
        queue_fill_level = review_counter / K_LIMIT if K_LIMIT > 0 else 1.0

        # Rule 1: Absolute Reject
        if risk_score > 0.95:
            decision = "REJECT"

        # Rule 2: Queue is Full
        elif review_counter >= K_LIMIT:
            decision = "REJECT" if risk_score > 0.85 else "ACCEPT"

        # Rule 3: Congestion Mode (Throttling)
        elif queue_fill_level >= 0.70:
            if risk_score > 0.70 or expected_fraud_loss > 100:
                decision = "REVIEW"
                review_counter += 1
            else:
                decision = "ACCEPT"

        # Rule 4: Normal State
        else:
            if expected_fraud_loss > FRICTION_COST:
                decision = "REVIEW"
                review_counter += 1
            else:
                decision = "ACCEPT"

        # 6. OUTPUT & FILE LOGGING
        timestamp = datetime.now().strftime("%H:%M:%S")

        # CONSOLE OUTPUT (Restored Queue counter)
        print(
            f"[{timestamp}] Trans #{transaction_count:03} | Amt: ${amount:7.2f} | Risk: {risk_score:.4f} | Decision: {decision:7} | Queue: {review_counter}/{K_LIMIT}"
        )

        # LOG TO TEXT FILE IF REVIEWED
        if decision == "REVIEW":
            with open(REVIEW_LOG_TXT, "a") as f:
                f.write(
                    f"{timestamp:<10} | #{transaction_count:<9} | ${amount:<9.2f} | {risk_score:.4f}\n"
                )

        # LOG TO CSV FOR ACTIVE LEARNING
        log_entry = pd.DataFrame([row.to_dict()])
        log_entry["Risk_Score"] = risk_score
        log_entry["Decision"] = decision
        log_entry["Timestamp"] = timestamp
        log_entry.to_csv(LOG_FILE_CSV, mode="a", header=False, index=False)

        transaction_count += 1
        time.sleep(0.1)

except KeyboardInterrupt:
    print(
        f"\n[INFO] Stream stopped. Summary logs saved to {LOG_FILE_CSV} and {REVIEW_LOG_TXT}"
    )
