# 🛡️ Fourcaster: A Cost-Aware Fraud Detection System

This repository implements a real-time, cost-aware credit card fraud detection pipeline. Unlike traditional binary classifiers that prioritize raw accuracy, this system incorporates a decision-making layer that balances the financial risk of fraud against the operational cost of customer friction.


## 💡 Project Premise
The primary challenge in fraud detection is the high cost of **False Positives**. Blocking a legitimate transaction for a small amount (e.g., a **$5** coffee) can cause significant customer dissatisfaction and long-term loss of revenue.

Our solution employs a **Cost-Aware Decision Engine**. It calculates the **Expected Loss** for each transaction and only intervenes if the risk outweighs the "Friction Cost" (the estimated loss in customer loyalty/lifetime value when a card is incorrectly declined).

### Mathematical Framework
The decision logic is governed by:

$$L_{expected} = P_{fraud} \times \text{Amount}$$

The system approves transactions by default unless $L_{expected} > \text{Friction Cost}$, in which case it is routed for manual review or immediate rejection based on the model's confidence and current system capacity.


## ✨ Key Features
* **Hybrid Ensemble Architecture:** Combines **XGBoost** for non-linear pattern recognition and **Logistic Regression** for statistical stability, using an **80/20** weighted scoring system.
* **Dynamic Review Throttling:** An adaptive operational layer that adjusts the manual review threshold based on real-time analyst capacity.
* **Stratified 5-Fold Cross-Validation:** Rigorous validation on the **ULB Credit Card Fraud dataset** to ensure performance stability on extremely skewed data (0.17% fraud).
* **Operational Dashboard:** A **Streamlit** interface for real-time monitoring, allowing users to adjust review capacity and window sizes on the fly.


## 🚀 Execution Guide

### 1. Installation
Clone the repository and install the necessary requirements:
```bash
git clone https://github.com/Speedbird-One/Fourcaster.git
cd Fourcaster
pip install -r requirements.txt
```
**Note:** The dataset is too large to be hosted on GitHub. You must download the `creditcard.csv` file from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory.

If you have the [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) configured, you can use:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
```

### 2. Model Training & Validation
Run the training script to perform cross-validation and generate the serialized model files (`scaler.joblib`, `logistic_regression_model.joblib`, and `xgboost_model.joblib`):
```bash
python train_model.py
```

### 3. CLI Stream Simulation
For a lightweight, terminal-based simulation of the production environment, use the stream processor. You can set the manual review capacity (K) via the CLI:
```bash
# Sets a capacity of 5 manual reviews per window
python stream_processor.py --capacity 5
```
Upon running `stream_processor.py` the system generates two primary logs during execution to support operations and future development:

* `active_learning_log.csv`: A comprehensive record of every processed transaction, including feature values, assigned risk scores, and the final system decision. This file is designed for batch-retraining the model (Active Learning).

* `manual_review_queue.txt`: A clean, human-readable log of every transaction flagged for **REVIEW**. This simulates an analyst's workflow, providing timestamps, transaction IDs, and risk scores.

### 4. Visual Dashboard
Launch the interactive control center to visualize the feed and adjust system parameters (K-limit and Window Size) on the fly:
```bash
streamlit run dashboard.py
```


## 📅 Future Development
* **Active Learning Integration:** Implement an automated retraining loop using the active_learning_log.csv generated during simulation.
* **Variable Friction Costs:** Dynamically adjust friction costs based on individual customer profiles and historical spending habits.
* **API Exposure:** Wrap the scoring engine in a FastAPI or Flask wrapper for integration with external payment gateways.
