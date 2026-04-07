import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime

# --- SETTINGS & ASSET LOADING ---
st.set_page_config(page_title="Fraud Sentinel Dashboard", layout="wide")


@st.cache_resource
def load_assets():
    scaler = joblib.load("./models/scaler.joblib")
    lr = joblib.load("./models/logistic_regression_model.joblib")
    xgb = joblib.load("./models/xgboost_model.joblib")
    data = pd.read_csv("creditcard.csv")
    return scaler, lr, xgb, data


try:
    scaler, lr_model, xgb_model, prod_data = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
if "index" not in st.session_state:
    st.session_state.index = 0
    st.session_state.review_counter = 0
    st.session_state.history = []  # Full live feed
    st.session_state.flagged_list = []  # Permanent log of Rejects/Reviews
    st.session_state.current_queue = []  # Items currently waiting for an analyst
    st.session_state.running = False

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Control Panel")
k_limit = st.sidebar.slider("No of analysts (Reviews per Window)", 1, 20, 5)
window_size = st.sidebar.slider("Window Size (Transactions)", 5, 120, 10)
sim_speed = st.sidebar.slider("Simulation Speed (sec)", 0.1, 2.0, 1.0)

col1, col2 = st.sidebar.columns(2)
if col1.button("▶️ Start"):
    st.session_state.running = True
if col2.button("⏸️ Stop"):
    st.session_state.running = False
if st.sidebar.button("🔄 Reset Session"):
    for key in ["index", "review_counter", "history", "flagged_list", "current_queue"]:
        st.session_state[key] = [] if isinstance(st.session_state[key], list) else 0
    st.rerun()

# --- UI LAYOUT ---
st.title("Fourcaster: Real-Time Decision Engine")

# Top Metrics Row
m1, m2, m3, m4 = st.columns(4)
queue_fill = st.session_state.review_counter / k_limit
m1.metric(
    "Queue Capacity",
    f"{st.session_state.review_counter} / {k_limit}",
    f"{int(queue_fill*100)}% Full",
    delta_color="inverse",
)
m2.metric("Total Processed", st.session_state.index)
m3.metric("System State", "CONGESTED" if queue_fill >= 0.7 else "HEALTHY")
m4.metric("Active Reviews", len(st.session_state.current_queue))

# Main Content Area
feed_col, queue_col = st.columns([2, 1])

with feed_col:
    st.subheader("Live Transaction Feed")
    feed_container = st.empty()

with queue_col:
    st.subheader("Active Review Queue")
    st.caption("Transactions awaiting manual analyst verification")
    queue_container = st.empty()

    st.divider()

    st.subheader("🚫 Blocked / Logged Actions")
    flagged_container = st.empty()

# --- SIMULATION LOGIC ---
if st.session_state.running and st.session_state.index < len(prod_data):
    row = prod_data.iloc[st.session_state.index]

    # Window Reset Logic
    if st.session_state.index % window_size == 0 and st.session_state.index > 0:
        st.session_state.review_counter = 0
        st.session_state.current_queue = []  # Clear the active queue for the new window

    # Model Inference
    features_df = pd.DataFrame([row.drop("Class")])
    amount = row["Amount"]
    features_scaled = scaler.transform(features_df)
    p_lr = lr_model.predict_proba(features_scaled)[:, 1][0]
    p_xgb = xgb_model.predict_proba(features_df)[:, 1][0]
    risk_score = (0.8 * p_xgb) + (0.2 * p_lr)

    # Decision Engine Logic
    decision = "ACCEPT"
    expected_loss = risk_score * amount

    if risk_score > 0.95:
        decision = "REJECT"
    elif st.session_state.review_counter >= k_limit:
        decision = "REJECT" if risk_score > 0.85 else "ACCEPT"
    elif (st.session_state.review_counter / k_limit) >= 0.70:
        if risk_score > 0.70 or expected_loss > 100:
            decision = "REVIEW"
            st.session_state.review_counter += 1
    else:
        if expected_loss > 20.0:
            decision = "REVIEW"
            st.session_state.review_counter += 1

    # Update Data Lists
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {
        "ID": f"#{st.session_state.index:03}",
        "Time": timestamp,
        "Amount": f"${amount:.2f}",
        "Risk": f"{risk_score:.4f}",
        "Decision": decision,
    }

    # Add to History (Reverse Chronological)
    st.session_state.history.insert(0, entry)

    # Add to Active Queue if it's a review
    if decision == "REVIEW":
        st.session_state.current_queue.insert(0, entry)
        st.session_state.flagged_list.insert(0, entry)
    elif decision == "REJECT":
        st.session_state.flagged_list.insert(0, entry)

    # Render Visuals
    feed_container.table(pd.DataFrame(st.session_state.history).head(15))

    if st.session_state.current_queue:
        queue_container.table(pd.DataFrame(st.session_state.current_queue))
    else:
        queue_container.write("Queue is currently empty.")

    if st.session_state.flagged_list:
        flagged_container.table(pd.DataFrame(st.session_state.flagged_list).head(10))

    st.session_state.index += 1
    time.sleep(sim_speed)
    st.rerun()
