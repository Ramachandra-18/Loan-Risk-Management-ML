import os
import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "loan_risk_model.pkl")
PREPROCESSOR_PATH = os.path.join(ROOT, "models", "loan_risk_preprocessor.pkl")

st.set_page_config(page_title="Loan Risk Management System", page_icon="💰", layout="wide")

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Add loan_risk_model.pkl to the models/ folder. See README.md.")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Add loan_risk_preprocessor.pkl to the models/ folder. See README.md.")
    return joblib.load(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)

st.title("💰 Intelligent Loan Risk Management System")
st.caption("Machine-learning demonstration using Logistic Regression.")
st.info("Academic prototype only. Predictions are estimates and should not be the sole basis for lending decisions.")

try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("1. Applicant information")
a, b, c = st.columns(3)
with a:
    age = st.number_input("Applicant age (years)", 18, 100, 35)
with b:
    account_type = st.selectbox("Bank account type", ["Savings", "Current", "Other"])
with c:
    employment = st.selectbox("Employment status", ["Permanent", "Self-Employed", "Student", "Unemployed", "Other", "Unknown"])
a, b = st.columns(2)
with a:
    education = st.selectbox("Education level", ["Primary", "Secondary", "Graduate", "Post-Graduate", "Other", "Unknown"])
with b:
    bank_name = st.text_input("Bank name", "GT Bank")

st.subheader("2. Current loan information")
a, b, c = st.columns(3)
with a:
    loan_amount = st.number_input("Loan amount", min_value=1.0, value=20000.0, step=1000.0)
with b:
    total_due = st.number_input("Total amount due", min_value=1.0, value=23000.0, step=1000.0)
with c:
    term_days = st.number_input("Repayment term (days)", min_value=1, value=30)

st.subheader("3. Previous loan history")
a, b, c = st.columns(3)
with a:
    previous_count = st.number_input("Number of previous loans", min_value=0, value=3)
with b:
    previous_avg_amount = st.number_input("Average previous loan amount", min_value=0.0, value=15000.0, step=1000.0)
with c:
    previous_total_amount = st.number_input("Total previous loan amount", min_value=0.0, value=45000.0, step=1000.0)
a, b, c = st.columns(3)
with a:
    previous_avg_due = st.number_input("Average previous total due", min_value=0.0, value=17000.0, step=1000.0)
with b:
    previous_total_due = st.number_input("Total previous due", min_value=0.0, value=51000.0, step=1000.0)
with c:
    previous_avg_term = st.number_input("Average previous term (days)", min_value=0.0, value=30.0)

st.subheader("4. Location and processing information")
a, b, c = st.columns(3)
with a:
    longitude = st.number_input("Longitude", value=5.5, format="%.6f")
with b:
    latitude = st.number_input("Latitude", value=7.2, format="%.6f")
with c:
    bank_branch = st.text_input("Bank branch", "Unknown")
processing_hours = st.number_input("Approval processing time (hours)", min_value=0.0, value=1.0, step=0.25)

if st.button("🔍 Predict loan risk", type="primary", use_container_width=True):
    if loan_amount <= 0 or term_days <= 0:
        st.error("Loan amount and repayment term must be greater than zero.")
        st.stop()

    row = {
        "loanamount": loan_amount, "totaldue": total_due, "termdays": term_days,
        "longitude_gps": longitude, "latitude_gps": latitude,
        "previous_loan_count": previous_count,
        "previous_avg_loan_amount": previous_avg_amount,
        "previous_total_loan_amount": previous_total_amount,
        "previous_avg_totaldue": previous_avg_due,
        "previous_total_due": previous_total_due,
        "previous_avg_term_days": previous_avg_term,
        "age_years": age,
        "approval_processing_hours": processing_hours,
        "loan_to_due_ratio": total_due / loan_amount,
        "loan_cost_ratio": (total_due - loan_amount) / loan_amount,
        "payment_burden": total_due / term_days,
        "previous_to_current_loan_ratio": previous_avg_amount / loan_amount if previous_count else 0.0,
        "has_previous_loan": int(previous_count > 0),
        "bank_account_type": account_type,
        "bank_name_clients": bank_name,
        "bank_branch_clients": bank_branch,
        "employment_status_clients": employment,
        "level_of_education_clients": education,
    }

    expected = list(getattr(preprocessor, "feature_names_in_", row.keys()))
    for col in expected:
        if col not in row:
            row[col] = "Unknown"
    input_df = pd.DataFrame([{col: row[col] for col in expected}])

    try:
        transformed = preprocessor.transform(input_df)
        predicted = int(model.predict(transformed)[0])
        probabilities = model.predict_proba(transformed)[0]
    except Exception as e:
        st.error("Prediction failed. The app input columns/categories may not match the exact training schema.")
        st.exception(e)
        st.stop()

    st.subheader("📋 Prediction result")
    if predicted == 0:
        st.success("🟢 GOOD RISK — The model classifies this application as Good Risk.")
    else:
        st.error("🔴 BAD RISK — The model classifies this application as Bad Risk.")
    x, y = st.columns(2)
    x.metric("Good Risk probability", f"{probabilities[0]:.2%}")
    y.metric("Bad Risk probability", f"{probabilities[1]:.2%}")
    st.caption("Probabilities are model estimates, not guarantees of repayment or default.")
