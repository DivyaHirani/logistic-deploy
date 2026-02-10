import streamlit as st
import joblib
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Purchase Predictor",
    page_icon="🛍️",
    layout="wide"
)

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------- CUSTOM STYLE ----------
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}
.card {
    padding:20px;
    border-radius:12px;
    background-color:#f5f7fa;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="big-title">🛍️ Customer Purchase Predictor</p>', unsafe_allow_html=True)
st.write("Predict whether a customer will purchase based on profile data.")

# ---------- SIDEBAR ----------
st.sidebar.header("Input Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
salary = st.sidebar.number_input("Salary", value=50000)
purchases = st.sidebar.slider("Previous Purchases", 0, 10, 2)

# ---------- MAIN CARD ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.metric("Age", age)
col2.metric("Salary", salary)
col3.metric("Past Purchases", purchases)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- PREDICTION ----------
if st.button("🔮 Predict Purchase"):

    data = np.array([[age, salary, purchases]])
    data = scaler.transform(data)

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    st.divider()

    if prediction[0] == 1:
        st.success(f"✅ Likely to Purchase (Confidence: {probability:.2%})")
        st.balloons()
    else:
        st.error(f"❌ Unlikely to Purchase (Confidence: {1-probability:.2%})")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("Built with Streamlit | Logistic Regression Demo")
