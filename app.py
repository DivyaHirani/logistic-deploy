import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Purchase Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background gradient */
.stApp {
    background: linear-gradient(120deg, #1f4037, #99f2c8);
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.15);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

/* Title style */
.title {
    font-size:48px;
    font-weight:800;
    text-align:center;
    color:white;
    letter-spacing:1px;
}

/* Subtitle */
.subtitle {
    text-align:center;
    color:white;
    font-size:18px;
    margin-bottom:30px;
}

/* Button styling */
div.stButton > button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color:white;
    border:none;
    border-radius:12px;
    padding:12px 25px;
    font-size:18px;
    font-weight:bold;
    updated-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">AI Customer Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict Purchase Behaviour using Machine Learning</div>', unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns([1,2])

with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("🎛 Customer Inputs")

    age = st.slider("Age", 18, 70, 30)
    salary = st.number_input("Salary", value=50000)
    purchases = st.slider("Previous Purchases", 0, 10, 2)

    predict = st.button("🚀 Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- OUTPUT SECTION ----------------
with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("📊 Prediction Result")

    if predict:
        data = np.array([[age, salary, purchases]])
        data = scaler.transform(data)

        pred = model.predict(data)
        prob = model.predict_proba(data)[0][1]

        st.progress(int(prob*100))

        if pred[0] == 1:
            st.success(f"🟢 Likely Buyer — Confidence {prob:.1%}")
            st.balloons()
        else:
            st.error(f"🔴 Unlikely Buyer — Confidence {(1-prob):.1%}")

    else:
        st.info("Enter inputs and click Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<center style='color:white;margin-top:40px'>
Built with ❤️ using Streamlit | ML Deployment Project
</center>
""", unsafe_allow_html=True)
