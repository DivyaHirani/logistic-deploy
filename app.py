import streamlit as st
import joblib
import numpy as np
import requests
from streamlit_lottie import st_lottie

# -------- PAGE CONFIG --------
st.set_page_config(page_title="AI Predictor", layout="wide")

# -------- LOAD MODEL --------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -------- LOAD ANIMATION --------
def load_lottie(url):
    r = requests.get(url)
    return r.json()

animation = load_lottie(
    "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
)

st.markdown("""
<style>

/* GLOBAL TEXT COLOR FIX */
html, body, [class*="css"]  {
    color: white !important;
}

/* Labels */
label {
    color: #ffffff !important;
    font-weight: 600;
}

/* Slider numbers */
span {
    color: #ffffff !important;
}

/* Metric text */
.stMetric {
    color: white !important;
}

/* Background */
.stApp {
    background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
}

/* Card */
.card {
    background: rgba(255,255,255,0.08);
    padding:25px;
    border-radius:18px;
    backdrop-filter: blur(14px);
    box-shadow: 0 0 25px rgba(0,0,0,0.5);
}

/* Button */
button[kind="primary"] {
    background: linear-gradient(90deg,#ff00cc,#3333ff);
    color:white;
    border:none;
    border-radius:10px;
    height:3em;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


# -------- HEADER --------
colA, colB = st.columns([1,2])

with colA:
    st_lottie(animation, height=250)

with colB:
    st.title("🧠 AI Purchase Intelligence")
    st.write("Interactive prediction powered by Machine Learning")

# -------- INPUT AREA --------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    age = st.slider("Age", 18, 70, 30)
    salary = st.slider("Salary", 20000, 150000, 60000)
    purchases = st.slider("Previous Purchases", 0, 10, 2)

    predict = st.button("Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- OUTPUT AREA --------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if predict:

        with st.spinner("AI thinking..."):
            data = np.array([[age, salary, purchases]])
            data = scaler.transform(data)

            pred = model.predict(data)
            prob = model.predict_proba(data)[0][1]

        st.subheader("Confidence Level")

        # Animated progress bar
        bar = st.progress(0)
        for i in range(int(prob*100)):
            bar.progress(i+1)

        if pred[0] == 1:
            st.success(f"Likely Buyer ({prob:.1%})")
            st.balloons()
        else:
            st.error(f"Unlikely Buyer ({1-prob:.1%})")
            st.snow()

    st.markdown('</div>', unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown("---")
st.caption("Next-Gen Interactive ML App")

