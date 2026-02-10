import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Purchase Predictor")

age = st.number_input("Age", 18, 100)
salary = st.number_input("Salary")
purchases = st.number_input("Previous Purchases")

if st.button("Predict"):

    data = np.array([[age, salary, purchases]])
    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Customer WILL Purchase")
    else:
        st.error("Customer will NOT Purchase")
