import streamlit as st
import requests

st.title("ML Prediction App 🚀")

st.write("Enter feature values:")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")


if st.button("Predict"):

    data = {
        "Employed": f1,
        "Bank_Balance": f2,
        "Annual_Salary": f3,
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=data
    )

    result = response.json()

    if "prediction" in result:
        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Confidence: {round(result['confidence']*100, 2)}%")
    else:
        st.error(f"API Error: {result}")
