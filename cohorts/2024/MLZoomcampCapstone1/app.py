import streamlit as st
import pandas as pd
import requests
import json

st.title("Heart Disease Prediction App")

st.write("""
### Enter Patient Information
This application predicts the likelihood of heart disease based on clinical parameters.
""")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=90, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=0.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    # Prepare features
    features = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    
    # Make prediction request
    try:
        response = requests.post(
            "http://0.0.0.0:3000/predict",
            headers={"content-type": "application/json"},
            data=json.dumps(features),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "error" in result:
                st.error(f"Prediction error: {result['error']}")
            else:
                prediction = result["prediction"]
                probability = result["probability"]
                
                st.write("### Prediction Results")
                if prediction == 1:
                    st.error(f"High risk of heart disease (Probability: {probability:.2%})")
                else:
                    st.success(f"Low risk of heart disease (Probability: {1-probability:.2%})")
        else:
            st.error(f"Error making prediction. Server returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to prediction service. Please ensure the service is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

