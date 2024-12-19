
import streamlit as st
import pandas as pd
import requests
import json

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    div.row-widget.stRadio > div{
        flex-direction: row;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title with emoji
st.title("❤️ Heart Disease Prediction")

# Introduction with markdown
st.markdown("""
    ### Welcome to the Heart Disease Prediction Tool
    This application uses machine learning to assess the likelihood of heart disease based on clinical parameters.
    Please fill in the following information to get your prediction.
    """)

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                      help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")

with col2:
    st.subheader("Clinical Measurements")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.subheader("Additional Clinical Data")
col3, col4 = st.columns(2)

with col3:
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=0.0, step=0.1)

with col4:
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Add a divider
st.markdown("---")

# Centered predict button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button("Predict", type="primary")

if predict_button:
    # Show a spinner while making prediction
    with st.spinner('Analyzing your data...'):
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
                    
                    st.markdown("### 📊 Prediction Results")
                    
                    if prediction == 1:
                        st.error(f"⚠️ High risk of heart disease\nProbability: {probability:.1%}")
                    else:
                        st.success(f"✅ Low risk of heart disease\nProbability: {1-probability:.1%}")
                    
                    # Add explanation
                    st.info("Note: This prediction is based on machine learning analysis of the provided parameters. Always consult with healthcare professionals for medical advice.")
            else:
                st.error(f"Error making prediction. Server returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to prediction service. Please ensure the service is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
