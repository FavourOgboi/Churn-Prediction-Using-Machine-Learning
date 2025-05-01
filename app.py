import streamlit as st
import pandas as pd
import joblib  # Import joblib
import os

# Absolute paths
model_path = "customer_churn_model.pkl"
encoder_path = "encoders.pkl"


# Load the saved model and feature names
model_data = joblib.load(model_path)  # Use joblib to load the model
loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# Load encoders using joblib
encoders = joblib.load(encoder_path)

# App Title
st.title("📊 Customer Churn Prediction App")

# App description / note
st.markdown("""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px;'>
<b>About this app:</b>  
This simple tool helps you predict whether a customer is likely to churn (leave the service) based on their personal and account details.  
Fill in the details below and click **Predict** to see the result.
</div>
""", unsafe_allow_html=True)

st.write("")  # spacer

# User input form
with st.form("churn_form"):
    gender = st.selectbox("Gender", ['Female', 'Male'])
    SeniorCitizen = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", [0, 1])
    Partner = st.selectbox("Partner", ['Yes', 'No'])
    Dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
    MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
    PaymentMethod = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=29.85)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=29.85)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Collect user inputs into a dataframe
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_data_df = pd.DataFrame([input_data])

    # Encode categorical features using saved encoders
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])

    # Ensure input data matches model's feature order
    input_data_df = input_data_df[feature_names]

    # Make a prediction
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)

    # Display the result
    st.write("---")
    if prediction[0] == 1:
        st.error(f"⚠️ The customer is likely to churn! (Probability: {pred_prob[0][1]:.2f})")
    else:
        st.success(f"✅ The customer is not likely to churn. (Probability: {pred_prob[0][0]:.2f})")
