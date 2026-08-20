import pandas as pd
import streamlit as st
from joblib import load

model = load("random_forest_model.pkl")

st.title("Customer Churn Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    value=12
)

usage_frequency = st.number_input(
    "Usage Frequency",
    min_value=0,
    value=20
)

support_calls = st.number_input(
    "Support Calls",
    min_value=0,
    value=5
)

payment_delay = st.number_input(
    "Payment Delay",
    min_value=0,
    value=10
)

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Premium", "Standard"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly"]
)

total_spend = st.number_input(
    "Total Spend",
    min_value=0,
    value=500
)

last_interaction = st.number_input(
    "Last Interaction",
    min_value=0,
    value=7
)

input_data = pd.DataFrame({
    'Age': [age],
    'Tenure': [tenure],
    'Usage Frequency': [usage_frequency],
    'Support Calls': [support_calls],
    'Payment Delay': [payment_delay],
    'Total Spend': [total_spend],
    'Last Interaction': [last_interaction],
    
    'Gender_Male': [1 if gender == 'Male' else 0],
    
    'Subscription Type_Premium': [
        1 if subscription_type == 'Premium' else 0
    ],
    
    'Subscription Type_Standard': [
        1 if subscription_type == 'Standard' else 0
    ],
    
    'Contract Length_Monthly': [
        1 if contract_length == 'Monthly' else 0
    ],
    
    'Contract Length_Quarterly': [
        1 if contract_length == 'Quarterly' else 0
    ]
})


if st.button("Predict Churn"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    churn_probability = probability[0][1] * 100

    if prediction[0] == 1:
        st.error("🔴 Customer is likely to churn")
    else:
        st.success("🟢 Customer is unlikely to churn")

    st.write(f"Churn Probability: {churn_probability:.2f}%")












