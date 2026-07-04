import streamlit as st
import pandas as pd
import joblib

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("encoder.pkl")
columns = joblib.load("columns.pkl")

st.title("🏦 Loan Approval Prediction")
st.write("Fill the applicant details below.")

applicant_income = st.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
age = st.number_input("Age", min_value=18)
dependents = st.number_input("Dependents", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
existing_loans = st.number_input("Existing Loans", min_value=0)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0)
savings = st.number_input("Savings", min_value=0.0)
collateral_value = st.number_input("Collateral Value", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Term", min_value=1)

employment = st.selectbox(
    "Employment Status",
    ["Contract", "Salaried", "Self-employed", "Unemployed"]
)

marital = st.selectbox(
    "Marital Status",
    ["Married", "Single"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

education = st.selectbox(
    "Education Level",
    ["Graduate", "Not Graduate"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

employer = st.selectbox(
    "Employer Category",
    ["Business", "Government", "MNC", "Private", "Unemployed"]
)

predict = st.button("Predict Loan Approval")

if predict:

    # Create a one-row dataframe from user input
    input_data = pd.DataFrame({
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Age": [age],
        "Dependents": [dependents],
        "Credit_Score": [credit_score],
        "Existing_Loans": [existing_loans],
        "DTI_Ratio": [dti_ratio],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Education_Level": [education],

        "Employment_Status": [employment],
        "Marital_Status": [marital],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Gender": [gender],
        "Employer_Category": [employer]
    })

    # List of categorical columns
    categorical_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category"
    ]

    # One-Hot Encode categorical columns
    encoded = ohe.transform(input_data[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(categorical_cols),
        index=input_data.index
    )

    # Remove original categorical columns
    input_data = input_data.drop(columns=categorical_cols)

    # Add encoded columns
    input_data = pd.concat([input_data, encoded_df], axis=1)

    # Convert Education Level into numeric
    if input_data["Education_Level"][0] == "Graduate":
        input_data["Education_Level"] = 0
    else:
        input_data["Education_Level"] = 1

    # Create squared features
    input_data["DTI_Ratio_sq"] = input_data["DTI_Ratio"] ** 2
    input_data["Credit_Score_sq"] = input_data["Credit_Score"] ** 2
    
    # Drop original features (to match training)
    input_data = input_data.drop(columns=["DTI_Ratio", "Credit_Score"])
    
    # Arrange columns in the same order as training
    input_data = input_data.reindex(columns=columns, fill_value=0)
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    # Display prediction
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"Approval Probability: {probability[0][1] * 100:.2f}%")