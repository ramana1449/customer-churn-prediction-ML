import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title
st.title("Customer Churn Prediction App")

# Tabs for input modes
option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV File"])

if option == "Manual Entry":
    st.subheader("🔘 Manual Entry")

    # Input form
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    # Convert inputs to DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    if st.button("Predict"):
        prediction = model.predict(input_data)
        result = "✅ Customer will **Churn**" if prediction[0] == "Yes" else "✅ Customer will **Stay**"
        st.success(result)

else:
    st.subheader("📁 Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            prediction = model.predict(data)

            results = data.copy()
            results['Churn Prediction'] = prediction

            st.write("### 🔍 Prediction Results")
            st.dataframe(results)

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
