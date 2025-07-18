# 📉 Customer Churn Prediction

This project uses Machine Learning to predict whether a customer will churn (leave) based on historical customer data.

---

## 🔍 Project Overview

Customer churn prediction helps companies retain customers by identifying those who are likely to stop using their services. In this project, we:

- Clean and preprocess telecom customer data
- Perform Exploratory Data Analysis (EDA)
- Train and evaluate Machine Learning models
- Predict churn and export the results

---

## 📁 Dataset

The dataset used in this project is `customer_churn.csv`, containing customer details like:

- `gender`, `SeniorCitizen`, `tenure`, `MonthlyCharges`, etc.
- `Churn` column indicates if the customer left (Yes/No)

---

## ⚙️ Project Structure

churn_project/
├── customer_churn.csv
├── clean_data.py
├── churn_model.pkl
├── predict_churn.py
├── app.py (if using Flask/Streamlit)
└── README.md
