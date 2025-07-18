import pandas as pd

# Columns your model was trained on (adjust if needed)
required_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'MonthlyCharges', 'TotalCharges'
]

# Read the original file
df = pd.read_csv('customer_churn.csv.csv')

# Keep only the required columns
df_cleaned = df[required_columns]

# Save to new file
df_cleaned.to_csv('customer_churn_cleaned.csv', index=False)

print("✅ Cleaned data saved to 'customer_churn_cleaned.csv'")
