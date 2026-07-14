"""
Epic 1 - Story 1: Download / prepare the dataset.

Since no real-world file was supplied, this script generates a realistic
SYNTHETIC credit-card-approval dataset whose columns mirror the ER diagram's
Applicant_Details and Credit_History entities. Swap this out for a real
dataset (e.g. the UCI/Kaggle "Credit Card Approval" dataset) by dropping a
CSV with the same column names into data/credit_card_approval.csv.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

income_types = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
education_types = ["Higher education", "Secondary", "Incomplete higher", "Lower secondary", "Academic degree"]
family_statuses = ["Married", "Single", "Civil marriage", "Separated", "Widow"]
housing_types = ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"]

df = pd.DataFrame({
    "ApplicantID": np.arange(1, N + 1),
    "UserID": np.random.randint(1, 200, N),
    "IncomeType": np.random.choice(income_types, N, p=[0.55, 0.20, 0.15, 0.07, 0.03]),
    "EducationType": np.random.choice(education_types, N, p=[0.5, 0.3, 0.12, 0.05, 0.03]),
    "FamilyStatus": np.random.choice(family_statuses, N, p=[0.55, 0.25, 0.1, 0.05, 0.05]),
    "HousingType": np.random.choice(housing_types, N, p=[0.65, 0.12, 0.1, 0.1, 0.03]),
    "EmploymentDays": np.random.randint(-15000, -30, N),  # negative = days employed (relative to today)
    "AnnualIncome": np.round(np.random.lognormal(mean=10.8, sigma=0.45, size=N), 2),
    "Age": np.random.randint(21, 65, N),
    "NumChildren": np.random.poisson(0.7, N).clip(0, 6),
    "OwnsCar": np.random.choice(["Y", "N"], N, p=[0.4, 0.6]),
    "OwnsRealty": np.random.choice(["Y", "N"], N, p=[0.6, 0.4]),
})

# ---- Credit_History fields ----
df["HistoryID"] = df["ApplicantID"]
df["PastDefaults"] = np.random.poisson(0.4, N).clip(0, 8)
df["OverdueMonths"] = np.random.poisson(1.0, N).clip(0, 24)
df["CreditUtilization"] = np.round(np.random.beta(2, 5, N), 3)          # 0-1
df["ExistingCreditLines"] = np.random.randint(0, 10, N)

# ---- Simple, explainable "ground truth" rule + noise -> Approval_Prediction.Result ----
score = (
    (df["AnnualIncome"] > df["AnnualIncome"].median()).astype(int) * 1.2
    + (df["PastDefaults"] == 0).astype(int) * 1.5
    + (df["OverdueMonths"] < 3).astype(int) * 1.3
    + (df["CreditUtilization"] < 0.4).astype(int) * 1.0
    + (df["EmploymentDays"] < -365).astype(int) * 0.8       # employed > 1 year
    + (df["ExistingCreditLines"] < 6).astype(int) * 0.4
    - (df["NumChildren"] > 3).astype(int) * 0.3
    + np.random.normal(0, 1.0, N)
)
df["Approved"] = (score > score.mean()).astype(int)  # 1 = Approved, 0 = Rejected

out_path = "/home/claude/credit_card_approval_prediction/data/credit_card_approval.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Approved"].value_counts(normalize=True))
