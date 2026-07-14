"""
Epic 3 - Data Pre-Processing
Story 1: remove duplicates
Story 2: handle missing values
Story 3: cleaning & merging (Applicant_Details + Credit_History -> single modeling table)
Story 4: feature engineering
Story 5: encode categorical variables
"""

import numpy as np
import pandas as pd

DATA_PATH = "/home/claude/credit_card_approval_prediction/data/credit_card_approval.csv"
OUT_PATH = "/home/claude/credit_card_approval_prediction/data/credit_card_approval_clean.csv"

df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# ---------- Story 1: remove duplicates ----------
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

# ---------- Story 2: handle missing values ----------
print("\nMissing values before imputation:\n", df.isnull().sum()[df.isnull().sum() > 0])
df["AnnualIncome"] = df["AnnualIncome"].fillna(df["AnnualIncome"].median())
df["EmploymentDays"] = df["EmploymentDays"].fillna(df["EmploymentDays"].median())
df["HousingType"] = df["HousingType"].fillna(df["HousingType"].mode()[0])
assert df.isnull().sum().sum() == 0, "Missing values remain!"
print("Missing values after imputation: 0")

# ---------- Story 3: cleaning & merging ----------
# (Applicant_Details and Credit_History are already merged 1:1 via ApplicantID/HistoryID
#  in the synthetic dataset; in a real system this would be a SQL JOIN.)
df["OwnsCar"] = df["OwnsCar"].str.strip().str.upper()
df["OwnsRealty"] = df["OwnsRealty"].str.strip().str.upper()

# ---------- Story 4: feature engineering ----------
df["YearsEmployed"] = (-df["EmploymentDays"] / 365.25).round(2)
df["IncomePerFamilyMember"] = (df["AnnualIncome"] / (df["NumChildren"] + 1)).round(2)
df["HasPastDefault"] = (df["PastDefaults"] > 0).astype(int)
df["HighUtilization"] = (df["CreditUtilization"] > 0.5).astype(int)

# ---------- Story 5: encode categorical variables ----------
binary_map = {"Y": 1, "N": 0}
df["OwnsCar"] = df["OwnsCar"].map(binary_map)
df["OwnsRealty"] = df["OwnsRealty"].map(binary_map)

categorical_cols = ["IncomeType", "EducationType", "FamilyStatus", "HousingType"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df_encoded.to_csv(OUT_PATH, index=False)
print(f"\nFinal cleaned & encoded shape: {df_encoded.shape}")
print(f"Saved to {OUT_PATH}")
