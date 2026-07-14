"""
Epic 2 - Visualizing and Analysing the Data
Story 1: import libraries
Story 2: read & explore dataset
Story 3: univariate analysis
Story 4: multivariate analysis
Story 5: descriptive statistics
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = "/home/claude/credit_card_approval_prediction/data/credit_card_approval.csv"
FIG_DIR = "/home/claude/credit_card_approval_prediction/docs/figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Story 2: read & explore ----------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ---------- Story 5: descriptive statistics ----------
print("\nDescriptive statistics (numeric):\n", df.describe())
desc_path = f"{FIG_DIR}/../descriptive_stats.csv"
df.describe(include="all").to_csv(desc_path)
print(f"Saved descriptive stats to {desc_path}")

# ---------- Story 3: univariate analysis ----------
plt.figure(figsize=(6, 4))
sns.countplot(x="Approved", data=df)
plt.title("Approval Distribution (Univariate)")
plt.savefig(f"{FIG_DIR}/univariate_approval_distribution.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4))
sns.histplot(df["AnnualIncome"], bins=40, kde=True)
plt.title("Annual Income Distribution (Univariate)")
plt.savefig(f"{FIG_DIR}/univariate_income_distribution.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4))
sns.countplot(y="IncomeType", data=df, order=df["IncomeType"].value_counts().index)
plt.title("Income Type Distribution (Univariate)")
plt.savefig(f"{FIG_DIR}/univariate_income_type.png", bbox_inches="tight")
plt.close()

# ---------- Story 4: multivariate analysis ----------
plt.figure(figsize=(8, 4))
sns.boxplot(x="Approved", y="AnnualIncome", data=df)
plt.title("Income vs Approval (Multivariate)")
plt.savefig(f"{FIG_DIR}/multivariate_income_vs_approval.png", bbox_inches="tight")
plt.close()

numeric_cols = ["AnnualIncome", "Age", "NumChildren", "PastDefaults",
                 "OverdueMonths", "CreditUtilization", "ExistingCreditLines", "Approved"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Multivariate)")
plt.savefig(f"{FIG_DIR}/multivariate_correlation_heatmap.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4))
sns.countplot(x="EducationType", hue="Approved", data=df)
plt.xticks(rotation=30, ha="right")
plt.title("Education Type vs Approval (Multivariate)")
plt.savefig(f"{FIG_DIR}/multivariate_education_vs_approval.png", bbox_inches="tight")
plt.close()

print(f"\nAll figures saved to {FIG_DIR}")
