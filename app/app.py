"""
Epic 5 - Application Building
Story 2: Build the Python (Flask) application and integrate the trained model.

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

app = Flask(__name__)

# ---- Load trained artifacts once, at startup ----
best_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLUMNS = json.load(f)
with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
    MODEL_NAME = f.read().strip()

INCOME_TYPES = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
EDUCATION_TYPES = ["Higher education", "Secondary", "Incomplete higher", "Lower secondary", "Academic degree"]
FAMILY_STATUSES = ["Married", "Single", "Civil marriage", "Separated", "Widow"]
HOUSING_TYPES = ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"]


def build_feature_row(form):
    """Turn the submitted form into a single-row DataFrame matching training features."""
    years_employed = float(form["years_employed"])
    employment_days = -years_employed * 365.25
    annual_income = float(form["annual_income"])
    num_children = int(form["num_children"])
    past_defaults = int(form["past_defaults"])
    credit_utilization = float(form["credit_utilization"])

    raw = {
        "EmploymentDays": employment_days,
        "AnnualIncome": annual_income,
        "Age": int(form["age"]),
        "NumChildren": num_children,
        "OwnsCar": 1 if form["owns_car"] == "Y" else 0,
        "OwnsRealty": 1 if form["owns_realty"] == "Y" else 0,
        "PastDefaults": past_defaults,
        "OverdueMonths": int(form["overdue_months"]),
        "CreditUtilization": credit_utilization,
        "ExistingCreditLines": int(form["existing_credit_lines"]),
        "YearsEmployed": round(years_employed, 2),
        "IncomePerFamilyMember": round(annual_income / (num_children + 1), 2),
        "HasPastDefault": int(past_defaults > 0),
        "HighUtilization": int(credit_utilization > 0.5),
        "IncomeType": form["income_type"],
        "EducationType": form["education_type"],
        "FamilyStatus": form["family_status"],
        "HousingType": form["housing_type"],
    }
    row = pd.DataFrame([raw])
    row_encoded = pd.get_dummies(row, columns=["IncomeType", "EducationType", "FamilyStatus", "HousingType"])

    # align to the exact training-time feature set
    for col in FEATURE_COLUMNS:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    row_encoded = row_encoded[FEATURE_COLUMNS]
    return row_encoded


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        income_types=INCOME_TYPES,
        education_types=EDUCATION_TYPES,
        family_statuses=FAMILY_STATUSES,
        housing_types=HOUSING_TYPES,
        model_name=MODEL_NAME,
    )


@app.route("/predict", methods=["POST"])
def predict():
    row = build_feature_row(request.form)

    if MODEL_NAME == "Logistic Regression":
        row_for_model = scaler.transform(row)
    else:
        row_for_model = row

    prediction = int(best_model.predict(row_for_model)[0])
    probability = float(best_model.predict_proba(row_for_model)[0][1])

    result = "Approved" if prediction == 1 else "Rejected"
    applicant_name = request.form.get("name", "Applicant").strip() or "Applicant"
    return render_template(
        "result.html",
        result=result,
        name=applicant_name,
        probability=round(probability * 100, 2),
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
