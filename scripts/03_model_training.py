"""
Epic 4 - Model Building
Story 1: Logistic Regression
Story 2: Random Forest
Story 3: Decision Tree
Story 4: Compare models, select the best, save it for deployment
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "/home/claude/credit_card_approval_prediction/data/credit_card_approval_clean.csv"
MODEL_DIR = "/home/claude/credit_card_approval_prediction/models"

df = pd.read_csv(DATA_PATH)

drop_cols = ["ApplicantID", "UserID", "HistoryID", "Approved"]
X = df.drop(columns=drop_cols)
y = df["Approved"]

FEATURE_COLUMNS = X.columns.tolist()  # needed by the Flask app at inference time

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
}

results = {}
fitted_models = {}

for name, model in models.items():
    # Logistic Regression benefits from scaled features; tree models don't need it
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    fitted_models[name] = model
    print(f"\n{name}:")
    for k, v in results[name].items():
        print(f"  {k}: {v}")

# ---------- Story 4: compare & select best ----------
best_name = max(results, key=lambda n: results[n]["f1_score"])
best_model = fitted_models[best_name]
print(f"\nBest performing model: {best_name}")

comparison_df = pd.DataFrame(results).T.sort_values("f1_score", ascending=False)
comparison_df.to_csv(f"{MODEL_DIR}/model_comparison.csv")
print("\nComparison table:\n", comparison_df)

# ---------- Save artifacts for deployment ----------
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f)
with open(f"{MODEL_DIR}/best_model_name.txt", "w") as f:
    f.write(best_name)

print(f"\nSaved best model ('{best_name}'), scaler, and feature list to {MODEL_DIR}")
