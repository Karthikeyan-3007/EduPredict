"""
Student Performance Prediction - ML Model Training
Features: attendance, assignment scores, midterm
Target: performance (Good / Average / Poor) + composite score
"""
import os
import warnings
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.insert(0, os.path.dirname(__file__))

from data.generate_data import generate_student_data
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ── 1. Load / generate data ─────────────────────────────────────────────────
print("=" * 60)
print("  STUDENT PERFORMANCE PREDICTION  ")
print("=" * 60)

df = generate_student_data(500)
df.to_csv("data/student_data.csv", index=False)
print(f"\n✓ Dataset: {len(df)} students, {df.shape[1]} features")
print(f"  Performance distribution:\n{df['performance'].value_counts().to_string()}\n")

# ── 2. Feature engineering ───────────────────────────────────────────────────
FEATURES = [
    "attendance_pct",
    "assignment1", "assignment2", "assignment3", "assignment4", "assignment5",
    "assignment_avg",
    "midterm_score",
    "final_exam_score",
]

X = df[FEATURES]
y = df["performance"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 3. Train & compare models ────────────────────────────────────────────────
models = {
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                  SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n── Model Comparison ────────────────────────────────────")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    cv_s = cross_val_score(model, X_scaled, y_enc, cv=cv, scoring="accuracy")
    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1_score": f1,
        "cv_mean": cv_s.mean(),
        "cv_std":  cv_s.std(),
        "y_pred":  y_pred,
    }
    print(f"  {name:<25}  Acc={acc:.3f}  F1={f1:.3f}  CV={cv_s.mean():.3f}±{cv_s.std():.3f}")

# ── 4. Best model ────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best      = results[best_name]
print(f"\n✓ Best model: {best_name}  (accuracy={best['accuracy']:.3f})\n")
print("── Classification Report ───────────────────────────────")
print(classification_report(y_test, best["y_pred"], target_names=le.classes_))

# ── 5. Persist artefacts ─────────────────────────────────────────────────────
joblib.dump(best["model"], "models/best_model.pkl")
joblib.dump(scaler,        "models/scaler.pkl")
joblib.dump(le,            "models/label_encoder.pkl")
joblib.dump(FEATURES,      "models/features.pkl")
joblib.dump(results,       "models/all_results.pkl")
joblib.dump({"X_test": X_test, "y_test": y_test, "X_train": X_train, "y_train": y_train,
             "df": df, "le": le, "best_name": best_name}, "models/eval_data.pkl")

print("\n✓ Models saved to models/")
