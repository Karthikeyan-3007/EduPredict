"""
Student Performance Predictor
Usage:
  python predict.py                        # predict for sample students
  python predict.py --interactive          # enter values manually
"""
import argparse, joblib, os
import numpy as np
import pandas as pd

def load_artefacts():
    model   = joblib.load("models/best_model.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    le      = joblib.load("models/label_encoder.pkl")
    features= joblib.load("models/features.pkl")
    return model, scaler, le, features

def predict_student(model, scaler, le, features, student_data: dict):
    df = pd.DataFrame([student_data])[features]
    X  = scaler.transform(df)
    pred_enc   = model.predict(X)[0]
    prob       = model.predict_proba(X)[0]
    label      = le.inverse_transform([pred_enc])[0]
    confidence = prob.max() * 100
    return label, confidence, dict(zip(le.classes_, (prob*100).round(1)))

def print_result(name, data, label, confidence, probs):
    grade_map = {"Good": "A/B", "Average": "C", "Poor": "D/F"}
    bar = "█" * int(confidence // 5) + "░" * (20 - int(confidence // 5))
    print(f"\n  Student : {name}")
    print(f"  Input   : Attendance={data['attendance_pct']}%  "
          f"Assignment avg={data['assignment_avg']:.1f}  "
          f"Midterm={data['midterm_score']}")
    print(f"  ▶ Predicted: {label} ({grade_map[label]})  "
          f"Confidence: {confidence:.1f}%")
    print(f"  [{bar}] {confidence:.1f}%")
    print(f"  Probabilities: " +
          "  ".join(f"{k}={v}%" for k, v in probs.items()))

def sample_predictions(model, scaler, le, features):
    students = [
        {"name": "Alice (High performer)",
         "attendance_pct": 95, "assignment1": 90, "assignment2": 88,
         "assignment3": 92, "assignment4": 85, "assignment5": 91,
         "assignment_avg": 89.2, "midterm_score": 88, "final_exam_score": 90},
        {"name": "Bob (Average student)",
         "attendance_pct": 72, "assignment1": 65, "assignment2": 70,
         "assignment3": 68, "assignment4": 72, "assignment5": 67,
         "assignment_avg": 68.4, "midterm_score": 63, "final_exam_score": 65},
        {"name": "Charlie (Struggling)",
         "attendance_pct": 42, "assignment1": 35, "assignment2": 40,
         "assignment3": 38, "assignment4": 42, "assignment5": 36,
         "assignment_avg": 38.2, "midterm_score": 35, "final_exam_score": 38},
    ]
    print("\n" + "="*60)
    print("  SAMPLE PREDICTIONS")
    print("="*60)
    for s in students:
        name = s.pop("name")
        label, conf, probs = predict_student(model, scaler, le, features, s)
        print_result(name, s, label, conf, probs)

def interactive_predict(model, scaler, le, features):
    print("\n" + "="*60)
    print("  INTERACTIVE PREDICTION")
    print("="*60)
    name = input("  Student name: ").strip() or "Student"
    att  = float(input("  Attendance %  (0-100): ") or 75)
    a1   = float(input("  Assignment 1  (0-100): ") or 70)
    a2   = float(input("  Assignment 2  (0-100): ") or 70)
    a3   = float(input("  Assignment 3  (0-100): ") or 70)
    a4   = float(input("  Assignment 4  (0-100): ") or 70)
    a5   = float(input("  Assignment 5  (0-100): ") or 70)
    mid  = float(input("  Midterm score (0-100): ") or 65)
    fin  = float(input("  Final exam    (0-100): ") or 65)
    avg  = round((a1+a2+a3+a4+a5)/5, 1)
    data = {"attendance_pct": att, "assignment1": a1, "assignment2": a2,
            "assignment3": a3, "assignment4": a4, "assignment5": a5,
            "assignment_avg": avg, "midterm_score": mid, "final_exam_score": fin}
    label, conf, probs = predict_student(model, scaler, le, features, data)
    print_result(name, data, label, conf, probs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model, scaler, le, features = load_artefacts()
    if args.interactive:
        interactive_predict(model, scaler, le, features)
    else:
        sample_predictions(model, scaler, le, features)
