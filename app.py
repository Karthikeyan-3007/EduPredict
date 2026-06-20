"""
Student Performance Prediction - Flask Web App
Run: python app.py
Visit: http://localhost:5000
"""
import os, sys, json, base64
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load models ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_models():
    try:
        model    = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        le       = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
        return model, scaler, le, features
    except FileNotFoundError:
        return None, None, None, None

model, scaler, le, features = load_models()

def img_to_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_ready = model is not None
    return render_template("index.html", model_ready=model_ready)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 503

    data = request.get_json()
    try:
        a1 = float(data["assignment1"])
        a2 = float(data["assignment2"])
        a3 = float(data["assignment3"])
        a4 = float(data["assignment4"])
        a5 = float(data["assignment5"])
        avg = round((a1+a2+a3+a4+a5)/5, 2)

        student = {
            "attendance_pct":   float(data["attendance"]),
            "assignment1":      a1,
            "assignment2":      a2,
            "assignment3":      a3,
            "assignment4":      a4,
            "assignment5":      a5,
            "assignment_avg":   avg,
            "midterm_score":    float(data["midterm"]),
            "final_exam_score": float(data["final_exam"]),
        }

        df_in = pd.DataFrame([student])[features]
        X     = scaler.transform(df_in)
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        label = le.inverse_transform([pred])[0]
        conf  = float(proba.max() * 100)

        probs_dict = {cls: round(float(p*100), 1) for cls, p in zip(le.classes_, proba)}

        grade_map   = {"Good": "A / B", "Average": "C", "Poor": "D / F"}
        advice_map  = {
            "Good":    "Excellent performance! Keep maintaining attendance and assignment quality.",
            "Average": "There's room to improve. Focus on assignments and don't miss classes.",
            "Poor":    "Immediate attention needed. Increase attendance and seek academic support.",
        }
        color_map   = {"Good": "#27ae60", "Average": "#f39c12", "Poor": "#e74c3c"}

        composite = round(
            0.20 * student["attendance_pct"] +
            0.25 * avg +
            0.25 * student["midterm_score"] +
            0.30 * student["final_exam_score"], 1
        )

        return jsonify({
            "performance":  label,
            "grade":        grade_map[label],
            "confidence":   round(conf, 1),
            "probabilities": probs_dict,
            "advice":       advice_map[label],
            "color":        color_map[label],
            "composite":    composite,
            "avg_assignment": avg,
        })

    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/dashboard")
def dashboard():
    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
    charts = {}
    chart_files = {
        "dist":       "01_performance_distribution.png",
        "heatmap":    "02_correlation_heatmap.png",
        "scatter":    "03_attendance_vs_score.png",
        "confusion":  "04_confusion_matrix.png",
        "comparison": "05_model_comparison.png",
        "boxplot":    "06_score_boxplots.png",
    }
    for key, fname in chart_files.items():
        b64 = img_to_b64(os.path.join(charts_dir, fname))
        if b64:
            charts[key] = f"data:image/png;base64,{b64}"

    # Stats from saved data
    stats = {}
    try:
        eval_data = joblib.load(os.path.join(MODEL_DIR, "eval_data.pkl"))
        results   = joblib.load(os.path.join(MODEL_DIR, "all_results.pkl"))
        df        = eval_data["df"]
        best_name = eval_data["best_name"]
        best_acc  = results[best_name]["accuracy"]
        stats = {
            "total":      len(df),
            "good":       int((df["performance"] == "Good").sum()),
            "average":    int((df["performance"] == "Average").sum()),
            "poor":       int((df["performance"] == "Poor").sum()),
            "avg_attend": round(df["attendance_pct"].mean(), 1),
            "avg_score":  round(df["composite_score"].mean(), 1),
            "best_model": best_name,
            "accuracy":   round(best_acc * 100, 1),
            "model_rows": [
                {"name": n, "acc": round(r["accuracy"]*100,1), "f1": round(r["f1_score"]*100,1)}
                for n, r in results.items()
            ]
        }
    except Exception:
        pass

    return render_template("dashboard.html", charts=charts, stats=stats)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🎓 Student Performance Prediction App")
    print("  Visit: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
