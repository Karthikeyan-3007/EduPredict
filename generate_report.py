"""
Generate an HTML report summarising the project results.
"""
import os, base64, joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def generate_report():
    results   = joblib.load("models/all_results.pkl")
    eval_data = joblib.load("models/eval_data.pkl")
    df        = eval_data["df"]
    le        = eval_data["le"]
    y_test    = eval_data["y_test"]
    best_name = eval_data["best_name"]
    best_res  = results[best_name]

    report_dict = classification_report(
        y_test, best_res["y_pred"], target_names=le.classes_, output_dict=True
    )

    rows_model = ""
    for name, r in results.items():
        bold = "font-weight:700;background:#eaf4fb;" if name == best_name else ""
        star = " ⭐" if name == best_name else ""
        rows_model += (
            f"<tr style='{bold}'><td>{name}{star}</td>"
            f"<td>{r['accuracy']:.3f}</td><td>{r['f1_score']:.3f}</td>"
            f"<td>{r['cv_mean']:.3f} ± {r['cv_std']:.3f}</td></tr>\n"
        )

    rows_class = ""
    for cls in le.classes_:
        m = report_dict[cls]
        rows_class += (
            f"<tr><td>{cls}</td><td>{m['precision']:.3f}</td>"
            f"<td>{m['recall']:.3f}</td><td>{m['f1-score']:.3f}</td>"
            f"<td>{int(m['support'])}</td></tr>\n"
        )

    charts = [
        ("01_performance_distribution.png", "Performance Distribution"),
        ("02_correlation_heatmap.png",       "Feature Correlation Heatmap"),
        ("03_attendance_vs_score.png",        "Attendance vs Composite Score"),
        ("04_confusion_matrix.png",           "Confusion Matrix"),
        ("05_model_comparison.png",           "Model Comparison"),
        ("06_score_boxplots.png",             "Score Distributions"),
    ]
    chart_html = ""
    for fname, title in charts:
        path = f"charts/{fname}"
        if os.path.exists(path):
            src = f"data:image/png;base64,{img_b64(path)}"
            chart_html += f"""
            <div class="chart-card">
              <h3>{title}</h3>
              <img src="{src}" alt="{title}">
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Student Performance Prediction – Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#f4f7fa;color:#2c3e50}}
  header{{background:linear-gradient(135deg,#2980b9,#1abc9c);color:#fff;padding:38px 40px;border-radius:0 0 16px 16px}}
  header h1{{font-size:2rem;margin-bottom:6px}}
  header p{{opacity:.85;font-size:1rem}}
  main{{max-width:1100px;margin:32px auto;padding:0 20px}}
  section{{background:#fff;border-radius:12px;padding:28px;margin-bottom:26px;
           box-shadow:0 2px 10px rgba(0,0,0,.07)}}
  h2{{font-size:1.3rem;color:#2980b9;margin-bottom:18px;border-left:4px solid #2980b9;padding-left:10px}}
  h3{{font-size:1rem;color:#555;margin-bottom:10px}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:6px}}
  .kpi{{background:#eaf4fb;border-radius:10px;padding:18px;text-align:center}}
  .kpi .val{{font-size:2rem;font-weight:700;color:#2980b9}}
  .kpi .lbl{{font-size:.82rem;color:#666;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;font-size:.92rem}}
  th{{background:#2980b9;color:#fff;padding:10px 14px;text-align:left}}
  td{{padding:9px 14px;border-bottom:1px solid #e8ecf0}}
  tr:last-child td{{border:none}}
  tr:hover td{{background:#f0f7ff}}
  .chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:22px}}
  .chart-card{{background:#f9fbfd;border:1px solid #dce6f0;border-radius:10px;padding:18px}}
  .chart-card img{{width:100%;border-radius:6px;margin-top:8px}}
  footer{{text-align:center;padding:20px;color:#888;font-size:.85rem}}
</style>
</head>
<body>
<header>
  <h1>🎓 Student Performance Prediction</h1>
  <p>Machine Learning Analysis – Attendance · Assignments · Marks</p>
</header>
<main>

<!-- KPIs -->
<section>
  <h2>Dataset Overview</h2>
  <div class="kpi-grid">
    <div class="kpi"><div class="val">{len(df)}</div><div class="lbl">Total Students</div></div>
    <div class="kpi"><div class="val">{df['performance'].value_counts()['Good']}</div><div class="lbl">Good Performers</div></div>
    <div class="kpi"><div class="val">{df['performance'].value_counts()['Average']}</div><div class="lbl">Average Performers</div></div>
    <div class="kpi"><div class="val">{df['performance'].value_counts()['Poor']}</div><div class="lbl">Poor Performers</div></div>
    <div class="kpi"><div class="val">{df['attendance_pct'].mean():.1f}%</div><div class="lbl">Avg Attendance</div></div>
    <div class="kpi"><div class="val">{df['composite_score'].mean():.1f}</div><div class="lbl">Avg Composite Score</div></div>
  </div>
</section>

<!-- Model Comparison -->
<section>
  <h2>Model Comparison</h2>
  <table>
    <thead><tr><th>Model</th><th>Accuracy</th><th>F1 Score</th><th>CV Score</th></tr></thead>
    <tbody>{rows_model}</tbody>
  </table>
</section>

<!-- Classification Report -->
<section>
  <h2>Best Model – Classification Report ({best_name})</h2>
  <table>
    <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Support</th></tr></thead>
    <tbody>{rows_class}</tbody>
  </table>
</section>

<!-- Charts -->
<section>
  <h2>Visualizations</h2>
  <div class="chart-grid">{chart_html}</div>
</section>

</main>
<footer>Student Performance Prediction Project · Python · scikit-learn · matplotlib · seaborn</footer>
</body>
</html>"""

    with open("reports/report.html", "w") as f:
        f.write(html)
    print("✓ Report saved to reports/report.html")

if __name__ == "__main__":
    generate_report()
