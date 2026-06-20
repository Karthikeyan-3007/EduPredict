"""
Student Performance Prediction – Visualizations
Produces 6 charts saved to charts/
"""
import os, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
os.makedirs("charts", exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
PALETTE  = {"Good": "#2ecc71", "Average": "#f39c12", "Poor": "#e74c3c"}
BLUE     = "#3498db"
DARK     = "#2c3e50"
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── load artefacts ────────────────────────────────────────────────────────────
data    = joblib.load("models/eval_data.pkl")
results = joblib.load("models/all_results.pkl")
df      = data["df"]
le      = data["le"]
y_test  = data["y_test"]
best_name = data["best_name"]
best_res  = results[best_name]

print("Generating charts …")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 – Performance Distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
counts = df["performance"].value_counts().reindex(["Good", "Average", "Poor"])
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE[k] for k in counts.index], width=0.55, edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
            str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_title("Student Performance Distribution", fontsize=15, fontweight="bold", color=DARK)
ax.set_xlabel("Performance Category", fontsize=12)
ax.set_ylabel("Number of Students", fontsize=12)
ax.set_ylim(0, counts.max() * 1.18)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/01_performance_distribution.png", dpi=150)
plt.close()
print("  ✓ Chart 1 – performance distribution")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 – Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
feat_cols = ["attendance_pct","assignment_avg","midterm_score",
             "final_exam_score","composite_score"]
corr = df[feat_cols].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", color=DARK)
plt.tight_layout()
plt.savefig("charts/02_correlation_heatmap.png", dpi=150)
plt.close()
print("  ✓ Chart 2 – correlation heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 – Attendance vs Composite Score (scatter)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for perf, grp in df.groupby("performance"):
    ax.scatter(grp["attendance_pct"], grp["composite_score"],
               c=PALETTE[perf], alpha=0.65, label=perf, edgecolors="none", s=50)
m, b = np.polyfit(df["attendance_pct"], df["composite_score"], 1)
xs = np.linspace(df["attendance_pct"].min(), df["attendance_pct"].max(), 100)
ax.plot(xs, m*xs + b, color=DARK, lw=1.8, ls="--", label="Trend")
ax.set_xlabel("Attendance %", fontsize=12)
ax.set_ylabel("Composite Score", fontsize=12)
ax.set_title("Attendance vs Composite Score", fontsize=14, fontweight="bold", color=DARK)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/03_attendance_vs_score.png", dpi=150)
plt.close()
print("  ✓ Chart 3 – attendance vs score")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 – Confusion Matrix of best model
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, best_res["y_pred"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax)
ax.set_title(f"Confusion Matrix – {best_name}", fontsize=13, fontweight="bold", color=DARK)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
plt.tight_layout()
plt.savefig("charts/04_confusion_matrix.png", dpi=150)
plt.close()
print("  ✓ Chart 4 – confusion matrix")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 – Model Accuracy Comparison (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[m]["accuracy"]  for m in model_names]
f1_scores   = [results[m]["f1_score"]  for m in model_names]

x = np.arange(len(model_names))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, accuracies, w, label="Accuracy", color=BLUE,      alpha=0.85, edgecolor="white")
b2 = ax.bar(x + w/2, f1_scores,  w, label="F1 Score", color="#9b59b6", alpha=0.85, edgecolor="white")
for bar in list(b1) + list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison – Accuracy & F1 Score", fontsize=14, fontweight="bold", color=DARK)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/05_model_comparison.png", dpi=150)
plt.close()
print("  ✓ Chart 5 – model comparison")

# ─────────────────────────────────────────────────────────────────────────────
# Chart 6 – Assignment Avg Boxplot by Performance
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
order = ["Poor", "Average", "Good"]

sns.boxplot(data=df, x="performance", y="assignment_avg",
            order=order, palette=PALETTE, ax=axes[0],
            linewidth=1.2, fliersize=3)
axes[0].set_title("Assignment Avg by Performance", fontsize=12, fontweight="bold", color=DARK)
axes[0].set_xlabel(""); axes[0].set_ylabel("Assignment Average (%)", fontsize=11)

sns.boxplot(data=df, x="performance", y="midterm_score",
            order=order, palette=PALETTE, ax=axes[1],
            linewidth=1.2, fliersize=3)
axes[1].set_title("Midterm Score by Performance", fontsize=12, fontweight="bold", color=DARK)
axes[1].set_xlabel(""); axes[1].set_ylabel("Midterm Score (%)", fontsize=11)

for ax in axes:
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Score Distributions by Performance Category",
             fontsize=14, fontweight="bold", color=DARK, y=1.01)
plt.tight_layout()
plt.savefig("charts/06_score_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Chart 6 – score boxplots")

print("\nAll charts saved to charts/")
