"""
run_all.py – One-shot pipeline: generate data → train → visualize → report
"""
import subprocess, sys

steps = [
    ("Train model",       ["python", "train_model.py"]),
    ("Generate charts",   ["python", "visualize.py"]),
    ("Generate report",   ["python", "generate_report.py"]),
    ("Sample predictions",["python", "predict.py"]),
]

for title, cmd in steps:
    print(f"\n{'='*60}")
    print(f"  STEP: {title}")
    print(f"{'='*60}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"\n✗ Step failed: {title}")
        sys.exit(1)

print("\n" + "="*60)
print("  ALL STEPS COMPLETE")
print("  • data/student_data.csv  – dataset")
print("  • models/                – trained models")
print("  • charts/                – 6 PNG visualizations")
print("  • reports/report.html    – full HTML report")
print("  Run: python predict.py --interactive  for live prediction")
print("="*60)
