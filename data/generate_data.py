"""
Generate synthetic student performance dataset.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

def generate_student_data(n=N):
    student_ids = [f"STU{str(i+1).zfill(4)}" for i in range(n)]
    names = [f"Student_{i+1}" for i in range(n)]

    attendance = np.clip(np.random.normal(75, 15, n), 0, 100).round(1)

    assignment_scores = []
    for _ in range(n):
        scores = np.clip(np.random.normal(70, 18, 5), 0, 100).round(1)
        assignment_scores.append(scores.tolist())

    midterm = np.clip(attendance * 0.3 + np.random.normal(50, 10, n), 0, 100).round(1)
    final_exam = np.clip(attendance * 0.25 + midterm * 0.4 + np.random.normal(20, 8, n), 0, 100).round(1)

    assignment_avg = np.array([np.mean(a) for a in assignment_scores])

    # Composite score
    composite = (
        0.20 * attendance +
        0.25 * assignment_avg +
        0.25 * midterm +
        0.30 * final_exam
    )

    def grade(score):
        if score >= 85: return "A"
        elif score >= 70: return "B"
        elif score >= 55: return "C"
        elif score >= 40: return "D"
        else: return "F"

    def performance_label(score):
        if score >= 70: return "Good"
        elif score >= 50: return "Average"
        else: return "Poor"

    df = pd.DataFrame({
        "student_id": student_ids,
        "name": names,
        "attendance_pct": attendance,
        "assignment1": [a[0] for a in assignment_scores],
        "assignment2": [a[1] for a in assignment_scores],
        "assignment3": [a[2] for a in assignment_scores],
        "assignment4": [a[3] for a in assignment_scores],
        "assignment5": [a[4] for a in assignment_scores],
        "assignment_avg": assignment_avg.round(1),
        "midterm_score": midterm,
        "final_exam_score": final_exam,
        "composite_score": composite.round(1),
        "grade": [grade(s) for s in composite],
        "performance": [performance_label(s) for s in composite],
    })
    return df

if __name__ == "__main__":
    df = generate_student_data()
    df.to_csv("student_data.csv", index=False)
    print(f"Generated {len(df)} student records.")
    print(df.head())
    print("\nPerformance distribution:")
    print(df["performance"].value_counts())
