# 🌸 EduPredict — Student Performance Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-FF69B4?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-FF1493?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-FF69B4?style=for-the-badge)

<br>

> 🎯 **Built as a hands-on learning project to improve skills in Python, Machine Learning, Data Visualisation, and Full-Stack Web Development.**

<br>

**A full-stack Machine Learning web application that predicts student academic performance using attendance, assignment scores, and exam marks — featuring a beautiful pink-themed dark UI.**

[Overview](#-overview) • [Skills Learned](#-skills-learned) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [ML Models](#-ml-models) • [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

EduPredict is a **complete end-to-end machine learning project** built from scratch to strengthen real-world development skills. It covers everything from raw data generation and model training, to building a live web app with a REST API and a polished UI.

The app predicts a student's academic performance as **Good / Average / Poor** based on:
- 📅 Attendance percentage
- 📝 Assignment scores (5 assignments)
- 📖 Midterm exam score
- 🎓 Final exam score

---

## 🧠 Skills Learned

This project was built specifically to **practise and improve the following skills**:

### 🐍 Python
- Writing clean, modular Python scripts
- Using virtual environments and managing packages
- File I/O, data generation, and pipeline scripting

### 🤖 Machine Learning
- Understanding and comparing classification algorithms
- Feature engineering and data preprocessing
- Model training, evaluation, and cross-validation
- Saving and loading models with `joblib`
- Reading confusion matrices and classification reports

### 📊 Data Analysis & Visualisation
- Working with `pandas` DataFrames
- Creating analytical charts with `matplotlib` and `seaborn`
- Understanding feature correlation and data distributions
- Interpreting model performance visually

### 🌐 Web Development (Flask)
- Building a Flask web server with routes and REST API
- Connecting a Python ML backend to an HTML/CSS/JS frontend
- Handling JSON requests and responses
- Serving dynamic HTML with Jinja2 templates

### 🎨 Frontend (HTML / CSS / JavaScript)
- Designing a responsive dark-themed UI from scratch
- Building interactive forms with real-time validation
- Animated progress bars and probability breakdowns
- Async fetch API calls to the Flask backend

### 🗂️ Project Structure & Best Practices
- Organising a multi-file Python project cleanly
- Writing reusable, well-commented code
- Using `.gitignore` correctly for ML projects
- Writing proper documentation

---

## ✨ Features

- 🔮 **Live Prediction** — Instant ML-powered performance prediction from the browser
- 📊 **Analytics Dashboard** — KPI cards, model comparison table, and 6 charts
- 🤖 **4 ML Models Compared** — Random Forest, Gradient Boosting, Logistic Regression, SVM
- 📈 **6 Visualisations** — Distribution, heatmap, scatter, confusion matrix, boxplots, model comparison
- 💡 **Personalised Advice** — Each prediction includes an actionable recommendation
- 🌸 **Pink Dark UI** — Custom-designed professional interface
- ⚡ **REST API** — `/predict` endpoint returns JSON for easy integration
- 📄 **HTML Report** — Auto-generated standalone report with all charts embedded

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Core logic and ML |
| **Web Framework** | Flask 2.3+ | Backend server and API |
| **ML Library** | scikit-learn | Model training and evaluation |
| **Data Processing** | pandas, numpy | Dataset handling |
| **Visualisation** | matplotlib, seaborn | Chart generation |
| **Model Saving** | joblib | Persist trained models |
| **Frontend** | HTML5, CSS3, Vanilla JS | Pink-themed dark UI |
| **Templating** | Jinja2 (via Flask) | Dynamic HTML rendering |

---

## ⚙️ Installation

### Prerequisites
- Python **3.11** or **3.12**
  > ⚠️ Python 3.14 is NOT yet supported by pandas and scikit-learn. Use 3.11 or 3.12.
- pip

### 1. Clone the repository
```bash
git clone https://github.com/your-username/edupredict.git
cd edupredict/student_performance
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If you get a permissions error, run this once first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal line.

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the model (run once)
```bash
python run_all.py
```
This will:
- ✅ Generate a 500-student synthetic dataset
- ✅ Train and compare 4 ML models
- ✅ Save 6 PNG charts to `charts/`
- ✅ Generate a full HTML report in `reports/`
- ✅ Print sample predictions in the terminal

### Step 2 — Start the web app
```bash
python app.py
```

### Step 3 — Open in your browser
```
http://localhost:5000
```

> Press `Ctrl + C` in the terminal to stop the server.

### Optional — Terminal predictor (no browser needed)
```bash
python predict.py                  # runs 3 sample students
python predict.py --interactive    # enter your own values
```

---

## 🌐 Web App Pages

| Route | Page | What you can do |
|---|---|---|
| `/` | **Predict** | Enter student data and get instant ML prediction |
| `/dashboard` | **Dashboard** | View KPIs, model comparison, and all 6 charts |
| `/predict` *(POST)* | **API** | Send JSON, get prediction response |

### 🔌 API Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "attendance": 85,
    "assignment1": 78,
    "assignment2": 82,
    "assignment3": 75,
    "assignment4": 88,
    "assignment5": 80,
    "midterm": 74,
    "final_exam": 79
  }'
```

**Response:**
```json
{
  "performance":    "Good",
  "grade":          "A / B",
  "confidence":     97.3,
  "composite":      80.1,
  "avg_assignment": 80.6,
  "probabilities":  { "Good": 97.3, "Average": 2.6, "Poor": 0.1 },
  "advice":         "Excellent performance! Keep maintaining attendance and assignment quality.",
  "color":          "#ff6fb0"
}
```

---

## 🤖 ML Models

| Model | Accuracy | F1 Score | Notes |
|---|---|---|---|
| **Logistic Regression** ⭐ | **99.0%** | **98.5%** | Best — auto selected |
| Gradient Boosting | 95.0% | 95.0% | Strong performer |
| Random Forest | 91.0% | 90.6% | Good ensemble baseline |
| SVM (RBF kernel) | 92.0% | 91.8% | Solid on structured data |

> All models use **5-fold stratified cross-validation**. The best model is auto-saved to `models/best_model.pkl`.

---

## 📊 Composite Score Formula

```
Composite = (Attendance      × 20%)
          + (Assignment Avg  × 25%)
          + (Midterm Score   × 25%)
          + (Final Exam      × 30%)
```

| Component | Weight | Reason |
|---|---|---|
| Attendance | 20% | Consistency indicator |
| Assignments | 25% | Daily effort and understanding |
| Midterm | 25% | Mid-course comprehension |
| Final Exam | 30% | Most significant single assessment |

---

## 📈 Charts Generated

| # | Chart | What it shows |
|---|---|---|
| 1 | Performance Distribution | Count of Good / Average / Poor students |
| 2 | Correlation Heatmap | Which features are most related to each other |
| 3 | Attendance vs Score | Scatter plot showing attendance impact |
| 4 | Confusion Matrix | Where the model is correct and where it is wrong |
| 5 | Model Comparison | Accuracy and F1 bar chart for all 4 models |
| 6 | Score Boxplots | Assignment and midterm ranges by performance group |

---

## 📁 Project Structure

```
student_performance/
│
├── app.py                  # 🌐 Flask web server + REST API
├── train_model.py          # 🤖 Train and compare 4 ML models
├── visualize.py            # 📊 Generate 6 analytical charts
├── predict.py              # 💻 Terminal-based predictor
├── generate_report.py      # 📄 Standalone HTML report generator
├── run_all.py              # ⚡ One-shot full pipeline runner
├── requirements.txt        # 📦 All Python dependencies
├── README.md               # 📖 This file
│
├── data/
│   ├── generate_data.py    # Synthetic 500-student dataset generator
│   └── student_data.csv    # Generated CSV (created after run_all.py)
│
├── models/                 # Saved model files (created after training)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── features.pkl
│   ├── all_results.pkl
│   └── eval_data.pkl
│
├── templates/              # Flask HTML templates (pink theme)
│   ├── index.html          # Predict page
│   └── dashboard.html      # Analytics dashboard
│
├── charts/                 # 6 PNG charts (created after run_all.py)
│   ├── 01_performance_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_attendance_vs_score.png
│   ├── 04_confusion_matrix.png
│   ├── 05_model_comparison.png
│   └── 06_score_boxplots.png
│
└── reports/
    └── report.html         # Full embedded HTML report
```

---

## 🙅 .gitignore (recommended)

Create a `.gitignore` file in the project root:

```
.venv/
__pycache__/
*.pyc
*.pyo
models/*.pkl
data/student_data.csv
charts/*.png
reports/report.html
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
```bash
git checkout -b feature/your-feature
```
3. Commit your changes
```bash
git commit -m "Add: your feature description"
```
4. Push and open a Pull Request
```bash
git push origin feature/your-feature
```

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

<div align="center">

🌸 Built with passion to **learn, practise, and grow** as a developer 🌸

Made using Python · Flask · scikit-learn · matplotlib · seaborn

</div>
