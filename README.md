# Employee Performance Predictor using Data Analytics 📊🏢

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-brightgreen.svg)

## 📌 Project Overview
The **Employee Performance Predictor** is a fully scalable Machine Learning pipeline built to assist Human Resource departments in analytically forecasting employee performance. Rather than using subjective evaluations, this system takes multidimensional parameters (experience, education, department, salary, and training metrics) to classify an employee's future performance as **Low, Medium, or High**.

This project relies on programmatic synthetic data generation that identically mimics real corporate patterns, maintaining maximum data privacy while demonstrating robust predictive analytical capabilities.

---

## 🚀 Problem Statement & Business Value
**Problem:** Annual employee reviews are subjective, biased, and often delayed. Managers lack actionable data to intervene and assist struggling employees before it impacts company productivity.

**Business Value:**
1. **Talent Retention:** Spot highly efficient workers and reward them properly.
2. **Predictive Intervention:** Identify "Low" trajectory performers explicitly mapped to lacking variables (like "Training Hours").
3. **Data-Driven Culture:** Elevate the Human Resources department from an administrative function to a key strategic player in business operations.

---

## 🛠 Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest Classifier, Scaling, Label Encoding)
- **Data Visualization:** Seaborn, Matplotlib

---

## 🏗 Architecture & Workflow

```text
Synthetic Dataset Generator → Data Preprocessing Pipeline → Exploratory Data Analysis (EDA) → Random Forest Classifier → Predicted Classes & Feature Importances
```
1. **Data Load:** `data_generation.py` constructs a realistic DataFrame simulating hundreds of corporate employees across IT, Sales, HR, etc.
2. **Preprocessing:** Clean columns, map categorical variables via One-Hot-Encoding.
3. **Model Flow:** Data scaled systematically. Splits into 80/20 train-test arrays.
4. **Evaluation:** Confirmed with Accuracy Metrics and visually understood through Confusion Matrices.

---

## 💻 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Employee-Performance-Predictor.git
   cd Employee-Performance-Predictor
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. **Install standard dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Execute the Intelligence Pipeline:**
   ```bash
   python main.py
   ```
   *Note: This command sequentially runs the synthetic data builder, the EDA graphical generator, and the model trainer.*

---

## 📈 Results & Visuals

The model outputs highly accurate estimations across three different target variables successfully determining feature importance thresholds.

### Correlation Heatmap
*(Insert screenshot from `images/correlation_heatmap.png` here on GitHub)*

### Training Hours vs Performance Ratings
*(Insert screenshot from `images/training_vs_performance.png` here on GitHub)*

### Classification Confusion Matrix
*(Insert screenshot from `images/confusion_matrix.png` here on GitHub)*

For extensive documentation, review the [Project Guide](./docs/Project_Guide.md) focusing on virtual simulation and deeper business intelligence setups!
