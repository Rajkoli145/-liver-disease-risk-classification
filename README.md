# Liver Disease Risk Classification Using Machine Learning
**Case Study 131 | B.Tech CSE Semester V**

A predictive clinical decision support system that uses patient demographic information and clinical measurements from Liver Function Tests (LFT) to classify liver disease risk.

---

## 📌 Project Overview
- **Dataset**: Indian Liver Patient Dataset (583 patient records)
- **Problem Type**: Binary Classification (1: Liver Disease, 0: Healthy)
- **Objective**: Identify key clinical biomarkers, compare multiple classification models prioritizing diagnostic Recall (minimizing False Negatives), and deploy an interactive decision support tool.

---

## 🔬 Machine Learning Algorithms Evaluated
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**
4. **Random Forest** *(Best Overall Model)*
5. **Gradient Boosting**
6. **Support Vector Machine (SVM)**

---

## 📊 Comparative Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Selected)** | **71.79%** | **75.00%** | **90.36%** | **0.8197** |
| **Logistic Regression** | 73.50% | 74.07% | 96.39% | 0.8377 |
| **SVM (RBF Kernel)** | 70.94% | 70.94% | 100.00% | 0.8300 |
| **Gradient Boosting** | 68.38% | 73.00% | 87.95% | 0.7978 |
| **KNN (k=5)** | 68.38% | 73.96% | 85.54% | 0.7933 |
| **Decision Tree** | 59.83% | 70.93% | 73.49% | 0.7219 |

---

## 🔑 Key Findings & Clinical Predictors
- **Top Biomarkers**: Alkaline Phosphatase (ALP), Aspartate Aminotransferase (AST/SGOT), Alamine Aminotransferase (ALT/SGPT), Total Bilirubin, and Age.
- **Why Recall is Critical**: In medical screening, missing a diseased patient (False Negative) is life-threatening. A False Positive simply prompts a safe secondary test (e.g. ultrasound). Hence, high sensitivity/recall is prioritized.

---

## 🚀 How to Run

### 1. Run the Jupyter Notebook:
```bash
jupyter notebook liver_disease_classification.ipynb
```

### 2. Launch the Streamlit Web Application:
```bash
streamlit run app.py
```
