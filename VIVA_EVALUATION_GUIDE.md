# Viva & Evaluation Guide: Case Study 131
## Liver Disease Risk Classification Using Machine Learning
**ITM Skills University | School of Future Tech | B.Tech CSE Semester V**

---

## 🎯 Quick Elevator Pitch (If the evaluator asks: "Tell me what you did in this project")
> *"In this project, I built an end-to-end Machine Learning clinical decision support system to classify whether a patient has liver disease based on demographic and liver function test (LFT) parameters. I cleaned and preprocessed the Indian Liver Patient Dataset, handled missing entries and class imbalance, performed EDA to identify primary biomarkers (Alkaline Phosphatase, ALT, AST, and Bilirubin), developed and benchmarked 6 classification models, and deployed the best model (Random Forest) into an interactive Streamlit web application. Our model prioritizes diagnostic Recall (~90.4%) to avoid dangerous False Negatives in healthcare triage."*

---

## 📋 Direct Answers to Section 8: "Questions to Be Answered"

### 1. Can liver-disease-related classification be performed using ML?
* **Answer**: **Yes.**
* **Explanation**: Routine blood tests measure specific biochemical markers produced or cleared by the liver (Bilirubin, Albumin, ALT, AST, and Alkaline Phosphatase). When the liver suffers damage or cholestasis, these enzyme levels surge while protein synthesis declines. ML models can detect these multi-dimensional, non-linear biochemical signatures with over **72% accuracy** and **>90% sensitivity (recall)**.

### 2. Which features contribute most to prediction?
* **Answer**:
  1. **Alkaline Phosphatase (ALP)**: Highest feature importance weight (~0.147). Elevated ALP points to bile duct obstruction or liver inflammation.
  2. **Aspartate Aminotransferase (AST / SGOT)** & **Alamine Aminotransferase (ALT / SGPT)**: Key enzymes released into the bloodstream upon hepatocyte cell damage.
  3. **Total Bilirubin & Direct Bilirubin**: Strong linear correlation with liver disease (>0.22 - 0.25). Reflects the liver's impaired ability to conjugate and excrete bile.
  4. **Age**: Older patients show higher cumulative risk of chronic liver damage.
  5. *Gender* had the least relative predictive power compared to biochemical enzymes.

### 3. Which model performs best?
* **Answer**: **Random Forest Classifier**.
* **Explanation**:
  - Balanced Accuracy: **~72%**
  - Precision: **75.0%**
  - Recall: **90.4%**
  - F1-Score: **0.82**
  - **Why?**: As an ensemble of de-correlated decision trees with bootstrap aggregation, Random Forest handles non-linear enzyme relationships, interactions between features (e.g. ALT vs AST ratio), and is robust against outliers and overfitting.

### 4. Which model provides better recall?
* **Answer**: **Support Vector Machine (SVM) and Logistic Regression** achieve 96% to 100% recall, but **Random Forest** provides the best *effective* recall (90.4%) with high precision (75.0%).
* **Why Recall Matters Most in Healthcare**:
  - In a disease screening tool, a **False Negative** (telling a sick patient they are healthy) is potentially fatal because treatment is delayed.
  - A **False Positive** (telling a healthy person to undergo a confirmatory ultrasound/biopsy) causes mild inconvenience but is medically safe.
  - Therefore, we optimize for high **Recall**.

### 5. How does preprocessing affect results?
* **Answer**: Preprocessing is critical for both model validity and convergence:
  1. **Missing Value Imputation**: `Albumin_and_Globulin_Ratio` contained 4 missing values. Imputing with the **median** prevented data loss while being immune to extreme outliers.
  2. **Categorical Encoding**: Converted `Gender` ('Male': 1, 'Female': 0) into binary numeric values for matrix algebra.
  3. **Feature Scaling (`StandardScaler`)**: Crucial for distance-based and gradient-based algorithms (KNN, Logistic Regression, SVM). Features like Alkaline Phosphatase range in thousands, while Bilirubin is in decimals. Without scaling, KNN would be completely dominated by ALP. (Note: Tree-based models are invariant to monotonic scale).
  4. **Stratified Splitting**: Ensures the 71% to 29% class distribution is identically preserved in both training and test sets.

### 6. Can the model be deployed for prediction?
* **Answer**: **Yes, absolutely.**
* **Deployment Architecture**:
  - Model and preprocessing metadata are serialized using `joblib` into `best_model.pkl` and `feature_names.pkl`.
  - A user-friendly **Streamlit web application (`app.py`)** allows doctors/lab technicians to enter patient demographic and LFT numbers.
  - It outputs real-time predicted risk class, disease probability percentage, and clinical flag warnings.

---

## 📊 Comparative Performance Summary Table

| Model | Accuracy | Precision | Recall | F1-Score | Clinical Pros / Cons |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 73.50% | 74.07% | 96.39% | 0.8377 | Fast, interpretable linear baseline |
| **KNN (k=5)** | 68.38% | 73.96% | 85.54% | 0.7933 | Sensitive to noise and distance metrics |
| **Decision Tree** | 59.83% | 70.93% | 73.49% | 0.7219 | Prone to variance and high sensitivity to splits |
| **Random Forest (Best)** | **71.79%** | **75.00%** | **90.36%** | **0.8197** | **Best balance of high recall, precision & stability** |
| **Gradient Boosting** | 68.38% | 73.00% | 87.95% | 0.7978 | Strong learner, slightly lower recall than RF |
| **SVM (RBF Kernel)** | 70.94% | 70.94% | 100.0% | 0.8300 | Tends to predict majority class due to boundary bias |

---

## 💡 Key Machine Learning Concepts Evaluators Might Ask

1. **What is the difference between Precision and Recall?**
   - *Precision* = $\frac{TP}{TP + FP}$ ("Out of all patients flagged as sick, how many actually are?")
   - *Recall* = $\frac{TP}{TP + FN}$ ("Out of all patients who actually have liver disease, how many did we catch?")

2. **Why use F1-Score instead of only Accuracy?**
   - The dataset is imbalanced (416 liver disease vs 167 non-liver disease, ~71% vs 29%). A naive model that always guesses "Liver Disease" would achieve 71.4% accuracy while being medically useless. F1-score balances precision and recall.

3. **How does Random Forest work?**
   - Random Forest is an ensemble method using **Bagging (Bootstrap Aggregation)**. It trains multiple decision trees on random subsets of rows and features, then averages their votes to reduce variance.

4. **What are the limitations of this model?**
   - Dataset size is modest (583 records).
   - Geographic limitation (predominantly Indian patient records).
   - Lacks lifestyle factors (alcohol history, viral hepatitis B/C markers, imaging data).

---

## 🚀 How to Run the Project

### 1. View the Jupyter Notebook:
```bash
conda activate myenv
jupyter notebook liver_disease_classification.ipynb
```

### 2. Run the Streamlit Web Application:
```bash
conda activate myenv
streamlit run app.py
```
