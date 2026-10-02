import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Liver Disease Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Load model and artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

model, feature_names = load_artifacts()

# Title and header
st.title("🩺 Liver Disease Risk Classification System")
st.markdown("""
**Case Study 131: Clinical Machine Learning Decision Support**  
Enter the patient's demographic information and routine Liver Function Test (LFT) values to assess liver disease risk.
""")

st.markdown("---")

# Layout into two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Demographics")
    age = st.slider("Age (years)", min_value=4, max_value=95, value=45, step=1)
    gender_str = st.radio("Gender", options=["Male", "Female"], horizontal=True)
    gender = 1 if gender_str == "Male" else 0

    st.subheader("🧪 Bilirubin & Protein Markers")
    total_bilirubin = st.number_input(
        "Total Bilirubin (mg/dL)", 
        min_value=0.1, max_value=80.0, value=1.0, step=0.1,
        help="Normal reference range: 0.2 to 1.2 mg/dL"
    )
    direct_bilirubin = st.number_input(
        "Direct Bilirubin (mg/dL)", 
        min_value=0.1, max_value=25.0, value=0.4, step=0.1,
        help="Normal reference range: 0.0 to 0.3 mg/dL"
    )
    total_protiens = st.number_input(
        "Total Proteins (g/dL)", 
        min_value=2.0, max_value=10.0, value=6.5, step=0.1,
        help="Normal reference range: 6.0 to 8.3 g/dL"
    )
    albumin = st.number_input(
        "Albumin (g/dL)", 
        min_value=0.5, max_value=6.0, value=3.2, step=0.1,
        help="Normal reference range: 3.5 to 5.5 g/dL"
    )
    ag_ratio = st.number_input(
        "Albumin and Globulin Ratio (A/G Ratio)", 
        min_value=0.1, max_value=3.0, value=0.95, step=0.05,
        help="Normal reference range: 1.0 to 2.0"
    )

with col2:
    st.subheader("⚡ Serum Liver Enzymes")
    alp = st.number_input(
        "Alkaline Phosphatase - ALP (IU/L)", 
        min_value=50, max_value=2500, value=200, step=5,
        help="Normal reference range: 44 to 147 IU/L"
    )
    alt = st.number_input(
        "Alamine Aminotransferase - ALT / SGPT (IU/L)", 
        min_value=5, max_value=2000, value=35, step=1,
        help="Normal reference range: 7 to 56 IU/L"
    )
    ast = st.number_input(
        "Aspartate Aminotransferase - AST / SGOT (IU/L)", 
        min_value=5, max_value=5000, value=40, step=1,
        help="Normal reference range: 10 to 40 IU/L"
    )

    st.markdown("### 🔍 Model Information")
    st.info("""
    - **Classifier**: Random Forest Classifier (Trained on ILPD)
    - **Sensitivity / Recall**: ~90.4%
    - **Optimization Goal**: Minimizing False Negatives in diagnostic triage.
    """)

st.markdown("---")

# Predict button
if st.button("🔮 Predict Liver Disease Risk", type="primary", use_container_width=True):
    input_data = pd.DataFrame([[
        age, gender, total_bilirubin, direct_bilirubin, 
        alp, alt, ast, total_protiens, albumin, ag_ratio
    ]], columns=feature_names)
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    risk_score = probabilities[1] * 100
    healthy_score = probabilities[0] * 100

    st.subheader("📊 Diagnostic Assessment")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.error(f"🚨 **High Risk of Liver Disease Detected**")
            st.metric(label="Predicted Class", value="Patient At Risk (Class 1)")
            st.warning(f"**Confidence / Disease Risk Probability**: **{risk_score:.1f}%**")
        else:
            st.success(f"✅ **Low Risk / Healthy Profile (Class 0)**")
            st.metric(label="Predicted Class", value="Healthy (Class 0)")
            st.info(f"**Healthy Confidence Probability**: **{healthy_score:.1f}%**")

    with res_col2:
        st.write("**Biomarker Summary:**")
        reasons = []
        if total_bilirubin > 1.2:
            reasons.append(f"• High Total Bilirubin ({total_bilirubin} mg/dL vs normal < 1.2)")
        if alp > 147:
            reasons.append(f"• Elevated Alkaline Phosphatase ({alp} IU/L vs normal < 147)")
        if alt > 56:
            reasons.append(f"• Elevated ALT / SGPT ({alt} IU/L vs normal < 56)")
        if ast > 40:
            reasons.append(f"• Elevated AST / SGOT ({ast} IU/L vs normal < 40)")
        if albumin < 3.5:
            reasons.append(f"• Low Albumin ({albumin} g/dL vs normal > 3.5)")

        if reasons:
            st.write("Abnormal markers detected:")
            for r in reasons:
                st.write(r)
        else:
            st.write("All primary biomarkers are within standard baseline ranges.")

st.markdown("---")
st.caption("Developed for ITM Skills University | School of Future Tech | B.Tech CSE Semester V")
