import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Liver Disease Risk Classifier",
    layout="centered"
)

# Custom minimal typography and structure
st.markdown("""
<style>
    .block-container {
        max-width: 780px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, p, span, label {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .header-tag {
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }
    .title-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.4rem;
        line-height: 1.25;
    }
    .subtitle-text {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 2rem;
        line-height: 1.5;
    }
    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #334155;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card {
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.5rem;
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .result-prob {
        font-size: 0.9rem;
        color: #475569;
    }
    @media (prefers-color-scheme: dark) {
        .title-text { color: #f8fafc; }
        .subtitle-text { color: #94a3b8; }
        .section-title { color: #cbd5e1; border-color: #334155; }
        .result-card { background-color: #0f172a; border-color: #1e293b; }
        .result-prob { color: #94a3b8; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    clf = joblib.load('best_model.pkl')
    cols = joblib.load('feature_names.pkl')
    return clf, cols

model, feature_names = load_model()

# Header
st.markdown('<div class="header-tag">Clinical Decision Support &middot; Case Study 131</div>', unsafe_allow_html=True)
st.markdown('<div class="title-text">Liver Disease Risk Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Enter standard patient demographics and serum liver function test (LFT) values to assess risk probability.</div>', unsafe_allow_html=True)

# Form
with st.form("patient_assessment_form"):
    st.markdown('<div class="section-title">Patient Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=4, max_value=100, value=45, step=1)
    with c2:
        gender_selection = st.selectbox("Gender", options=["Male", "Female"])
        gender = 1 if gender_selection == "Male" else 0

    st.markdown('<div class="section-title">Serum Enzymes (IU/L)</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        alp = st.number_input("Alkaline Phosphatase (ALP)", min_value=10, max_value=3000, value=200, step=5)
    with c4:
        alt = st.number_input("ALT / SGPT", min_value=5, max_value=2500, value=35, step=1)
    with c5:
        ast = st.number_input("AST / SGOT", min_value=5, max_value=5000, value=40, step=1)

    st.markdown('<div class="section-title">Bilirubin & Proteins</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6:
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.1, max_value=80.0, value=1.0, step=0.1)
        total_proteins = st.number_input("Total Proteins (g/dL)", min_value=1.0, max_value=12.0, value=6.5, step=0.1)
    with c7:
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=30.0, value=0.4, step=0.1)
        albumin = st.number_input("Albumin (g/dL)", min_value=0.5, max_value=8.0, value=3.2, step=0.1)

    ag_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.1, max_value=5.0, value=0.95, step=0.05)

    submitted = st.form_submit_button("Compute Assessment", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([[
        age, gender, total_bilirubin, direct_bilirubin,
        alp, alt, ast, total_proteins, albumin, ag_ratio
    ]], columns=feature_names)

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    disease_prob = prob[1] * 100
    healthy_prob = prob[0] * 100

    if pred == 1:
        color = "#b91c1c"
        status_text = "Elevated Risk Detected"
        detail_text = f"The model classifies this profile as high risk for liver disease with a calculated probability of {disease_prob:.1f}%."
    else:
        color = "#15803d"
        status_text = "Low Risk Profile"
        detail_text = f"The model classifies this profile within standard non-disease parameters ({healthy_prob:.1f}% confidence)."

    st.markdown(f"""
    <div class="result-card" style="border-left: 4px solid {color};">
        <div class="result-title" style="color: {color};">{status_text}</div>
        <div class="result-prob">{detail_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Clean clinical indicator table
    indicators = []
    if total_bilirubin > 1.2:
        indicators.append(("Total Bilirubin", f"{total_bilirubin} mg/dL", "Above reference (< 1.2)"))
    if alp > 147:
        indicators.append(("Alkaline Phosphatase", f"{alp} IU/L", "Above reference (44 - 147)"))
    if alt > 56:
        indicators.append(("ALT / SGPT", f"{alt} IU/L", "Above reference (7 - 56)"))
    if ast > 40:
        indicators.append(("AST / SGOT", f"{ast} IU/L", "Above reference (10 - 40)"))
    if albumin < 3.5:
        indicators.append(("Albumin", f"{albumin} g/dL", "Below reference (3.5 - 5.5)"))

    if indicators:
        st.markdown('<div class="section-title">Clinical Flags</div>', unsafe_allow_html=True)
        summary_table = pd.DataFrame(indicators, columns=["Biomarker", "Patient Value", "Reference Status"])
        st.table(summary_table)
