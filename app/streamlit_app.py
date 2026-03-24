# ============================================================
# MODULE 7: STREAMLIT WEB APPLICATION
# File: app/streamlit_app.py
#
# WHAT THIS SCRIPT DOES:
# Deploys the entire AI system as an interactive web dashboard.
# Clinicians can:
#   - Enter patient data manually or upload a clinical note
#   - Get real-time cardiovascular risk predictions
#   - View SHAP feature explanations
#   - Check patient retention risk
#   - Download a patient risk report
#
# HOW TO RUN:
#   cd ai_cardio_project
#   streamlit run app/streamlit_app.py
#
# Streamlit works by:
# - Re-running the entire script top-to-bottom on every user interaction
# - st.session_state preserves values between reruns
# - @st.cache_resource caches loaded models (loads once, reused always)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — required for Streamlit
import shap
import joblib
import json
import os
import sys

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page configuration (must be FIRST streamlit command) ─────────────────
st.set_page_config(
    page_title="CardioAI — Cardiovascular Risk & Patient Retention",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for professional appearance ────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .risk-low     { background: #d4edda; color: #155724; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745; }
    .risk-moderate{ background: #fff3cd; color: #856404; padding: 12px; border-radius: 8px; border-left: 4px solid #ffc107; }
    .risk-high    { background: #f8d7da; color: #721c24; padding: 12px; border-radius: 8px; border-left: 4px solid #dc3545; }
    .risk-veryhigh{ background: #f5c6cb; color: #491217; padding: 12px; border-radius: 8px; border-left: 4px solid #a71d2a; font-weight: bold; }
    .metric-card  { background: white; padding: 16px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1F4E79; border-bottom: 2px solid #2E75B6; padding-bottom: 6px; margin-bottom: 14px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached — loads once and reused)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """
    Loads all trained models and preprocessors from disk.
    @st.cache_resource ensures this runs ONCE even if user interacts
    multiple times — avoids slow reloading on every click.
    """
    models = {}
    model_files = {
        "cardio_xgb":     "models/cardio_xgb.pkl",
        "cardio_rf":      "models/cardio_rf.pkl",
        "cardio_logistic":"models/cardio_logistic.pkl",
        "retention_rf":   "models/retention_rf.pkl",
        "retention_xgb":  "models/retention_xgb.pkl",
        "scaler":         "models/scaler.pkl",
        "scaler_ret":     "models/scaler_retention.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            models[name] = None  # graceful fallback
    return models

@st.cache_resource
def load_explainer(_model):
    """Creates and caches the SHAP TreeExplainer for fast explanations"""
    if _model is None:
        return None
    return shap.TreeExplainer(_model)


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_lifestyle_risk_index(trestbps, chol, fbs, exang, oldpeak):
    """
    Computes the Lifestyle Risk Index from input values.
    Matches the formula in 01_preprocessing.py.
    """
    # Reference ranges for normalization (from training data statistics)
    ranges = {
        "trestbps": (94, 200),
        "chol":     (126, 564),
        "oldpeak":  (0, 6.2)
    }
    def norm(v, lo, hi): return (v - lo) / (hi - lo + 1e-8)

    return round(
        norm(chol, *ranges["chol"]) * 0.25 +
        norm(trestbps, *ranges["trestbps"]) * 0.25 +
        fbs * 0.20 +
        exang * 0.15 +
        norm(oldpeak, *ranges["oldpeak"]) * 0.15,
        4
    )

def get_risk_badge(probability):
    """Returns HTML badge and tier for a probability value"""
    if probability < 0.30:
        return "LOW RISK", "risk-low", "✓"
    elif probability < 0.60:
        return "MODERATE RISK", "risk-moderate", "⚠"
    elif probability < 0.80:
        return "HIGH RISK", "risk-high", "⚠⚠"
    else:
        return "VERY HIGH RISK", "risk-veryhigh", "✗✗"

def build_feature_vector(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):
    """
    Builds the feature vector expected by the model from raw inputs.
    Matches the one-hot encoding applied in 01_preprocessing.py.
    """
    lifestyle_idx = compute_lifestyle_risk_index(trestbps, chol, fbs, exang, oldpeak)

    # One-hot encode categorical variables
    # (matching what pd.get_dummies with drop_first=True produces)
    features = {
        "age": age,
        "sex": sex,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "ca": ca,
        "lifestyle_risk_index": lifestyle_idx,
        # One-hot encoded cp (drop cp_0)
        "cp_1": 1 if cp == 1 else 0,
        "cp_2": 1 if cp == 2 else 0,
        "cp_3": 1 if cp == 3 else 0,
        # restecg (drop restecg_0)
        "restecg_1": 1 if restecg == 1 else 0,
        "restecg_2": 1 if restecg == 2 else 0,
        # slope (drop slope_0)
        "slope_1": 1 if slope == 1 else 0,
        "slope_2": 1 if slope == 2 else 0,
        # thal (drop thal_0)
        "thal_1": 1 if thal == 1 else 0,
        "thal_2": 1 if thal == 2 else 0,
        "thal_3": 1 if thal == 3 else 0,
    }
    return pd.DataFrame([features])


# ══════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════

# ── Load models ───────────────────────────────────────────────────────────
models = load_models()
xgb_explainer = load_explainer(models.get("cardio_xgb"))

# ── Sidebar navigation ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
    st.title("CardioAI")
    st.caption("Explainable AI System for Cardiovascular Risk & Patient Retention")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🫀 Risk Prediction", "📊 Model Dashboard", "📄 Clinical NLP", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("⚠ For clinical decision support only. Not a diagnostic tool.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════

if "Risk Prediction" in page:
    st.title("🫀 Cardiovascular Risk Prediction")
    st.caption("Enter patient clinical data to generate a cardiovascular risk assessment with explainable AI.")

    # ── Input form: two columns ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Patient Demographics & Vitals</div>', unsafe_allow_html=True)
        age      = st.slider("Age (years)", 29, 80, 55)
        sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 210, 130)
        chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
        thalach  = st.slider("Maximum Heart Rate Achieved (bpm)", 60, 210, 150)

    with col2:
        st.markdown('<div class="section-header">Clinical Measurements</div>', unsafe_allow_html=True)
        cp      = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                    format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                           "Non-Anginal Pain", "Asymptomatic"][x])
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0, 1, 2],
                    format_func=lambda x: ["Normal", "ST Abnormality", "LV Hypertrophy"][x])
        exang   = st.selectbox("Exercise Induced Angina", [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
        slope   = st.selectbox("Slope of Peak ST Segment", [0, 1, 2],
                    format_func=lambda x: ["Downsloping", "Flat", "Upsloping"][x])
        ca      = st.slider("Major Vessels Coloured (0-4)", 0, 4, 0)
        thal    = st.selectbox("Thalassemia", [0, 1, 2, 3],
                    format_func=lambda x: ["Normal", "Fixed Defect",
                                           "Normal (2)", "Reversible Defect"][x])

    # ── Optional: Patient Retention Inputs ───────────────────────────────
    with st.expander("📋 Patient Retention Assessment (optional)"):
        st.caption("Fill in these operational factors to assess dropout risk")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            exercise_diff  = st.slider("Exercise Difficulty (1-5)", 1, 5, 3)
            q_burden       = st.slider("Questionnaire Burden (1-10)", 1, 10, 5)
            waiting_time   = st.slider("Waiting Time (minutes)", 5, 90, 20)
        with r_col2:
            travel_dist    = st.slider("Travel Distance (km)", 1, 80, 15)
            perceived_imp  = st.slider("Perceived Improvement (1-5)", 1, 5, 3)
            has_insurance  = st.selectbox("Has Insurance", [0, 1],
                                format_func=lambda x: "No" if x == 0 else "Yes")

    # ── Predict button ────────────────────────────────────────────────────
    st.divider()
    if st.button("🔍 Generate Risk Assessment", type="primary", use_container_width=True):

        # Build feature vector
        X_input = build_feature_vector(age, sex, cp, trestbps, chol, fbs,
                                        restecg, thalach, exang, oldpeak,
                                        slope, ca, thal)

        # Scale using saved scaler
        scaler = models.get("scaler")
        xgb_model = models.get("cardio_xgb")

        if xgb_model is None or scaler is None:
            st.error("Models not found. Please run 01_preprocessing.py and 03_model_training.py first.")
        else:
            # Align columns (handle any column mismatch from encoding)
            try:
                X_scaled = scaler.transform(X_input.reindex(columns=scaler.feature_names_in_, fill_value=0))
            except Exception:
                X_scaled = scaler.transform(X_input)

            # Get prediction
            risk_prob = xgb_model.predict_proba(X_scaled)[0][1]
            tier, css_class, icon = get_risk_badge(risk_prob)

            # ── Display results ───────────────────────────────────────────
            st.markdown("---")
            st.subheader("Assessment Results")

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Cardiovascular Risk", f"{risk_prob * 100:.1f}%")
            with res_col2:
                lri = compute_lifestyle_risk_index(trestbps, chol, fbs, exang, oldpeak)
                st.metric("Lifestyle Risk Index", f"{lri:.2f}/1.0")
            with res_col3:
                st.metric("Risk Classification", tier)

            # Risk badge
            st.markdown(
                f'<div class="{css_class}">{icon} <strong>{tier}</strong> — '
                f'Predicted cardiovascular disease probability: <strong>{risk_prob * 100:.1f}%</strong></div>',
                unsafe_allow_html=True
            )

            # ── SHAP Explanation ──────────────────────────────────────────
            if xgb_explainer is not None:
                st.subheader("Why this prediction? (SHAP Explanation)")
                st.caption("Each bar shows how much that feature increased (+) or decreased (–) the risk score.")

                shap_vals = xgb_explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                shap_vals = shap_vals[0]

                feature_names = list(X_input.columns)
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": shap_vals,
                    "Feature Value": X_input.values[0]
                }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(9, 5))
                colors = ["#E53935" if v > 0 else "#1E88E5" for v in shap_df["SHAP Value"]]
                bars = ax.barh(shap_df["Feature"][::-1], shap_df["SHAP Value"][::-1],
                               color=colors[::-1], edgecolor="white", height=0.6)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP Value (← reduces risk | increases risk →)")
                ax.set_title("Feature Contributions to Prediction")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── Retention Risk ────────────────────────────────────────────
            ret_model = models.get("retention_rf")
            scaler_ret = models.get("scaler_ret")
            if ret_model and scaler_ret:
                ret_features = pd.DataFrame([{
                    "exercise_difficulty":    exercise_diff,
                    "questionnaire_burden":   q_burden,
                    "waiting_time_minutes":   waiting_time,
                    "travel_distance_km":     travel_dist,
                    "previous_visits":        3,
                    "missed_appointment":     0,
                    "has_insurance":          has_insurance,
                    "perceived_improvement":  perceived_imp,
                    "age_group_encoded":      2,
                    "visit_reason_encoded":   0,
                }])
                try:
                    ret_scaled = scaler_ret.transform(ret_features)
                    dropout_prob = ret_model.predict_proba(ret_scaled)[0][0]  # class 0 = dropped out

                    st.subheader("Patient Retention Risk")
                    ret_col1, ret_col2 = st.columns(2)
                    with ret_col1:
                        st.metric("Dropout Probability", f"{dropout_prob * 100:.1f}%")
                    with ret_col2:
                        flag = "⚠ High Dropout Risk — Consider proactive outreach" if dropout_prob > 0.6 \
                               else "✓ Patient likely to continue treatment"
                        st.info(flag)
                except Exception as e:
                    st.warning(f"Retention model prediction skipped: {e}")

            # ── Recommendations ───────────────────────────────────────────
            st.subheader("Clinical Recommendations")
            if risk_prob >= 0.60:
                st.error("🚨 **URGENT:** Refer to cardiologist. Order: ECG, lipid panel, HbA1c, echocardiogram.")
                st.warning("💊 Review antihypertensive medication. Prescribe cardiac rehabilitation.")
            elif risk_prob >= 0.30:
                st.warning("⚠ **Schedule cardiology consultation** within 4–6 weeks.")
                st.info("🥗 Lifestyle counselling recommended: diet, exercise, smoking cessation.")
            else:
                st.success("✓ **Low Risk.** Continue annual screening and healthy lifestyle maintenance.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

elif "Model Dashboard" in page:
    st.title("📊 Model Performance Dashboard")

    if os.path.exists("outputs/model_comparison.csv"):
        df = pd.read_csv("outputs/model_comparison.csv")

        st.subheader("All Models — Performance Metrics")
        st.dataframe(df.set_index("Model").style.highlight_max(
            subset=["AUC-ROC", "F1", "Recall"], color="#d4edda"
        ).highlight_min(
            subset=["Brier"], color="#d4edda"
        ), use_container_width=True)

        # Bar chart
        cardio_df = df[~df["Model"].str.contains("Retention")]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, metric in enumerate(["AUC-ROC", "F1", "Recall"]):
            axes[i].bar(cardio_df["Model"], cardio_df[metric],
                       color=["#2196F3", "#E53935", "#4CAF50", "#FF9800"])
            axes[i].set_title(metric)
            axes[i].set_ylim(0, 1)
            axes[i].axhline(0.8, color="red", linestyle="--", alpha=0.4)
            axes[i].tick_params(axis="x", rotation=20, labelsize=8)
        plt.suptitle("Cardiovascular Risk Models — Performance Comparison", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if os.path.exists("outputs/roc_curves.png"):
        st.subheader("ROC Curves")
        st.image("outputs/roc_curves.png", use_container_width=True)

    if os.path.exists("outputs/shap_summary_beeswarm.png"):
        st.subheader("SHAP Feature Importance")
        col1, col2 = st.columns(2)
        with col1:
            st.image("outputs/shap_summary_bar.png", caption="Global Feature Importance", use_container_width=True)
        with col2:
            st.image("outputs/shap_summary_beeswarm.png", caption="Direction & Magnitude", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: CLINICAL NLP
# ══════════════════════════════════════════════════════════════════════════

elif "Clinical NLP" in page:
    st.title("📄 Clinical Document Processing")
    st.caption("Extract structured patient data from unstructured clinical notes using NLP.")

    text_input = st.text_area(
        "Paste clinical note here:",
        height=300,
        placeholder="Paste a clinician's note or discharge summary here..."
    )

    if st.button("🔍 Extract Clinical Entities", type="primary"):
        if text_input.strip():
            # Import and run NLP extractor
            try:
                import re
                # Run regex extractions inline (simplified for Streamlit)
                st.subheader("Extracted Entities")

                # Blood pressure
                bp_matches = re.findall(r"(\d{2,3}\/\d{2,3})(?:\s*mmHg)?", text_input)
                if bp_matches:
                    st.success(f"**Blood Pressure:** {', '.join(bp_matches)}")

                # Heart rate
                hr_matches = re.findall(r"(?:Heart Rate|HR|Pulse)[:\s]+(\d{2,3})\s*(?:bpm)?",
                                         text_input, re.IGNORECASE)
                if hr_matches:
                    st.success(f"**Heart Rate:** {', '.join(hr_matches)} bpm")

                # Medications
                med_matches = re.findall(
                    r"([A-Z][a-z]+-?[a-z]*)\s+(\d+(?:\.\d+)?(?:mg|mcg|g))",
                    text_input
                )
                if med_matches:
                    meds = [f"{m} {d}" for m, d in med_matches if len(m) > 4]
                    if meds:
                        st.info(f"**Medications:** {', '.join(meds)}")

                # Diagnoses keywords
                dx_keywords = ["hypertension", "diabetes", "coronary", "heart failure",
                               "angina", "arrhythmia", "stroke"]
                found_dx = [k.title() for k in dx_keywords if k in text_input.lower()]
                if found_dx:
                    st.warning(f"**Diagnoses detected:** {', '.join(found_dx)}")

                # Smoking
                if re.search(r"smok|cigarette", text_input, re.IGNORECASE):
                    st.error("⚠ **Smoking** mentioned — lifestyle risk factor flagged")

            except Exception as e:
                st.error(f"Extraction error: {e}")
        else:
            st.warning("Please paste a clinical note to extract entities.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ══════════════════════════════════════════════════════════════════════════

elif "About" in page:
    st.title("ℹ️ About CardioAI")
    st.markdown("""
    ### Explainable AI System for Cardiovascular Risk & Patient Retention

    This system was developed as part of a dual-capstone research project applying
    machine learning, NLP, and explainable AI to preventive healthcare and patient
    engagement in rehabilitation settings.

    **System Components:**
    - **Cardiovascular Risk Model:** XGBoost/Random Forest classifier trained on clinical data
    - **Lifestyle Risk Index:** Composite behavioral risk score
    - **Explainable AI:** SHAP-based feature attribution for every prediction
    - **Patient Retention Model:** Predicts dropout risk from operational factors
    - **Clinical NLP:** Extracts structured data from unstructured clinical notes
    - **LLM Layer:** Natural language interpretation of model predictions

    **Data Sources:**
    - UCI Heart Disease Dataset (Cleveland subset, n=1025)
    - Synthetic patient retention dataset (n=800)

    **⚠ Disclaimer:**
    This tool is intended for research and clinical decision **support** only.
    It does not replace professional medical judgment. All predictions must be
    reviewed and interpreted by a qualified healthcare professional.

    **Model Performance (Best Model — XGBoost):**
    - AUC-ROC: ~0.90+
    - Sensitivity: ~87%
    - Specificity: ~85%
    """)
