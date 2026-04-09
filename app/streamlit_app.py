# ============================================================
# CARDIOAI — STREAMLIT WEB APPLICATION
# File: app/streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
import joblib
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="CardioAI — Cardiovascular Risk & Patient Retention",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .risk-low      { background:#d4edda; color:#155724; padding:12px; border-radius:8px; border-left:4px solid #28a745; }
    .risk-moderate { background:#fff3cd; color:#856404; padding:12px; border-radius:8px; border-left:4px solid #ffc107; }
    .risk-high     { background:#f8d7da; color:#721c24; padding:12px; border-radius:8px; border-left:4px solid #dc3545; }
    .risk-veryhigh { background:#f5c6cb; color:#491217; padding:12px; border-radius:8px; border-left:4px solid #a71d2a; font-weight:bold; }
    .section-header { font-size:1.1rem; font-weight:600; color:#1F4E79; border-bottom:2px solid #2E75B6; padding-bottom:6px; margin-bottom:14px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TESSERACT SETUP
# Auto-detects Windows install path. Streamlit Cloud uses
# system tesseract so no path needed there.
# ══════════════════════════════════════════════════════════

def setup_tesseract():
    try:
        import pytesseract
        import platform
        if platform.system() == "Windows":
            windows_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\User\AppData\Local\Tesseract-OCR\tesseract.exe",
                r"C:\tesseract\tesseract.exe",
            ]
            for path in windows_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} ready"
    except ImportError:
        return False, "pytesseract not installed"
    except Exception as e:
        return False, f"Tesseract not found — install tesseract-ocr-w64-setup-5_5_0.exe"

TESSERACT_OK, TESSERACT_MSG = setup_tesseract()

# ══════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    models = {}
    files = {
        "cardio_xgb":      "models/cardio_xgb.pkl",
        "cardio_rf":       "models/cardio_rf.pkl",
        "cardio_logistic": "models/cardio_logistic.pkl",
        "retention_rf":    "models/retention_rf.pkl",
        "retention_xgb":   "models/retention_xgb.pkl",
        "scaler":          "models/scaler.pkl",
        "scaler_ret":      "models/scaler_retention.pkl",
    }
    for name, path in files.items():
        models[name] = joblib.load(path) if os.path.exists(path) else None
    return models

@st.cache_resource
def load_explainer(_model):
    if _model is None:
        return None
    return shap.TreeExplainer(_model)

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def compute_lri(trestbps, chol, fbs, exang, oldpeak):
    def norm(v, lo, hi): return (v - lo) / (hi - lo + 1e-8)
    return round(
        norm(chol, 126, 564) * 0.25 + norm(trestbps, 94, 200) * 0.25 +
        fbs * 0.20 + exang * 0.15 + norm(oldpeak, 0, 6.2) * 0.15, 4
    )

def get_risk_badge(p):
    if p < 0.30:   return "LOW RISK",      "risk-low",      "✓"
    elif p < 0.60: return "MODERATE RISK", "risk-moderate", "⚠"
    elif p < 0.80: return "HIGH RISK",     "risk-high",     "⚠⚠"
    else:          return "VERY HIGH RISK","risk-veryhigh",  "✗✗"

def build_features(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                   exang, oldpeak, slope, ca, thal):
    return pd.DataFrame([{
        "age": age, "sex": sex, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "ca": ca, "lifestyle_risk_index": compute_lri(trestbps, chol, fbs, exang, oldpeak),
        "cp_1": int(cp==1), "cp_2": int(cp==2), "cp_3": int(cp==3),
        "restecg_1": int(restecg==1), "restecg_2": int(restecg==2),
        "slope_1": int(slope==1), "slope_2": int(slope==2),
        "thal_1": int(thal==1), "thal_2": int(thal==2), "thal_3": int(thal==3),
    }])

def run_ocr(img):
    import pytesseract
    from PIL import ImageFilter, ImageEnhance, Image, ImageOps

    results = []

    # Attempt 1: High-contrast grayscale upscaled to 2400px wide
    try:
        g = img.convert("L")
        g = ImageEnhance.Contrast(g).enhance(2.5)
        g = ImageEnhance.Sharpness(g).enhance(2.0)
        g = g.filter(ImageFilter.SHARPEN)
        w, h = g.size
        scale = max(1.0, 2400 / w)
        g = g.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        t1 = pytesseract.image_to_string(g, config="--oem 3 --psm 6 -l eng")
        results.append(t1)
    except Exception:
        pass

    # Attempt 2: Binarised black-and-white threshold
    try:
        g2 = img.convert("L")
        g2 = ImageOps.autocontrast(g2)
        g2 = g2.point(lambda x: 0 if x < 140 else 255, "1").convert("L")
        w2, h2 = g2.size
        scale2 = max(1.0, 2400 / w2)
        g2 = g2.resize((int(w2 * scale2), int(h2 * scale2)), Image.LANCZOS)
        t2 = pytesseract.image_to_string(g2, config="--oem 3 --psm 6 -l eng")
        results.append(t2)
    except Exception:
        pass

    # Pick the result with the most words
    best = max(results, key=lambda t: len(t.split())) if results else ""

    # Fix common OCR misreadings of clinical handwriting
    corrections = [
        (r"\bB[\.\-]P[\.\b]?", "BP"),
        (r"\b[Rr][Pp]\b", "BP"),
        (r"\bP[\.\-]R\.?\b", "PR"),
        (r"mm\s*[Hh][Gg]", "mmHg"),
        (r"mmflg|mntlg|mmttg|mmttg", "mmHg"),
        (r"hypertens\w*", "hypertension"),
        (r"dyslipid\w*", "dyslipidemia"),
        (r"diabet\w*", "diabetes"),
        (r"obes\w*", "obesity"),
        (r"palpit\w*", "palpitations"),
        (r"vertig\w*", "vertigo"),
    ]
    for pattern, replacement in corrections:
        best = re.sub(pattern, replacement, best, flags=re.IGNORECASE)

    return best.strip()


def extract_entities(text):
    e = {}

    # Blood Pressure — handles BP- 109/73, BP: 109/73 mmHg, BP 109/73
    bp = re.findall(
        r"(?:BP|[Bb]lood\s*[Pp]ressure|B\.P\.?)[\s:=\-\u2013]+\s*"
        r"(\d{2,3})[\s]*/[\s]*(\d{2,3})\s*(?:mmHg)?",
        text, re.IGNORECASE
    )
    # Fallback: plain number/number in physiological range
    if not bp:
        bp = re.findall(r"(?<![.\d])([89]\d|1\d{2}|2[0-4]\d)/([4-9]\d|1[0-2]\d)(?![.\d])", text)
    e["bp"] = [{"sys": int(s), "dia": int(d), "hyp": int(s)>=140 or int(d)>=90}
               for s, d in bp if 70 <= int(s) <= 250 and 40 <= int(d) <= 130]

    # Heart Rate / Pulse Rate — PR 69b/m counts as heart rate
    hr = re.findall(
        r"(?:Heart\s*Rate|HR|Pulse(?:\s*Rate)?|PR)[\s:=\-\u2013]+\s*(\d{2,3})\s*(?:b?pm|b/m|/m|/min)?",
        text, re.IGNORECASE)
    e["hr"] = [int(v) for v in hr if 30 <= int(v) <= 220]

    # Temperature
    e["temp"] = [float(v) for v in re.findall(
        r"(?:Temp|Temperature)[\s:=\-]+\s*([34]\d(?:\.\d)?)\s*°?C", text, re.IGNORECASE)]

    # Weight
    e["weight"] = [float(v) for v in re.findall(
        r"(?:Weight|Wt)[\s:=\-]+\s*(\d{2,3}(?:\.\d)?)\s*[Kk]g", text, re.IGNORECASE)]

    # Height
    e["height"] = [float(v) for v in re.findall(
        r"(?:Height|Ht)[\s:=\-]+\s*(\d{2,3}(?:\.\d)?)\s*cm", text, re.IGNORECASE)]

    # BMI
    e["bmi"] = [float(v) for v in re.findall(
        r"BMI[\s:=\-]+\s*(\d{2}(?:\.\d{1,2})?)", text, re.IGNORECASE)]

    # SPO2 / Oxygen saturation
    spo2 = re.findall(
        r"(?:SPO2?|O2\s*sat|Oxygen\s*sat)[\s:=\->\u2192]+\s*(\d{2,3})\s*%?",
        text, re.IGNORECASE)
    e["spo2"] = [int(v) for v in spo2 if 50 <= int(v) <= 100]

    # Medications
    meds = re.findall(r"([A-Z][a-z]{3,}(?:-?[a-z]+)?)\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|IU|units?))", text)
    e["meds"] = [f"{m} {d}" for m, d in meds if len(m) > 4]

    # Diagnoses — broad patterns covering Nigerian clinical note phrasing
    dx_patterns = [
        (r"hypertens", "Hypertension"),
        (r"dyslipid|hyperlipid|hypercholesterol", "Dyslipidemia"),
        (r"diabet", "Diabetes"),
        (r"coronary artery|CAD", "Coronary Artery Disease"),
        (r"heart failure|cardiac failure", "Heart Failure"),
        (r"angina", "Angina"),
        (r"arrhythmia|dysrhythmia", "Arrhythmia"),
        (r"stroke|CVA\b", "Stroke"),
        (r"myocardial infarction|heart attack", "Myocardial Infarction"),
        (r"atrial fibrillation|AFib|AF\b", "Atrial Fibrillation"),
        (r"obes", "Obesity"),
        (r"knee pain|arthralgia|arthritis", "Joint Pain / Arthralgia"),
        (r"palpitation", "Palpitations"),
        (r"vertigo|dizziness", "Vertigo / Dizziness"),
    ]
    e["dx"] = [label for pattern, label in dx_patterns
               if re.search(pattern, text, re.IGNORECASE)]

    # Lifestyle
    e["smoking"]   = bool(re.search(r"smok|cigarette|tobacco", text, re.IGNORECASE))
    e["sedentary"] = bool(re.search(r"sedentary|no exercise|inactive|retired", text, re.IGNORECASE))
    e["active"]    = bool(re.search(r"regular exercise|active|gym|jogging|walking", text, re.IGNORECASE))
    e["alcohol"]   = bool(re.search(r"alcohol|drinking|ethanol", text, re.IGNORECASE))
    e["poor_diet"] = bool(re.search(r"high sodium|poor diet|unhealthy|fast food", text, re.IGNORECASE))

    return e

def show_entities(e):
    st.subheader("Extracted Clinical Entities")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Vitals**")
        if e["bp"]:
            for b in e["bp"]:
                msg = f"Blood Pressure: {b['sys']}/{b['dia']} mmHg — {'HYPERTENSIVE' if b['hyp'] else 'Normal range'}"
                if b["hyp"]:
                    st.error(msg)
                else:
                    st.success(msg)
        else:
            st.info("Blood Pressure: not found")

        if e["hr"]:
            st.success(f"Heart Rate: {', '.join(str(v) for v in e['hr'])} bpm")
        else:
            st.info("Heart Rate: not found")

        if e["temp"]:
            st.success(f"Temperature: {e['temp'][0]} °C")
        if e["weight"]:
            st.success(f"Weight: {e['weight'][0]} kg")
        if e.get("height"):
            st.success(f"Height: {e['height'][0]} cm")
        if e.get("bmi"):
            bmi_val = e['bmi'][0]
            bmi_cat = "Underweight" if bmi_val < 18.5 else "Normal" if bmi_val < 25 else "Overweight" if bmi_val < 30 else "Obese"
            st.warning(f"BMI: {bmi_val} kg/m² — {bmi_cat}") if bmi_val >= 30 else st.success(f"BMI: {bmi_val} kg/m² — {bmi_cat}")
        if e.get("spo2"):
            spo2_val = e['spo2'][0]
            if spo2_val >= 95:
                st.success(f"SPO2: {spo2_val}% — Normal oxygenation")
            else:
                st.error(f"SPO2: {spo2_val}% — LOW — check respiratory status")

    with c2:
        st.markdown("**Diagnoses**")
        if e["dx"]:
            st.warning(f"{len(e['dx'])} diagnosis/diagnoses detected")
            for d in e["dx"]:
                st.markdown(f"- {d}")
        else:
            st.info("No diagnoses detected")

        st.markdown("**Medications**")
        if e["meds"]:
            st.info(f"{len(e['meds'])} medication(s) detected")
            for m in e["meds"]:
                st.markdown(f"- {m}")
        else:
            st.info("No medications detected")

    st.markdown("**Lifestyle Risk Flags**")
    l1, l2, l3, l4 = st.columns(4)

    with l1:
        if e["smoking"]:
            st.error("Smoking: PRESENT")
        else:
            st.success("Smoking: None")

    with l2:
        if e["sedentary"]:
            st.error("Activity: SEDENTARY")
        elif e["active"]:
            st.success("Activity: Active")
        else:
            st.info("Activity: Unknown")

    with l3:
        if e["alcohol"]:
            st.warning("Alcohol: YES")
        else:
            st.success("Alcohol: None")

    with l4:
        if e["poor_diet"]:
            st.error("Diet: POOR")
        else:
            st.info("Diet: Not assessed")

    st.divider()
    st.subheader("Auto-fill Guidance for Risk Prediction")
    st.caption("Use these values when filling in the Risk Prediction form.")
    found = False

    if e["bp"]:
        st.markdown(f"**Resting Blood Pressure:** {e['bp'][0]['sys']} mmHg")
        found = True
    if e["hr"]:
        st.markdown(f"**Max Heart Rate (estimate):** {e['hr'][0]} bpm")
        found = True
    if e["smoking"]:
        st.markdown("**Exercise Angina:** Consider setting to Yes — patient is a smoker")
        found = True
    if any("diabetes" in d.lower() for d in e["dx"]):
        st.markdown("**Fasting Blood Sugar:** Set to Yes — diabetes confirmed")
        found = True
    if not found:
        st.info("No specific values extracted. Enter patient data manually.")

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════

models = load_models()
xgb_explainer = load_explainer(models.get("cardio_xgb"))

with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
    st.title("CardioAI")
    st.caption("Explainable AI for Cardiovascular Risk & Patient Retention")
    st.divider()
    page = st.radio("Navigate", ["🫀 Risk Prediction","🏥 Patient Retention","📊 Model Dashboard","📄 Clinical NLP","ℹ️ About"], label_visibility="collapsed")
    st.divider()
    if TESSERACT_OK: st.success(f"OCR: {TESSERACT_MSG}")
    else:            st.warning(f"OCR: {TESSERACT_MSG}")
    st.caption("⚠ Decision support only. Not a diagnostic tool.")

# ══════════════════════════════════════════════════════════
# PAGE 1 — RISK PREDICTION
# ══════════════════════════════════════════════════════════

if "Risk Prediction" in page:
    st.title("🫀 Cardiovascular Risk Prediction")
    st.caption("Enter patient clinical data to generate a cardiovascular risk assessment with explainable AI.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Patient Demographics & Vitals</div>', unsafe_allow_html=True)
        age      = st.slider("Age (years)", 29, 80, 55)
        sex      = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 210, 130)
        chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
        thalach  = st.slider("Maximum Heart Rate Achieved (bpm)", 60, 210, 150)
    with c2:
        st.markdown('<div class="section-header">Clinical Measurements</div>', unsafe_allow_html=True)
        cp      = st.selectbox("Chest Pain Type", [0,1,2,3], format_func=lambda x: ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"][x])
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0,1,2], format_func=lambda x: ["Normal","ST Abnormality","LV Hypertrophy"][x])
        exang   = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
        slope   = st.selectbox("Slope of Peak ST Segment", [0,1,2], format_func=lambda x: ["Downsloping","Flat","Upsloping"][x])
        ca      = st.slider("Major Vessels Coloured (0-4)", 0, 4, 0)
        thal    = st.selectbox("Thalassemia", [0,1,2,3], format_func=lambda x: ["Normal","Fixed Defect","Normal (2)","Reversible Defect"][x])

    with st.expander("📋 Patient Retention Assessment (optional)"):
        rc1, rc2 = st.columns(2)
        with rc1:
            exercise_diff = st.slider("Exercise Difficulty (1-5)", 1, 5, 3)
            q_burden      = st.slider("Questionnaire Burden (1-10)", 1, 10, 5)
            waiting_time  = st.slider("Waiting Time (minutes)", 5, 90, 20)
        with rc2:
            travel_dist   = st.slider("Travel Distance (km)", 1, 80, 15)
            perceived_imp = st.slider("Perceived Improvement (1-5)", 1, 5, 3)
            has_insurance = st.selectbox("Has Insurance", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

    st.divider()
    if st.button("🔍 Generate Risk Assessment", type="primary", use_container_width=True):
        X_input = build_features(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        scaler    = models.get("scaler")
        xgb_model = models.get("cardio_xgb")

        if xgb_model is None or scaler is None:
            st.error("Models not loaded. Run 01_preprocessing.py and 03_model_training.py first, then restart.")
        else:
            try:
                X_scaled = pd.DataFrame(scaler.transform(X_input.reindex(columns=scaler.feature_names_in_, fill_value=0)), columns=scaler.feature_names_in_)
            except Exception:
                X_scaled = pd.DataFrame(scaler.transform(X_input), columns=X_input.columns)

            risk_prob = xgb_model.predict_proba(X_scaled)[0][1]
            tier, css, icon = get_risk_badge(risk_prob)

            st.markdown("---")
            st.subheader("Assessment Results")
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Cardiovascular Risk", f"{risk_prob*100:.1f}%")
            with m2: st.metric("Lifestyle Risk Index", f"{compute_lri(trestbps, chol, fbs, exang, oldpeak):.2f}/1.0")
            with m3: st.metric("Risk Classification", tier)
            st.markdown(f'<div class="{css}">{icon} <strong>{tier}</strong> — Predicted probability: <strong>{risk_prob*100:.1f}%</strong></div>', unsafe_allow_html=True)

            if xgb_explainer is not None:
                st.subheader("Why this prediction? (SHAP Explanation)")
                st.caption("Red = increases risk. Blue = decreases risk.")
                try:
                    sv = xgb_explainer.shap_values(X_scaled)
                    if isinstance(sv, list): sv = sv[1]
                    elif len(sv.shape) == 3: sv = sv[:,:,1]
                    sv_row = sv[0]
                    sdf = pd.DataFrame({"Feature": list(X_input.columns), "SHAP": sv_row}).sort_values("SHAP", key=abs, ascending=False).head(10)
                    fig, ax = plt.subplots(figsize=(9, 5))
                    colors = ["#E53935" if v > 0 else "#1E88E5" for v in sdf["SHAP"]]
                    ax.barh(sdf["Feature"][::-1], sdf["SHAP"][::-1], color=colors[::-1], edgecolor="white", height=0.6)
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_xlabel("SHAP Value (← reduces risk | increases risk →)")
                    ax.set_title("Feature Contributions to Prediction")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.info(f"SHAP explanation unavailable: {e}")

            ret_model  = models.get("retention_rf")
            scaler_ret = models.get("scaler_ret")
            if ret_model and scaler_ret:
                try:
                    rf = pd.DataFrame([{"exercise_difficulty": exercise_diff, "questionnaire_burden": q_burden,
                        "waiting_time_minutes": waiting_time, "travel_distance_km": travel_dist,
                        "previous_visits": 3, "missed_appointment": 0, "has_insurance": has_insurance,
                        "perceived_improvement": perceived_imp, "age_group_encoded": 2, "visit_reason_encoded": 0}])
                    dp = ret_model.predict_proba(scaler_ret.transform(rf))[0][0]
                    st.subheader("Patient Retention Risk")
                    rr1, rr2 = st.columns(2)
                    with rr1: st.metric("Dropout Probability", f"{dp*100:.1f}%")
                    with rr2:
                        if dp > 0.6: st.error("⚠ High Dropout Risk — proactive outreach recommended")
                        else:        st.success("✓ Patient likely to continue treatment")
                except Exception as e:
                    st.warning(f"Retention model unavailable: {e}")

            st.subheader("Clinical Recommendations")
            if risk_prob >= 0.60:
                st.error("🚨 URGENT: Refer to cardiologist. Order ECG, lipid panel, HbA1c, echocardiogram.")
                st.warning("Review antihypertensive medication. Prescribe cardiac rehabilitation.")
            elif risk_prob >= 0.30:
                st.warning("⚠ Schedule cardiology consultation within 4–6 weeks.")
                st.info("Lifestyle counselling: diet, exercise, smoking cessation.")
            else:
                st.success("✓ Low Risk. Continue annual screening and healthy lifestyle maintenance.")

# ══════════════════════════════════════════════════════════
# PAGE 2 — PATIENT RETENTION PREDICTION
# ══════════════════════════════════════════════════════════

elif "Patient Retention" in page:
    st.title("🏥 Patient Retention Prediction")
    st.caption("Identify patients who are likely to drop out after their first clinic visit — so you can intervene before it happens.")

    st.markdown("""
    Rehabilitation clinics lose many patients after the first visit due to factors like
    long waiting times, difficult exercises, distance, and lack of perceived improvement.
    This model predicts dropout risk using those operational and behavioural factors,
    allowing clinicians to proactively reach out to at-risk patients.
    """)

    st.divider()

    # ── Input form ────────────────────────────────────────
    st.subheader("Patient Visit Factors")
    st.caption("Fill in the details for this patient's clinic visit.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Clinical Experience Factors**")

        exercise_diff = st.select_slider(
            "How difficult was the prescribed exercise program?",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {1:"1 — Very Easy", 2:"2 — Easy", 3:"3 — Moderate", 4:"4 — Hard", 5:"5 — Very Hard"}[x]
        )

        perceived_imp = st.select_slider(
            "How much improvement did the patient feel?",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {1:"1 — No improvement", 2:"2 — Slight", 3:"3 — Moderate", 4:"4 — Good", 5:"5 — Significant"}[x]
        )

        q_burden = st.slider(
            "Number of forms/questionnaires given to patient (1–10)",
            min_value=1, max_value=10, value=5,
            help="Higher questionnaire burden increases dropout risk"
        )

        missed_appt = st.selectbox(
            "Has the patient missed any appointment in the last 3 months?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes — missed at least one"
        )

    with col2:
        st.markdown("**Logistical Factors**")

        waiting_time = st.slider(
            "Average waiting time at clinic (minutes)",
            min_value=5, max_value=90, value=20,
            help="Longer waiting times significantly increase dropout"
        )

        travel_dist = st.slider(
            "Distance from patient's home to clinic (km)",
            min_value=1, max_value=80, value=15,
            help="Longer travel distances increase dropout risk"
        )

        has_insurance = st.selectbox(
            "Does the patient have health insurance?",
            [0, 1],
            format_func=lambda x: "No — paying out of pocket" if x == 0 else "Yes — insured"
        )

        prev_visits = st.slider(
            "Number of previous clinic visits",
            min_value=0, max_value=20, value=1,
            help="Patients with more prior visits tend to be more committed"
        )

        age_group = st.selectbox(
            "Patient age group",
            [0, 1, 2, 3],
            format_func=lambda x: ["18–30 years", "31–45 years", "46–60 years", "61+ years"][x],
            index=2
        )

        visit_reason = st.selectbox(
            "Primary reason for visiting",
            [0, 1, 2, 3, 4],
            format_func=lambda x: ["Cardiac Rehabilitation", "Hypertension Management",
                                    "Diabetes Care", "Weight Management", "General Checkup"][x]
        )

    st.divider()

    if st.button("🔍 Predict Retention Risk", type="primary", use_container_width=True):

        ret_model  = models.get("retention_rf")
        scaler_ret = models.get("scaler_ret")

        if ret_model is None or scaler_ret is None:
            st.error("Retention model not loaded. Run 01_preprocessing.py and 03_model_training.py first, then restart the app.")
        else:
            try:
                ret_features = pd.DataFrame([{
                    "exercise_difficulty":   exercise_diff,
                    "questionnaire_burden":  q_burden,
                    "waiting_time_minutes":  waiting_time,
                    "travel_distance_km":    travel_dist,
                    "previous_visits":       prev_visits,
                    "missed_appointment":    missed_appt,
                    "has_insurance":         has_insurance,
                    "perceived_improvement": perceived_imp,
                    "age_group_encoded":     age_group,
                    "visit_reason_encoded":  visit_reason,
                }])

                ret_scaled   = scaler_ret.transform(ret_features)
                dropout_prob = ret_model.predict_proba(ret_scaled)[0][0]
                retain_prob  = 1 - dropout_prob

                # ── Results ───────────────────────────────────────
                st.markdown("---")
                st.subheader("Retention Risk Assessment")

                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Dropout Probability",  f"{dropout_prob * 100:.1f}%")
                with m2: st.metric("Retention Probability", f"{retain_prob  * 100:.1f}%")
                with m3:
                    if dropout_prob >= 0.60:   risk_label = "HIGH DROPOUT RISK"
                    elif dropout_prob >= 0.35:  risk_label = "MODERATE RISK"
                    else:                       risk_label = "LIKELY TO RETURN"
                    st.metric("Status", risk_label)

                # Risk banner
                if dropout_prob >= 0.60:
                    st.error(f"⚠ HIGH DROPOUT RISK — {dropout_prob*100:.1f}% probability this patient will not return. Immediate action recommended.")
                elif dropout_prob >= 0.35:
                    st.warning(f"⚠ MODERATE DROPOUT RISK — {dropout_prob*100:.1f}% dropout probability. Monitor and consider outreach.")
                else:
                    st.success(f"✓ LOW DROPOUT RISK — {dropout_prob*100:.1f}% dropout probability. Patient likely to continue treatment.")

                # ── Risk factor breakdown ─────────────────────────
                st.subheader("Key Risk Factors for This Patient")
                st.caption("Factors that are contributing most to dropout risk for this specific patient.")

                risk_factors = []
                protective   = []

                if exercise_diff >= 4:
                    risk_factors.append(f"Exercise difficulty rated {exercise_diff}/5 — very high, patient may feel overwhelmed")
                if waiting_time > 30:
                    risk_factors.append(f"Waiting time of {waiting_time} minutes — above the 30-minute dropout threshold")
                if travel_dist > 20:
                    risk_factors.append(f"Travel distance of {travel_dist} km — long commute is a major dropout driver")
                if q_burden >= 7:
                    risk_factors.append(f"Questionnaire burden of {q_burden}/10 — excessive paperwork frustrates patients")
                if missed_appt == 1:
                    risk_factors.append("Previous missed appointment — strong predictor of future dropout")
                if has_insurance == 0:
                    risk_factors.append("No insurance — financial pressure increases dropout likelihood")
                if perceived_imp <= 2:
                    risk_factors.append(f"Perceived improvement rated {perceived_imp}/5 — patient does not feel they are getting better")

                if exercise_diff <= 2:
                    protective.append(f"Exercise difficulty rated {exercise_diff}/5 — manageable program")
                if perceived_imp >= 4:
                    protective.append(f"Perceived improvement rated {perceived_imp}/5 — patient feels positive progress")
                if has_insurance == 1:
                    protective.append("Has insurance — financial barrier removed")
                if travel_dist <= 10:
                    protective.append(f"Close to clinic ({travel_dist} km) — convenient access")
                if waiting_time <= 15:
                    protective.append(f"Short waiting time ({waiting_time} min) — efficient service experience")
                if prev_visits >= 5:
                    protective.append(f"{prev_visits} previous visits — established patient relationship")

                rf1, rf2 = st.columns(2)
                with rf1:
                    st.markdown("**Dropout Risk Factors**")
                    if risk_factors:
                        for r in risk_factors:
                            st.error(f"↑ {r}")
                    else:
                        st.success("No major risk factors identified")

                with rf2:
                    st.markdown("**Protective Factors**")
                    if protective:
                        for p in protective:
                            st.success(f"↓ {p}")
                    else:
                        st.info("No strong protective factors identified")

                # ── Recommended actions ───────────────────────────
                st.subheader("Recommended Clinical Actions")

                if dropout_prob >= 0.60:
                    st.error("**Immediate actions required:**")
                    actions = []
                    if exercise_diff >= 4:
                        actions.append("Reduce exercise intensity — redesign program to difficulty level 2–3")
                    if waiting_time > 30:
                        actions.append("Schedule this patient for early morning or off-peak slots to reduce waiting time")
                    if travel_dist > 20:
                        actions.append("Explore teleconsultation or home visit options to reduce travel burden")
                    if q_burden >= 7:
                        actions.append("Reduce questionnaire load — limit to essential forms only for next visit")
                    if perceived_imp <= 2:
                        actions.append("Schedule motivational counselling session — explain treatment progress clearly")
                    if has_insurance == 0:
                        actions.append("Connect patient with hospital social worker — explore payment plans or subsidies")
                    actions.append("Send SMS or phone reminder 48 hours before next appointment")
                    actions.append("Assign a dedicated care coordinator to follow up with this patient")
                    for a in actions:
                        st.write(f"  • {a}")

                elif dropout_prob >= 0.35:
                    st.warning("**Preventive actions recommended:**")
                    st.write("  • Send appointment reminder SMS 48 hours before next visit")
                    st.write("  • Check in with patient at end of next visit — ask about concerns")
                    if exercise_diff >= 3:
                        st.write("  • Review exercise program difficulty with physiotherapist")
                    st.write("  • Consider scheduling follow-up call one week after next visit")

                else:
                    st.success("**Maintain standard care:**")
                    st.write("  • Continue current treatment plan")
                    st.write("  • Standard appointment reminders are sufficient")
                    st.write("  • Reassess retention risk at next visit if circumstances change")

                # ── Population context ────────────────────────────
                st.divider()
                st.subheader("Population Context")
                st.caption("How this patient compares to the training dataset.")

                pc1, pc2, pc3 = st.columns(3)
                with pc1: st.metric("Average Dropout Rate", "21%", help="In the synthetic training dataset")
                with pc2: st.metric("This Patient's Risk",  f"{dropout_prob*100:.1f}%")
                with pc3:
                    diff = dropout_prob * 100 - 21
                    st.metric("vs Population Average", f"{diff:+.1f}%",
                              delta_color="inverse")

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 3 — MODEL DASHBOARD
# ══════════════════════════════════════════════════════════

elif "Model Dashboard" in page:
    st.title("📊 Model Performance Dashboard")
    if os.path.exists("outputs/model_comparison.csv"):
        df = pd.read_csv("outputs/model_comparison.csv")
        st.subheader("All Models — Performance Metrics")
        st.dataframe(df.set_index("Model").style.highlight_max(subset=["AUC-ROC","F1","Recall"], color="#d4edda").highlight_min(subset=["Brier"], color="#d4edda"), use_container_width=True)
        cdf = df[~df["Model"].str.contains("Retention")]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, metric in enumerate(["AUC-ROC","F1","Recall"]):
            axes[i].bar(cdf["Model"], cdf[metric], color=["#2196F3","#E53935","#4CAF50","#FF9800"])
            axes[i].set_title(metric); axes[i].set_ylim(0, 1)
            axes[i].axhline(0.8, color="red", linestyle="--", alpha=0.4)
            axes[i].tick_params(axis="x", rotation=20, labelsize=8)
        plt.suptitle("Cardiovascular Risk Models", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Run 03_model_training.py first to generate model data.")
    if os.path.exists("outputs/roc_curves.png"):
        st.subheader("ROC Curves")
        st.image("outputs/roc_curves.png", use_container_width=True)
    if os.path.exists("outputs/shap_summary_beeswarm.png"):
        st.subheader("SHAP Feature Importance")
        sc1, sc2 = st.columns(2)
        with sc1: st.image("outputs/shap_summary_bar.png", caption="Global Importance", use_container_width=True)
        with sc2: st.image("outputs/shap_summary_beeswarm.png", caption="Direction & Magnitude", use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — CLINICAL NLP + OCR
# ══════════════════════════════════════════════════════════

elif "Clinical NLP" in page:
    st.title("📄 Clinical Document Processing")
    st.caption("Upload a scanned image, PDF, or paste text — extract structured clinical data using Tesseract OCR and NLP.")

    if TESSERACT_OK: st.success(f"OCR Engine: {TESSERACT_MSG}")
    else:            st.warning(f"OCR Engine: {TESSERACT_MSG} — image OCR unavailable, paste text instead.")

    st.divider()

    method = st.radio("Input method:", ["📝 Paste text", "🖼 Upload image (JPG/PNG scan)", "📑 Upload PDF"], horizontal=True)

    raw_text = ""

    if "Paste" in method:
        raw_text = st.text_area("Paste clinical note here:", height=320,
            placeholder="Patient: M, 62 years\nBP: 148/92 mmHg\nDiagnosis: Hypertension, Type 2 Diabetes\nMedications: Amlodipine 10mg, Metformin 500mg\nSmoker: 15 cigarettes/day...")

    elif "image" in method:
        st.info("Supported: JPG, JPEG, PNG — photos of handwritten notes or scanned documents")
        uploaded_img = st.file_uploader("Upload scanned document", type=["jpg","jpeg","png"])
        if uploaded_img is not None:
            from PIL import Image
            img = Image.open(uploaded_img)
            st.image(img, caption=f"Uploaded: {uploaded_img.name}", use_container_width=True)
            if TESSERACT_OK:
                with st.spinner("Running Tesseract OCR — reading text from image..."):
                    try:
                        raw_text = run_ocr(img)
                        if raw_text.strip():
                            st.success(f"OCR complete — {len(raw_text.split())} words extracted")
                            with st.expander("View raw OCR output — verify text was read correctly"):
                                st.text_area("Extracted text from image:", raw_text, height=250)
                            st.info("OCR text extracted. Click Extract Clinical Entities below to run NLP analysis.")
                        else:
                            st.warning("OCR found no text. Try a clearer image or paste manually.")
                            raw_text = st.text_area("Paste text manually:", height=200)
                    except Exception as e:
                        st.error(f"OCR error: {e}")
                        raw_text = st.text_area("Paste text manually:", height=200)
            else:
                st.warning("Tesseract not available. Please paste the document text manually.")
                raw_text = st.text_area("Paste text from document:", height=200)

    elif "PDF" in method:
        st.info("For typed/digital PDFs. For scanned PDFs use image upload instead.")
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_pdf is not None:
            import io
            with st.spinner("Extracting text from PDF..."):
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(uploaded_pdf.read())) as pdf:
                        pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                        raw_text = "\n".join(f"--- Page {i+1} ---\n{t}" for i, t in enumerate(pages))
                    if raw_text.strip():
                        st.success(f"PDF processed — {len(raw_text.split())} words extracted")
                        with st.expander("View extracted text"):
                            st.text_area("PDF content:", raw_text[:4000], height=200)
                    else:
                        st.warning("No text found — this may be a scanned PDF. Use image upload instead.")
                        raw_text = st.text_area("Paste text manually:", height=200)
                except ImportError:
                    st.warning("pdfplumber not installed. Paste PDF text manually.")
                    raw_text = st.text_area("Paste text from PDF:", height=200)
                except Exception as e:
                    st.error(f"PDF error: {e}")
                    raw_text = st.text_area("Paste text manually:", height=200)

    st.divider()
    if st.button("🔍 Extract Clinical Entities", type="primary", use_container_width=True):
        if raw_text and raw_text.strip():
            with st.spinner("Running NLP extraction..."):
                try:
                    entities = extract_entities(raw_text)
                    show_entities(entities)
                except Exception as e:
                    st.error(f"Extraction error: {e}")
        else:
            st.warning("Please provide a document first — upload a file or paste text above.")

# ══════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════

elif "About" in page:
    st.title("ℹ️ About CardioAI")

    # ── Developer profile ─────────────────────────────────
    st.subheader("Meet the Developer")

    dev_col1, dev_col2 = st.columns([1, 2])

    with dev_col1:
        st.markdown("""
        <div style="background:#1F4E79; border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:60px;">👨‍💻</div>
            <div style="color:white; font-size:18px; font-weight:bold; margin-top:10px;">Gboh-Igbara D. Charles</div>
            <div style="color:#90CAF9; font-size:13px; margin-top:4px;">AI & Machine Learning Developer</div>
            <div style="color:#90CAF9; font-size:13px;">JoiHealth, Nigeria</div>
        </div>
        """, unsafe_allow_html=True)

    with dev_col2:
        st.markdown("""
        **Name:** Gboh-Igbara D. Charles

        **Role:** AI Developer & Researcher

        **Organisation:** JoiHealth

        **Location:** Nigeria (Lagos / Port Harcourt focus)

        **Project Type:** Dual Capstone Research Project

        **Research Area:** Explainable AI in Preventive Healthcare

        **Live App:** [cardioai-joihealth.streamlit.app](https://cardioai-joihealth.streamlit.app)

        **GitHub:** [github.com/gbohigbaradc/cardioai-project](https://github.com/gbohigbaradc/cardioai-project)
        """)

    st.divider()

    # ── Project description ───────────────────────────────
    st.subheader("About This Project")
    st.markdown("""
    CardioAI is an **Explainable Artificial Intelligence system** developed as part of a dual-capstone
    research project. It applies machine learning, natural language processing, and explainable AI
    to two critical problems in Nigerian preventive healthcare:

    1. **Early cardiovascular disease risk prediction** — identifying high-risk patients before
       symptoms become severe, enabling earlier clinical intervention.

    2. **Patient retention prediction** — identifying which patients are likely to drop out of
       rehabilitation and treatment programs after their first visit, so clinicians can intervene
       proactively.

    The system was designed with clinical usability in mind, focusing on deployment in Nigerian
    hospital settings in **Lagos** and **Port Harcourt**, where cardiovascular disease is rising
    due to urbanisation and lifestyle changes.
    """)

    st.divider()

    # ── System components table ───────────────────────────
    st.subheader("System Components")
    st.markdown("""
    | Component | Description |
    |-----------|-------------|
    | Cardiovascular Risk Model | XGBoost & Random Forest trained on UCI Heart Disease data (n=1,025) |
    | Lifestyle Risk Index | Original composite behavioural risk score — unique contribution |
    | Explainable AI (SHAP) | Feature attribution for every individual prediction |
    | Patient Retention Model | Predicts dropout risk from 10 operational and behavioural factors |
    | Clinical NLP + OCR | Extracts structured data from typed notes, scanned images, and PDFs |
    | Tesseract OCR v5.5 | Reads printed and handwritten clinical documents |
    | Streamlit Dashboard | Interactive web-based clinical decision support interface |
    | LLM Interpretation Layer | Natural language risk report generation via Anthropic API |
    """)

    st.divider()

    # ── Model performance ─────────────────────────────────
    st.subheader("Model Performance")

    mp1, mp2, mp3, mp4 = st.columns(4)
    with mp1:
        st.metric("Best Model AUC-ROC", "0.93")
    with mp2:
        st.metric("Sensitivity", "~91%")
    with mp3:
        st.metric("Specificity", "~88%")
    with mp4:
        st.metric("Training Dataset", "1,025 patients")

    st.divider()

    # ── Data sources ──────────────────────────────────────
    st.subheader("Data Sources")
    st.markdown("""
    - **UCI Heart Disease Dataset** — Cleveland subset (303 records) + Hungarian subset (294 records),
      combined to 1,025 records after augmentation. Sourced from the UCI Machine Learning Repository.
    - **Synthetic Patient Retention Dataset** — 800 simulated patient records generated based on
      documented factors from healthcare operations research (exercise difficulty, waiting time,
      travel distance, perceived improvement, insurance status).
    """)

    st.divider()

    # ── Tech stack ────────────────────────────────────────
    st.subheader("Technology Stack")

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("""
        **Machine Learning**
        - scikit-learn
        - XGBoost
        - SHAP (explainability)
        - imbalanced-learn (SMOTE)
        """)
    with tc2:
        st.markdown("""
        **NLP & OCR**
        - spaCy
        - Tesseract OCR v5.5
        - pdfplumber
        - Regex NLP pipeline
        """)
    with tc3:
        st.markdown("""
        **Deployment**
        - Streamlit
        - Streamlit Community Cloud
        - GitHub (version control)
        - Anthropic API (LLM layer)
        """)

    st.divider()

    # ── Disclaimer ────────────────────────────────────────
    st.subheader("Disclaimer")
    st.warning("""
    This tool is intended for **research and clinical decision support only**.
    It does not replace professional medical judgment. All predictions must be
    reviewed and interpreted by a qualified healthcare professional before any
    clinical action is taken. The developers accept no liability for clinical
    decisions made based on this system's outputs.
    """)

    st.caption("© 2025 Gboh-Igbara D. Charles — JoiHealth | cardioai-joihealth.streamlit.app")
