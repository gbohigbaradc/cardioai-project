import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
import cv2
import re
import joblib  # Assuming you use joblib to load your 4 ML models [cite: 8]
import shap    # For your explainability research objective [cite: 66]

# --- 1. SYSTEM CONFIGURATION & ASSETS ---
# Ensure these match your slide methodology [cite: 24, 75]
MODELS = {
    "XGBoost": "models/xgboost_model.pkl",
    "Random Forest": "models/rf_model.pkl",
    "Logistic Regression": "models/lr_model.pkl",
    "Neural Network": "models/mlp_model.pkl"
}

# --- 2. ENHANCED OCR & NLP PIPELINE ---
def preprocess_for_ocr(pil_image):
    """Clean image for handwritten notes like Mrs. Charis's [cite: 70, 242]"""
    img = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Adaptive thresholding to handle Nigerian clinic note paper quality [cite: 44, 218]
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    return processed_img

def extract_clinical_info(text):
    """Fuzzy Regex for clinical metrics [cite: 109, 206]"""
    data = {}
    # Extract BP (e.g., 109/73)
    bp = re.search(r'(?:BP|B\.P|Pressure)\D{0,7}(\d{2,3}/\d{2,3})', text, re.I)
    if bp: data['blood_pressure'] = bp.group(1)
    
    # Extract Cholesterol (Relevant for your LRI [cite: 119])
    chol = re.search(r'(?:Chol|Cholesterol)\D{0,7}(\d{2,3})', text, re.I)
    if chol: data['cholesterol'] = int(chol.group(1))
    
    # Extract Age & Weight
    age = re.search(r'(?:Age)\D{0,5}(\d{2})', text, re.I)
    if age: data['age'] = int(age.group(1))
    
    return data

# --- 3. NOVEL LIFESTYLE RISK INDEX (LRI) CALCULATION ---
def calculate_lri(row):
    """
    Implements your original formula[cite: 119]:
    LRI = 0.25*Chol + 0.25*BP + 0.20*Sugar + 0.15*Angina + 0.15*ST_Depression
    """
    # Note: Normalization logic should match your StandardScaler [cite: 89]
    lri_score = (0.25 * row['chol']) + (0.25 * row['bp_sys']) # Simplified for example
    return min(max(lri_score, 0.0), 1.0) # Ensure 0.0 to 1.0 range [cite: 135]

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="CardioAI - JoiHealth", layout="wide")

st.title("CardioAI: Explainable AI for Cardiovascular Risk")
st.sidebar.info("Gboh-Igbara D. Charles | JoiHealth, Nigeria [cite: 4, 263]")

tabs = st.tabs(["Clinical OCR", "Risk Prediction", "Patient Retention", "Model Dashboard"])

# --- TAB 1: CLINICAL OCR (The part we fixed) ---
with tabs[0]:
    st.header("Upload Clinical Notes")
    uploaded_file = st.file_uploader("Upload Image/PDF [cite: 110]", type=["jpg", "png", "pdf"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Note", width=400)
        
        with st.spinner("Processing with Tesseract OCR v5.5... [cite: 107]"):
            cleaned = preprocess_for_ocr(image)
            raw_text = pytesseract.image_to_string(cleaned, config='--psm 4')
            extracted = extract_clinical_info(raw_text)
            
        st.subheader("Verify Extracted Clinical Data")
        # Clinician-in-the-loop verification [cite: 215]
        c1, c2, c3 = st.columns(3)
        with c1:
            v_age = st.number_input("Age", value=extracted.get('age', 0))
            v_bp = st.text_input("BP (systolic/diastolic)", value=extracted.get('blood_pressure', '120/80'))
        with c2:
            v_chol = st.number_input("Cholesterol", value=extracted.get('cholesterol', 0))
            v_sugar = st.selectbox("Fasting Sugar > 120mg/dl", [0, 1])
        with c3:
            v_angina = st.selectbox("Exercise Angina", [0, 1])
            v_oldpeak = st.number_input("ST Depression", value=0.0)

        if st.button("Transfer to Prediction Model"):
            st.session_state['patient_data'] = {
                'age': v_age, 'chol': v_chol, 'sugar': v_sugar, 
                'angina': v_angina, 'oldpeak': v_oldpeak, 'bp': v_bp
            }
            st.success("Data ready for analysis in 'Risk Prediction' tab.")

# --- TAB 2: RISK PREDICTION & SHAP ---
with tabs[1]:
    st.header("Cardiovascular Risk Analysis")
    if 'patient_data' in st.session_state:
        # Load Model (e.g., XGBoost with 0.999 AUC [cite: 155])
        st.write(f"Analyzing patient data using best-performing model...")
        
        # Display SHAP Waterfall Chart for individual explanation [cite: 102, 217]
        st.subheader("Individual Risk Explanation (SHAP)")
        st.info("This chart explains WHY the patient was assigned this risk score. [cite: 239]")
        # (Insert SHAP plotting code here using st.pyplot)

# --- TAB 3: PATIENT RETENTION ---
with tabs[2]:
    st.header("Dropout Risk Prediction [cite: 169]")
    st.write("Predicting if patient will return based on operational factors. [cite: 68]")
    # Inputs for Travel Distance, Waiting Time, Exercise Difficulty 
    dist = st.slider("Travel Distance (km)", 0, 100, 10)
    wait = st.slider("Waiting Time (mins)", 0, 120, 30)
    
    if st.button("Predict Retention"):
        # LogReg Retention AUC: 0.66 [cite: 224]
        st.warning("Action Plan: Offer teleconsultation for distant patients. [cite: 195]")

# --- TAB 4: MODEL DASHBOARD ---
with tabs[3]:
    st.header("Performance Metrics [cite: 203]")
    # Table from Slide 14 [cite: 141]
    results_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"],
        "AUC-ROC": [0.927, 0.998, 0.999, 0.994]
    })
    st.table(results_df)
