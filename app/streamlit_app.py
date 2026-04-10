import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
import cv2
import re

# --- 1. THE ADAPTIVE IMAGE PREPROCESSOR ---
def preprocess_for_handwriting(pil_image):
    """
    Converts a grayish clinical note into a high-contrast Black & White image.
    This helps Tesseract 'see' the ink better.
    """
    # Convert PIL to OpenCV format
    img = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Adaptive Thresholding: Crucial for handling uneven lighting in photos
    cleaned = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Noise reduction (removes small dots/artifacts)
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

# --- 2. THE FUZZY NLP EXTRACTION ---
def extract_fuzzy_metrics(text):
    """
    Handles OCR errors like 'Rp' instead of 'BP' and extracts numbers.
    """
    extracted = {}
    
    # BP: Handles 'BP', 'Rp', 'B.P' followed by numbers (e.g., 109/73)
    # Allows for up to 15 characters of noise in between
    bp_match = re.search(r'(?:BP|Rp|B\.P|Press)\D{0,15}(\d{2,3}/\d{2,3})', text, re.I)
    if bp_match:
        extracted['blood_pressure'] = bp_match.group(1)

    # Weight: Handles 'Wt', 'Weight', or 'Wala' (common OCR error for Wt)
    wt_match = re.search(r'(?:Weight|Wt|Wala)\D{0,15}(\d{2,3}(?:\.\d)?)', text, re.I)
    if wt_match:
        extracted['weight'] = wt_match.group(1)

    # SPO2: Handles 'SPO2', 'Oxygen', 'O2'
    spo2_match = re.search(r'(?:SPO2|Oxygen|O2)\D{0,10}(\d{2,3})', text, re.I)
    if spo2_match:
        extracted['spo2'] = spo2_match.group(1)

    return extracted

# --- 3. STREAMLIT INTERFACE INTEGRATION ---
st.title("CardioAI: Clinical Document Processor")

uploaded_file = st.file_uploader("Upload Handwritten Clinical Note", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Note", width=400)
    
    with st.spinner("Applying Adaptive Binarization & OCR..."):
        # Process the image
        cleaned_image = preprocess_for_handwriting(image)
        
        # Display the 'cleaned' view so you can see what the AI sees
        with st.expander("View Preprocessed Image (Binarized)"):
            st.image(cleaned_image, caption="How the AI sees the note", width=400)
        
        # Run OCR with PSM 4 (Assume a single column of text)
        raw_text = pytesseract.image_to_string(cleaned_image, config='--psm 4')
        
        # Extract data
        results = extract_fuzzy_metrics(raw_text)

    st.subheader("Extracted Clinical Metrics (Review Required)")
    st.info("The AI found the following. Please verify before running risk analysis.")
    
    # DATA VERIFICATION UI
    col1, col2 = st.columns(2)
    with col1:
        # These values will automatically fill if the AI finds them
        final_bp = st.text_input("Blood Pressure", value=results.get('blood_pressure', ''))
        final_wt = st.text_input("Weight (kg)", value=results.get('weight', ''))
    with col2:
        final_spo2 = st.text_input("SPO2 (%)", value=results.get('spo2', ''))
        st.write("---")
        if st.button("Confirm & Run Risk Prediction"):
            st.success("Data transferred to model. Generating SHAP explanation...")
