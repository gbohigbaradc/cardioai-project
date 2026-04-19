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
    page_title="CardioAI — JoiHealth",
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

# ── Secret helper ─────────────────────────────────────────────────────────
# Streamlit Cloud stores secrets in st.secrets, not os.environ.
# This helper checks both so the app works locally AND on Streamlit Cloud.
def get_secret(key):
    """Get API key from st.secrets (Streamlit Cloud) or os.environ (local)."""
    try:
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, "")

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
# MULTIMODAL VISION — DOCUMENT UNDERSTANDING
# Primary: Google Gemini Flash (free tier — 1000 req/day)
# Secondary: Claude Vision (if Anthropic key available)
# Both read the image directly — no OCR pipeline needed.
# The AI understands handwriting, layout, and clinical context.
# ══════════════════════════════════════════════════════════

CLINICAL_EXTRACTION_PROMPT = """You are a medical data extraction assistant analysing a handwritten clinical note.

Look at this image carefully and extract ALL clinical values you can see written on the page.

Return ONLY a plain text summary in this exact format (skip any line where the value is not visible):
Name: [patient name]
Age: [age in years]
Sex: [M or F]
BP: [systolic/diastolic mmHg — e.g. 109/73]
PR: [pulse rate bpm — e.g. 69]
SPO2: [oxygen saturation percent — e.g. 97]
Weight: [kg — e.g. 104.7]
Height: [cm — e.g. 160]
BMI: [value — e.g. 40.6]
Temperature: [celsius if visible]
Diagnoses: [comma separated list of all diagnoses and conditions mentioned]
Medications: [comma separated list of medications with doses]
Smoking: [Yes or No if mentioned]
Alcohol: [Yes or No if mentioned]
Chief Complaint: [brief description]

Rules:
- Only report values you can clearly read. Do not guess.
- If a value is not written on the page, skip that line entirely.
- For BP write as numbers only e.g. 109/73 not "one hundred and nine over seventy three"
- Read carefully — handwriting may be cursive or stylised"""


def extract_with_gemini_vision(img):
    """
    Gemini Vision multimodal extraction using google-generativeai SDK.
    Tries current 2026 model names in order until one works.
    Free tier: 1500 requests/day at aistudio.google.com
    """
    import io

    api_key = get_secret("GOOGLE_API_KEY")
    if not api_key:
        return None, "No GOOGLE_API_KEY in Streamlit secrets"

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Convert PIL image to bytes
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=95)
        img_bytes = buffer.getvalue()

        # Image part as dict — works across all SDK versions
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}

        # Try model names in order — 2026 current models first
        # The old SDK needs "models/" prefix for some versions
        models_to_try = [
            "gemini-2.0-flash",
            "models/gemini-2.0-flash",
            "gemini-2.5-flash",
            "models/gemini-2.5-flash",
            "gemini-1.5-flash",
            "models/gemini-1.5-flash",
            "gemini-flash-latest",
        ]

        errors = []
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    [CLINICAL_EXTRACTION_PROMPT, image_part]
                )
                return response.text, f"Gemini Vision ({model_name})"
            except Exception as me:
                errors.append(f"{model_name}: {str(me)[:80]}")
                continue

        return None, "All models failed: " + " | ".join(errors[:3])

    except ImportError:
        return None, "google-generativeai not installed"
    except Exception as e:
        return None, f"Gemini setup error: {type(e).__name__}: {str(e)}"


def extract_with_claude_vision(img):
    """
    Secondary multimodal option using Claude Vision.
    Used as fallback if Gemini is unavailable.
    """
    import base64
    import io

    api_key = get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "No Anthropic API key"

    try:
        import anthropic

        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=95)
        img_b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": CLINICAL_EXTRACTION_PROMPT}
            ]}]
        )
        return message.content[0].text, "Claude Vision"

    except ImportError:
        return None, "anthropic library not installed"
    except Exception as e:
        return None, f"Claude Vision error: {str(e)}"


def extract_with_vision(img):
    """
    Master function: tries Gemini first, Claude second, returns best result.
    Exposes full error details so failures can be diagnosed.
    """
    gemini_result, gemini_status = extract_with_gemini_vision(img)
    if gemini_result and gemini_result.strip():
        return gemini_result, gemini_status

    claude_result, claude_status = extract_with_claude_vision(img)
    if claude_result and claude_result.strip():
        return claude_result, claude_status

    # Both failed — return combined error details
    combined = f"Gemini: [{gemini_status}] | Claude: [{claude_status}]"
    return None, combined


# ══════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """
    Load trained models. If model files are missing (e.g. on Streamlit Cloud),
    automatically run the training pipeline to generate them.
    """
    files = {
        "cardio_xgb":      "models/cardio_xgb.pkl",
        "cardio_rf":       "models/cardio_rf.pkl",
        "cardio_logistic": "models/cardio_logistic.pkl",
        "retention_rf":    "models/retention_rf.pkl",
        "retention_xgb":   "models/retention_xgb.pkl",
        "scaler":          "models/scaler.pkl",
        "scaler_ret":      "models/scaler_retention.pkl",
    }

    # Check if models exist
    missing = [p for p in files.values() if not os.path.exists(p)]

    if missing:
        # Models not found — run training pipeline automatically
        st.info("First run detected — training models now. This takes 2-3 minutes...")
        progress = st.progress(0, text="Setting up...")

        try:
            import subprocess, sys
            os.makedirs("models", exist_ok=True)
            os.makedirs("outputs", exist_ok=True)
            os.makedirs("data", exist_ok=True)

            # Run preprocessing
            progress.progress(10, text="Running preprocessing...")
            result = subprocess.run(
                [sys.executable, "01_preprocessing.py"],
                capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                st.warning(f"Preprocessing warning: {result.stderr[-200:]}")

            # Run model training
            progress.progress(40, text="Training models (this takes ~2 minutes)...")
            result = subprocess.run(
                [sys.executable, "03_model_training.py"],
                capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                st.warning(f"Training warning: {result.stderr[-200:]}")

            progress.progress(100, text="Models ready!")
            st.success("Models trained successfully. Loading now...")
            progress.empty()

        except Exception as e:
            st.error(f"Auto-training failed: {e}. Please run scripts locally and upload model files.")
            progress.empty()
            return {name: None for name in files}

    # Load all models
    models = {}
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
    """
    OpenCV-powered OCR pipeline for handwritten Nigerian clinical notes.
    Runs 4 preprocessing strategies and picks the best result.
    Falls back to PIL-only if OpenCV is unavailable.
    """
    import pytesseract
    from PIL import ImageFilter, ImageEnhance, Image, ImageOps
    import numpy as np

    results = []

    # ── Convert PIL image to numpy array for OpenCV processing ──────────
    img_np = np.array(img.convert("RGB"))

    # ── STRATEGY 1: Gemini-recommended adaptive threshold ───────────────
    # blockSize=11 gives finer local contrast — better for handwriting detail
    # MORPH_OPEN removes noise without thickening strokes (cleaner than dilate)
    try:
        import cv2
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        scale = max(1.0, 2400 / w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_LANCZOS4)
        # Fine-grained adaptive threshold — turns paper texture pure white
        # blockSize=11 recommended for handwriting (Gemini + literature)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # small blocks = fine detail on handwriting
            C=2            # standard subtraction constant
        )
        # MORPH_OPEN: erodes then dilates — removes tiny noise specks
        # without merging or thickening handwriting strokes
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        pil_cv1 = Image.fromarray(cleaned)
        t1 = pytesseract.image_to_string(
            pil_cv1, config="--oem 3 --psm 6 -l eng")
        if t1.strip():
            results.append(("opencv_adaptive_fine", t1))
    except ImportError:
        pass  # OpenCV not available — PIL fallbacks below will run
    except Exception:
        pass

    # ── STRATEGY 1B: Wider blockSize for low-contrast scans ─────────────
    try:
        import cv2
        gray_b = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h_b, w_b = gray_b.shape
        scale_b = max(1.0, 2400 / w_b)
        gray_b = cv2.resize(gray_b, (int(w_b * scale_b), int(h_b * scale_b)),
                            interpolation=cv2.INTER_LANCZOS4)
        binary_b = cv2.adaptiveThreshold(
            gray_b, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31, C=10)
        kernel_b = np.ones((1, 1), np.uint8)
        cleaned_b = cv2.morphologyEx(binary_b, cv2.MORPH_OPEN, kernel_b)
        t1b = pytesseract.image_to_string(
            Image.fromarray(cleaned_b), config="--oem 3 --psm 6 -l eng")
        if t1b.strip():
            results.append(("opencv_adaptive_wide", t1b))
    except Exception:
        pass

    # ── STRATEGY 2: OpenCV Otsu Thresholding ────────────────────────────
    # Otsu automatically finds the optimal global threshold value
    try:
        import cv2
        gray2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h2, w2 = gray2.shape
        scale2 = max(1.0, 2400 / w2)
        gray2 = cv2.resize(gray2, (int(w2 * scale2), int(h2 * scale2)),
                           interpolation=cv2.INTER_LANCZOS4)
        # Gaussian blur before Otsu reduces noise
        blurred = cv2.GaussianBlur(gray2, (3, 3), 0)
        _, otsu = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_cv2 = Image.fromarray(otsu)
        t2 = pytesseract.image_to_string(
            pil_cv2, config="--oem 3 --psm 6 -l eng")
        if t2.strip():
            results.append(("opencv_otsu", t2))
    except Exception:
        pass

    # ── STRATEGY 3: PIL High-Contrast (fallback if no OpenCV) ───────────
    try:
        g3 = img.convert("L")
        g3 = ImageEnhance.Contrast(g3).enhance(3.0)
        g3 = ImageEnhance.Sharpness(g3).enhance(2.5)
        g3 = g3.filter(ImageFilter.SHARPEN)
        w3, h3 = g3.size
        scale3 = max(1.0, 2400 / w3)
        g3 = g3.resize((int(w3 * scale3), int(h3 * scale3)), Image.LANCZOS)
        t3 = pytesseract.image_to_string(
            g3, config="--oem 3 --psm 6 -l eng")
        if t3.strip():
            results.append(("pil_contrast", t3))
    except Exception:
        pass

    # ── STRATEGY 4: PIL Hard Binarisation ───────────────────────────────
    try:
        g4 = img.convert("L")
        g4 = ImageOps.autocontrast(g4)
        g4 = g4.point(lambda x: 0 if x < 128 else 255, "1").convert("L")
        w4, h4 = g4.size
        scale4 = max(1.0, 2400 / w4)
        g4 = g4.resize((int(w4 * scale4), int(h4 * scale4)), Image.LANCZOS)
        t4 = pytesseract.image_to_string(
            g4, config="--oem 3 --psm 6 -l eng")
        if t4.strip():
            results.append(("pil_binary", t4))
    except Exception:
        pass

    # ── Pick the strategy that extracted the most words ──────────────────
    if not results:
        return ""
    best_name, best = max(results, key=lambda x: len(x[1].split()))

    # ── Post-processing: fix common OCR misreadings ──────────────────────
    # These patterns fix the most common handwriting OCR errors seen in
    # Nigerian clinical notes (e.g. "8P" for "BP", "Rp" for "BP")
    corrections = [
        # Blood Pressure label variants
        (r"8P",              "BP"),
        (r"B\.P\.?",         "BP"),
        (r"B-P",             "BP"),
        (r"[Rr][Pp]",        "BP"),
        # Pulse Rate label variants
        (r"P\.R\.?",         "PR"),
        (r"P-R",             "PR"),
        # mmHg corruptions
        (r"mm\s*[Hh][Gg]",       "mmHg"),
        (r"mm[Ff]lg|mm[Tt]+g",   "mmHg"),
        (r"mnHg|nnHg",           "mmHg"),
        # Digit confusions inside numbers
        (r"(?<=\d)[lIi](?=\d)",  "1"),
        (r"(?<=\d)[Oo](?=\d)",   "0"),
        # Clinical keyword recovery
        (r"hypertens\w+",        "hypertension"),
        (r"dyslipid\w+",         "dyslipidemia"),
        (r"diabet\w+",           "diabetes"),
        (r"obes\w+",             "obesity"),
        (r"palpit\w+",           "palpitations"),
        (r"vertig\w+",           "vertigo"),
        (r"cholester\w+",        "cholesterol"),
        (r"cardiac\w*",          "cardiac"),
    ]
    for pattern, replacement in corrections:
        best = re.sub(pattern, replacement, best, flags=re.IGNORECASE)

    return best.strip()

def extract_entities(text):
    """
    Flexible NLP extractor handling both clean typed text and
    OCR-corrected handwritten clinical notes.
    Covers Nigerian clinical note conventions: BP-, PR, SPO2, etc.
    """
    e = {}

    # ── Blood Pressure ─────────────────────────────────────────────────
    # Strategy A: Gemini fuzzy — BP + any OCR noise + NNN/NN
    # Handles "BP Wala mmHg 109/73" or "BP -> 109/73" or "BP: 109/73"
    bp_m = re.search(
        r"(?:BP|[Bb]lood\s*[Pp]ressure|B\.P\.?)\D{0,25}(\d{2,3})\s*/\s*(\d{2,3})",
        text, re.IGNORECASE)
    bp = [(bp_m.group(1), bp_m.group(2))] if bp_m else []
    # Strategy B: standard labelled pattern
    if not bp:
        bp = re.findall(
            r"(?:BP|[Bb]lood\s*[Pp]ressure|B\.P\.?)[\s:=\-\u2013]+\s*"
            r"(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmHg)?",
            text, re.IGNORECASE)
    # Strategy C: plain NNN/NN fallback
    if not bp:
        bp = re.findall(
            r"(?<![.\d])([89]\d|1\d{2}|2[0-4]\d)/([4-9]\d|1[0-2]\d)(?![.\d])",
            text)
    e["bp"] = [{"sys": int(s), "dia": int(d),
                "hyp": int(s) >= 140 or int(d) >= 90}
               for s, d in bp
               if 70 <= int(s) <= 250 and 40 <= int(d) <= 130]

    # ── Heart Rate / Pulse Rate ────────────────────────────────────────
    # PR 69 bpm, Heart Rate: 88, HR 72
    hr = re.findall(
        r"(?:Heart\s*Rate|HR|Pulse(?:\s*Rate)?|PR)[\s:=\-\u2013]+\s*"
        r"(\d{2,3})\s*(?:b?pm|b/m|/m|/min)?",
        text, re.IGNORECASE)
    e["hr"] = [int(v) for v in hr if 30 <= int(v) <= 220]

    # ── Temperature ────────────────────────────────────────────────────
    e["temp"] = [float(v) for v in re.findall(
        r"(?:Temp|Temperature)[\s:=\-]+\s*([34]\d(?:\.\d)?)\s*°?C",
        text, re.IGNORECASE)]

    # ── Weight ─────────────────────────────────────────────────────────
    # With label: Weight: 104.7kg  |  Without: 104.7 kg (fallback)
    weight = re.findall(
        r"(?:Weight|Wt)[\s:=\-]+\s*(\d{2,3}(?:\.\d{1,2})?)\s*[Kk]g",
        text, re.IGNORECASE)
    if not weight:
        weight = re.findall(r"(\d{2,3}(?:\.\d{1,2})?)\s*[Kk]g", text, re.IGNORECASE)
    e["weight"] = [float(v) for v in weight if 20 <= float(v) <= 300]

    # ── Height ─────────────────────────────────────────────────────────
    height = re.findall(
        r"(?:Height|Ht)[\s:=\-]+\s*(\d{2,3}(?:\.\d)?)\s*cm",
        text, re.IGNORECASE)
    if not height:
        height = re.findall(r"(\d{2,3}(?:\.\d)?)\s*cm", text, re.IGNORECASE)
    e["height"] = [float(v) for v in height if 50 <= float(v) <= 250]

    # ── BMI ────────────────────────────────────────────────────────────
    e["bmi"] = [float(v) for v in re.findall(
        r"BMI[\s:=\-]+\s*(\d{2}(?:\.\d{1,2})?)", text, re.IGNORECASE)]

    # ── SPO2 / Oxygen Saturation ───────────────────────────────────────
    # Gemini fuzzy: SPO2 + any noise + 2-3 digits
    spo2_m = re.search(
        r"(?:SPO2?|O2\s*sat(?:uration)?)\D{0,10}(\d{2,3})\s*%?",
        text, re.IGNORECASE)
    spo2 = [spo2_m.group(1)] if spo2_m else []
    if not spo2:
        for line in text.split("\n"):
            if re.search(r"spo|oxygen|o2", line, re.IGNORECASE):
                pct = re.findall(r"(\d{2,3})\s*%", line)
                spo2.extend(pct)
    e["spo2"] = [int(v) for v in spo2 if 50 <= int(v) <= 100]

    # ── Medications ────────────────────────────────────────────────────
    meds = re.findall(
        r"([A-Z][a-z]{3,}(?:-?[a-z]+)?)\s+"
        r"(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|IU|units?))",
        text)
    e["meds"] = [f"{m} {d}" for m, d in meds if len(m) > 4]

    # ── Diagnoses — broad patterns for Nigerian clinical notes ─────────
    dx_patterns = [
        (r"hypertens",                      "Hypertension"),
        (r"dyslipid|hyperlipid|hypercholest","Dyslipidemia"),
        (r"diabet",                          "Diabetes"),
        (r"coronary artery|CAD\b",           "Coronary Artery Disease"),
        (r"heart failure|cardiac failure",   "Heart Failure"),
        (r"\bangina\b",                      "Angina"),
        (r"arrhythmia|dysrhythmia",          "Arrhythmia"),
        (r"\bstroke\b|CVA\b",                "Stroke"),
        (r"myocardial infarction|heart attack","Myocardial Infarction"),
        (r"atrial fibrillation|AFib\b|AF\b", "Atrial Fibrillation"),
        (r"\bobes",                          "Obesity"),
        (r"knee pain|arthralgia|arthritis",  "Joint Pain / Arthralgia"),
        (r"palpitat",                        "Palpitations"),
        (r"vertigo|dizziness",               "Vertigo / Dizziness"),
    ]
    e["dx"] = [label for pattern, label in dx_patterns
               if re.search(pattern, text, re.IGNORECASE)]

    # ── Lifestyle ──────────────────────────────────────────────────────
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
    # ── JoiHealth Logo ────────────────────────────────────────
    import os
    logo_paths = [
        "Joi_Health_PM.jpeg",
        "app/Joi_Health_PM.jpeg",
        os.path.join(os.path.dirname(__file__), "Joi_Health_PM.jpeg"),
        os.path.join(os.path.dirname(__file__), "..", "Joi_Health_PM.jpeg"),
    ]
    logo_loaded = False
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
            logo_loaded = True
            break
    if not logo_loaded:
        # Fallback: show text branding if logo file not found
        st.markdown("""
        <div style='text-align:center;padding:8px 0 4px;'>
          <span style='font-size:22px;font-weight:700;color:#0D1B2A;font-family:Georgia,serif;'>
            Joi Health
          </span><br>
          <span style='font-size:10px;color:#475569;letter-spacing:2px;'>
            COMMITMENT TO CARE
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;margin:-8px 0 4px;'><span style='font-size:13px;font-weight:600;color:#0D1B2A;'>CardioAI</span></div>", unsafe_allow_html=True)
    st.caption("Explainable AI for Cardiovascular Risk & Patient Retention")
    st.divider()
    page = st.radio("Navigate", ["🫀 Risk Prediction","🏥 Patient Retention","📊 Model Dashboard","📄 Clinical NLP","🔬 Medical Imaging","ℹ️ About"], label_visibility="collapsed")
    st.divider()
    if get_secret("GOOGLE_API_KEY"):
        st.success("Gemini Vision: Ready")
    elif get_secret("ANTHROPIC_API_KEY"):
        st.success("Claude Vision: Ready")
    else:
        st.warning("Vision AI: Add GOOGLE_API_KEY to secrets")
    if TESSERACT_OK: st.success(f"OCR Fallback: {TESSERACT_MSG}")
    else:            st.info("OCR Fallback: Tesseract not found")
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

            raw_text = ""
            extraction_method = ""

            # ── PRIMARY: Vision AI (Gemini or Claude — reads like a human) ──
            google_key = get_secret("GOOGLE_API_KEY")
            anthropic_key = get_secret("ANTHROPIC_API_KEY")

            if google_key or anthropic_key:
                spinner_msg = ("Gemini Vision is reading the document..."
                               if google_key else "Claude Vision is reading the document...")
                with st.spinner(spinner_msg):
                    vision_text, vision_status = extract_with_vision(img)

                if vision_text and vision_text.strip():
                    st.success(f"Multimodal AI extraction complete — {vision_status}")
                    extraction_method = f"{vision_status} (Multimodal AI)"
                    raw_text = vision_text
                    with st.expander(f"View {vision_status} extraction — verify before proceeding"):
                        st.text_area("AI extracted:", raw_text, height=280)
                else:
                    st.warning(f"Vision AI unavailable: {vision_status} — falling back to Tesseract OCR.")

            # ── FALLBACK: Tesseract OCR (if no API key or Claude fails) ──
            if not raw_text.strip():
                if TESSERACT_OK:
                    with st.spinner("Running Tesseract OCR — preprocessing image..."):
                        try:
                            ocr_raw = run_ocr(img)
                        except Exception as e:
                            ocr_raw = ""
                            st.warning(f"OCR error: {e}")

                    if ocr_raw.strip():
                        st.info(f"Tesseract OCR extracted {len(ocr_raw.split())} words. "
                                "Review and correct below before extracting entities.")
                        extraction_method = "Tesseract OCR (review recommended)"
                        raw_text = st.text_area(
                            "OCR output — correct any errors before extracting:",
                            value=ocr_raw, height=300,
                            help="Fix any OCR mistakes, especially numbers like BP (e.g. 109/73)."
                        )
                    else:
                        st.warning("OCR could not read this image. Please type key values manually.")
                        extraction_method = "Manual entry"
                        raw_text = st.text_area(
                            "Type clinical values manually:",
                            height=300,
                            placeholder="BP: 109/73 mmHg\nPR: 69 bpm\nWeight: 104.7 kg\n"
                                        "Height: 160 cm\nBMI: 40.6\nSPO2: 97%\n"
                                        "Diagnosis: Hypertension, Dyslipidemia, Obesity"
                        )
                else:
                    st.warning(
                        "Neither Claude Vision (no API key) nor Tesseract OCR is available. "
                        "Set ANTHROPIC_API_KEY in Streamlit secrets for best results, "
                        "or type the clinical values below."
                    )
                    extraction_method = "Manual entry"
                    raw_text = st.text_area(
                        "Enter clinical values manually:",
                        height=300,
                        placeholder="BP: 109/73 mmHg\nPR: 69 bpm\nWeight: 104.7 kg\n"
                                    "Height: 160 cm\nBMI: 40.6\nSPO2: 97%\n"
                                    "Diagnosis: Hypertension, Dyslipidemia"
                    )

            # Show which method was used
            if extraction_method:
                st.caption(f"Extraction method: {extraction_method}")

    elif "PDF" in method:
        st.info("Upload any PDF — typed or scanned. Scanned PDFs will be processed with Gemini Vision or OCR automatically.")
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_pdf is not None:
            import io
            pdf_bytes = uploaded_pdf.read()

            with st.spinner("Processing PDF..."):
                raw_text = ""

                # Step 1: Try pdfplumber for digital/typed PDFs
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                        pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                        raw_text = "\n".join(f"--- Page {i+1} ---\n{t}" for i, t in enumerate(pages))
                    if raw_text.strip():
                        st.success(f"Digital PDF processed — {len(raw_text.split())} words extracted")
                        with st.expander("View extracted text"):
                            st.text_area("PDF content:", raw_text[:4000], height=200)
                except Exception:
                    pass

                # Step 2: If no text found, it is a scanned PDF — convert to image and use Vision AI
                if not raw_text.strip():
                    st.info("This appears to be a scanned PDF. Converting to image and running Vision AI...")
                    try:
                        # Convert first page of PDF to image using pypdf + PIL
                        from PIL import Image
                        import base64

                        # Try pdf2image first
                        try:
                            from pdf2image import convert_from_bytes
                            images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=3)
                            st.success(f"PDF converted to {len(images)} image(s)")
                        except Exception:
                            # Fallback: use PyMuPDF if available
                            try:
                                import fitz
                                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                                images = []
                                for page_num in range(min(3, len(doc))):
                                    page = doc[page_num]
                                    mat = fitz.Matrix(2, 2)
                                    pix = page.get_pixmap(matrix=mat)
                                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                    images.append(img)
                                st.success(f"PDF converted to {len(images)} image(s) via PyMuPDF")
                            except Exception as fe:
                                images = []
                                st.warning(f"Could not convert PDF to images: {fe}. Please use the Image upload option instead.")

                        # Run vision on each page and combine results
                        if images:
                            all_text = []
                            for i, page_img in enumerate(images):
                                st.image(page_img, caption=f"Page {i+1}", use_container_width=True)

                                # Try Gemini Vision first
                                google_key = get_secret("GOOGLE_API_KEY")
                                anthropic_key = get_secret("ANTHROPIC_API_KEY")

                                if google_key or anthropic_key:
                                    with st.spinner(f"Vision AI reading page {i+1}..."):
                                        page_text, method_used = extract_with_vision(page_img)
                                    if page_text and page_text.strip():
                                        all_text.append(f"--- Page {i+1} (Vision AI) ---\n{page_text}")
                                        continue

                                # Fallback to Tesseract OCR
                                if TESSERACT_OK:
                                    with st.spinner(f"OCR reading page {i+1}..."):
                                        page_text = run_ocr(page_img)
                                    if page_text.strip():
                                        all_text.append(f"--- Page {i+1} (OCR) ---\n{page_text}")

                            raw_text = "\n\n".join(all_text)
                            if raw_text.strip():
                                st.success(f"Scanned PDF processed — extracted from {len(images)} page(s)")
                                with st.expander("View extracted text — verify before extracting"):
                                    st.text_area("Extracted:", raw_text[:4000], height=250)
                            else:
                                st.warning("Could not extract text from PDF. Please type key values manually below.")
                                raw_text = st.text_area("Type clinical values manually:", height=250,
                                    placeholder="BP: 109/73 mmHg\nPR: 69 bpm\nWeight: 104.7 kg\nHeight: 160 cm\nDiagnosis: Hypertension")

                    except Exception as e:
                        st.error(f"PDF processing error: {e}")
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
# PAGE 5 — MEDICAL IMAGING (CNN)
# ══════════════════════════════════════════════════════════

elif "Medical Imaging" in page:
    st.title("🔬 Medical Imaging — CNN Analysis")
    st.caption("Upload a chest X-ray to detect 18 pathologies, segment anatomy, compute cardiothoracic ratio, and generate Grad-CAM heatmaps.")

    st.info(
        "**Powered by DenseNet-121** pretrained on 100,000+ chest X-rays "
        "(NIH ChestX-ray14, CheXpert, MIMIC-CXR). "
        "Detects 18 pathologies with pixel-level Grad-CAM explainability."
    )

    # ── Check dependencies ─────────────────────────────────
    cnn_ready = False
    try:
        import torch
        import torchxrayvision as xrv
        import skimage
        cnn_ready = True
    except ImportError:
        st.warning(
            "CNN imaging requires additional packages. "
            "Add to requirements.txt: `torchxrayvision scikit-image`"
        )

    uploaded_xray = st.file_uploader(
        "Upload chest X-ray (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="PA (posterior-anterior) or AP view chest X-ray. "
             "DICOM: export as PNG/JPEG first."
    )

    if uploaded_xray and cnn_ready:
        from PIL import Image as PILImage
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        import torchxrayvision as xrv
        import skimage.transform
        import numpy as np
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm_module

        img_pil = PILImage.open(uploaded_xray)
        st.image(img_pil, caption=f"Uploaded: {uploaded_xray.name}", use_container_width=True)

        # ── Thresholds ─────────────────────────────────────
        THRESHOLDS = {
            "Cardiomegaly": 0.35, "Effusion": 0.40, "Pneumonia": 0.30,
            "Atelectasis": 0.40, "Consolidation": 0.40, "Pneumothorax": 0.28,
            "Edema": 0.35, "Emphysema": 0.40, "Fibrosis": 0.40,
            "Nodule": 0.45, "Mass": 0.38, "Infiltration": 0.40,
            "Pleural_Thickening": 0.40, "Hernia": 0.30,
        }
        URGENT  = {"Pneumothorax", "Mass", "Edema", "Effusion"}
        CARDIAC = {"Cardiomegaly", "Effusion", "Edema", "Consolidation"}

        # ── Preprocess ─────────────────────────────────────
        with st.spinner("Preprocessing image..."):
            img_np = np.array(img_pil.convert("L")).astype(np.float32)
            img_norm = xrv.datasets.normalize(img_np, img_np.max() if img_np.max() > 0 else 255)
            img_norm = img_norm[None, ...]
            transform = transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224)
            ])
            img_tensor = torch.from_numpy(transform(img_norm)).float()

        # ── Classification ─────────────────────────────────
        with st.spinner("Running DenseNet-121 CNN — classifying 18 pathologies..."):
            try:
                model = xrv.models.DenseNet(weights="densenet121-res224-all")
                model.eval()
                with torch.no_grad():
                    outputs = model(img_tensor[None, ...])
                pathologies = model.targets
                scores = dict(zip(pathologies, outputs[0].detach().numpy().tolist()))
                scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
                st.success(f"Classification complete — {len(scores)} pathologies scored")
            except Exception as e:
                st.error(f"Classification error: {e}")
                scores = {}
                model = None

        # ── Segmentation ───────────────────────────────────
        masks, seg_targets = None, None
        with st.spinner("Running anatomical segmentation..."):
            try:
                seg_model = xrv.baseline_models.chestx_det.PSPNet()
                seg_model.eval()
                with torch.no_grad():
                    seg_out = seg_model(img_tensor[None, ...])
                masks = seg_out[0].detach().numpy()
                seg_targets = seg_model.targets
                st.success(f"Segmented {len(seg_targets)} anatomical structures")
            except Exception as e:
                st.info(f"Segmentation unavailable: {e}")

        # ── Cardiothoracic Ratio ────────────────────────────
        ctr, cardiomegaly = None, None
        if masks is not None and seg_targets is not None:
            try:
                hi = seg_targets.index("Heart")
                li = seg_targets.index("Left Lung")
                ri = seg_targets.index("Right Lung")
                hm = masks[hi] > 0.5
                chest_m = (masks[li] > 0.5) | (masks[ri] > 0.5) | hm
                hc = np.where(hm.any(axis=0))[0]
                cc = np.where(chest_m.any(axis=0))[0]
                if len(hc) >= 2 and len(cc) >= 2:
                    ctr = round(float((hc[-1]-hc[0]) / (cc[-1]-cc[0])), 4)
                    cardiomegaly = ctr >= 0.50
            except Exception:
                pass

        # ── Flags ──────────────────────────────────────────
        flags = []
        for p, s in scores.items():
            t = THRESHOLDS.get(p, 0.45)
            if s >= t:
                flags.append({
                    "pathology": p, "score": round(s, 3),
                    "severity": "Mild" if s < 0.50 else ("Moderate" if s < 0.70 else "Significant"),
                    "urgent": p in URGENT, "cardiac": p in CARDIAC
                })
        flags.sort(key=lambda x: (not x["urgent"], -x["score"]))

        # ── Grad-CAM ───────────────────────────────────────
        gcam = None
        top_path = flags[0]["pathology"] if flags else (list(scores.keys())[0] if scores else None)
        if model and top_path:
            with st.spinner(f"Generating Grad-CAM for {top_path}..."):
                try:
                    grads, acts = [], []
                    def bwd(m, gi, go): grads.append(go[0])
                    def fwd(m, i, o):   acts.append(o)
                    layer = model.model.features.denseblock4
                    h1 = layer.register_forward_hook(fwd)
                    h2 = layer.register_backward_hook(bwd)
                    inp = img_tensor[None, ...].requires_grad_(True)
                    out = model(inp)
                    model.zero_grad()
                    out[0, model.targets.index(top_path)].backward()
                    g = grads[0][0]; a = acts[0][0]
                    c = F.relu((g.mean(dim=[1,2])[:, None, None] * a).sum(0)).detach().numpy()
                    if c.max() > 0: c = (c - c.min()) / (c.max() - c.min())
                    gcam = skimage.transform.resize(c, (224, 224))
                    h1.remove(); h2.remove()
                    st.success(f"Grad-CAM generated for: {top_path}")
                except Exception as e:
                    st.info(f"Grad-CAM unavailable: {e}")

        # ── RESULTS DISPLAY ────────────────────────────────
        st.divider()
        st.subheader("Results")

        # CTR metric card
        if ctr is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cardiothoracic Ratio", f"{ctr:.3f}",
                          delta="CARDIOMEGALY" if cardiomegaly else "Normal")
            with col2:
                st.metric("Findings Flagged", len(flags))
            with col3:
                urgent_count = sum(1 for f in flags if f["urgent"])
                st.metric("Urgent Findings", urgent_count)

        # Urgent alerts
        for f in flags:
            if f["urgent"]:
                st.error(f"⚠ URGENT — {f['pathology']}: Score {f['score']:.3f} [{f['severity']}]")

        # Cardiomegaly alert
        if cardiomegaly:
            st.error(f"⚠ CARDIOMEGALY DETECTED — CTR = {ctr:.3f} (Normal < 0.50). Recommend echocardiogram.")

        # ── Visual grid ────────────────────────────────────
        img_display = img_tensor[0].numpy()

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Original X-ray**")
            st.image(((img_display - img_display.min()) /
                      (img_display.max() - img_display.min() + 1e-8) * 255).astype(np.uint8),
                     use_container_width=True)

        with col_b:
            st.markdown("**Anatomical Segmentation**")
            if masks is not None:
                ov = np.stack([img_display] * 3, axis=-1)
                ov = (ov - ov.min()) / (ov.max() - ov.min() + 1e-8)
                cmap_seg = {"Heart": [1,.2,.2], "Left Lung": [.2,.6,1],
                            "Right Lung": [.2,.9,.4], "Aorta": [1,.8,0]}
                for struct, col in cmap_seg.items():
                    if struct in seg_targets:
                        m = skimage.transform.resize(masks[seg_targets.index(struct)], (224,224)) > 0.5
                        for ch in range(3):
                            ov[:,:,ch] = np.where(m, ov[:,:,ch]*.35 + col[ch]*.65, ov[:,:,ch])
                st.image((ov * 255).astype(np.uint8), use_container_width=True)
                st.caption("Red=Heart  Blue=L.Lung  Green=R.Lung  Yellow=Aorta")
            else:
                st.info("Segmentation unavailable")

        with col_c:
            st.markdown(f"**Grad-CAM: {top_path}**")
            if gcam is not None:
                norm = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
                rgb  = np.stack([norm]*3, axis=-1)
                heat = cm_module.jet(gcam)[:,:,:3]
                blended = (rgb * 0.45 + heat * 0.55)
                st.image((blended * 255).astype(np.uint8), use_container_width=True)
                st.caption("Red = region driving prediction")
            else:
                st.info("Grad-CAM unavailable")

        # ── Scores table ───────────────────────────────────
        st.divider()
        st.subheader("Pathology Scores")
        flag_set = {f["pathology"] for f in flags}

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Flagged Abnormalities**")
            if flags:
                for f in flags:
                    label = f"{f['pathology']} — {f['score']:.3f} [{f['severity']}]"
                    if f["urgent"]:
                        st.error(f"⚠ {label} [URGENT]")
                    elif f["cardiac"]:
                        st.warning(f"♥ {label} [CARDIAC]")
                    else:
                        st.warning(label)
            else:
                st.success("No abnormalities detected above threshold")

        with c2:
            st.markdown("**All Scores (sorted)**")
            import pandas as pd
            df = pd.DataFrame([
                {"Pathology": p, "Score": round(s, 4),
                 "Flagged": "⚠ Yes" if p in flag_set else "No",
                 "Threshold": THRESHOLDS.get(p, 0.45)}
                for p, s in scores.items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Clinical report ────────────────────────────────
        st.divider()
        st.subheader("Clinical Report")
        report_lines = [
            "CARDIOAI — CHEST X-RAY ANALYSIS REPORT",
            "AI-Assisted Screening — NOT a diagnostic replacement",
            "=" * 50,
        ]
        if not flags:
            report_lines.append("IMPRESSION: No significant findings detected.")
        else:
            urgent = [f for f in flags if f["urgent"]]
            report_lines.append(f"IMPRESSION: {'URGENT — ' if urgent else ''}{len(flags)} finding(s). Clinical review advised.")

        if ctr:
            report_lines.append(f"\nCARDIOTHORACIC RATIO: {ctr:.3f}")
            report_lines.append("  CARDIOMEGALY DETECTED" if cardiomegaly else "  Normal (CTR < 0.50)")

        if flags:
            report_lines.append(f"\nFINDINGS ({len(flags)}):")
            for f in flags:
                report_lines.append(
                    f"  {'!' if f['urgent'] else '-'} {f['pathology']:<22} "
                    f"Score: {f['score']:.3f}  [{f['severity']}]"
                    f"{' URGENT' if f['urgent'] else ''}"
                    f"{' CARDIAC' if f['cardiac'] else ''}"
                )
        report_lines.append("\nIMPORTANT: This is an AI screening tool.")
        report_lines.append("All findings must be confirmed by a radiologist.")

        st.text("\n".join(report_lines))


        st.warning(
            "This AI analysis is for screening assistance only. "
            "It does not replace clinical judgement or radiologist review. "
            "Do not make treatment decisions based solely on this output."
        )

    elif uploaded_xray and not cnn_ready:
        st.error("Install torchxrayvision and scikit-image to use this feature.")

    if not uploaded_xray:
        st.info(
            "Upload a chest PA or AP X-ray to begin analysis. "
            "The system will automatically detect pathologies, segment anatomy, "
            "compute cardiothoracic ratio, and generate Grad-CAM heatmaps."
        )
        st.markdown("""
        **What this module detects:**
        - Cardiomegaly, Pleural Effusion, Pulmonary Oedema (cardiac conditions)
        - Pneumonia, Atelectasis, Consolidation (lung infections)
        - Pneumothorax, Mass, Nodule (urgent findings)
        - Emphysema, Fibrosis, Pleural Thickening (chronic lung disease)
        - Full anatomical segmentation: Heart, Lungs, Aorta, Spine
        - Cardiothoracic ratio (CTR) with cardiomegaly threshold
        """)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════

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
