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
    page = st.radio("Navigate", ["🫀 Risk Prediction","🏥 Patient Retention","📊 Model Dashboard","📄 Clinical NLP","🔬 Medical Imaging","💊 Pharmaco-Intelligence","ℹ️ About"], label_visibility="collapsed")
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

            # ══════════════════════════════════════════════
            # RISK TRAJECTORY FORECASTING
            # ══════════════════════════════════════════════
            st.divider()
            st.subheader("📈 Risk Trajectory Forecast")
            st.caption(
                "Projected cardiovascular risk over the next 5 years under four scenarios, "
                "using Monte Carlo simulation with clinical biomarker progression rates."
            )

            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            # ── Biomarker progression rates (evidence-based) ───────────
            # Sources: AHA/ACC guidelines, Framingham Heart Study,
            # INTERHEART Africa sub-study for Nigerian populations
            # Each tuple: (mean annual change, std dev)
            # v2.1 — lri_delta keys present in all scenarios
            PROGRESSION = {
                # Without any intervention
                "no_intervention": {
                    "trestbps_delta": (2.1, 1.5),    # BP rises ~2 mmHg/yr untreated
                    "chol_delta":     (3.5, 2.8),    # Cholesterol rises 3.5 mg/dl/yr
                    "thalach_delta":  (-1.2, 0.9),   # Max HR declines ~1.2/yr
                    "oldpeak_delta":  (0.08, 0.05),  # ST depression worsens
                    "lri_delta":      (0.025, 0.015),# Lifestyle Risk Index worsens
                },
                # With lifestyle modification (diet + exercise)
                "lifestyle": {
                    "trestbps_delta": (-1.8, 1.2),
                    "chol_delta":     (-5.5, 3.0),
                    "thalach_delta":  (0.5, 0.8),
                    "oldpeak_delta":  (-0.04, 0.03),
                    "lri_delta":      (-0.03, 0.012),
                },
                # With medication (antihypertensive + statin)
                "medication": {
                    "trestbps_delta": (-4.5, 1.8),
                    "chol_delta":     (-18.0, 5.0),
                    "thalach_delta":  (-0.5, 0.6),
                    "oldpeak_delta":  (-0.02, 0.02),
                    "lri_delta":      (-0.01, 0.01),
                },
                # Full JoiHealth cardiac rehabilitation program
                "rehabilitation": {
                    "trestbps_delta": (-6.2, 1.5),
                    "chol_delta":     (-22.0, 4.5),
                    "thalach_delta":  (2.8, 1.0),
                    "oldpeak_delta":  (-0.07, 0.03),
                    "lri_delta":      (-0.06, 0.015),
                },
            }

            SCENARIO_LABELS = {
                "no_intervention": "No intervention",
                "lifestyle":       "Lifestyle changes",
                "medication":      "Medication",
                "rehabilitation":  "JoiHealth Rehabilitation",
            }

            SCENARIO_COLORS = {
                "no_intervention": "#E63946",
                "lifestyle":       "#FFB703",
                "medication":      "#00B4D8",
                "rehabilitation":  "#06D6A0",
            }

            N_SIM    = 600    # Monte Carlo paths
            N_MONTHS = 60     # 5-year horizon
            MONTHS   = list(range(0, N_MONTHS + 1, 3))  # quarterly

            np.random.seed(42)

            # ── Capture all patient values for simulation ──────────
            _base = {
                "age": age, "sex": sex, "cp": cp,
                "trestbps": trestbps, "chol": chol, "fbs": fbs,
                "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
                "lri_base": compute_lri(trestbps, chol, fbs, exang, oldpeak),
                "risk_now": risk_prob,
            }

            def simulate_trajectory(base, scenario_key, n_sim, months,
                                    _scaler=scaler, _model=xgb_model):
                """
                Monte Carlo simulation — all patient variables passed explicitly
                to avoid NameError from outer scope references.
                """
                deltas = PROGRESSION[scenario_key]
                paths  = np.zeros((n_sim, len(months)))

                for sim in range(n_sim):
                    bp_d   = np.random.normal(*deltas.get("trestbps_delta", (0, 0.5)))
                    chol_d = np.random.normal(*deltas.get("chol_delta",     (0, 1.0)))
                    hr_d   = np.random.normal(*deltas.get("thalach_delta",  (0, 0.5)))
                    op_d   = np.random.normal(*deltas.get("oldpeak_delta",  (0, 0.02)))
                    lri_d  = np.random.normal(*deltas.get("lri_delta",      (0, 0.01)))

                    for mi, month in enumerate(months):
                        yr = month / 12.0

                        bp_t   = float(np.clip(base["trestbps"] + bp_d   * yr, 90,  220))
                        chol_t = float(np.clip(base["chol"]     + chol_d * yr, 100, 600))
                        hr_t   = float(np.clip(base["thalach"]  + hr_d   * yr, 60,  202))
                        op_t   = float(np.clip(base["oldpeak"]  + op_d   * yr, 0,   6.2))

                        # Recompute LRI from projected biomarkers
                        lri_t = float(np.clip(
                            compute_lri(bp_t, chol_t, base["fbs"],
                                        base["exang"], op_t) + lri_d * yr * 0.3,
                            0, 1.0))

                        features_t = {
                            "age":                  base["age"] + yr,
                            "sex":                  base["sex"],
                            "cp":                   base["cp"],
                            "trestbps":             bp_t,
                            "chol":                 chol_t,
                            "fbs":                  base["fbs"],
                            "restecg":              base["restecg"],
                            "thalach":              hr_t,
                            "exang":                base["exang"],
                            "oldpeak":              op_t,
                            "slope":                base["slope"],
                            "ca":                   base["ca"],
                            "thal":                 base["thal"],
                            "lifestyle_risk_index": lri_t,
                        }

                        X_t = pd.DataFrame([features_t])
                        try:
                            X_s = _scaler.transform(X_t)
                            p   = float(_model.predict_proba(X_s)[0][1])
                        except Exception:
                            p   = float(base["risk_now"])
                        paths[sim, mi] = p

                return paths

            # ── Run all 4 scenarios ─────────────────────────────────
            with st.spinner("Running Monte Carlo simulation (600 paths × 4 scenarios)..."):
                all_paths = {}
                for scenario in PROGRESSION.keys():
                    all_paths[scenario] = simulate_trajectory(
                        _base, scenario, N_SIM, MONTHS)

            # ── Plot ────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            fig.patch.set_facecolor("#0D1B2A")

            for ax in axes:
                ax.set_facecolor("#111927")
                ax.tick_params(colors="white")
                for sp in ax.spines.values():
                    sp.set_color("#2C3E50")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")

            # ── Left: trajectory fan chart ───────────────────────
            ax1 = axes[0]

            # Risk zone bands
            ax1.axhspan(0.0,  0.30, alpha=0.06, color="#06D6A0")
            ax1.axhspan(0.30, 0.60, alpha=0.06, color="#FFB703")
            ax1.axhspan(0.60, 1.00, alpha=0.06, color="#E63946")
            ax1.axhline(0.30, color="#06D6A0", lw=0.8, ls="--", alpha=0.4)
            ax1.axhline(0.60, color="#E63946", lw=0.8, ls="--", alpha=0.4)
            ax1.text(1, 0.15, "Low risk", color="#06D6A0", fontsize=8, alpha=0.7)
            ax1.text(1, 0.44, "Moderate risk", color="#FFB703", fontsize=8, alpha=0.7)
            ax1.text(1, 0.75, "High risk", color="#E63946", fontsize=8, alpha=0.7)

            # Current risk dot
            ax1.scatter([0], [risk_prob], color="white", s=60, zorder=10,
                        label=f"Today ({risk_prob*100:.1f}%)")

            for scenario, paths in all_paths.items():
                color = SCENARIO_COLORS[scenario]
                label = SCENARIO_LABELS[scenario]
                median = np.median(paths, axis=0)
                p25    = np.percentile(paths, 25, axis=0)
                p75    = np.percentile(paths, 75, axis=0)
                p10    = np.percentile(paths, 10, axis=0)
                p90    = np.percentile(paths, 90, axis=0)

                x = [m/12 for m in MONTHS]
                ax1.fill_between(x, p10, p90, alpha=0.08, color=color)
                ax1.fill_between(x, p25, p75, alpha=0.18, color=color)
                ax1.plot(x, median, color=color, lw=2.2, label=label)

                # End-point label
                ax1.annotate(
                    f"{median[-1]*100:.0f}%",
                    xy=(x[-1], median[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    color=color, fontsize=8.5, fontweight="bold", va="center"
                )

            ax1.set_xlim(0, 5.3)
            ax1.set_ylim(0, 1.0)
            ax1.set_xlabel("Years from now", color="white", fontsize=10)
            ax1.set_ylabel("Cardiovascular risk probability", color="white", fontsize=10)
            ax1.set_title("5-Year Risk Trajectory (Monte Carlo)", color="white",
                          fontsize=12, pad=10)
            ax1.set_xticks([0, 1, 2, 3, 4, 5])
            ax1.set_xticklabels(["Now", "1yr", "2yr", "3yr", "4yr", "5yr"])
            ax1.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
            ax1.legend(loc="upper left", fontsize=8,
                       facecolor="#1A2A3A", labelcolor="white",
                       framealpha=0.85, edgecolor="#2C3E50")

            # ── Right: 5-year endpoint comparison bar chart ──────
            ax2 = axes[1]

            scenario_names = list(SCENARIO_LABELS.values())
            end_medians    = [np.median(all_paths[s][:, -1]) * 100
                               for s in PROGRESSION.keys()]
            end_p25        = [np.percentile(all_paths[s][:, -1], 25) * 100
                               for s in PROGRESSION.keys()]
            end_p75        = [np.percentile(all_paths[s][:, -1], 75) * 100
                               for s in PROGRESSION.keys()]
            colors_bar     = list(SCENARIO_COLORS.values())

            bars = ax2.barh(range(4), end_medians,
                            xerr=[
                                [m - p25 for m, p25 in zip(end_medians, end_p25)],
                                [p75 - m for m, p75 in zip(end_medians, end_p75)]
                            ],
                            color=colors_bar, height=0.55,
                            error_kw={"ecolor": "white", "capsize": 4,
                                      "alpha": 0.6, "lw": 1.2})

            ax2.set_yticks(range(4))
            ax2.set_yticklabels(scenario_names, color="white", fontsize=9)
            ax2.set_xlabel("Predicted risk at 5 years (%)", color="white", fontsize=10)
            ax2.set_title("5-Year Risk Endpoint Comparison", color="white",
                          fontsize=12, pad=10)
            ax2.axvline(x=30, color="#06D6A0", ls="--", lw=0.9, alpha=0.5)
            ax2.axvline(x=60, color="#E63946", ls="--", lw=0.9, alpha=0.5)
            ax2.set_xlim(0, 105)

            for i, (val, color) in enumerate(zip(end_medians, colors_bar)):
                ax2.text(val + 2, i, f"{val:.0f}%",
                         va="center", color=color, fontsize=10, fontweight="bold")

            # Benefit annotation
            worst = end_medians[0]
            best  = end_medians[3]
            reduction = worst - best
            if reduction > 2:
                ax2.annotate(
                    f"Rehab reduces 5yr risk by {reduction:.0f} pts",
                    xy=(best, 3), xytext=(best + 8, 1.5),
                    fontsize=8, color="#06D6A0",
                    arrowprops=dict(arrowstyle="->", color="#06D6A0", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D3A2A",
                              edgecolor="#06D6A0", alpha=0.8)
                )

            plt.tight_layout(pad=2.5)
            st.pyplot(fig)
            plt.close()

            # ── Key forecast metrics ────────────────────────────
            st.divider()
            st.markdown("**Forecast Summary — Key Numbers**")

            fc1, fc2, fc3, fc4 = st.columns(4)

            no_int_5yr  = np.median(all_paths["no_intervention"][:, -1]) * 100
            rehab_5yr   = np.median(all_paths["rehabilitation"][:, -1])  * 100
            lifestyle_1yr = np.median(all_paths["lifestyle"][:,
                            MONTHS.index(12) if 12 in MONTHS else 4]) * 100
            med_2yr     = np.median(all_paths["medication"][:,
                          MONTHS.index(24) if 24 in MONTHS else 8]) * 100

            with fc1:
                st.metric(
                    "5yr risk — no action",
                    f"{no_int_5yr:.0f}%",
                    delta=f"+{no_int_5yr - risk_prob*100:.0f}% from today",
                    delta_color="inverse"
                )
            with fc2:
                st.metric(
                    "5yr risk — with rehab",
                    f"{rehab_5yr:.0f}%",
                    delta=f"{rehab_5yr - risk_prob*100:+.0f}% from today",
                    delta_color="inverse"
                )
            with fc3:
                st.metric(
                    "Risk reduction — rehab vs no action",
                    f"{no_int_5yr - rehab_5yr:.0f} pts",
                    delta="at 5 years"
                )
            with fc4:
                # Time to cross 60% threshold without intervention
                medians_no_int = np.median(all_paths["no_intervention"], axis=0)
                threshold_crossings = [i for i, m in enumerate(medians_no_int)
                                       if m >= 0.60]
                if threshold_crossings:
                    cross_months = MONTHS[threshold_crossings[0]]
                    if cross_months < 12:
                        cross_label = f"{cross_months}m"
                    else:
                        cross_label = f"{cross_months//12}yr {cross_months%12}m" if cross_months%12 else f"{cross_months//12}yr"
                    st.metric("High-risk threshold", cross_label,
                              delta="without intervention", delta_color="inverse")
                else:
                    st.metric("High-risk threshold", "Not reached",
                              delta="within 5 years")

            # ── Narrative forecast interpretation ───────────────
            st.divider()
            st.markdown("**Clinical Forecast Interpretation**")

            # Select scenario text based on current risk
            if risk_prob < 0.30:
                outlook = "currently low"
                trajectory_warning = (
                    f"However, without lifestyle modification, the model projects "
                    f"risk increasing to **{no_int_5yr:.0f}%** within 5 years. "
                    f"This patient is on a trajectory toward moderate risk."
                )
            elif risk_prob < 0.60:
                outlook = "currently moderate"
                trajectory_warning = (
                    f"Without intervention, risk is projected to reach "
                    f"**{no_int_5yr:.0f}%** — entering high-risk territory — "
                    f"within 5 years. Early action now has the highest impact."
                )
            else:
                outlook = "currently high"
                trajectory_warning = (
                    f"Without urgent intervention, risk remains above 60% "
                    f"and could reach **{no_int_5yr:.0f}%** within 5 years. "
                    f"Immediate cardiology referral is indicated."
                )

            rehab_benefit = no_int_5yr - rehab_5yr
            med_benefit   = no_int_5yr - (np.median(all_paths["medication"][:,-1])*100)

            st.markdown(f"""
This patient's cardiovascular risk is **{outlook}** at **{risk_prob*100:.1f}%**.
{trajectory_warning}

**Scenario outcomes at 5 years:**
- **No intervention:** {no_int_5yr:.0f}% predicted risk — risk continues to rise with age and biomarker progression
- **Lifestyle changes alone:** {np.median(all_paths['lifestyle'][:,-1])*100:.0f}% — diet and exercise slow progression significantly
- **Medication (antihypertensive + statin):** {np.median(all_paths['medication'][:,-1])*100:.0f}% — medication produces the sharpest early reduction
- **JoiHealth Cardiac Rehabilitation:** {rehab_5yr:.0f}% — combined programme achieves the best long-term outcome, reducing 5-year risk by **{rehab_benefit:.0f} percentage points** compared to no action

The shaded bands represent the 25th–75th and 10th–90th percentile uncertainty range across 600 simulated patient trajectories, reflecting natural variability in biomarker progression. Wider bands indicate higher uncertainty.

> ⚠ **Note:** This forecast uses evidence-based biomarker progression rates from the Framingham Heart Study and AHA/ACC guidelines, combined with the patient's current ML-predicted risk score. It is a decision support tool — not a clinical prediction. Actual outcomes depend on adherence, comorbidities, and factors not captured in this model.
""")

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

# ══════════════════════════════════════════════════════════
# PAGE 6 — PHARMACO-INTELLIGENCE
# ══════════════════════════════════════════════════════════

elif "Pharmaco-Intelligence" in page:
    st.title("💊 Pharmaco-Intelligence")
    st.caption(
        "Evidence-based drug recommendations, interaction checking, lab interpretation, "
        "pharmacy claims analysis, and AI-generated clinical documentation — "
        "grounded in 2026 ACC/AHA and ESC/EAS guidelines."
    )
    st.warning(
        "⚕ This module is a **clinical decision support tool only**. "
        "All outputs must be reviewed and approved by a licensed pharmacist or physician "
        "before any prescribing or dispensing decision is made."
    )

    # ── Clinical rules engine ──────────────────────────────────────
    # Hard-coded evidence base: 2026 ACC/AHA, 2025 AACE, ESC/EAS 2025
    # ──────────────────────────────────────────────────────────────

    DRUG_DB = {
        # ── ANTIHYPERTENSIVES ──────────────────────────────────
        "Amlodipine": {
            "class": "Calcium Channel Blocker (CCB)",
            "indications": ["Hypertension", "Angina"],
            "standard_dose": "5–10 mg once daily",
            "mechanism": "Blocks L-type voltage-gated Ca²⁺ channels in vascular smooth muscle → peripheral vasodilation → BP reduction",
            "rehab_notes": "Safe during exercise. Does not blunt heart rate response. May cause peripheral oedema — monitor in rehab patients.",
            "contraindications": ["Severe aortic stenosis", "Cardiogenic shock"],
            "cautions": ["Heart failure with reduced EF (use with caution)", "Elderly — start 2.5mg"],
            "nigeria_availability": "High — generic widely available",
            "cost_tier": "Low (₦500–₦1,500/month)",
            "interactions": {
                "Simvastatin": ("Major", "Amlodipine inhibits CYP3A4 — simvastatin plasma levels increase up to 77%. Cap simvastatin at 20mg/day or switch to rosuvastatin."),
                "Cyclosporine": ("Major", "Amlodipine increases cyclosporine levels significantly."),
                "Rifampicin": ("Moderate", "CYP3A4 inducer reduces amlodipine efficacy — may need dose increase."),
            },
            "lab_monitoring": ["Renal function (eGFR)", "Electrolytes (oedema risk)"],
            "guideline_ref": "2025 AHA/ACC Hypertension Guideline — First-line agent for HTN",
        },
        "Lisinopril": {
            "class": "ACE Inhibitor (ACEi)",
            "indications": ["Hypertension", "Heart Failure", "Post-MI cardioprotection", "Diabetic nephropathy"],
            "standard_dose": "5–40 mg once daily",
            "mechanism": "Inhibits angiotensin-converting enzyme → ↓ angiotensin II → vasodilation + ↓ aldosterone → ↓ BP + ↓ cardiac remodelling",
            "rehab_notes": "First-choice post-MI rehab drug. Reduces exercise-induced BP surge. Monitor for dry cough (10–15% of patients — more common in African patients, up to 30–40%).",
            "contraindications": ["Pregnancy", "Bilateral renal artery stenosis", "History of angioedema with ACEi", "Hyperkalaemia (K⁺ >5.5 mmol/L)"],
            "cautions": ["CKD (reduce dose if eGFR <30)", "Concurrent NSAID use (↓ efficacy + ↑ renal risk)", "Elderly — risk of first-dose hypotension"],
            "nigeria_availability": "High — generic available, on NHIS formulary",
            "cost_tier": "Low (₦400–₦1,200/month)",
            "interactions": {
                "Spironolactone": ("Major", "Combined with ACEi — significant hyperkalaemia risk. Monitor K⁺ within 1 week of starting."),
                "Ibuprofen/NSAIDs": ("Major", "NSAIDs reduce ACEi antihypertensive effect and ↑ acute kidney injury risk. Avoid combination."),
                "Allopurinol": ("Moderate", "↑ risk of hypersensitivity reactions and Stevens-Johnson syndrome."),
                "Potassium supplements": ("Major", "Combined hyperkalaemia risk. Avoid unless K⁺ <3.5."),
            },
            "lab_monitoring": ["eGFR and creatinine (baseline, 1–2 weeks after start)", "Serum potassium", "Urinalysis (proteinuria in diabetes)"],
            "guideline_ref": "2026 ACC/AHA — Preferred agent in HF with reduced EF, post-MI, diabetic nephropathy",
        },
        "Losartan": {
            "class": "Angiotensin Receptor Blocker (ARB)",
            "indications": ["Hypertension", "Heart Failure (ACEi-intolerant)", "Diabetic nephropathy", "Stroke prevention in LVH"],
            "standard_dose": "25–100 mg once daily",
            "mechanism": "Blocks AT₁ receptor → prevents angiotensin II vasoconstriction. Unlike ACEi, does not accumulate bradykinin — lower cough rate.",
            "rehab_notes": "Preferred over ACEi in patients who develop ACEi cough (common in Nigerian patients). Identical cardiac protection benefits.",
            "contraindications": ["Pregnancy", "Bilateral renal artery stenosis", "Hyperkalaemia"],
            "cautions": ["CKD (reduce dose)", "Volume depletion (risk of hypotension)", "Do NOT combine with ACEi or aliskiren (ONTARGET trial showed harm)"],
            "nigeria_availability": "Moderate — available in major pharmacies",
            "cost_tier": "Low-Medium (₦1,000–₦3,000/month)",
            "interactions": {
                "ACE Inhibitors": ("Contraindicated", "Dual RAS blockade — ↑ hypotension, hyperkalaemia, renal failure. ONTARGET trial confirmed harm."),
                "NSAIDs": ("Major", "Reduce ARB efficacy and ↑ renal toxicity."),
                "Lithium": ("Moderate", "ARBs increase lithium levels — toxicity risk."),
            },
            "lab_monitoring": ["eGFR", "Serum potassium", "Blood pressure"],
            "guideline_ref": "2025 AHA/ACC Hypertension — First-line alternative when ACEi not tolerated",
        },
        "Atenolol": {
            "class": "Beta-Blocker (β₁-selective)",
            "indications": ["Hypertension", "Angina", "Post-MI rate control", "Arrhythmia"],
            "standard_dose": "25–100 mg once daily",
            "mechanism": "Selective β₁-adrenoceptor blockade → ↓ heart rate, ↓ cardiac output, ↓ renin release → ↓ BP",
            "rehab_notes": "⚠ CRITICAL REHAB NOTE: Beta-blockers blunt heart rate response to exercise. Use Karvonen formula with resting HR correction. Target perceived exertion (RPE 11–14) rather than heart rate targets in patients on beta-blockers.",
            "contraindications": ["Bradycardia (<50 bpm)", "Heart block (2nd/3rd degree)", "Severe asthma/COPD (use bisoprolol if needed)", "Cardiogenic shock"],
            "cautions": ["Diabetes — may mask hypoglycaemia symptoms", "Peripheral vascular disease", "Abrupt withdrawal — rebound tachycardia/angina"],
            "nigeria_availability": "Very High — cheapest beta-blocker in Nigeria",
            "cost_tier": "Very Low (₦200–₦600/month)",
            "interactions": {
                "Verapamil/Diltiazem": ("Contraindicated", "Combined negative chronotropic effect — severe bradycardia, AV block, asystole."),
                "Clonidine": ("Major", "Abrupt clonidine withdrawal during beta-blockade → severe hypertensive rebound."),
                "Insulin": ("Moderate", "Masks tachycardia of hypoglycaemia. Non-selective beta-blockers also impair glycogen mobilisation."),
                "NSAIDs": ("Moderate", "NSAIDs reduce antihypertensive effect of beta-blockers."),
            },
            "lab_monitoring": ["Resting heart rate", "Blood glucose (in diabetes)", "Lipid panel (beta-blockers can raise TGs slightly)"],
            "guideline_ref": "2026 ACC/AHA — Preferred post-MI; note: no longer first-line for uncomplicated HTN alone",
        },
        "Bisoprolol": {
            "class": "Beta-Blocker (β₁-highly selective)",
            "indications": ["Heart Failure with reduced EF", "Hypertension", "Angina", "Rate control in AF"],
            "standard_dose": "1.25–10 mg once daily (start low, titrate slowly)",
            "mechanism": "Highly selective β₁ blockade → ↓ HR + ↓ cardiac remodelling. Safer in COPD/asthma than atenolol.",
            "rehab_notes": "Gold-standard beta-blocker in cardiac rehab HFrEF patients. Start at 1.25mg and uptitrate every 2 weeks as tolerated. Blunts HR — use RPE scale for exercise intensity.",
            "contraindications": ["Bradycardia", "Significant AV block", "Decompensated HF (acute phase)"],
            "cautions": ["COPD (monitor for bronchospasm — less risk than non-selective)", "Peripheral arterial disease"],
            "nigeria_availability": "Moderate — available but more expensive than atenolol",
            "cost_tier": "Medium (₦1,500–₦4,000/month)",
            "interactions": {
                "Verapamil": ("Contraindicated", "Severe bradycardia and AV block."),
                "Amiodarone": ("Major", "Additive bradycardia and risk of complete heart block."),
                "Digoxin": ("Moderate", "Additive bradycardia. Monitor HR closely."),
            },
            "lab_monitoring": ["Heart rate", "Blood pressure", "LVEF (echocardiogram)"],
            "guideline_ref": "CIBIS-II trial, ESC HF Guidelines 2025 — Class I for HFrEF",
        },
        # ── LIPID-LOWERING ─────────────────────────────────────
        "Atorvastatin": {
            "class": "Statin (HMG-CoA Reductase Inhibitor)",
            "indications": ["Dyslipidaemia", "Primary/Secondary ASCVD prevention", "Post-MI statin therapy"],
            "standard_dose": "10–80 mg once daily (evening preferred)",
            "mechanism": "Inhibits HMG-CoA reductase → ↓ hepatic cholesterol synthesis → ↑ LDL receptor expression → ↓ LDL-C 35–55%",
            "rehab_notes": "⚠ Monitor for statin-induced myopathy during increased exercise in rehab. Symptoms: muscle aches, weakness, dark urine. Check CK if symptomatic. Risk ↑ with high-intensity exercise.",
            "contraindications": ["Active liver disease", "Pregnancy/breastfeeding", "Unexplained persistent ↑ transaminases"],
            "cautions": ["Concurrent CYP3A4 inhibitors (clarithromycin, amlodipine, grapefruit)", "Hypothyroidism (↑ myopathy risk)", "Heavy alcohol use"],
            "nigeria_availability": "High — generic available",
            "cost_tier": "Low-Medium (₦800–₦2,500/month)",
            "interactions": {
                "Amlodipine": ("Moderate", "CYP3A4 inhibition increases atorvastatin AUC ~18%. Monitor for myopathy."),
                "Clarithromycin": ("Major", "Strong CYP3A4 inhibitor — atorvastatin levels ↑ dramatically. Suspend statin during course."),
                "Warfarin": ("Moderate", "Statins can ↑ INR. Monitor INR when starting/stopping."),
                "Fibrates": ("Major", "Combination ↑ myopathy risk significantly. If needed, use fenofibrate (lower risk than gemfibrozil)."),
                "Colchicine": ("Moderate", "↑ myopathy risk — particularly in renal impairment."),
            },
            "lab_monitoring": ["LFTs (baseline, then if symptomatic)", "CK (if muscle symptoms)", "Lipid panel (4–12 weeks after start)", "HbA1c (statins slightly ↑ diabetes risk)"],
            "guideline_ref": "2026 ACC/AHA Dyslipidaemia — High-intensity statin for ASCVD risk ≥7.5% or LDL >190",
        },
        "Rosuvastatin": {
            "class": "Statin (High-intensity)",
            "indications": ["Dyslipidaemia", "ASCVD prevention", "Familial hypercholesterolaemia"],
            "standard_dose": "5–40 mg once daily",
            "mechanism": "More potent HMG-CoA inhibitor than atorvastatin. Not metabolised by CYP3A4 — fewer drug interactions.",
            "rehab_notes": "Preferred statin in patients on multiple cardiac drugs (avoids CYP3A4 interactions). LDL reduction: 38–65%.",
            "contraindications": ["Active liver disease", "Pregnancy", "Myopathy with previous statin"],
            "cautions": ["Asian patients — use lower doses (↑ bioavailability)", "Severe renal impairment — avoid >10mg"],
            "nigeria_availability": "Moderate — more expensive than atorvastatin",
            "cost_tier": "Medium (₦2,000–₦6,000/month)",
            "interactions": {
                "Antacids (aluminium/magnesium)": ("Moderate", "Reduce rosuvastatin absorption by ~50%. Give 2 hours apart."),
                "Warfarin": ("Moderate", "May increase INR. Monitor."),
                "Fibrates": ("Major", "↑ myopathy risk."),
            },
            "lab_monitoring": ["Lipid panel", "LFTs", "CK if muscle symptoms", "eGFR (dose-adjust in renal impairment)"],
            "guideline_ref": "2026 ACC/AHA — Preferred when CYP3A4 interactions are a concern",
        },
        "Metformin": {
            "class": "Biguanide (Antidiabetic)",
            "indications": ["Type 2 Diabetes", "Pre-diabetes", "Insulin resistance"],
            "standard_dose": "500–2000 mg daily in divided doses with meals",
            "mechanism": "Activates AMPK → ↓ hepatic gluconeogenesis + ↑ peripheral insulin sensitivity. Modest LDL-lowering and weight benefit.",
            "rehab_notes": "Safe during exercise. Monitor for exercise-induced lactic acidosis in patients with renal impairment. Hold 48h before contrast procedures.",
            "contraindications": ["eGFR <30 mL/min/1.73m²", "Acute heart failure", "Severe liver disease", "IV contrast within 48 hours"],
            "cautions": ["eGFR 30–45 — reduce dose, monitor closely", "Heavy alcohol use (lactic acidosis risk)", "Vitamin B12 deficiency with long-term use"],
            "nigeria_availability": "Very High — cheap, on NHIS formulary",
            "cost_tier": "Very Low (₦200–₦800/month)",
            "interactions": {
                "Alcohol": ("Major", "↑ lactic acidosis risk."),
                "Contrast dye": ("Major", "Nephrotoxicity + lactic acidosis. Hold 48h before and after."),
                "Corticosteroids": ("Moderate", "Steroids antagonise metformin's glucose-lowering effect."),
            },
            "lab_monitoring": ["eGFR (every 6 months)", "HbA1c (3-monthly)", "Vitamin B12 (annual)", "Fasting glucose"],
            "guideline_ref": "2025 ADA Standards — First-line T2DM with cardiovascular benefit",
        },
        "Aspirin": {
            "class": "Antiplatelet (COX-1 inhibitor)",
            "indications": ["Secondary ASCVD prevention (post-MI, stroke)", "ACS", "Stable angina"],
            "standard_dose": "75–100 mg once daily (low-dose)",
            "mechanism": "Irreversibly acetylates COX-1 in platelets → ↓ thromboxane A₂ → ↓ platelet aggregation. Lasts platelet lifetime (7–10 days).",
            "rehab_notes": "Continue throughout rehab. No exercise restriction. Watch for GI bleeding — prescribe with PPI (omeprazole 20mg) if high GI risk.",
            "contraindications": ["Active peptic ulcer/GI bleeding", "Aspirin hypersensitivity", "Haemophilia", "Note: No longer recommended for PRIMARY prevention in most patients — 2022 USPSTF"],
            "cautions": ["↑ bleeding with warfarin or DOACs", "Asthma (aspirin-exacerbated respiratory disease in ~10%)"],
            "nigeria_availability": "Very High — widely available OTC",
            "cost_tier": "Very Low (₦100–₦400/month)",
            "interactions": {
                "Warfarin": ("Major", "Significantly ↑ bleeding risk. Monitor INR closely. Usually intentional in high-risk patients."),
                "Ibuprofen/NSAIDs": ("Major", "Ibuprofen can competitively antagonise aspirin's antiplatelet effect. Use paracetamol for analgesia instead."),
                "Clopidogrel": ("Moderate/Intentional", "Dual antiplatelet used post-ACS/stent. ↑ bleeding risk — PPI recommended."),
            },
            "lab_monitoring": ["FBC (bleeding)", "Faecal occult blood (if GI symptoms)", "INR (if on warfarin)"],
            "guideline_ref": "2026 ACC/AHA — Recommended for all established ASCVD (secondary prevention)",
        },
        "Spironolactone": {
            "class": "Aldosterone Antagonist / Potassium-sparing Diuretic",
            "indications": ["Heart Failure with reduced EF (HFrEF)", "Resistant hypertension", "Hypokalaemia", "Primary hyperaldosteronism"],
            "standard_dose": "12.5–50 mg once daily",
            "mechanism": "Blocks mineralocorticoid receptor → ↓ sodium/water retention + ↓ potassium excretion + ↓ cardiac fibrosis",
            "rehab_notes": "Monitor electrolytes before and during rehab — hyperkalaemia can cause fatal arrhythmias. Check K⁺ and creatinine at 1 week, 1 month, then 3-monthly.",
            "contraindications": ["Hyperkalaemia (K⁺ >5.0 mmol/L)", "Severe renal impairment (eGFR <30)", "Addison's disease"],
            "cautions": ["Concurrent ACEi/ARB — highest hyperkalaemia risk", "Gynaecomastia (common — consider eplerenone)", "Menstrual irregularities in women"],
            "nigeria_availability": "Moderate",
            "cost_tier": "Low-Medium (₦800–₦2,500/month)",
            "interactions": {
                "ACE Inhibitors/ARBs": ("Major", "Hyperkalaemia. Check K⁺ within 1 week of combination. Life-threatening if unmonitored."),
                "NSAIDs": ("Moderate", "Reduce diuretic efficacy and ↑ renal toxicity."),
                "Digoxin": ("Moderate", "Spironolactone can ↑ digoxin levels."),
            },
            "lab_monitoring": ["Serum potassium (CRITICAL)", "Creatinine/eGFR", "Blood pressure"],
            "guideline_ref": "RALES + EMPHASIS-HF trials. ESC HF 2025 — Class I for HFrEF if EF <35%",
        },
        "Warfarin": {
            "class": "Vitamin K Antagonist (Anticoagulant)",
            "indications": ["Atrial fibrillation (stroke prevention)", "Mechanical heart valves", "VTE treatment/prevention", "Post-MI with LV thrombus"],
            "standard_dose": "Dose individualised to INR — typically 2–10 mg daily",
            "mechanism": "Inhibits vitamin K epoxide reductase → ↓ synthesis of clotting factors II, VII, IX, X and proteins C and S",
            "rehab_notes": "⚠ Exercise intensity monitoring essential. High-intensity exercise contraindicated with supratherapeutic INR. Avoid contact sports. Head injury risk during exercise — educate patient. Check INR before starting rehab programme.",
            "contraindications": ["Active major bleeding", "Pregnancy (1st/3rd trimester)", "Severe liver disease", "INR >3.5 — hold and review"],
            "cautions": ["Many drug and food interactions", "Narrow therapeutic window", "Elderly — ↑ bleeding risk", "Green leafy vegetables (vitamin K) — counsel on consistent dietary intake"],
            "nigeria_availability": "Moderate — requires INR monitoring which is challenging outside major cities",
            "cost_tier": "Low (₦300–₦800/month) but INR monitoring adds cost",
            "interactions": {
                "Aspirin": ("Major", "↑ bleeding significantly."),
                "NSAIDs": ("Major", "↑ bleeding and GI injury."),
                "Statins": ("Moderate", "Most statins ↑ INR — monitor when starting."),
                "Clarithromycin/Azithromycin": ("Major", "Antibiotics kill gut flora producing vitamin K → INR rises sharply."),
                "Fluconazole": ("Major", "Strong CYP2C9 inhibitor — warfarin levels increase substantially."),
                "Amiodarone": ("Major", "Dramatically ↑ INR. Reduce warfarin dose by 30–50% when starting amiodarone."),
                "Paracetamol": ("Moderate", ">2g/day paracetamol ↑ INR — use lowest effective dose."),
            },
            "lab_monitoring": ["INR (target 2.0–3.0 for AF; 2.5–3.5 for mechanical valves)", "FBC", "LFTs", "Creatinine"],
            "guideline_ref": "2023 ACC/AHA AFib — Preferred in mechanical valve patients; DOACs preferred in AF",
        },
    }

    # ── LAB REFERENCE RANGES ──────────────────────────────────────
    LAB_RANGES = {
        "Total Cholesterol":  {"unit": "mg/dL", "optimal": "<200",   "borderline": "200–239", "high": "≥240",  "lo": 0,   "hi": 500,  "optimal_max": 200, "high_min": 240},
        "LDL Cholesterol":    {"unit": "mg/dL", "optimal": "<70 (ASCVD) / <100 (moderate risk)", "borderline": "100–129", "high": "≥130", "lo": 0, "hi": 300, "optimal_max": 100, "high_min": 130},
        "HDL Cholesterol":    {"unit": "mg/dL", "optimal": ">60 (M) / >50 (F)", "borderline": "40–60", "high": "<40 (↑CVD risk)", "lo": 0, "hi": 150, "optimal_max": 999, "high_min": 999},
        "Triglycerides":      {"unit": "mg/dL", "optimal": "<150",   "borderline": "150–199", "high": "≥200",  "lo": 0,   "hi": 2000, "optimal_max": 150, "high_min": 200},
        "Fasting Glucose":    {"unit": "mg/dL", "optimal": "70–99",  "borderline": "100–125 (pre-DM)", "high": "≥126 (DM)", "lo": 0, "hi": 600, "optimal_max": 99, "high_min": 126},
        "HbA1c":              {"unit": "%",     "optimal": "<5.7",   "borderline": "5.7–6.4 (pre-DM)", "high": "≥6.5 (DM)", "lo": 0, "hi": 20, "optimal_max": 5.7, "high_min": 6.5},
        "eGFR":               {"unit": "mL/min/1.73m²", "optimal": "≥90", "borderline": "60–89 (mild ↓)", "high": "<60 (CKD)", "lo": 0, "hi": 150, "optimal_max": 999, "high_min": 999},
        "Serum Creatinine":   {"unit": "mg/dL", "optimal": "0.6–1.2 (M) / 0.5–1.1 (F)", "borderline": "1.2–2.0", "high": ">2.0", "lo": 0, "hi": 20, "optimal_max": 1.2, "high_min": 2.0},
        "Serum Potassium":    {"unit": "mmol/L","optimal": "3.5–5.0","borderline": "3.0–3.5 or 5.0–5.5", "high": "<3.0 or >5.5 (dangerous)", "lo": 0, "hi": 10, "optimal_max": 5.0, "high_min": 5.5},
        "Serum Sodium":       {"unit": "mmol/L","optimal": "135–145","borderline": "130–135 or 145–150", "high": "<130 or >150 (critical)", "lo": 0, "hi": 200, "optimal_max": 145, "high_min": 150},
        "Haemoglobin":        {"unit": "g/dL",  "optimal": "13–17 (M) / 12–16 (F)", "borderline": "10–12", "high": "<10 (anaemia — limits rehab)", "lo": 0, "hi": 25, "optimal_max": 999, "high_min": 999},
        "CK (Creatine Kinase)":{"unit": "U/L",  "optimal": "20–200", "borderline": "200–1000 (↑ statin risk)", "high": ">1000 (myopathy)", "lo": 0, "hi": 10000, "optimal_max": 200, "high_min": 1000},
        "INR":                {"unit": "",      "optimal": "2.0–3.0 (AF) / 2.5–3.5 (valve)", "borderline": "1.5–2.0 or 3.0–3.5", "high": ">3.5 (↑ bleeding)", "lo": 0, "hi": 10, "optimal_max": 3.0, "high_min": 3.5},
        "ALT (liver)":        {"unit": "U/L",   "optimal": "<40",    "borderline": "40–120 (3×ULN)", "high": ">120 (hold statin/drug)", "lo": 0, "hi": 2000, "optimal_max": 40, "high_min": 120},
        "BNP / NT-proBNP":    {"unit": "pg/mL", "optimal": "<100 (BNP)", "borderline": "100–400", "high": ">400 (likely HF)", "lo": 0, "hi": 35000, "optimal_max": 100, "high_min": 400},
    }

    # ─────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💊 Drug Recommender",
        "⚠ Interaction Checker",
        "🔬 Lab Interpreter",
        "🧬 Drug-Condition Analyser",
        "🏪 Pharmacy Claims",
        "📋 Documentation Generator",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — DRUG RECOMMENDER
    # ════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Evidence-Based Drug Recommendation")
        st.caption("Enter patient profile to receive guideline-recommended drug options with mechanism explanations.")

        c1, c2 = st.columns(2)
        with c1:
            pt_age   = st.number_input("Age (years)", 18, 100, 60, key="dr_age")
            pt_sex   = st.selectbox("Sex", ["Male", "Female"], key="dr_sex")
            pt_bmi   = st.number_input("BMI (kg/m²)", 15.0, 60.0, 28.0, step=0.1, key="dr_bmi")
            pt_sbp   = st.number_input("Systolic BP (mmHg)", 80, 220, 145, key="dr_sbp")
            pt_dbp   = st.number_input("Diastolic BP (mmHg)", 40, 140, 90, key="dr_dbp")
        with c2:
            pt_ldl   = st.number_input("LDL Cholesterol (mg/dL)", 0, 400, 135, key="dr_ldl")
            pt_egfr  = st.number_input("eGFR (mL/min/1.73m²)", 5, 150, 72, key="dr_egfr")
            pt_k     = st.number_input("Serum Potassium (mmol/L)", 2.0, 7.0, 4.2, step=0.1, key="dr_k")
            pt_diag  = st.multiselect("Active diagnoses", [
                "Hypertension", "Heart Failure (HFrEF)", "Post-MI", "Atrial Fibrillation",
                "Type 2 Diabetes", "Dyslipidaemia", "Stable Angina", "CKD", "Stroke/TIA",
                "Obesity", "COPD/Asthma"
            ], default=["Hypertension", "Dyslipidaemia"], key="dr_diag")
            pt_allergy = st.text_input("Known drug allergies / intolerances", "ACE inhibitor cough", key="dr_allergy")

        if st.button("🔍 Generate Drug Recommendations", type="primary", use_container_width=True, key="btn_recommend"):
            recs = []
            reasons = []

            # ── Hypertension logic ──────────────────────────────
            if "Hypertension" in pt_diag:
                has_hf      = "Heart Failure (HFrEF)" in pt_diag
                has_dm      = "Type 2 Diabetes" in pt_diag
                has_ckd     = "CKD" in pt_diag or pt_egfr < 60
                has_post_mi = "Post-MI" in pt_diag
                has_asthma  = "COPD/Asthma" in pt_diag
                ace_intol   = any(x in pt_allergy.lower() for x in ["ace", "lisinopril", "cough", "perindopril", "ramipril"])
                high_k      = pt_k > 5.0

                # First-line HTN
                if not ace_intol and not high_k:
                    recs.append(("Lisinopril", "First-line ACEi — particularly beneficial with diabetes/CKD/HF/post-MI", "★★★★★"))
                else:
                    recs.append(("Losartan", "ARB preferred — ACEi intolerance or hyperkalaemia risk", "★★★★★"))
                    reasons.append(f"⚠ ACEi avoided: {'ACEi cough/intolerance noted' if ace_intol else 'K⁺ = ' + str(pt_k) + ' mmol/L — hyperkalaemia risk'}")

                if pt_sbp > 150 or pt_ldl > 100:
                    recs.append(("Amlodipine", "Add CCB for additional BP lowering or when ACEi/ARB alone insufficient", "★★★★☆"))

                if has_post_mi or has_hf:
                    if not has_asthma:
                        recs.append(("Bisoprolol", "Beta-blocker — cardioprotective post-MI and in HFrEF. Start low, uptitrate.", "★★★★★"))
                    else:
                        reasons.append("⚠ Beta-blocker caution: COPD/Asthma listed — use bisoprolol (most selective) only if benefit outweighs risk")
                elif not has_asthma:
                    recs.append(("Atenolol", "Beta-blocker option if rate control or angina component", "★★★☆☆"))

                if has_hf and pt_egfr >= 30 and not high_k:
                    recs.append(("Spironolactone", "Add aldosterone antagonist in HFrEF (EF <35%) — reduces mortality 30%", "★★★★★"))

            # ── Dyslipidaemia logic ─────────────────────────────
            if "Dyslipidaemia" in pt_diag or pt_ldl > 130:
                has_ascvd = any(x in pt_diag for x in ["Post-MI", "Stroke/TIA", "Stable Angina", "Heart Failure (HFrEF)"])
                # Check for CYP3A4 interactions
                on_amlodipine = any("amlodipine" in r[0].lower() for r in recs)
                if on_amlodipine:
                    recs.append(("Rosuvastatin", "Preferred statin — no CYP3A4 interaction with amlodipine. Avoids atorvastatin dose cap.", "★★★★★"))
                    reasons.append("ℹ Rosuvastatin chosen over atorvastatin: amlodipine CYP3A4 interaction limits atorvastatin to 20mg max")
                else:
                    recs.append(("Atorvastatin", f"{'High-intensity statin 40-80mg — ASCVD established' if has_ascvd else 'Moderate-intensity statin 20-40mg — primary prevention'}", "★★★★★"))

            # ── Diabetes logic ──────────────────────────────────
            if "Type 2 Diabetes" in pt_diag:
                if pt_egfr >= 30:
                    recs.append(("Metformin", "First-line T2DM — weight neutral, CV-safe, low cost, widely available in Nigeria", "★★★★★"))
                else:
                    reasons.append("⚠ Metformin: eGFR <30 — contraindicated. Consider alternative antidiabetic.")

            # ── Post-MI antiplatelet ────────────────────────────
            if "Post-MI" in pt_diag or "Stable Angina" in pt_diag:
                recs.append(("Aspirin", "Antiplatelet — lifelong secondary prevention post-MI. Prescribe with PPI if GI risk.", "★★★★★"))

            # ── eGFR adjustments ────────────────────────────────
            if pt_egfr < 30:
                reasons.append("⚠ eGFR <30: Avoid metformin, NSAIDs, spironolactone. Reduce ACEi/ARB dose. Avoid high-dose statins.")
            elif pt_egfr < 60:
                reasons.append("ℹ eGFR 30–60: Monitor creatinine/K⁺ closely on ACEi/ARB. Reduce metformin dose.")

            # ── Display recommendations ─────────────────────────
            if recs:
                st.success(f"**{len(recs)} drug recommendation(s) generated** based on patient profile")
                for drug, rationale, stars in recs:
                    info = DRUG_DB.get(drug, {})
                    with st.expander(f"{stars}  **{drug}** — {info.get('class','')}", expanded=True):
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            st.markdown(f"**Standard dose:** {info.get('standard_dose','See BNF')}")
                            st.markdown(f"**Rationale:** {rationale}")
                            st.markdown(f"**Mechanism:** {info.get('mechanism','')}")
                            if info.get("rehab_notes"):
                                st.info(f"🏃 **Rehab note:** {info['rehab_notes']}")
                        with col_b:
                            st.markdown(f"**Nigeria availability:** {info.get('nigeria_availability','')}")
                            st.markdown(f"**Cost tier:** {info.get('cost_tier','')}")
                            if info.get("contraindications"):
                                st.markdown("**Contraindications:**")
                                for c in info["contraindications"][:3]:
                                    st.markdown(f"  - {c}")
                            st.caption(f"📚 {info.get('guideline_ref','')}")

                if reasons:
                    st.divider()
                    st.markdown("**Clinical decision notes:**")
                    for r in reasons:
                        st.markdown(r)
            else:
                st.info("Please select at least one diagnosis to generate recommendations.")

    # ════════════════════════════════════════════════════════
    # TAB 2 — DRUG INTERACTION CHECKER
    # ════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Drug-Drug Interaction Checker")
        st.caption("Enter the patient's current medication list to screen for interactions.")

        all_drug_names = list(DRUG_DB.keys())
        selected_drugs = st.multiselect(
            "Select all medications the patient is currently taking:",
            all_drug_names + ["Ibuprofen/NSAIDs", "Clarithromycin", "Amiodarone",
                               "Digoxin", "Verapamil/Diltiazem", "Colchicine",
                               "Fluconazole", "Rifampicin", "Cyclosporine",
                               "Insulin", "Clonidine", "Allopurinol"],
            key="int_drugs"
        )

        if st.button("🔍 Check Interactions", type="primary", use_container_width=True, key="btn_interact"):
            if len(selected_drugs) < 2:
                st.warning("Add at least 2 medications to check for interactions.")
            else:
                found = []
                for i, drug_a in enumerate(selected_drugs):
                    info_a = DRUG_DB.get(drug_a, {})
                    interactions = info_a.get("interactions", {})
                    for drug_b in selected_drugs[i+1:]:
                        if drug_b in interactions:
                            sev, desc = interactions[drug_b]
                            found.append((drug_a, drug_b, sev, desc))
                        # Check reverse
                        info_b = DRUG_DB.get(drug_b, {})
                        if drug_a in info_b.get("interactions", {}):
                            sev, desc = info_b["interactions"][drug_a]
                            if not any(f[0]==drug_b and f[1]==drug_a for f in found):
                                found.append((drug_b, drug_a, sev, desc))

                if not found:
                    st.success("✓ No significant interactions detected among selected medications.")
                    st.caption("Note: This checker covers common cardiac drug interactions. Always verify with a pharmacist.")
                else:
                    # Sort by severity
                    sev_order = {"Contraindicated": 0, "Major": 1, "Moderate": 2, "Minor": 3}
                    found.sort(key=lambda x: sev_order.get(x[2], 4))

                    contraind = [f for f in found if f[2] == "Contraindicated"]
                    major     = [f for f in found if f[2] == "Major"]
                    moderate  = [f for f in found if f[2] == "Moderate"]
                    minor     = [f for f in found if f[2] in ("Minor", "Moderate/Intentional")]

                    if contraind:
                        st.error(f"🚫 **{len(contraind)} CONTRAINDICATED combination(s)**")
                    if major:
                        st.error(f"⚠ **{len(major)} MAJOR interaction(s)**")
                    if moderate:
                        st.warning(f"⚠ **{len(moderate)} MODERATE interaction(s)**")

                    for drug_a, drug_b, sev, desc in found:
                        color = "🚫" if sev=="Contraindicated" else ("🔴" if sev=="Major" else "🟡")
                        with st.expander(f"{color} **{drug_a}** + **{drug_b}** — [{sev}]", expanded=(sev in ["Contraindicated","Major"])):
                            st.markdown(f"**Severity:** {sev}")
                            st.markdown(f"**Clinical significance:** {desc}")
                            if sev == "Contraindicated":
                                st.error("Action required: Do NOT co-prescribe. Choose an alternative.")
                            elif sev == "Major":
                                st.warning("Action required: Enhanced monitoring, dose adjustment, or consider alternative.")
                            else:
                                st.info("Action: Monitor clinically. Usually manageable with dose adjustment or timing.")

    # ════════════════════════════════════════════════════════
    # TAB 3 — LAB INTERPRETER
    # ════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Lab Result Interpreter")
        st.caption("Enter patient lab values. The system interprets each result, flags abnormalities, and generates drug adjustment recommendations.")

        lab_values = {}
        st.markdown("**Enter available lab results (leave at 0 to skip):**")

        cols = st.columns(3)
        lab_items = list(LAB_RANGES.items())
        for i, (lab, ref) in enumerate(lab_items):
            with cols[i % 3]:
                val = st.number_input(
                    f"{lab} ({ref['unit']})",
                    min_value=float(ref["lo"]),
                    max_value=float(ref["hi"]),
                    value=0.0, step=0.1,
                    key=f"lab_{lab}"
                )
                if val > 0:
                    lab_values[lab] = val

        if st.button("🔬 Interpret Lab Results", type="primary", use_container_width=True, key="btn_labs"):
            if not lab_values:
                st.info("Enter at least one lab value to interpret.")
            else:
                st.divider()
                st.markdown("### Lab Interpretation Report")

                drug_adjustments = []
                critical_flags = []

                for lab, val in lab_values.items():
                    ref = LAB_RANGES[lab]
                    opt_max = ref["optimal_max"]
                    hi_min  = ref["high_min"]

                    # Determine status
                    if lab in ["eGFR", "HDL Cholesterol", "Haemoglobin"]:
                        # Higher is better
                        if val >= 60:
                            status, color, icon = "Normal", "success", "✅"
                        elif val >= 30:
                            status, color, icon = "Low — monitor", "warning", "⚠"
                        else:
                            status, color, icon = "Critically Low", "error", "🔴"
                    elif lab == "Serum Potassium":
                        if 3.5 <= val <= 5.0:
                            status, color, icon = "Normal", "success", "✅"
                        elif (3.0 <= val < 3.5) or (5.0 < val <= 5.5):
                            status, color, icon = "Borderline — monitor", "warning", "⚠"
                        else:
                            status, color, icon = "CRITICAL — dangerous arrhythmia risk", "error", "🚨"
                            critical_flags.append(f"K⁺ = {val} — {'HYPERKALAEMIA' if val > 5.5 else 'HYPOKALAEMIA'}: Stop/reduce ACEi/ARB/spironolactone immediately if K⁺ >5.5. Consider IV correction if K⁺ <3.0.")
                    elif lab == "INR":
                        if 2.0 <= val <= 3.0:
                            status, color, icon = "Therapeutic (AF target)", "success", "✅"
                        elif val < 2.0:
                            status, color, icon = "Sub-therapeutic — ↑ clot risk", "warning", "⚠"
                        elif val <= 3.5:
                            status, color, icon = "High — ↑ bleeding risk", "warning", "⚠"
                        else:
                            status, color, icon = "DANGEROUS — hold warfarin", "error", "🚨"
                            critical_flags.append(f"INR = {val} — SUPRATHERAPEUTIC: Hold warfarin. Assess for active bleeding. Consider vitamin K.")
                    else:
                        if val <= opt_max:
                            status, color, icon = "Optimal", "success", "✅"
                        elif val < hi_min:
                            status, color, icon = "Borderline — review", "warning", "⚠"
                        else:
                            status, color, icon = "Elevated — action needed", "error", "🔴"

                    # Drug adjustment triggers
                    if lab == "eGFR":
                        if val < 30:
                            drug_adjustments.append(f"eGFR {val:.0f}: **Stop metformin** (lactic acidosis). **Halve ACEi/ARB dose**. Avoid spironolactone. **Do not use NSAIDs**.")
                        elif val < 60:
                            drug_adjustments.append(f"eGFR {val:.0f}: **Reduce metformin dose** (max 1g/day). Monitor K⁺ and creatinine on ACEi/ARB weekly.")
                    if lab == "Serum Potassium" and val > 5.5:
                        drug_adjustments.append(f"K⁺ {val}: **Stop spironolactone immediately**. **Reduce/stop ACEi or ARB**. **Stop potassium supplements**. Recheck K⁺ in 24–48h.")
                    if lab == "Serum Potassium" and val < 3.5:
                        drug_adjustments.append(f"K⁺ {val}: Risk of digoxin toxicity and arrhythmia. Consider **potassium supplementation**. Check if thiazide diuretic contributing.")
                    if lab == "CK (Creatine Kinase)" and val > 1000:
                        drug_adjustments.append(f"CK {val:.0f} U/L: **Suspend statin immediately** (myopathy/rhabdomyolysis risk). Recheck in 1 week. Hydrate. Monitor renal function.")
                    elif lab == "CK (Creatine Kinase)" and val > 200:
                        drug_adjustments.append(f"CK {val:.0f} U/L (elevated, <5×ULN): **Reduce statin intensity**. Avoid high-intensity exercise until resolved.")
                    if lab == "ALT (liver)" and val > 120:
                        drug_adjustments.append(f"ALT {val:.0f} U/L (>3×ULN): **Hold statin and hepatotoxic drugs**. Recheck LFTs in 4–6 weeks. Review alcohol history.")
                    if lab == "LDL Cholesterol" and val > 190:
                        drug_adjustments.append(f"LDL {val:.0f}: Severely elevated — screen for **familial hypercholesterolaemia**. High-intensity statin + ezetimibe. Consider PCSK9 inhibitor referral.")
                    if lab == "HbA1c" and val >= 6.5:
                        drug_adjustments.append(f"HbA1c {val}% — Diabetes range: **Initiate or intensify antidiabetic therapy**. Metformin first-line (if eGFR permits).")
                    if lab == "BNP / NT-proBNP" and val > 400:
                        drug_adjustments.append(f"BNP/NT-proBNP {val:.0f}: Consistent with **Heart Failure**. Urgent cardiology review. Initiate ACEi/ARB + beta-blocker + loop diuretic.")

                    # Display
                    getattr(st, color)(f"{icon} **{lab}:** {val} {ref['unit']} — {status} | Reference: Optimal {ref['optimal']}")

                if critical_flags:
                    st.divider()
                    st.error("🚨 CRITICAL VALUES — IMMEDIATE ACTION REQUIRED")
                    for flag in critical_flags:
                        st.error(flag)

                if drug_adjustments:
                    st.divider()
                    st.subheader("Drug Adjustment Recommendations")
                    for adj in drug_adjustments:
                        st.warning(f"💊 {adj}")

                if not drug_adjustments and not critical_flags:
                    st.success("✓ No drug adjustments required based on entered lab values.")

    # ════════════════════════════════════════════════════════
    # TAB 4 — DRUG-CONDITION ANALYSER
    # ════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Drug-Condition Suitability Analyser")
        st.caption("Understand WHY a specific drug is or is not appropriate for this patient's exact clinical situation.")

        col1, col2 = st.columns(2)
        with col1:
            selected_drug = st.selectbox("Select drug to analyse:", list(DRUG_DB.keys()), key="dca_drug")
        with col2:
            patient_context = st.text_area(
                "Patient clinical context (diagnoses, vitals, labs, current meds):",
                placeholder="e.g. 66yr female, HTN, BMI 40.6, BP 109/73, eGFR 72, K+ 4.2, on amlodipine, has ACE inhibitor cough",
                height=80, key="dca_context"
            )

        if st.button("🧬 Analyse Drug-Condition Fit", type="primary", use_container_width=True, key="btn_dca"):
            drug_info = DRUG_DB.get(selected_drug, {})

            # Rules-based suitability assessment
            st.subheader(f"Analysis: {selected_drug}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Drug class:** {drug_info.get('class','')}")
                st.markdown(f"**Mechanism:** {drug_info.get('mechanism','')}")
                st.markdown(f"**Standard dose:** {drug_info.get('standard_dose','')}")
                st.markdown(f"**Licensed indications:** {', '.join(drug_info.get('indications',[]))}")
            with col_b:
                st.markdown(f"**Nigeria availability:** {drug_info.get('nigeria_availability','')}")
                st.markdown(f"**Cost tier:** {drug_info.get('cost_tier','')}")
                st.info(f"🏃 **Cardiac Rehab note:** {drug_info.get('rehab_notes','No specific rehab notes')}")

            st.divider()
            st.markdown("**Contraindications:**")
            for ci in drug_info.get("contraindications", []):
                st.error(f"🚫 {ci}")

            st.markdown("**Clinical cautions:**")
            for ca in drug_info.get("cautions", []):
                st.warning(f"⚠ {ca}")

            st.markdown("**Lab monitoring required:**")
            for lm in drug_info.get("lab_monitoring", []):
                st.info(f"🔬 {lm}")

            if drug_info.get("interactions"):
                st.markdown("**Key interactions:**")
                for drug_b, (sev, desc) in list(drug_info["interactions"].items())[:5]:
                    icon = "🚫" if sev=="Contraindicated" else ("🔴" if sev=="Major" else "🟡")
                    st.markdown(f"  {icon} **+ {drug_b}** [{sev}]: {desc}")

            st.divider()
            st.caption(f"📚 Guideline reference: {drug_info.get('guideline_ref','')}")

            # AI-enhanced explanation if Gemini available
            if get_secret("GOOGLE_API_KEY") and patient_context.strip():
                with st.spinner("Generating AI clinical explanation..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=get_secret("GOOGLE_API_KEY"))
                        model_ai = genai.GenerativeModel("gemini-2.0-flash")
                        prompt = f"""You are a clinical pharmacist specialising in cardiac rehabilitation.

Patient context: {patient_context}

Drug being considered: {selected_drug}
Drug class: {drug_info.get('class','')}
Mechanism: {drug_info.get('mechanism','')}
Indications: {', '.join(drug_info.get('indications',[]))}
Contraindications: {', '.join(drug_info.get('contraindications',[]))}

Write a concise clinical explanation (3-4 paragraphs) for a Nigerian clinician covering:
1. Whether this drug is appropriate for THIS specific patient and why
2. How this patient's specific clinical features (age, BMI, diagnoses, labs, co-meds) affect the drug's pharmacokinetics and pharmacodynamics
3. What to monitor in this patient specifically
4. Any Nigeria-specific considerations (cost, availability, alternatives)

Be specific to the patient context provided. Be direct and clinical. Do not give generic information."""
                        response = model_ai.generate_content(prompt)
                        st.divider()
                        st.markdown("**AI Clinical Pharmacist Explanation:**")
                        st.markdown(response.text)
                        st.caption("AI-generated — verify with licensed pharmacist before prescribing decisions")
                    except Exception as e:
                        st.info(f"AI explanation unavailable: {e}")

    # ════════════════════════════════════════════════════════
    # TAB 5 — PHARMACY CLAIMS INTELLIGENCE
    # ════════════════════════════════════════════════════════
    with tab5:
        st.subheader("Pharmacy Claims Intelligence")
        st.caption("Enter the patient's current prescription regimen for a comprehensive medication review.")

        st.markdown("**Current prescription regimen:**")
        claims = []
        for i in range(6):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
            with c1:
                drug_name = st.text_input(f"Drug {i+1} name", key=f"cl_name_{i}",
                                          placeholder="e.g. Amlodipine")
            with c2:
                dose = st.text_input(f"Dose", key=f"cl_dose_{i}", placeholder="5mg")
            with c3:
                freq = st.selectbox(f"Frequency", ["Once daily","Twice daily","Three times daily","As needed","Weekly"],
                                    key=f"cl_freq_{i}")
            with c4:
                duration = st.text_input(f"Duration / Supply", key=f"cl_dur_{i}", placeholder="30 days")
            if drug_name.strip():
                claims.append({"name": drug_name.strip(), "dose": dose, "freq": freq, "duration": duration})

        diag_ctx = st.text_input("Patient diagnoses (for appropriateness check):",
                                  placeholder="Hypertension, Dyslipidaemia, Type 2 Diabetes, Post-MI",
                                  key="cl_diag")

        if st.button("🏪 Analyse Prescription", type="primary", use_container_width=True, key="btn_claims"):
            if not claims:
                st.info("Enter at least one drug to analyse.")
            else:
                st.divider()
                st.markdown(f"### Medication Review — {len(claims)} drug(s)")

                total_cost_low = 0
                total_cost_high = 0

                for claim in claims:
                    drug = claim["name"]
                    info = DRUG_DB.get(drug, {})

                    with st.expander(f"**{drug}** — {claim['dose']} {claim['freq']}", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            if info:
                                st.success(f"✓ Drug recognised: {info.get('class','')}")
                                st.markdown(f"**Standard dose:** {info.get('standard_dose','')}")
                                st.markdown(f"**Nigeria availability:** {info.get('nigeria_availability','')}")
                                st.markdown(f"**Cost tier:** {info.get('cost_tier','')}")

                                # Rough cost estimate
                                cost_str = info.get("cost_tier","")
                                import re
                                nums = re.findall(r'[\d,]+', cost_str.replace(",",""))
                                if len(nums) >= 2:
                                    total_cost_low  += int(nums[0])
                                    total_cost_high += int(nums[1])
                            else:
                                st.warning(f"⚠ Drug not found in database — manual review required")
                        with c2:
                            if info:
                                st.markdown("**Rehab considerations:**")
                                st.info(info.get("rehab_notes","No specific rehab notes"))
                                if info.get("lab_monitoring"):
                                    st.markdown("**Monitoring required:**")
                                    for lm in info["lab_monitoring"]:
                                        st.caption(f"• {lm}")

                if total_cost_low > 0:
                    st.divider()
                    st.markdown(f"**Estimated monthly medication cost:** ₦{total_cost_low:,} – ₦{total_cost_high:,}")
                    if total_cost_high > 20000:
                        st.warning("💰 High medication burden — consider NHIS coverage or generic alternatives.")

                # Cross-drug interaction check
                drug_names = [c["name"] for c in claims if c["name"] in DRUG_DB]
                if len(drug_names) >= 2:
                    st.divider()
                    st.markdown("**Auto interaction scan:**")
                    any_found = False
                    for i, da in enumerate(drug_names):
                        for db in drug_names[i+1:]:
                            info_a = DRUG_DB.get(da, {})
                            if db in info_a.get("interactions", {}):
                                sev, desc = info_a["interactions"][db]
                                icon = "🚫" if sev=="Contraindicated" else ("🔴" if sev=="Major" else "🟡")
                                st.warning(f"{icon} **{da} + {db}** [{sev}]: {desc}")
                                any_found = True
                    if not any_found:
                        st.success("✓ No interactions detected in current prescription")

    # ════════════════════════════════════════════════════════
    # TAB 6 — DOCUMENTATION GENERATOR
    # ════════════════════════════════════════════════════════
    with tab6:
        st.subheader("AI Documentation Generator")
        st.caption("Auto-generate pharmaceutical care plans, discharge summaries, and patient counselling notes.")

        doc_type = st.selectbox("Document type:", [
            "Pharmaceutical Care Plan",
            "Discharge Medication Summary",
            "Patient Drug Counselling Sheet",
            "Medication Reconciliation Report",
            "Cardiac Rehab Pharmacy Review",
        ], key="doc_type")

        c1, c2 = st.columns(2)
        with c1:
            doc_patient_name    = st.text_input("Patient name", key="doc_name")
            doc_patient_age     = st.number_input("Age", 18, 100, 60, key="doc_age")
            doc_patient_sex     = st.selectbox("Sex", ["Male", "Female"], key="doc_sex")
            doc_diagnoses       = st.text_area("Diagnoses", height=80,
                                                placeholder="Hypertension Stage 2, Dyslipidaemia, T2DM",
                                                key="doc_diag")
        with c2:
            doc_medications     = st.text_area("Current medications + doses",
                                               placeholder="Amlodipine 5mg OD\nLisinopril 10mg OD\nAtorvastatin 40mg ON\nMetformin 1g BD\nAspirin 75mg OD",
                                               height=100, key="doc_meds")
            doc_labs            = st.text_area("Recent labs (optional)",
                                               placeholder="BP 145/90, LDL 135, HbA1c 7.2%, eGFR 72, K+ 4.3",
                                               height=80, key="doc_labs")
            doc_additional      = st.text_area("Additional clinical notes",
                                               placeholder="Patient on cardiac rehab programme. Statin myopathy concern.",
                                               height=60, key="doc_additional")

        if st.button("📋 Generate Document", type="primary", use_container_width=True, key="btn_doc"):
            if not doc_medications.strip():
                st.warning("Please enter medications to generate documentation.")
            elif not get_secret("GOOGLE_API_KEY"):
                st.error("GOOGLE_API_KEY required for AI documentation generation. Add to Streamlit secrets.")
            else:
                with st.spinner(f"Generating {doc_type}..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=get_secret("GOOGLE_API_KEY"))
                        model_ai = genai.GenerativeModel("gemini-2.0-flash")

                        prompts = {
                            "Pharmaceutical Care Plan": f"""You are a clinical pharmacist at JoiHealth Polyclinics, Nigeria's premier cardiac rehabilitation centre.

Generate a complete, professional Pharmaceutical Care Plan for:
Patient: {doc_patient_name or 'Patient'}, {doc_patient_age}yr {doc_patient_sex}
Diagnoses: {doc_diagnoses}
Medications: {doc_medications}
Labs: {doc_labs}
Notes: {doc_additional}

Include these sections:
1. Drug Therapy Problem Identification (actual and potential problems)
2. Pharmacist Goals of Therapy
3. Pharmaceutical Care Plan (drug, indication, goal, monitoring parameter, timeframe)
4. Drug Interaction Assessment
5. Patient Counselling Points
6. Monitoring Plan (what to check, when, target values)
7. Follow-up plan

Format professionally for a Nigerian hospital record. Be specific and clinical.""",

                            "Discharge Medication Summary": f"""You are a clinical pharmacist at JoiHealth Polyclinics.

Generate a Discharge Medication Summary for:
Patient: {doc_patient_name or 'Patient'}, {doc_patient_age}yr {doc_patient_sex}
Diagnoses: {doc_diagnoses}
Discharge Medications: {doc_medications}
Labs at discharge: {doc_labs}
Notes: {doc_additional}

Include: medication name/dose/frequency/route, indication for each drug, key counselling points, what to avoid, when to seek urgent review, follow-up date recommendation.""",

                            "Patient Drug Counselling Sheet": f"""Generate a patient-friendly Drug Counselling Sheet in plain English (avoid jargon) for a Nigerian patient.

Patient: {doc_patient_name or 'Patient'}, {doc_patient_age}yr {doc_patient_sex}
Medications: {doc_medications}
Diagnoses: {doc_diagnoses}
Notes: {doc_additional}

For each medication: what it is, what it does in simple terms, how/when to take it, common side effects to watch for, what foods/drinks to avoid, what to do if a dose is missed. Include emergency warning signs that require going to hospital immediately.""",

                            "Medication Reconciliation Report": f"""Generate a Medication Reconciliation Report for:
Patient: {doc_patient_name or 'Patient'}, {doc_patient_age}yr {doc_patient_sex}
Diagnoses: {doc_diagnoses}
Current medication list: {doc_medications}
Labs: {doc_labs}
Notes: {doc_additional}

Identify: discrepancies between expected therapy and current regimen, omissions (drugs that should be present given diagnoses), duplications, inappropriate drugs, dose discrepancies. Provide recommendations for each finding.""",

                            "Cardiac Rehab Pharmacy Review": f"""You are the pharmacist for JoiHealth cardiac rehabilitation programme.

Generate a Cardiac Rehab Pharmacy Review for:
Patient: {doc_patient_name or 'Patient'}, {doc_patient_age}yr {doc_patient_sex}
Diagnoses: {doc_diagnoses}
Medications: {doc_medications}
Labs: {doc_labs}
Notes: {doc_additional}

Cover: appropriateness of current regimen for cardiac rehab, exercise-drug interactions (especially beta-blockers and heart rate targets, statins and myopathy risk, anticoagulants and exercise intensity), medication optimisation recommendations, monitoring plan for rehab duration, patient adherence strategies.""",
                        }

                        response = model_ai.generate_content(prompts[doc_type])
                        doc_text = response.text

                        st.divider()
                        st.markdown(f"### {doc_type}")
                        st.markdown(f"**Patient:** {doc_patient_name or 'Patient'} | **Date:** {pd.Timestamp.now().strftime('%d %B %Y')}")
                        st.markdown(f"**Generated by:** CardioAI Pharmaco-Intelligence | JoiHealth Polyclinics")
                        st.divider()
                        st.markdown(doc_text)
                        st.divider()
                        st.caption("⚕ This document is AI-generated and must be reviewed and signed by a licensed pharmacist before use in clinical practice.")

                        # Download as text
                        full_doc = f"{doc_type}\nPatient: {doc_patient_name or 'Patient'} | Date: {pd.Timestamp.now().strftime('%d %B %Y')}\n{'='*60}\n\n{doc_text}\n\n[Generated by CardioAI Pharmaco-Intelligence — JoiHealth Polyclinics]"
                        st.download_button(
                            "⬇ Download Document",
                            full_doc,
                            file_name=f"pharma_{doc_type.replace(' ','_').lower()}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Document generation error: {e}")

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

    st.caption("© 2025 Gboh-Igbara D. Charles — JoiHealth | cardioai-joihealth.streamlit.app")# ============================================================
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

            # ══════════════════════════════════════════════
            # RISK TRAJECTORY FORECASTING
            # ══════════════════════════════════════════════
            st.divider()
            st.subheader("📈 Risk Trajectory Forecast")
            st.caption(
                "Projected cardiovascular risk over the next 5 years under four scenarios, "
                "using Monte Carlo simulation with clinical biomarker progression rates."
            )

            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            # ── Biomarker progression rates (evidence-based) ───────────
            # Sources: AHA/ACC guidelines, Framingham Heart Study,
            # INTERHEART Africa sub-study for Nigerian populations
            # Each tuple: (mean annual change, std dev)
            # v2.1 — lri_delta keys present in all scenarios
            PROGRESSION = {
                # Without any intervention
                "no_intervention": {
                    "trestbps_delta": (2.1, 1.5),    # BP rises ~2 mmHg/yr untreated
                    "chol_delta":     (3.5, 2.8),    # Cholesterol rises 3.5 mg/dl/yr
                    "thalach_delta":  (-1.2, 0.9),   # Max HR declines ~1.2/yr
                    "oldpeak_delta":  (0.08, 0.05),  # ST depression worsens
                    "lri_delta":      (0.025, 0.015),# Lifestyle Risk Index worsens
                },
                # With lifestyle modification (diet + exercise)
                "lifestyle": {
                    "trestbps_delta": (-1.8, 1.2),
                    "chol_delta":     (-5.5, 3.0),
                    "thalach_delta":  (0.5, 0.8),
                    "oldpeak_delta":  (-0.04, 0.03),
                    "lri_delta":      (-0.03, 0.012),
                },
                # With medication (antihypertensive + statin)
                "medication": {
                    "trestbps_delta": (-4.5, 1.8),
                    "chol_delta":     (-18.0, 5.0),
                    "thalach_delta":  (-0.5, 0.6),
                    "oldpeak_delta":  (-0.02, 0.02),
                    "lri_delta":      (-0.01, 0.01),
                },
                # Full JoiHealth cardiac rehabilitation program
                "rehabilitation": {
                    "trestbps_delta": (-6.2, 1.5),
                    "chol_delta":     (-22.0, 4.5),
                    "thalach_delta":  (2.8, 1.0),
                    "oldpeak_delta":  (-0.07, 0.03),
                    "lri_delta":      (-0.06, 0.015),
                },
            }

            SCENARIO_LABELS = {
                "no_intervention": "No intervention",
                "lifestyle":       "Lifestyle changes",
                "medication":      "Medication",
                "rehabilitation":  "JoiHealth Rehabilitation",
            }

            SCENARIO_COLORS = {
                "no_intervention": "#E63946",
                "lifestyle":       "#FFB703",
                "medication":      "#00B4D8",
                "rehabilitation":  "#06D6A0",
            }

            N_SIM    = 600    # Monte Carlo paths
            N_MONTHS = 60     # 5-year horizon
            MONTHS   = list(range(0, N_MONTHS + 1, 3))  # quarterly

            np.random.seed(42)

            # ── Capture all patient values for simulation ──────────
            _base = {
                "age": age, "sex": sex, "cp": cp,
                "trestbps": trestbps, "chol": chol, "fbs": fbs,
                "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
                "lri_base": compute_lri(trestbps, chol, fbs, exang, oldpeak),
                "risk_now": risk_prob,
            }

            def simulate_trajectory(base, scenario_key, n_sim, months,
                                    _scaler=scaler, _model=xgb_model):
                """
                Monte Carlo simulation — all patient variables passed explicitly
                to avoid NameError from outer scope references.
                """
                deltas = PROGRESSION[scenario_key]
                paths  = np.zeros((n_sim, len(months)))

                for sim in range(n_sim):
                    bp_d   = np.random.normal(*deltas.get("trestbps_delta", (0, 0.5)))
                    chol_d = np.random.normal(*deltas.get("chol_delta",     (0, 1.0)))
                    hr_d   = np.random.normal(*deltas.get("thalach_delta",  (0, 0.5)))
                    op_d   = np.random.normal(*deltas.get("oldpeak_delta",  (0, 0.02)))
                    lri_d  = np.random.normal(*deltas.get("lri_delta",      (0, 0.01)))

                    for mi, month in enumerate(months):
                        yr = month / 12.0

                        bp_t   = float(np.clip(base["trestbps"] + bp_d   * yr, 90,  220))
                        chol_t = float(np.clip(base["chol"]     + chol_d * yr, 100, 600))
                        hr_t   = float(np.clip(base["thalach"]  + hr_d   * yr, 60,  202))
                        op_t   = float(np.clip(base["oldpeak"]  + op_d   * yr, 0,   6.2))

                        # Recompute LRI from projected biomarkers
                        lri_t = float(np.clip(
                            compute_lri(bp_t, chol_t, base["fbs"],
                                        base["exang"], op_t) + lri_d * yr * 0.3,
                            0, 1.0))

                        features_t = {
                            "age":                  base["age"] + yr,
                            "sex":                  base["sex"],
                            "cp":                   base["cp"],
                            "trestbps":             bp_t,
                            "chol":                 chol_t,
                            "fbs":                  base["fbs"],
                            "restecg":              base["restecg"],
                            "thalach":              hr_t,
                            "exang":                base["exang"],
                            "oldpeak":              op_t,
                            "slope":                base["slope"],
                            "ca":                   base["ca"],
                            "thal":                 base["thal"],
                            "lifestyle_risk_index": lri_t,
                        }

                        X_t = pd.DataFrame([features_t])
                        try:
                            X_s = _scaler.transform(X_t)
                            p   = float(_model.predict_proba(X_s)[0][1])
                        except Exception:
                            p   = float(base["risk_now"])
                        paths[sim, mi] = p

                return paths

            # ── Run all 4 scenarios ─────────────────────────────────
            with st.spinner("Running Monte Carlo simulation (600 paths × 4 scenarios)..."):
                all_paths = {}
                for scenario in PROGRESSION.keys():
                    all_paths[scenario] = simulate_trajectory(
                        _base, scenario, N_SIM, MONTHS)

            # ── Plot ────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            fig.patch.set_facecolor("#0D1B2A")

            for ax in axes:
                ax.set_facecolor("#111927")
                ax.tick_params(colors="white")
                for sp in ax.spines.values():
                    sp.set_color("#2C3E50")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")

            # ── Left: trajectory fan chart ───────────────────────
            ax1 = axes[0]

            # Risk zone bands
            ax1.axhspan(0.0,  0.30, alpha=0.06, color="#06D6A0")
            ax1.axhspan(0.30, 0.60, alpha=0.06, color="#FFB703")
            ax1.axhspan(0.60, 1.00, alpha=0.06, color="#E63946")
            ax1.axhline(0.30, color="#06D6A0", lw=0.8, ls="--", alpha=0.4)
            ax1.axhline(0.60, color="#E63946", lw=0.8, ls="--", alpha=0.4)
            ax1.text(1, 0.15, "Low risk", color="#06D6A0", fontsize=8, alpha=0.7)
            ax1.text(1, 0.44, "Moderate risk", color="#FFB703", fontsize=8, alpha=0.7)
            ax1.text(1, 0.75, "High risk", color="#E63946", fontsize=8, alpha=0.7)

            # Current risk dot
            ax1.scatter([0], [risk_prob], color="white", s=60, zorder=10,
                        label=f"Today ({risk_prob*100:.1f}%)")

            for scenario, paths in all_paths.items():
                color = SCENARIO_COLORS[scenario]
                label = SCENARIO_LABELS[scenario]
                median = np.median(paths, axis=0)
                p25    = np.percentile(paths, 25, axis=0)
                p75    = np.percentile(paths, 75, axis=0)
                p10    = np.percentile(paths, 10, axis=0)
                p90    = np.percentile(paths, 90, axis=0)

                x = [m/12 for m in MONTHS]
                ax1.fill_between(x, p10, p90, alpha=0.08, color=color)
                ax1.fill_between(x, p25, p75, alpha=0.18, color=color)
                ax1.plot(x, median, color=color, lw=2.2, label=label)

                # End-point label
                ax1.annotate(
                    f"{median[-1]*100:.0f}%",
                    xy=(x[-1], median[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    color=color, fontsize=8.5, fontweight="bold", va="center"
                )

            ax1.set_xlim(0, 5.3)
            ax1.set_ylim(0, 1.0)
            ax1.set_xlabel("Years from now", color="white", fontsize=10)
            ax1.set_ylabel("Cardiovascular risk probability", color="white", fontsize=10)
            ax1.set_title("5-Year Risk Trajectory (Monte Carlo)", color="white",
                          fontsize=12, pad=10)
            ax1.set_xticks([0, 1, 2, 3, 4, 5])
            ax1.set_xticklabels(["Now", "1yr", "2yr", "3yr", "4yr", "5yr"])
            ax1.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
            ax1.legend(loc="upper left", fontsize=8,
                       facecolor="#1A2A3A", labelcolor="white",
                       framealpha=0.85, edgecolor="#2C3E50")

            # ── Right: 5-year endpoint comparison bar chart ──────
            ax2 = axes[1]

            scenario_names = list(SCENARIO_LABELS.values())
            end_medians    = [np.median(all_paths[s][:, -1]) * 100
                               for s in PROGRESSION.keys()]
            end_p25        = [np.percentile(all_paths[s][:, -1], 25) * 100
                               for s in PROGRESSION.keys()]
            end_p75        = [np.percentile(all_paths[s][:, -1], 75) * 100
                               for s in PROGRESSION.keys()]
            colors_bar     = list(SCENARIO_COLORS.values())

            bars = ax2.barh(range(4), end_medians,
                            xerr=[
                                [m - p25 for m, p25 in zip(end_medians, end_p25)],
                                [p75 - m for m, p75 in zip(end_medians, end_p75)]
                            ],
                            color=colors_bar, height=0.55,
                            error_kw={"ecolor": "white", "capsize": 4,
                                      "alpha": 0.6, "lw": 1.2})

            ax2.set_yticks(range(4))
            ax2.set_yticklabels(scenario_names, color="white", fontsize=9)
            ax2.set_xlabel("Predicted risk at 5 years (%)", color="white", fontsize=10)
            ax2.set_title("5-Year Risk Endpoint Comparison", color="white",
                          fontsize=12, pad=10)
            ax2.axvline(x=30, color="#06D6A0", ls="--", lw=0.9, alpha=0.5)
            ax2.axvline(x=60, color="#E63946", ls="--", lw=0.9, alpha=0.5)
            ax2.set_xlim(0, 105)

            for i, (val, color) in enumerate(zip(end_medians, colors_bar)):
                ax2.text(val + 2, i, f"{val:.0f}%",
                         va="center", color=color, fontsize=10, fontweight="bold")

            # Benefit annotation
            worst = end_medians[0]
            best  = end_medians[3]
            reduction = worst - best
            if reduction > 2:
                ax2.annotate(
                    f"Rehab reduces 5yr risk by {reduction:.0f} pts",
                    xy=(best, 3), xytext=(best + 8, 1.5),
                    fontsize=8, color="#06D6A0",
                    arrowprops=dict(arrowstyle="->", color="#06D6A0", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D3A2A",
                              edgecolor="#06D6A0", alpha=0.8)
                )

            plt.tight_layout(pad=2.5)
            st.pyplot(fig)
            plt.close()

            # ── Key forecast metrics ────────────────────────────
            st.divider()
            st.markdown("**Forecast Summary — Key Numbers**")

            fc1, fc2, fc3, fc4 = st.columns(4)

            no_int_5yr  = np.median(all_paths["no_intervention"][:, -1]) * 100
            rehab_5yr   = np.median(all_paths["rehabilitation"][:, -1])  * 100
            lifestyle_1yr = np.median(all_paths["lifestyle"][:,
                            MONTHS.index(12) if 12 in MONTHS else 4]) * 100
            med_2yr     = np.median(all_paths["medication"][:,
                          MONTHS.index(24) if 24 in MONTHS else 8]) * 100

            with fc1:
                st.metric(
                    "5yr risk — no action",
                    f"{no_int_5yr:.0f}%",
                    delta=f"+{no_int_5yr - risk_prob*100:.0f}% from today",
                    delta_color="inverse"
                )
            with fc2:
                st.metric(
                    "5yr risk — with rehab",
                    f"{rehab_5yr:.0f}%",
                    delta=f"{rehab_5yr - risk_prob*100:+.0f}% from today",
                    delta_color="inverse"
                )
            with fc3:
                st.metric(
                    "Risk reduction — rehab vs no action",
                    f"{no_int_5yr - rehab_5yr:.0f} pts",
                    delta="at 5 years"
                )
            with fc4:
                # Time to cross 60% threshold without intervention
                medians_no_int = np.median(all_paths["no_intervention"], axis=0)
                threshold_crossings = [i for i, m in enumerate(medians_no_int)
                                       if m >= 0.60]
                if threshold_crossings:
                    cross_months = MONTHS[threshold_crossings[0]]
                    if cross_months < 12:
                        cross_label = f"{cross_months}m"
                    else:
                        cross_label = f"{cross_months//12}yr {cross_months%12}m" if cross_months%12 else f"{cross_months//12}yr"
                    st.metric("High-risk threshold", cross_label,
                              delta="without intervention", delta_color="inverse")
                else:
                    st.metric("High-risk threshold", "Not reached",
                              delta="within 5 years")

            # ── Narrative forecast interpretation ───────────────
            st.divider()
            st.markdown("**Clinical Forecast Interpretation**")

            # Select scenario text based on current risk
            if risk_prob < 0.30:
                outlook = "currently low"
                trajectory_warning = (
                    f"However, without lifestyle modification, the model projects "
                    f"risk increasing to **{no_int_5yr:.0f}%** within 5 years. "
                    f"This patient is on a trajectory toward moderate risk."
                )
            elif risk_prob < 0.60:
                outlook = "currently moderate"
                trajectory_warning = (
                    f"Without intervention, risk is projected to reach "
                    f"**{no_int_5yr:.0f}%** — entering high-risk territory — "
                    f"within 5 years. Early action now has the highest impact."
                )
            else:
                outlook = "currently high"
                trajectory_warning = (
                    f"Without urgent intervention, risk remains above 60% "
                    f"and could reach **{no_int_5yr:.0f}%** within 5 years. "
                    f"Immediate cardiology referral is indicated."
                )

            rehab_benefit = no_int_5yr - rehab_5yr
            med_benefit   = no_int_5yr - (np.median(all_paths["medication"][:,-1])*100)

            st.markdown(f"""
This patient's cardiovascular risk is **{outlook}** at **{risk_prob*100:.1f}%**.
{trajectory_warning}

**Scenario outcomes at 5 years:**
- **No intervention:** {no_int_5yr:.0f}% predicted risk — risk continues to rise with age and biomarker progression
- **Lifestyle changes alone:** {np.median(all_paths['lifestyle'][:,-1])*100:.0f}% — diet and exercise slow progression significantly
- **Medication (antihypertensive + statin):** {np.median(all_paths['medication'][:,-1])*100:.0f}% — medication produces the sharpest early reduction
- **JoiHealth Cardiac Rehabilitation:** {rehab_5yr:.0f}% — combined programme achieves the best long-term outcome, reducing 5-year risk by **{rehab_benefit:.0f} percentage points** compared to no action

The shaded bands represent the 25th–75th and 10th–90th percentile uncertainty range across 600 simulated patient trajectories, reflecting natural variability in biomarker progression. Wider bands indicate higher uncertainty.

> ⚠ **Note:** This forecast uses evidence-based biomarker progression rates from the Framingham Heart Study and AHA/ACC guidelines, combined with the patient's current ML-predicted risk score. It is a decision support tool — not a clinical prediction. Actual outcomes depend on adherence, comorbidities, and factors not captured in this model.
""")

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
