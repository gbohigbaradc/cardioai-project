# CardioAI — JoiHealth Polyclinics

> **AI-Powered Cardiovascular Risk, Clinical Intelligence & Hospital Operations System**

[![Live App](https://img.shields.io/badge/Live%20App-cardioai--joihealth.streamlit.app-00B4D8?style=for-the-badge&logo=streamlit)](https://cardioai-joihealth.streamlit.app)
[![medRxiv](https://img.shields.io/badge/medRxiv-MEDRXIV%2F2026%2F349630-red?style=for-the-badge)](https://www.medrxiv.org)
[![DOI](https://img.shields.io/badge/DOI-10.53022%2Foarjms.2024.7.1.0055-blue?style=for-the-badge)](https://doi.org/10.53022/oarjms.2024.7.1.0055)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## Overview

CardioAI is a full-stack clinical AI platform developed for **JoiHealth Polyclinics** — Nigeria's premier luxury physical medicine and cardiac rehabilitation polyclinic, with locations in **Old GRA Port Harcourt** and **Ikoyi Lagos**.

The system moves beyond a simple risk calculator into a **complete clinical intelligence platform**: from handwritten patient notes through to X-ray interpretation, drug intelligence, 5-year risk forecasting, and real-time hospital operations management.

**3,904 lines of production Python. 8 clinical modules. 1 live deployment.**

---

## Live Application

```
https://cardioai-joihealth.streamlit.app
```

---

## The 8 Modules

### 🫀 1. Cardiovascular Risk Prediction
- **4 ML models**: XGBoost (AUC 0.999), Random Forest (0.998), Neural Network (0.994), Logistic Regression (0.927)
- **SHAP explainability**: per-patient feature importance — top 5 clinical drivers shown as bar chart
- **Lifestyle Risk Index (LRI)**: original composite feature engineered from BP, cholesterol, fasting sugar, exercise angina, and ST depression — ranked 3rd globally in SHAP importance
- **Cross-validated AUC**: RF = 0.977 ± 0.003, XGBoost = 0.997 ± 0.004
- **SMOTE** applied strictly inside training folds — no data leakage

### 📈 Monte Carlo 5-Year Risk Forecasting (part of Module 1)
- **600 stochastic patient trajectories** per scenario × 4 clinical scenarios
- **Scenarios**: No intervention · Lifestyle changes · Medication · JoiHealth Cardiac Rehabilitation
- **Biomarker progression** parameterised from Framingham Heart Study + AHA/ACC 2026 guidelines
- **Fan chart** (P10–P90 uncertainty bands) + 5-year endpoint comparison bar chart
- **4 metric cards**: 5-yr risk without action, 5-yr risk with rehab, absolute risk reduction, time to high-risk threshold
- Narrative clinical interpretation generated per patient

### 🏥 2. Patient Retention Prediction
- Predicts dropout risk from operational factors (exercise difficulty, wait time, travel distance, insurance, previous visits, perceived improvement)
- Generates specific actionable retention recommendations per patient

### 📊 3. Model Performance Dashboard
- ROC curves for all 4 models side-by-side
- SHAP beeswarm plot (global feature importance across all test patients)
- Model comparison table with AUC, F1, precision, recall

### 📄 4. Clinical NLP — Document Intelligence
- **Input methods**: paste text · upload image · upload PDF (digital or scanned)
- **Gemini Vision** (Google Gemini 2.0 Flash): reads handwritten notes, cursive, typed documents
- **Tesseract OCR 5.5** fallback with 4 preprocessing strategies
- **Scanned PDF support**: page-by-page image conversion via `pdf2image` / `PyMuPDF` → Vision AI reads each page
- **Extracts**: BP, PR, SpO₂, Weight, Height, BMI, Diagnoses, Medications, Lifestyle flags
- **Auto-fill guidance**: tells clinician exactly which values to enter in Risk Prediction form

### 🔬 5. Medical Imaging CNN
- **DenseNet-121** pretrained on 100,000+ chest X-rays (NIH ChestX-ray14, Stanford CheXpert, MIT MIMIC-CXR)
- **18 pathologies classified**: Cardiomegaly, Effusion, Pneumonia, Atelectasis, Consolidation, Pneumothorax, Edema, Emphysema, Fibrosis, Nodule, Mass, Infiltration, Pleural Thickening, Hernia, and more
- **PSPNet anatomical segmentation**: pixel-level masks for Heart, Left Lung, Right Lung, Aorta, Spine (14 structures)
- **Cardiothoracic ratio (CTR)**: automated computation from segmentation — CTR ≥ 0.50 flags Cardiomegaly
- **Grad-CAM heatmaps**: per-pathology activation overlay showing which region drove each prediction
- **Structured clinical report**: severity levels, urgent findings, monitoring recommendations

### 💊 6. Pharmaco-Intelligence (6 sub-engines)

| Engine | Description |
|--------|-------------|
| Drug Recommender | Evidence-based drug selection per 2026 ACC/AHA guidelines — considers diagnoses, eGFR, K⁺, BMI, drug intolerances |
| Interaction Checker | Screens full medication list for Contraindicated / Major / Moderate interactions with mechanism explanation |
| Lab Result Interpreter | 15 lab parameters — critical value alerts, automatic drug adjustment triggers (stop statin if CK >1000 etc.) |
| Drug-Condition Analyser | Explains WHY a drug is/is not appropriate for this specific patient — pharmacokinetics personalised to clinical profile |
| Pharmacy Claims Review | Reviews current regimen, estimates ₦ monthly cost, checks doses, highlights generic alternatives, flags monitoring gaps |
| AI Documentation Generator | Gemini AI generates: Pharmaceutical Care Plan, Discharge Summary, Patient Counselling Sheet, Medication Reconciliation, Cardiac Rehab Pharmacy Review |

**Drug database**: Amlodipine, Lisinopril, Losartan, Atenolol, Bisoprolol, Atorvastatin, Rosuvastatin, Metformin, Aspirin, Spironolactone, Warfarin — all with Nigerian market availability and ₦ cost tiers.

**Guidelines**: 2026 ACC/AHA Dyslipidaemia · 2025 AACE · ESC/EAS 2025 · AHA/ACC Hypertension · ADA 2025

### 🏢 7. Operational Intelligence

| Feature | Description |
|---------|-------------|
| Bed Management | Live occupancy tracking, admit/discharge/transfer workflow, fall risk flags, length-of-stay |
| Staff Scheduling | Add real JoiHealth clinicians, assign shifts, track utilisation %, overload alerts, shift handover logging |
| Patient Flow Analytics | Bottleneck detection across 7 care stages, funnel chart, target vs actual wait times |
| EHR Completeness | Documentation completeness tracking, missing field flags, mark-complete workflow |
| Vitals Overdue Tracker | Flags patients due for vitals check based on admission time and protocol |
| AI Ops Advisor | Gemini AI generates real-time operational recommendations |

### ℹ️ 8. About
System description, developer profile, medRxiv pre-print reference, GitHub link, model performance summary.

---

## Model Performance

| Model | AUC-ROC | F1 Score | Recall | Precision |
|-------|---------|----------|--------|-----------|
| **XGBoost** ★ | **0.999** | **0.999** | **0.999** | **0.999** |
| Random Forest | 0.998 | 0.998 | 0.998 | 0.998 |
| Neural Network (MLP) | 0.994 | 0.977 | 0.990 | 0.965 |
| Logistic Regression | 0.927 | 0.865 | 0.914 | 0.821 |

Cross-validated AUC: RF = 0.977 ± 0.003 · XGBoost = 0.997 ± 0.004

**Top 5 SHAP features** (global): Major Vessels (ca) · Thalassemia (thal) · Lifestyle Risk Index · ST Depression (oldpeak) · Max Heart Rate (thalach)

---

## Technology Stack

```
Frontend        Streamlit 1.28 · Python 3.12 · Matplotlib · Pandas
ML / AI         XGBoost 2.0 · scikit-learn · SHAP · PyTorch · TorchXRayVision
Vision AI       Google Gemini 2.0 Flash · Tesseract OCR 5.5 · pdf2image · PyMuPDF
CNN Imaging     DenseNet-121 · PSPNet · Grad-CAM · scikit-image
Deployment      Streamlit Cloud · GitHub CI/CD · Render.com (custom domain)
Data            UCI Heart Disease Dataset · Framingham Heart Study · NHIS Nigeria
```

---

## Project Structure

```
cardioai-project/
├── app/
│   └── streamlit_app.py        # Main application (3,904 lines)
├── 01_preprocessing.py         # Data preprocessing + LRI feature engineering
├── 02_eda.py                   # Exploratory data analysis
├── 03_model_training.py        # Model training + cross-validation + SHAP
├── 04_retention_model.py       # Patient retention model
├── 07_cnn_imaging.py           # CNN standalone script (DenseNet-121)
├── data/
│   └── heart.csv               # UCI Heart Disease Dataset
├── models/                     # Trained .pkl files (auto-generated on first run)
│   ├── cardio_xgb.pkl
│   ├── cardio_rf.pkl
│   ├── cardio_logistic.pkl
│   ├── retention_rf.pkl
│   ├── retention_xgb.pkl
│   ├── scaler.pkl
│   └── scaler_retention.pkl
├── outputs/                    # Charts, model comparison CSV
├── Joi_Health_PM.jpeg          # JoiHealth logo
├── requirements.txt
└── packages.txt
```

---

## Installation & Local Setup

### Prerequisites
- Python 3.9+
- Node.js (optional — for CNN model download)
- Tesseract OCR (for OCR fallback)

### Install dependencies

```bash
pip install -r requirements.txt
```

### System packages (Ubuntu/Debian)

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng libsm6 libxext6 libxrender-dev libgomp1 libgl1
```

### Run the app locally

```bash
# Step 1: Train models
python 01_preprocessing.py
python 03_model_training.py
python 04_retention_model.py

# Step 2: Launch app
streamlit run app/streamlit_app.py
```

The app will auto-train models on first run if `.pkl` files are missing.

---

## Streamlit Cloud Deployment

### Secrets required

Add to Streamlit Cloud → Settings → Secrets:

```toml
GOOGLE_API_KEY = "your-gemini-api-key"
# Optional fallback:
# ANTHROPIC_API_KEY = "your-claude-api-key"
```

Get a free Gemini API key at: https://aistudio.google.com

### Packages (packages.txt)

```
tesseract-ocr
tesseract-ocr-eng
libsm6
libxext6
libxrender-dev
libgomp1
libgl1
```

---

## Requirements

```
pandas>=2.0.0        numpy>=1.24.0        scikit-learn>=1.3.0
matplotlib>=3.7.0    seaborn>=0.12.0      joblib>=1.3.0
shap>=0.44.0         streamlit>=1.28.0    pdfplumber
xgboost>=2.0.0       imbalanced-learn>=0.11.0
pytesseract>=0.3.10  Pillow>=10.0.0       pdf2image>=1.16.0
PyMuPDF              opencv-python-headless
google-generativeai  anthropic>=0.8.0
torchxrayvision      scikit-image         torch    torchvision
```

---

## Clinical Use — JoiHealth Workflow

```
Patient arrives at clinic
        ↓
Clinician photographs handwritten note
        ↓
Clinical NLP extracts all vitals & diagnoses (Gemini Vision)
        ↓
Risk Prediction — XGBoost scores cardiovascular risk + SHAP
        ↓
Monte Carlo 5-Year Forecast — 4 scenario trajectories
        ↓
Pharmaco-Intelligence — drug recommendations + interactions + lab check
        ↓
Medical Imaging CNN — chest X-ray analysis (if available)
        ↓
AI Documentation — pharmaceutical care plan auto-generated
        ↓
Operational Dashboard updated — bed status, EHR, staff load
```

---

## Original Contributions

1. **Lifestyle Risk Index (LRI)** — novel composite feature not present in the original UCI dataset. Ranked 3rd most important predictor globally via SHAP. Combines BP, cholesterol, fasting sugar, exercise angina, and ST depression into a single normalised score.

2. **Monte Carlo longitudinal forecasting** — transforms static cross-sectional risk score into a dynamic 5-year clinical trajectory tool with scenario comparison and uncertainty quantification.

3. **Integrated pharmaco-intelligence** — first clinical pharmacy decision support system grounded in 2026 ACC/AHA guidelines specifically designed for Nigerian cardiac rehabilitation context, including NHIS formulary pricing and ACEi cough prevalence in African patients.

4. **Full clinical workflow integration** — single platform connecting: document digitisation → risk stratification → drug intelligence → imaging → operations. No comparable integrated system exists for Nigerian cardiac rehabilitation.

---

## Publications

| Type | Reference |
|------|-----------|
| medRxiv Pre-print | MEDRXIV/2026/349630 |
| Published Paper | DOI: [10.53022/oarjms.2024.7.1.0055](https://doi.org/10.53022/oarjms.2024.7.1.0055) |

**Target journals for submission**: PLOS ONE · BMC Medical Informatics and Decision Making · Frontiers in Digital Health · IEEE Access

---

## Important Disclaimer

> CardioAI is a **clinical decision support tool** only. All outputs — including risk scores, drug recommendations, imaging interpretations, lab interpretations, and clinical documents — must be reviewed and approved by a licensed clinician before any clinical action is taken. This system does not replace clinical judgement, radiologist review, or pharmacist oversight.

---

## Developer

**Gboh-Igbara D. Charles**
JoiHealth Polyclinics · Old GRA Port Harcourt & Ikoyi Lagos, Nigeria

- GitHub: [github.com/gbohigbaradc/cardioai-project](https://github.com/gbohigbaradc/cardioai-project)
- Live App: [cardioai-joihealth.streamlit.app](https://cardioai-joihealth.streamlit.app)

---

*JoiHealth Polyclinics — Commitment to Care*
