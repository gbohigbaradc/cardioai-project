# ============================================================
# MODULE 5: NLP + OCR — CLINICAL DOCUMENT PROCESSING
# File: 05_nlp_ocr.py
#
# WHAT THIS SCRIPT DOES:
# Clinicians write notes and treatment plans — sometimes typed,
# sometimes scanned PDFs or images of handwritten notes.
# This module:
#   1. OCR (Optical Character Recognition): extracts raw text
#      from image-based documents using Tesseract
#   2. NLP (Natural Language Processing): uses spaCy to extract
#      structured medical entities from that raw text
#      (blood pressure values, diagnoses, medications, symptoms)
#
# WHY THIS MATTERS:
# Instead of manually reading every clinical note, the system
# can automatically extract: "BP: 145/90", "Diagnosis: Hypertension",
# "Medication: Lisinopril 10mg" and feed these into the ML pipeline.
#
# SETUP NEEDED (before running):
#   pip install pytesseract pillow pdf2image spacy
#   python -m spacy download en_core_web_sm
#   sudo apt-get install tesseract-ocr    (Linux)
#   brew install tesseract                (Mac)
#
# OUTPUT:
# - outputs/extracted_text_sample.txt
# - outputs/nlp_entities_sample.json
# ============================================================

import re
import json
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# PART A: OCR — Extract text from scanned medical documents
# ══════════════════════════════════════════════════════════════════════════

def extract_text_from_image(image_path):
    """
    Extracts raw text from an image file using Tesseract OCR.

    Tesseract works by:
    1. Converting the image to grayscale
    2. Applying adaptive thresholding to separate text from background
    3. Segmenting text blocks and lines
    4. Using a trained neural network to recognize characters

    Parameters:
    - image_path: path to PNG, JPG, or TIFF image

    Returns:
    - extracted text as string
    """
    try:
        import pytesseract
        from PIL import Image, ImageFilter, ImageEnhance

        print(f"  Processing image: {image_path}")
        img = Image.open(image_path)

        # ── Pre-process image to improve OCR accuracy ────────────────────
        # Convert to grayscale (removes color noise)
        img = img.convert("L")

        # Enhance contrast (makes text stand out more)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Apply slight sharpening to clarify blurry handwriting
        img = img.filter(ImageFilter.SHARPEN)

        # Resize to at least 300 DPI equivalent (Tesseract prefers 300+ DPI)
        width, height = img.size
        if width < 1500:
            scale = 1500 / width
            img = img.resize((int(width * scale), int(height * scale)))

        # ── OCR configuration ─────────────────────────────────────────────
        # --oem 3: Use LSTM neural network (most accurate)
        # --psm 6: Assume uniform block of text
        # -l eng: Use English language model
        config = "--oem 3 --psm 6 -l eng"
        text = pytesseract.image_to_string(img, config=config)

        return text.strip()

    except ImportError:
        print("  ⚠ pytesseract not installed. Using sample text for demonstration.")
        return None
    except Exception as e:
        print(f"  ⚠ OCR error: {e}. Using sample text.")
        return None


def extract_text_from_pdf(pdf_path):
    """
    Converts each PDF page to an image, then runs OCR on each page.

    Parameters:
    - pdf_path: path to PDF file

    Returns:
    - full extracted text from all pages
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract

        print(f"  Converting PDF: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=300)  # 300 DPI = high quality

        full_text = []
        for i, page in enumerate(pages):
            page_text = pytesseract.image_to_string(page, config="--oem 3 --psm 6")
            full_text.append(f"--- Page {i+1} ---\n{page_text}")

        return "\n".join(full_text)

    except ImportError:
        print("  ⚠ pdf2image not installed. Using sample text.")
        return None


# ── Sample clinical note for demonstration ───────────────────────────────
# When no real document is provided, we use this realistic sample
# to demonstrate the full NLP pipeline end-to-end.
SAMPLE_CLINICAL_NOTE = """
PATIENT CLINICAL NOTES
Date: 15/03/2025
Patient ID: PT-00482
Clinician: Dr. A. Adewale

PRESENTING COMPLAINT:
Patient presents with chest tightness and shortness of breath on exertion.
Reports episodes of palpitations over the past 3 weeks.

VITAL SIGNS:
Blood Pressure: 148/92 mmHg
Heart Rate: 88 bpm
SpO2: 97%
Temperature: 36.8°C
Weight: 84 kg

EXAMINATION FINDINGS:
Mild peripheral oedema noted in lower limbs.
No murmurs detected on auscultation.
ECG shows ST-segment changes in leads V4-V6.

DIAGNOSES:
1. Hypertension (Grade 2)
2. Suspected coronary artery disease
3. Type 2 Diabetes Mellitus — poorly controlled

CURRENT MEDICATIONS:
- Amlodipine 10mg once daily
- Metformin 500mg twice daily
- Aspirin 75mg once daily
- Atorvastatin 40mg at night

LIFESTYLE ASSESSMENT:
Patient reports sedentary lifestyle. Smoker (15 cigarettes/day for 12 years).
Alcohol: moderate (3-4 units/week). Diet: high sodium, low fibre.
Exercise: minimal — states difficulty walking more than 200m without breathlessness.

PLAN:
- Refer to cardiology for further workup and possible angiography
- Increase Metformin to 1000mg twice daily
- Initiate Losartan 50mg for blood pressure management
- Refer to dietitian and cardiac rehabilitation program
- Repeat HbA1c and lipid panel in 8 weeks

NEXT APPOINTMENT: 4 weeks
"""


# ══════════════════════════════════════════════════════════════════════════
# PART B: NLP — Extract structured entities from clinical text
# ══════════════════════════════════════════════════════════════════════════

class ClinicalNLPExtractor:
    """
    Extracts structured medical information from unstructured clinical text.

    Uses two approaches:
    1. Rule-based extraction: regex patterns for known medical formats
       (e.g., blood pressure always appears as NNN/NN mmHg)
    2. spaCy NER: for general medical entity recognition (diagnoses,
       medications, anatomy)
    """

    def __init__(self):
        # Try to load spaCy model. Fall back to regex-only if not available.
        self.nlp = None
        self.spacy = None   # FIX: store spacy module reference on self
        try:
            import spacy
            self.spacy = spacy          # save reference so methods can use it
            self.nlp = spacy.load("en_core_web_sm")
            print("  ✓ spaCy model loaded: en_core_web_sm")
        except Exception:
            print("  ⚠ spaCy not available — using regex extraction only")

    def extract_blood_pressure(self, text):
        """
        Extracts blood pressure readings using regex.
        Pattern matches realistic BP values: systolic 90-250, diastolic 50-130.
        Excludes dates like 15/03 by requiring systolic >= 90.
        """
        # Require systolic 90-250 mmHg, diastolic 50-130 mmHg
        # (?<!\d) = not preceded by a digit (avoids matching mid-number)
        pattern = r"(?<!\d)(1\d{2}|2[0-4]\d|9\d)\/((?:[5-9]\d)|(?:1[0-2]\d))(?!\d)(?:\s*mmHg)?"
        matches = re.findall(pattern, text)

        results = []
        for systolic, diastolic in matches:
            results.append({
                "raw": f"{systolic}/{diastolic}",
                "systolic": int(systolic),
                "diastolic": int(diastolic),
                "hypertensive": int(systolic) >= 140 or int(diastolic) >= 90
            })
        return results

    def extract_heart_rate(self, text):
        """Extracts heart rate values (e.g., 'Heart Rate: 88 bpm')"""
        pattern = r"(?:Heart Rate|HR|Pulse)[:\s]+(\d{2,3})\s*(?:bpm|beats)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [{"value": int(m), "unit": "bpm"} for m in matches]

    def extract_diagnoses(self, text):
        """
        Extracts diagnoses from numbered or bulleted diagnosis sections.
        Looks for patterns like:
        - "Diagnosis:" followed by text
        - "DIAGNOSES:" section
        - ICD-style entries
        """
        # Strategy 1: Look for explicit diagnosis section
        diag_section = re.search(
            r"DIAGNOS[EIS]+[:\s]+(.*?)(?=\n[A-Z]{3,}:|$)",
            text, re.IGNORECASE | re.DOTALL
        )

        diagnoses = []
        if diag_section:
            # Extract individual diagnoses from numbered list
            items = re.findall(r"\d+\.\s*([^\n]+)", diag_section.group(1))
            diagnoses.extend([item.strip() for item in items])

        # Strategy 2: Common diagnosis keywords
        common_dx_patterns = [
            r"(hypertension(?:\s+grade\s+\d)?)",
            r"(diabetes\s+mellitus(?:\s+type\s+[12])?)",
            r"(coronary\s+artery\s+disease)",
            r"(heart\s+failure)",
            r"(atrial\s+fibrillation)",
            r"(myocardial\s+infarction)",
        ]
        for pattern in common_dx_patterns:
            found = re.findall(pattern, text, re.IGNORECASE)
            diagnoses.extend(found)

        return list(set([d.strip().title() for d in diagnoses if d.strip()]))

    def extract_medications(self, text):
        """
        Extracts medication names and dosages.
        Pattern: drug name + dosage (e.g., Amlodipine 10mg, Metformin 500mg)
        """
        # Common medication suffix patterns
        med_pattern = r"([A-Z][a-z]+-?[a-z]*)\s+(\d+(?:\.\d+)?(?:mg|mcg|g|ml|IU))"
        matches = re.findall(med_pattern, text)

        medications = []
        for drug, dose in matches:
            # Filter out false positives (common English words)
            if len(drug) > 4 and not drug.lower() in ["with", "from", "this", "that"]:
                medications.append({"name": drug, "dose": dose})
        return medications

    def extract_lifestyle_factors(self, text):
        """
        Extracts key lifestyle risk factors.
        These are fed into the Lifestyle Risk Index calculation.
        """
        factors = {}

        # Smoking
        if re.search(r"smok|cigarette|tobacco", text, re.IGNORECASE):
            cigarettes = re.search(r"(\d+)\s*cigarette", text, re.IGNORECASE)
            factors["smoking"] = {
                "present": True,
                "daily_count": int(cigarettes.group(1)) if cigarettes else None
            }
        else:
            factors["smoking"] = {"present": False}

        # Alcohol
        alcohol_match = re.search(r"alcohol[:\s]+(\w+)", text, re.IGNORECASE)
        if alcohol_match:
            factors["alcohol"] = alcohol_match.group(1).lower()

        # Exercise
        if re.search(r"sedentary|minimal\s+exercise|inactive", text, re.IGNORECASE):
            factors["physical_activity"] = "sedentary"
        elif re.search(r"regular\s+exercise|active", text, re.IGNORECASE):
            factors["physical_activity"] = "active"
        else:
            factors["physical_activity"] = "unknown"

        # Diet
        if re.search(r"high\s+sodium|poor\s+diet|unhealthy", text, re.IGNORECASE):
            factors["diet"] = "unhealthy"
        elif re.search(r"balanced|healthy\s+diet", text, re.IGNORECASE):
            factors["diet"] = "healthy"
        else:
            factors["diet"] = "unknown"

        return factors

    def extract_all(self, text):
        """
        Master extraction function — runs all extractors and returns
        a structured dictionary of all extracted clinical entities.
        """
        print("  Running clinical entity extraction...")
        entities = {
            "blood_pressure":    self.extract_blood_pressure(text),
            "heart_rate":        self.extract_heart_rate(text),
            "diagnoses":         self.extract_diagnoses(text),
            "medications":       self.extract_medications(text),
            "lifestyle_factors": self.extract_lifestyle_factors(text),
        }

        # spaCy general entity recognition (if available)
        if self.nlp and self.spacy:
            doc = self.nlp(text)
            entities["spacy_entities"] = [
                {"text": ent.text, "label": ent.label_,
                 "description": self.spacy.explain(ent.label_)}  # FIX: use self.spacy
                for ent in doc.ents
                if ent.label_ in ["ORG", "PERSON", "GPE", "DATE", "CARDINAL", "QUANTITY"]
            ]

        return entities


# ══════════════════════════════════════════════════════════════════════════
# PART C: RUN THE PIPELINE
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("NLP + OCR CLINICAL DOCUMENT PROCESSING PIPELINE")
print("=" * 60)

# ── Step 1: Attempt OCR on any provided document ─────────────────────────
# In real deployment, user uploads a scanned PDF or image.
# For demonstration, we use the sample note directly.
sample_image_path = "data/sample_clinical_note.png"  # placeholder
sample_pdf_path   = "data/sample_clinical_note.pdf"  # placeholder

if os.path.exists(sample_image_path):
    print("\nProcessing image document via OCR...")
    raw_text = extract_text_from_image(sample_image_path)
elif os.path.exists(sample_pdf_path):
    print("\nProcessing PDF document via OCR...")
    raw_text = extract_text_from_pdf(sample_pdf_path)
else:
    print("\nNo document file found — using sample clinical note for demonstration...")
    raw_text = SAMPLE_CLINICAL_NOTE

# Save raw extracted text
with open("outputs/extracted_text_sample.txt", "w", encoding="utf-8") as f:
    f.write(raw_text)
print(f"✓ Raw text saved: outputs/extracted_text_sample.txt")

# ── Step 2: Run NLP entity extraction ─────────────────────────────────────
print("\nRunning NLP extraction pipeline...")
extractor = ClinicalNLPExtractor()
entities = extractor.extract_all(raw_text)

# ── Step 3: Display and save results ─────────────────────────────────────
print("\n── Extraction Results ──")

print(f"\n  Blood Pressure readings found: {len(entities['blood_pressure'])}")
for bp in entities["blood_pressure"]:
    flag = "⚠ HYPERTENSIVE" if bp["hypertensive"] else "✓ Normal"
    print(f"    {bp['raw']} mmHg — {flag}")

print(f"\n  Heart Rate readings: {entities['heart_rate']}")

print(f"\n  Diagnoses extracted ({len(entities['diagnoses'])}):")
for dx in entities["diagnoses"]:
    print(f"    - {dx}")

print(f"\n  Medications extracted ({len(entities['medications'])}):")
for med in entities["medications"]:
    print(f"    - {med['name']} {med['dose']}")

print(f"\n  Lifestyle Factors:")
for k, v in entities["lifestyle_factors"].items():
    print(f"    {k}: {v}")

# Save structured entities as JSON
with open("outputs/nlp_entities_sample.json", "w", encoding="utf-8") as f:
    json.dump(entities, f, indent=2)
print(f"\n✓ Structured entities saved: outputs/nlp_entities_sample.json")

# ── Step 4: Map extracted data to model features ─────────────────────────
# This is the INTEGRATION step — turning NLP output into model input.
# Shows how a clinician's note can automatically populate the ML features.

print("\n── Mapping Extracted Data to ML Model Features ──")
model_features_from_nlp = {}

# Blood pressure → trestbps feature
if entities["blood_pressure"]:
    bp = entities["blood_pressure"][0]
    model_features_from_nlp["trestbps"] = bp["systolic"]
    print(f"  trestbps (resting BP): {bp['systolic']} mmHg")

# Heart rate → thalach proxy
if entities["heart_rate"]:
    model_features_from_nlp["thalach_estimate"] = entities["heart_rate"][0]["value"]
    print(f"  thalach estimate (HR): {entities['heart_rate'][0]['value']} bpm")

# Lifestyle factors → lifestyle risk index inputs
lf = entities["lifestyle_factors"]
model_features_from_nlp["smoking_flag"] = 1 if lf.get("smoking", {}).get("present") else 0
model_features_from_nlp["physical_activity_sedentary"] = 1 if lf.get("physical_activity") == "sedentary" else 0
print(f"  smoking_flag: {model_features_from_nlp['smoking_flag']}")
print(f"  sedentary_flag: {model_features_from_nlp['physical_activity_sedentary']}")

with open("outputs/nlp_model_features.json", "w", encoding="utf-8") as f:
    json.dump(model_features_from_nlp, f, indent=2)
print(f"\n✓ Model-ready features saved: outputs/nlp_model_features.json")

print("\n" + "=" * 60)
print("NLP + OCR PIPELINE COMPLETE")
print("=" * 60)
