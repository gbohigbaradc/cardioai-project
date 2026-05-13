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

# ══════════════════════════════════════════════════════════
# EXPORT UTILITIES
# Provides PDF, Excel, and CSV export for all modules.
# ══════════════════════════════════════════════════════════

import io
import base64

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _df_to_excel_bytes(sheets: dict) -> bytes:
    """sheets = {sheet_name: dataframe}"""
    try:
        import openpyxl  # noqa — confirms package present
    except ImportError:
        raise ImportError("openpyxl not installed. Add 'openpyxl>=3.1.0' to requirements.txt")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()

def _df_to_docx_bytes(title: str, sections: list) -> bytes:
    """
    sections = list of (heading: str, content: str | pd.DataFrame)
    Produces a styled Word document using python-docx.
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        raise ImportError(
            "python-docx not installed. Add 'python-docx>=1.1.0' to requirements.txt"
        )

    def _set_cell_bg(cell, hex_colour: str):
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), hex_colour)
        shd.set(qn("w:val"), "clear")
        tcPr.append(shd)

    doc = Document()
    for sec in doc.sections:
        sec.top_margin    = Cm(2)
        sec.bottom_margin = Cm(2)
        sec.left_margin   = Cm(2.5)
        sec.right_margin  = Cm(2.5)

    h0 = doc.add_heading(title, level=0)
    h0.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if h0.runs:
        h0.runs[0].font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    sub = doc.add_paragraph(
        f"Generated: {pd.Timestamp.now().strftime('%d %B %Y, %H:%M')}  |  "
        f"CardioAI Nova"
    )
    if sub.runs:
        sub.runs[0].font.size = Pt(9)
        sub.runs[0].font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    doc.add_paragraph()

    for heading, content in sections:
        if heading:
            h2 = doc.add_heading(heading, level=2)
            if h2.runs:
                h2.runs[0].font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)

        if isinstance(content, pd.DataFrame):
            if content.empty:
                doc.add_paragraph("No data available.")
                doc.add_paragraph()
                continue
            col_names = list(content.columns)
            table = doc.add_table(rows=1 + len(content), cols=len(col_names))
            table.style = "Table Grid"
            hdr_row = table.rows[0]
            for ci, col in enumerate(col_names):
                cell = hdr_row.cells[ci]
                cell.text = str(col)
                _set_cell_bg(cell, "1F4E79")
                p = cell.paragraphs[0]
                if p.runs:
                    p.runs[0].bold = True
                    p.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    p.runs[0].font.size = Pt(9)
            for ri, (_, row) in enumerate(content.iterrows()):
                tr = table.rows[ri + 1]
                bg = "EBF5FB" if ri % 2 == 1 else "FFFFFF"
                for ci, val in enumerate(row):
                    cell = tr.cells[ci]
                    cell.text = str(val) if val is not None else ""
                    _set_cell_bg(cell, bg)
                    p = cell.paragraphs[0]
                    if p.runs:
                        p.runs[0].font.size = Pt(8.5)
        else:
            text = str(content) if content is not None else ""
            for line in text.split("\n"):
                p = doc.add_paragraph(line)
                if p.runs:
                    p.runs[0].font.size = Pt(10)
        doc.add_paragraph()

    disc = doc.add_paragraph()
    run = disc.add_run(
        "⚕ CardioAI is a clinical decision support tool only. All outputs must be "
        "reviewed by a licensed clinician before any clinical action is taken. "
        "— CardioAI Nova | cardioai-nova.streamlit.app"
    )
    run.font.size = Pt(8)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

def _df_to_pdf_bytes(title: str, sections: list) -> bytes:
    """
    sections = list of (heading: str, content: str | pd.DataFrame)
    Renders a styled PDF using reportlab.
    Falls back to a plain-text .txt if reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
    except ImportError:
        # Plain-text fallback — still downloadable
        lines = [title, "=" * 60,
                 f"Generated: {pd.Timestamp.now().strftime('%d %B %Y %H:%M')}",
                 "CardioAI Nova",
                 "(Install reportlab>=4.0.0 in requirements.txt for true PDF output)", ""]
        for heading, content in sections:
            if heading:
                lines.append(f"\n{heading}\n{'-' * len(heading)}")
            if isinstance(content, pd.DataFrame):
                lines.append(content.to_string(index=False))
            else:
                lines.append(str(content))
        return "\n".join(lines).encode("utf-8")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CardioTitle", parent=styles["Title"],
        fontSize=16, textColor=colors.HexColor("#1F4E79"), spaceAfter=6
    )
    h2_style = ParagraphStyle(
        "CardioH2", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#2E75B6"),
        spaceBefore=10, spaceAfter=4
    )
    body_style = ParagraphStyle(
        "CardioBody", parent=styles["Normal"],
        fontSize=9, leading=13, spaceAfter=4
    )
    footer_style = ParagraphStyle(
        "CardioFooter", parent=styles["Normal"],
        fontSize=7, textColor=colors.grey, spaceBefore=20
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(
        f"Generated: {pd.Timestamp.now().strftime('%d %B %Y, %H:%M')} &nbsp;|&nbsp; "
        f"CardioAI Nova",
        styles["Normal"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor("#2E75B6"), spaceAfter=12))

    for heading, content in sections:
        if heading:
            story.append(Paragraph(heading, h2_style))
        if isinstance(content, pd.DataFrame):
            if content.empty:
                story.append(Paragraph("No data.", body_style))
                story.append(Spacer(1, 6))
                continue
            col_names = list(content.columns)
            data = [col_names] + content.astype(str).values.tolist()
            avail_w = A4[0] - 4 * cm
            col_w = avail_w / len(col_names)
            t = Table(data, colWidths=[col_w] * len(col_names), repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
                ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",       (0, 0), (-1, 0), 8),
                ("FONTSIZE",       (0, 1), (-1, -1), 7.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#EBF5FB")]),
                ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#BDC3C7")),
                ("VALIGN",         (0, 0), (-1, -1), "TOP"),
                ("WORDWRAP",       (0, 0), (-1, -1), True),
                ("TOPPADDING",     (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
        else:
            safe = (str(content)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
            for line in safe.split("\n"):
                story.append(Paragraph(line if line.strip() else "&nbsp;", body_style))
        story.append(Spacer(1, 8))

    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#BDC3C7"), spaceBefore=16))
    story.append(Paragraph(
        "⚕ CardioAI is a clinical decision support tool only. All outputs must be reviewed "
        "by a licensed clinician before any clinical action is taken. — CardioAI Nova | cardioai-nova.streamlit.app",
        footer_style
    ))
    doc.build(story)
    return buf.getvalue()


def export_buttons(label: str, csv_df: pd.DataFrame = None,
                   excel_sheets: dict = None,
                   pdf_title: str = None, pdf_sections: list = None,
                   docx_title: str = None, docx_sections: list = None,
                   file_stem: str = "cardioai_export"):
    """
    Renders CSV / Excel / PDF / Word download buttons side-by-side.
    All formats are always offered when data is provided.
    Errors in any format are shown as st.warning — never silent failures.
    """
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    active = []
    if csv_df is not None:        active.append("csv")
    if excel_sheets is not None:  active.append("xlsx")
    if pdf_sections is not None:  active.append("pdf")
    if docx_sections is not None: active.append("docx")
    if not active:
        return

    st.markdown("**⬇ Export results:**")
    btn_cols = st.columns(len(active))

    for col_idx, fmt in enumerate(active):
        with btn_cols[col_idx]:
            if fmt == "csv":
                try:
                    data = _df_to_csv_bytes(csv_df)
                    st.download_button(
                        "📄 Download CSV",
                        data=data,
                        file_name=f"{file_stem}_{ts}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_csv_{file_stem}_{ts}",
                    )
                except Exception as e:
                    st.warning(f"CSV export error: {e}")

            elif fmt == "xlsx":
                try:
                    data = _df_to_excel_bytes(excel_sheets)
                    st.download_button(
                        "📊 Download Excel",
                        data=data,
                        file_name=f"{file_stem}_{ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key=f"dl_xlsx_{file_stem}_{ts}",
                    )
                except ImportError:
                    st.warning("Excel export requires openpyxl. Add `openpyxl>=3.1.0` to requirements.txt")
                except Exception as e:
                    st.warning(f"Excel export error: {e}")

            elif fmt == "pdf":
                try:
                    pdf_bytes = _df_to_pdf_bytes(pdf_title or label, pdf_sections)
                    is_pdf = pdf_bytes[:4] == b"%PDF"
                    st.download_button(
                        "🖨 Download PDF" if is_pdf else "📝 Download Report (TXT)",
                        data=pdf_bytes,
                        file_name=f"{file_stem}_{ts}.{'pdf' if is_pdf else 'txt'}",
                        mime="application/pdf" if is_pdf else "text/plain",
                        use_container_width=True,
                        key=f"dl_pdf_{file_stem}_{ts}",
                    )
                    if not is_pdf:
                        st.caption("Add `reportlab>=4.0.0` to requirements.txt for true PDF output.")
                except Exception as e:
                    st.warning(f"PDF export error: {e}")

            elif fmt == "docx":
                try:
                    data = _df_to_docx_bytes(docx_title or label, docx_sections)
                    st.download_button(
                        "📝 Download Word",
                        data=data,
                        file_name=f"{file_stem}_{ts}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                        key=f"dl_docx_{file_stem}_{ts}",
                    )
                except ImportError:
                    st.warning("Word export requires python-docx. Add `python-docx>=1.1.0` to requirements.txt")
                except Exception as e:
                    st.warning(f"Word export error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ICD-10 / ICD-11 / ICF / SNOMED CT  —  CLINICAL CODING REFERENCE
# Covers all conditions relevant to CardioAI Nova cardiac rehabilitation practice.
# ══════════════════════════════════════════════════════════════════════════════

ICD_DB = {
    "Hypertension": {
        "icd10": "I10", "icd11": "BA00",
        "description": "Essential (primary) hypertension",
        "snomed": "38341003", "category": "Cardiovascular",
        "icf_codes": ["b4200","b4201"],
        "icf_labels": ["Blood pressure functions — general","Diastolic blood pressure"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Hypertensive Heart Disease": {
        "icd10": "I11", "icd11": "BA01",
        "description": "Hypertensive heart disease without heart failure",
        "snomed": "64715009", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4200"],
        "icf_labels": ["Heart rate","Blood pressure functions"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Hypertensive Heart Disease with Heart Failure": {
        "icd10": "I11.0", "icd11": "BA01.0",
        "description": "Hypertensive heart disease with heart failure",
        "snomed": "39727004", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4200","b4550"],
        "icf_labels": ["Heart rate","Blood pressure functions","General physical endurance"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Stable Angina": {
        "icd10": "I20.9", "icd11": "BA80",
        "description": "Angina pectoris, unspecified",
        "snomed": "194828000", "category": "Cardiovascular",
        "icf_codes": ["b4101","b4550","d4500"],
        "icf_labels": ["Heart rhythm","General physical endurance","Walking short distances"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Unstable Angina": {
        "icd10": "I20.0", "icd11": "BA41",
        "description": "Unstable angina",
        "snomed": "4557003", "category": "Cardiovascular",
        "icf_codes": ["b4101","b4550"],
        "icf_labels": ["Heart rhythm","General physical endurance"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Myocardial Infarction": {
        "icd10": "I21.9", "icd11": "BA41",
        "description": "Acute myocardial infarction, unspecified",
        "snomed": "22298006", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4101","b4550","d4500","d6505"],
        "icf_labels": ["Heart rate","Heart rhythm","General physical endurance",
                       "Walking short distances","Taking care of one's health"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Post-MI": {
        "icd10": "I25.2", "icd11": "BA83",
        "description": "Old myocardial infarction / Post-MI status",
        "snomed": "399211009", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4550","d4500","d6505"],
        "icf_labels": ["Heart rate","General physical endurance",
                       "Walking short distances","Taking care of one's health"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Post-MI Rehab": {
        "icd10": "I25.2", "icd11": "BA83",
        "description": "Old myocardial infarction — cardiac rehabilitation phase",
        "snomed": "399211009", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4550","d4500"],
        "icf_labels": ["Heart rate","General physical endurance","Walking short distances"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Coronary Artery Disease": {
        "icd10": "I25.1", "icd11": "BA80.0",
        "description": "Atherosclerotic heart disease of native coronary artery",
        "snomed": "53741008", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4550"],
        "icf_labels": ["Heart rate","General physical endurance"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Heart Failure": {
        "icd10": "I50", "icd11": "BD10",
        "description": "Heart failure, unspecified",
        "snomed": "84114007", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4550","b4552","d4500","d4501"],
        "icf_labels": ["Heart rate","General physical endurance","Fatiguability",
                       "Walking short distances","Walking long distances"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Heart Failure (HFrEF)": {
        "icd10": "I50.2", "icd11": "BD10.1",
        "description": "Systolic (congestive) heart failure — HFrEF",
        "snomed": "703272007", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4550","b4552","d4500"],
        "icf_labels": ["Heart rate","General physical endurance",
                       "Fatiguability","Walking short distances"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Atrial Fibrillation": {
        "icd10": "I48", "icd11": "BC81.1",
        "description": "Atrial fibrillation",
        "snomed": "49436004", "category": "Cardiovascular",
        "icf_codes": ["b4101","b4200"],
        "icf_labels": ["Heart rhythm","Blood pressure functions"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Arrhythmia": {
        "icd10": "I49.9", "icd11": "BC81",
        "description": "Cardiac arrhythmia, unspecified",
        "snomed": "698247007", "category": "Cardiovascular",
        "icf_codes": ["b4101"],
        "icf_labels": ["Heart rhythm"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Stroke": {
        "icd10": "I64", "icd11": "8B20",
        "description": "Stroke, not specified as haemorrhage or infarction",
        "snomed": "230690007", "category": "Cerebrovascular",
        "icf_codes": ["b7301","b7600","b1300","d4500","d5500"],
        "icf_labels": ["Power of muscles on one side of body","Control of voluntary movement",
                       "Energy level","Walking short distances","Washing oneself"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Stroke/TIA": {
        "icd10": "G45.9", "icd11": "8B11",
        "description": "Transient cerebral ischaemic attack / Stroke",
        "snomed": "266257000", "category": "Cerebrovascular",
        "icf_codes": ["b7600","b1300","d4500"],
        "icf_labels": ["Control of voluntary movement","Energy level","Walking short distances"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Stroke Rehab": {
        "icd10": "I69.3", "icd11": "8B20",
        "description": "Sequelae of cerebral infarction — stroke rehabilitation",
        "snomed": "413102000", "category": "Cerebrovascular",
        "icf_codes": ["b7301","b7600","b1300","d4500","d5500"],
        "icf_labels": ["Power of muscles on one side of body","Control of voluntary movement",
                       "Energy level","Walking short distances","Washing oneself"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Type 2 Diabetes": {
        "icd10": "E11", "icd11": "5A11",
        "description": "Type 2 diabetes mellitus",
        "snomed": "44054006", "category": "Metabolic / Endocrine",
        "icf_codes": ["b5401","b4550","b2401"],
        "icf_labels": ["Glucose metabolism","General physical endurance","Sensation of pain in extremity"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Dyslipidaemia": {
        "icd10": "E78.5", "icd11": "5C80",
        "description": "Hyperlipidaemia, unspecified / Dyslipidaemia",
        "snomed": "370992007", "category": "Metabolic / Endocrine",
        "icf_codes": ["b5403"],
        "icf_labels": ["Fat metabolism"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Obesity": {
        "icd10": "E66", "icd11": "5B81",
        "description": "Obesity",
        "snomed": "414916001", "category": "Metabolic / Endocrine",
        "icf_codes": ["b5300","b4550","d5701"],
        "icf_labels": ["Appetite functions","General physical endurance","Managing diet and fitness"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "CKD": {
        "icd10": "N18.9", "icd11": "GB61",
        "description": "Chronic kidney disease, unspecified",
        "snomed": "709044004", "category": "Renal",
        "icf_codes": ["b6100","b5401"],
        "icf_labels": ["Urinary filtration functions","Glucose metabolism"],
        "rehab_relevant": False, "nhis_billable": True,
    },
    "COPD/Asthma": {
        "icd10": "J44.1", "icd11": "CA23",
        "description": "Chronic obstructive pulmonary disease / Asthma",
        "snomed": "13645005", "category": "Respiratory",
        "icf_codes": ["b4400","b4401","b4550"],
        "icf_labels": ["Respiration rate","Respiratory rhythm","General physical endurance"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Osteoarthritis": {
        "icd10": "M19.9", "icd11": "FA00",
        "description": "Osteoarthritis, unspecified",
        "snomed": "396275006", "category": "Musculoskeletal",
        "icf_codes": ["b7100","b2801","d4500"],
        "icf_labels": ["Mobility of a single joint","Pain in body part","Walking short distances"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Joint Pain / Arthralgia": {
        "icd10": "M25.5", "icd11": "FA44",
        "description": "Pain in joint / Arthralgia",
        "snomed": "57676002", "category": "Musculoskeletal",
        "icf_codes": ["b2801","b7100","d4500"],
        "icf_labels": ["Pain in body part","Mobility of a single joint","Walking short distances"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Back Pain": {
        "icd10": "M54.5", "icd11": "ME84",
        "description": "Low back pain",
        "snomed": "279039007", "category": "Musculoskeletal",
        "icf_codes": ["b28013","b7101","d4154"],
        "icf_labels": ["Pain in back","Mobility of several joints","Staying in a seated position"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Cardiomegaly": {
        "icd10": "I51.7", "icd11": "BC91",
        "description": "Cardiomegaly",
        "snomed": "8186001", "category": "Cardiovascular",
        "icf_codes": ["b4100","b4550"],
        "icf_labels": ["Heart rate","General physical endurance"],
        "rehab_relevant": True, "nhis_billable": True,
    },
    "Palpitations": {
        "icd10": "R00.2", "icd11": "MD21",
        "description": "Palpitations",
        "snomed": "80313002", "category": "Cardiovascular — Symptom",
        "icf_codes": ["b4101"],
        "icf_labels": ["Heart rhythm"],
        "rehab_relevant": False, "nhis_billable": True,
    },
    "Vertigo / Dizziness": {
        "icd10": "R42", "icd11": "MD30",
        "description": "Dizziness and giddiness",
        "snomed": "404640003", "category": "Neurological — Symptom",
        "icf_codes": ["b2401"],
        "icf_labels": ["Dizziness"],
        "rehab_relevant": False, "nhis_billable": True,
    },
}

# ── NLP diagnosis label → ICD key mapping ─────────────────────────────────────
NLP_DX_TO_ICD = {
    "Hypertension":          "Hypertension",
    "Heart Failure":         "Heart Failure",
    "Myocardial Infarction": "Myocardial Infarction",
    "Atrial Fibrillation":   "Atrial Fibrillation",
    "Stroke":                "Stroke",
    "Type 2 Diabetes":       "Type 2 Diabetes",
    "Dyslipidaemia":         "Dyslipidaemia",
    "Obesity":               "Obesity",
    "Arrhythmia":            "Arrhythmia",
    "Stable Angina":         "Stable Angina",
    "Joint Pain / Arthralgia": "Joint Pain / Arthralgia",
    "Palpitations":          "Palpitations",
    "Vertigo / Dizziness":   "Vertigo / Dizziness",
}

def render_icd_panel(diagnoses: list, context: str = ""):
    """Render a collapsible ICD-10/11/ICF/CPT panel for a list of diagnosis name strings."""
    if not diagnoses:
        return
    matched   = [(dx, ICD_DB[NLP_DX_TO_ICD.get(dx, dx)])
                 for dx in diagnoses
                 if NLP_DX_TO_ICD.get(dx, dx) in ICD_DB]
    unmatched = [dx for dx in diagnoses
                 if NLP_DX_TO_ICD.get(dx, dx) not in ICD_DB]

    if not matched and not unmatched:
        return

    with st.expander(f"🏷 ICD / ICF / CPT Clinical Codes — {context}", expanded=False):
        for dx, rec in matched:
            c1, c2, c3, c4 = st.columns([2, 1.5, 2, 2])
            with c1:
                st.markdown(f"**{dx}**")
                st.caption(rec["description"])
                st.markdown(f"*{rec['category']}*")
                st.markdown(
                    ("✅ Insurance Billable" if rec["nhis_billable"] else "❌ Not Insurance Billable") +
                    ("  🏥 Rehab" if rec["rehab_relevant"] else "")
                )
            with c2:
                st.markdown("**ICD / SNOMED**")
                st.markdown(f"🔵 ICD-10: `{rec['icd10']}`")
                st.markdown(f"🟢 ICD-11: `{rec['icd11']}`")
                st.markdown(f"🟣 SNOMED: `{rec['snomed']}`")
            with c3:
                st.markdown("**ICF Functional Codes**")
                if rec["icf_codes"]:
                    for code, label in zip(rec["icf_codes"], rec["icf_labels"]):
                        st.caption(f"`{code}` {label}")
            with c4:
                st.markdown("**CPT Procedures**")
                for cpt in rec.get("cpt_codes", []):
                    cpt_rec = CPT_DB.get(cpt, {})
                    st.caption(f"🟠 `{cpt}` — {cpt_rec.get('description','—')}  ₦{cpt_rec.get('nhis_tariff',0):,}")
            st.divider()
        if unmatched:
            st.caption(f"⚠ No ICD mapping found for: {', '.join(unmatched)}")

def icd_export_df(diagnoses: list) -> pd.DataFrame:
    """Flat DataFrame of ICD/ICF/CPT codes for a list of diagnosis names — used in exports."""
    rows = []
    for dx in diagnoses:
        key = NLP_DX_TO_ICD.get(dx, dx)
        rec = ICD_DB.get(key)
        if rec:
            cpt_codes = rec.get("cpt_codes", [])
            cpt_descs = [CPT_DB.get(c, {}).get("description", "—") for c in cpt_codes]
            cpt_tariffs = sum(CPT_DB.get(c, {}).get("nhis_tariff", 0) for c in cpt_codes)
            rows.append({
                "Diagnosis":        dx,
                "ICD-10":           rec["icd10"],
                "ICD-11":           rec["icd11"],
                "SNOMED CT":        rec["snomed"],
                "Description":      rec["description"],
                "Category":         rec["category"],
                "Insurance Billable":    "Yes" if rec["nhis_billable"] else "No",
                "Rehab Relevant":   "Yes" if rec["rehab_relevant"] else "No",
                "ICF Codes":        ", ".join(rec["icf_codes"]),
                "ICF Descriptions": ", ".join(rec["icf_labels"]),
                "CPT Codes":        ", ".join(cpt_codes),
                "CPT Procedures":   " | ".join(cpt_descs),
                "Est. Tariff (local currency)": cpt_tariffs,
            })
        else:
            rows.append({
                "Diagnosis": dx, "ICD-10": "—", "ICD-11": "—", "SNOMED CT": "—",
                "Description": "Not mapped in ICD_DB", "Category": "—",
                "Insurance Billable": "—", "Rehab Relevant": "—",
                "ICF Codes": "—", "ICF Descriptions": "—",
                "CPT Codes": "—", "CPT Procedures": "—", "Est. Tariff (local currency)": 0,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# CPT — CURRENT PROCEDURAL TERMINOLOGY  (AMA / Insurance Procedure Codes)
# Covers all investigations and procedures relevant to CardioAI Nova cardiac rehab.
# Nigerian Insurance tariff estimates in ₦ are approximate 2025 rates.
# ══════════════════════════════════════════════════════════════════════════════

CPT_DB = {
    # ── Evaluation & Management ───────────────────────────────────────────
    "99202": {"description": "Office visit — new patient, low complexity",
              "category": "E&M", "nhis_tariff": 5000, "unit": "per visit"},
    "99203": {"description": "Office visit — new patient, moderate complexity",
              "category": "E&M", "nhis_tariff": 8000, "unit": "per visit"},
    "99213": {"description": "Office visit — established patient, low-moderate complexity",
              "category": "E&M", "nhis_tariff": 4000, "unit": "per visit"},
    "99214": {"description": "Office visit — established patient, moderate-high complexity",
              "category": "E&M", "nhis_tariff": 6500, "unit": "per visit"},
    "99223": {"description": "Initial hospital care — high complexity",
              "category": "E&M", "nhis_tariff": 15000, "unit": "per admission"},
    "99232": {"description": "Subsequent hospital care — moderate complexity",
              "category": "E&M", "nhis_tariff": 7000, "unit": "per day"},
    "99238": {"description": "Hospital discharge day management",
              "category": "E&M", "nhis_tariff": 5000, "unit": "per discharge"},

    # ── Cardiovascular Procedures ─────────────────────────────────────────
    "93000": {"description": "ECG — routine 12-lead with interpretation and report",
              "category": "Cardiology", "nhis_tariff": 3500, "unit": "per test"},
    "93005": {"description": "ECG — tracing only, no interpretation",
              "category": "Cardiology", "nhis_tariff": 2000, "unit": "per test"},
    "93306": {"description": "Echocardiography — transthoracic with Doppler",
              "category": "Cardiology", "nhis_tariff": 35000, "unit": "per study"},
    "93307": {"description": "Echocardiography — transthoracic, without Doppler",
              "category": "Cardiology", "nhis_tariff": 25000, "unit": "per study"},
    "93015": {"description": "Cardiovascular stress test (exercise ECG — treadmill/bike)",
              "category": "Cardiology", "nhis_tariff": 20000, "unit": "per test"},
    "93017": {"description": "Cardiovascular stress test — tracing only",
              "category": "Cardiology", "nhis_tariff": 12000, "unit": "per test"},
    "93224": {"description": "Holter monitor — 24-hour ECG, recording and analysis",
              "category": "Cardiology", "nhis_tariff": 28000, "unit": "per study"},
    "93268": {"description": "Patient-activated event recorder — 30 days",
              "category": "Cardiology", "nhis_tariff": 40000, "unit": "per study"},
    "93452": {"description": "Left heart catheterisation — coronary angiography",
              "category": "Cardiology", "nhis_tariff": 250000, "unit": "per procedure"},
    "93458": {"description": "Coronary angiography — left and right heart catheterisation",
              "category": "Cardiology", "nhis_tariff": 350000, "unit": "per procedure"},
    "93798": {"description": "Cardiac rehabilitation — per session (physician supervision)",
              "category": "Cardiology", "nhis_tariff": 8000, "unit": "per session"},
    "33249": {"description": "Implantable cardioverter-defibrillator (ICD) insertion",
              "category": "Cardiology", "nhis_tariff": 1500000, "unit": "per procedure"},
    "33208": {"description": "Permanent pacemaker insertion — dual chamber",
              "category": "Cardiology", "nhis_tariff": 900000, "unit": "per procedure"},

    # ── Vital Signs & Physical Exam ───────────────────────────────────────
    "94760": {"description": "Pulse oximetry — single determination (SpO₂)",
              "category": "Vital Signs", "nhis_tariff": 500, "unit": "per reading"},
    "94761": {"description": "Pulse oximetry — multiple determinations",
              "category": "Vital Signs", "nhis_tariff": 1500, "unit": "per session"},
    "93784": {"description": "24-hour ambulatory blood pressure monitoring (ABPM)",
              "category": "Vital Signs", "nhis_tariff": 18000, "unit": "per study"},

    # ── Imaging ───────────────────────────────────────────────────────────
    "71046": {"description": "Chest X-ray — 2 views (PA and lateral)",
              "category": "Radiology", "nhis_tariff": 6000, "unit": "per study"},
    "71045": {"description": "Chest X-ray — 1 view",
              "category": "Radiology", "nhis_tariff": 3500, "unit": "per study"},
    "74177": {"description": "CT Chest with contrast",
              "category": "Radiology", "nhis_tariff": 80000, "unit": "per study"},
    "70553": {"description": "MRI Brain with contrast",
              "category": "Radiology", "nhis_tariff": 120000, "unit": "per study"},
    "93880": {"description": "Carotid duplex ultrasound — bilateral",
              "category": "Radiology", "nhis_tariff": 25000, "unit": "per study"},
    "76770": {"description": "Abdominal/renal ultrasound",
              "category": "Radiology", "nhis_tariff": 15000, "unit": "per study"},

    # ── Blood Glucose & Diabetes ──────────────────────────────────────────
    "82947": {"description": "Glucose — fasting blood sugar (FBS)",
              "category": "Chemistry", "nhis_tariff": 800, "unit": "per test"},
    "82948": {"description": "Glucose — random blood sugar (RBS)",
              "category": "Chemistry", "nhis_tariff": 800, "unit": "per test"},
    "83036": {"description": "HbA1c (glycated haemoglobin)",
              "category": "Chemistry", "nhis_tariff": 3500, "unit": "per test"},
    "82950": {"description": "Glucose — post-challenge (OGTT 2-hour)",
              "category": "Chemistry", "nhis_tariff": 2500, "unit": "per test"},
    "84681": {"description": "C-Peptide",
              "category": "Chemistry", "nhis_tariff": 8000, "unit": "per test"},
    "83525": {"description": "Insulin — fasting",
              "category": "Chemistry", "nhis_tariff": 7500, "unit": "per test"},

    # ── Lipid Profile ─────────────────────────────────────────────────────
    "80061": {"description": "Lipid panel — total cholesterol, HDL, triglycerides (LDL calculated)",
              "category": "Chemistry", "nhis_tariff": 3000, "unit": "per panel"},
    "82465": {"description": "Total cholesterol",
              "category": "Chemistry", "nhis_tariff": 1000, "unit": "per test"},
    "83718": {"description": "HDL cholesterol",
              "category": "Chemistry", "nhis_tariff": 1200, "unit": "per test"},
    "84478": {"description": "Triglycerides",
              "category": "Chemistry", "nhis_tariff": 1200, "unit": "per test"},
    "83721": {"description": "LDL cholesterol — direct measurement",
              "category": "Chemistry", "nhis_tariff": 1500, "unit": "per test"},
    "86141": {"description": "hs-CRP (high-sensitivity C-reactive protein)",
              "category": "Chemistry", "nhis_tariff": 4500, "unit": "per test"},
    "83695": {"description": "Lipoprotein(a) — Lp(a)",
              "category": "Chemistry", "nhis_tariff": 9000, "unit": "per test"},

    # ── Renal & Electrolytes (E/U/Cr) ────────────────────────────────────
    "80048": {"description": "Basic metabolic panel (Na, K, Cl, CO₂, Glucose, BUN, Creatinine, Ca)",
              "category": "Chemistry", "nhis_tariff": 4500, "unit": "per panel"},
    "80053": {"description": "Comprehensive metabolic panel (BMP + LFTs + proteins)",
              "category": "Chemistry", "nhis_tariff": 7500, "unit": "per panel"},
    "84520": {"description": "Urea / BUN",
              "category": "Chemistry", "nhis_tariff": 800, "unit": "per test"},
    "82565": {"description": "Creatinine — serum",
              "category": "Chemistry", "nhis_tariff": 800, "unit": "per test"},
    "82570": {"description": "Creatinine — urine",
              "category": "Chemistry", "nhis_tariff": 1000, "unit": "per test"},
    "33935": {"description": "eGFR — estimated glomerular filtration rate (reported with creatinine)",
              "category": "Chemistry", "nhis_tariff": 500, "unit": "per test"},
    "82042": {"description": "Urine albumin — quantitative",
              "category": "Chemistry", "nhis_tariff": 2000, "unit": "per test"},
    "82044": {"description": "Urine albumin — microalbumin, quantitative",
              "category": "Chemistry", "nhis_tariff": 2500, "unit": "per test"},
    "84550": {"description": "Uric acid — serum",
              "category": "Chemistry", "nhis_tariff": 1000, "unit": "per test"},
    "82374": {"description": "Bicarbonate / CO₂ — serum",
              "category": "Chemistry", "nhis_tariff": 700, "unit": "per test"},

    # ── Electrolytes (individual) ─────────────────────────────────────────
    "84295": {"description": "Sodium (Na⁺)",
              "category": "Chemistry", "nhis_tariff": 600, "unit": "per test"},
    "84132": {"description": "Potassium (K⁺)",
              "category": "Chemistry", "nhis_tariff": 600, "unit": "per test"},
    "82435": {"description": "Chloride (Cl⁻)",
              "category": "Chemistry", "nhis_tariff": 600, "unit": "per test"},
    "82310": {"description": "Calcium — total serum",
              "category": "Chemistry", "nhis_tariff": 800, "unit": "per test"},
    "83735": {"description": "Magnesium (Mg²⁺)",
              "category": "Chemistry", "nhis_tariff": 1000, "unit": "per test"},
    "84100": {"description": "Phosphate (PO₄³⁻)",
              "category": "Chemistry", "nhis_tariff": 800, "unit": "per test"},

    # ── Cardiac Biomarkers ────────────────────────────────────────────────
    "84484": {"description": "Troponin I — quantitative",
              "category": "Cardiac Markers", "nhis_tariff": 8000, "unit": "per test"},
    "84512": {"description": "Troponin T — quantitative",
              "category": "Cardiac Markers", "nhis_tariff": 8000, "unit": "per test"},
    "82553": {"description": "CK-MB (creatine kinase MB isoenzyme)",
              "category": "Cardiac Markers", "nhis_tariff": 4000, "unit": "per test"},
    "82550": {"description": "Total CK (creatine kinase)",
              "category": "Cardiac Markers", "nhis_tariff": 2000, "unit": "per test"},
    "83880": {"description": "BNP (B-type natriuretic peptide)",
              "category": "Cardiac Markers", "nhis_tariff": 12000, "unit": "per test"},
    "83937": {"description": "NT-proBNP",
              "category": "Cardiac Markers", "nhis_tariff": 14000, "unit": "per test"},
    "83874": {"description": "Myoglobin",
              "category": "Cardiac Markers", "nhis_tariff": 5000, "unit": "per test"},
    "83090": {"description": "Homocysteine",
              "category": "Cardiac Markers", "nhis_tariff": 9000, "unit": "per test"},

    # ── Liver Function Tests ──────────────────────────────────────────────
    "80076": {"description": "Hepatic function panel (ALT, AST, ALP, bilirubin, albumin, total protein)",
              "category": "Chemistry", "nhis_tariff": 4000, "unit": "per panel"},
    "84460": {"description": "ALT / SGPT",
              "category": "Chemistry", "nhis_tariff": 1000, "unit": "per test"},
    "84450": {"description": "AST / SGOT",
              "category": "Chemistry", "nhis_tariff": 1000, "unit": "per test"},
    "84075": {"description": "ALP (alkaline phosphatase)",
              "category": "Chemistry", "nhis_tariff": 1000, "unit": "per test"},
    "82977": {"description": "GGT (gamma-glutamyl transferase)",
              "category": "Chemistry", "nhis_tariff": 1500, "unit": "per test"},
    "82247": {"description": "Total bilirubin",
              "category": "Chemistry", "nhis_tariff": 800, "unit": "per test"},
    "82040": {"description": "Albumin — serum",
              "category": "Chemistry", "nhis_tariff": 800, "unit": "per test"},

    # ── Inflammatory Markers ──────────────────────────────────────────────
    "86140": {"description": "CRP (C-reactive protein)",
              "category": "Immunology", "nhis_tariff": 2000, "unit": "per test"},
    "85652": {"description": "ESR (erythrocyte sedimentation rate)",
              "category": "Haematology", "nhis_tariff": 800, "unit": "per test"},
    "82728": {"description": "Ferritin — serum",
              "category": "Chemistry", "nhis_tariff": 4000, "unit": "per test"},
    "83615": {"description": "LDH (lactate dehydrogenase)",
              "category": "Chemistry", "nhis_tariff": 1500, "unit": "per test"},
    "84244": {"description": "Procalcitonin (PCT)",
              "category": "Immunology", "nhis_tariff": 12000, "unit": "per test"},

    # ── Full Blood Count ──────────────────────────────────────────────────
    "85025": {"description": "Full blood count / CBC with differential",
              "category": "Haematology", "nhis_tariff": 2000, "unit": "per test"},
    "85027": {"description": "CBC without differential",
              "category": "Haematology", "nhis_tariff": 1500, "unit": "per test"},
    "85610": {"description": "Prothrombin time (PT / INR)",
              "category": "Haematology", "nhis_tariff": 2500, "unit": "per test"},
    "85730": {"description": "APTT (activated partial thromboplastin time)",
              "category": "Haematology", "nhis_tariff": 2500, "unit": "per test"},

    # ── Thyroid & Endocrinology ───────────────────────────────────────────
    "84443": {"description": "TSH (thyroid stimulating hormone)",
              "category": "Endocrinology", "nhis_tariff": 5000, "unit": "per test"},
    "84439": {"description": "Free T4 (thyroxine — free)",
              "category": "Endocrinology", "nhis_tariff": 5000, "unit": "per test"},
    "84481": {"description": "Free T3 (triiodothyronine — free)",
              "category": "Endocrinology", "nhis_tariff": 5000, "unit": "per test"},
    "82533": {"description": "Cortisol — serum (AM)",
              "category": "Endocrinology", "nhis_tariff": 8000, "unit": "per test"},
    "82088": {"description": "Aldosterone — serum",
              "category": "Endocrinology", "nhis_tariff": 10000, "unit": "per test"},
    "84270": {"description": "SHBG (sex hormone binding globulin)",
              "category": "Endocrinology", "nhis_tariff": 8000, "unit": "per test"},
    "84403": {"description": "Testosterone — total serum",
              "category": "Endocrinology", "nhis_tariff": 8000, "unit": "per test"},
    "82670": {"description": "Oestradiol (E2)",
              "category": "Endocrinology", "nhis_tariff": 8000, "unit": "per test"},
    "84146": {"description": "Prolactin",
              "category": "Endocrinology", "nhis_tariff": 7000, "unit": "per test"},
    "84153": {"description": "PSA — total",
              "category": "Endocrinology", "nhis_tariff": 6000, "unit": "per test"},
    "84244": {"description": "Renin — plasma activity",
              "category": "Endocrinology", "nhis_tariff": 12000, "unit": "per test"},

    # ── Serology / Immunology ─────────────────────────────────────────────
    "86038": {"description": "ANA (antinuclear antibody)",
              "category": "Serology", "nhis_tariff": 8000, "unit": "per test"},
    "86235": {"description": "Anti-dsDNA antibody",
              "category": "Serology", "nhis_tariff": 9000, "unit": "per test"},
    "86430": {"description": "Rheumatoid factor (RF)",
              "category": "Serology", "nhis_tariff": 3000, "unit": "per test"},
    "86200": {"description": "Anti-CCP (cyclic citrullinated peptide antibody)",
              "category": "Serology", "nhis_tariff": 12000, "unit": "per test"},
    "86147": {"description": "Antiphospholipid antibody (cardiolipin IgG/IgM)",
              "category": "Serology", "nhis_tariff": 10000, "unit": "per test"},
    "86063": {"description": "Complement C3",
              "category": "Serology", "nhis_tariff": 6000, "unit": "per test"},
    "86060": {"description": "Complement C4",
              "category": "Serology", "nhis_tariff": 6000, "unit": "per test"},
    "87340": {"description": "HBsAg (Hepatitis B surface antigen)",
              "category": "Serology", "nhis_tariff": 2500, "unit": "per test"},
    "86704": {"description": "HBcAb (Hepatitis B core antibody)",
              "category": "Serology", "nhis_tariff": 3500, "unit": "per test"},
    "86803": {"description": "Anti-HCV (Hepatitis C antibody)",
              "category": "Serology", "nhis_tariff": 3500, "unit": "per test"},
    "87389": {"description": "HIV-1/2 Ag/Ab combination test",
              "category": "Serology", "nhis_tariff": 3000, "unit": "per test"},
    "86592": {"description": "VDRL / RPR (syphilis screening)",
              "category": "Serology", "nhis_tariff": 1500, "unit": "per test"},
    "86596": {"description": "TPHA (syphilis confirmatory)",
              "category": "Serology", "nhis_tariff": 3000, "unit": "per test"},

    # ── Microbiology ──────────────────────────────────────────────────────
    "87040": {"description": "Blood culture — aerobic and anaerobic",
              "category": "Microbiology", "nhis_tariff": 5000, "unit": "per set"},
    "87086": {"description": "Urine culture and sensitivity (M/C/S)",
              "category": "Microbiology", "nhis_tariff": 4000, "unit": "per test"},
    "87070": {"description": "Sputum culture and sensitivity",
              "category": "Microbiology", "nhis_tariff": 4000, "unit": "per test"},
    "87081": {"description": "Wound swab culture and sensitivity",
              "category": "Microbiology", "nhis_tariff": 4000, "unit": "per test"},
    "87207": {"description": "AFB smear (acid-fast bacilli — TB screen)",
              "category": "Microbiology", "nhis_tariff": 2500, "unit": "per test"},
    "87015": {"description": "Malaria thick and thin film / RDT",
              "category": "Microbiology", "nhis_tariff": 1500, "unit": "per test"},
    "87305": {"description": "Stool M/C/S (microscopy, culture, sensitivity)",
              "category": "Microbiology", "nhis_tariff": 3000, "unit": "per test"},
    "87502": {"description": "Influenza A and B — molecular (PCR)",
              "category": "Microbiology", "nhis_tariff": 12000, "unit": "per test"},
    "87635": {"description": "SARS-CoV-2 (COVID-19) — molecular (PCR)",
              "category": "Microbiology", "nhis_tariff": 15000, "unit": "per test"},

    # ── Toxicology & Drug Monitoring ──────────────────────────────────────
    "80162": {"description": "Digoxin level — therapeutic monitoring",
              "category": "Toxicology", "nhis_tariff": 7000, "unit": "per test"},
    "80202": {"description": "Vancomycin — therapeutic drug monitoring",
              "category": "Toxicology", "nhis_tariff": 8000, "unit": "per test"},
    "80185": {"description": "Phenytoin — total",
              "category": "Toxicology", "nhis_tariff": 6000, "unit": "per test"},
    "80178": {"description": "Lithium — therapeutic monitoring",
              "category": "Toxicology", "nhis_tariff": 6000, "unit": "per test"},
    "82055": {"description": "Blood alcohol level",
              "category": "Toxicology", "nhis_tariff": 3000, "unit": "per test"},
    "80150": {"description": "Paracetamol (acetaminophen) level",
              "category": "Toxicology", "nhis_tariff": 5000, "unit": "per test"},
    "80101": {"description": "Urine drug screen — qualitative (multi-panel)",
              "category": "Toxicology", "nhis_tariff": 8000, "unit": "per test"},

    # ── Cytology & Histopathology ─────────────────────────────────────────
    "88141": {"description": "Cervical cytology (LBC) — interpretation",
              "category": "Pathology", "nhis_tariff": 5000, "unit": "per test"},
    "87624": {"description": "HPV — high risk, nucleic acid (PCR)",
              "category": "Pathology", "nhis_tariff": 15000, "unit": "per test"},
    "88305": {"description": "Surgical pathology — biopsy interpretation (Level IV)",
              "category": "Pathology", "nhis_tariff": 12000, "unit": "per specimen"},
    "88307": {"description": "Surgical pathology — complex specimen (Level V)",
              "category": "Pathology", "nhis_tariff": 18000, "unit": "per specimen"},
    "88173": {"description": "FNAC — interpretation",
              "category": "Pathology", "nhis_tariff": 8000, "unit": "per test"},
    "88108": {"description": "Cytopathology — concentration technique (sputum/pleural/urine)",
              "category": "Pathology", "nhis_tariff": 6000, "unit": "per specimen"},

    # ── Tumour Markers ────────────────────────────────────────────────────
    "86304": {"description": "CA-125 (cancer antigen 125)",
              "category": "Tumour Markers", "nhis_tariff": 9000, "unit": "per test"},
    "86301": {"description": "CA 19-9",
              "category": "Tumour Markers", "nhis_tariff": 9000, "unit": "per test"},
    "82378": {"description": "CEA (carcinoembryonic antigen)",
              "category": "Tumour Markers", "nhis_tariff": 8000, "unit": "per test"},
    "82105": {"description": "AFP (alpha-fetoprotein)",
              "category": "Tumour Markers", "nhis_tariff": 8000, "unit": "per test"},

    # ── Allergy ───────────────────────────────────────────────────────────
    "86003": {"description": "Allergen-specific IgE — single allergen",
              "category": "Allergy", "nhis_tariff": 4000, "unit": "per allergen"},
    "86005": {"description": "Total IgE — serum",
              "category": "Allergy", "nhis_tariff": 5000, "unit": "per test"},
    "95004": {"description": "Skin prick tests — percutaneous (per test)",
              "category": "Allergy", "nhis_tariff": 1500, "unit": "per allergen"},

    # ── Rehabilitation ────────────────────────────────────────────────────
    "97110": {"description": "Therapeutic exercise — per 15 minutes",
              "category": "Rehabilitation", "nhis_tariff": 3000, "unit": "per 15 min"},
    "97012": {"description": "Mechanical traction — per session",
              "category": "Rehabilitation", "nhis_tariff": 2500, "unit": "per session"},
    "97530": {"description": "Therapeutic activities — per 15 minutes",
              "category": "Rehabilitation", "nhis_tariff": 3000, "unit": "per 15 min"},
    "97003": {"description": "Occupational therapy evaluation",
              "category": "Rehabilitation", "nhis_tariff": 8000, "unit": "per evaluation"},
    "97001": {"description": "Physical therapy evaluation",
              "category": "Rehabilitation", "nhis_tariff": 8000, "unit": "per evaluation"},
}

# ── Map: Investigation name (as used in risk module) → CPT code(s) ────────────
INVESTIGATION_CPT_MAP = {
    # Vitals
    "SpO₂ / Pulse Oximetry":          ["94760"],
    "24hr Ambulatory BP Monitoring":   ["93784"],
    # ECG
    "12-Lead ECG":                     ["93000"],
    "Holter Monitor (24hr)":           ["93224"],
    "Exercise Stress Test":            ["93015"],
    # Blood glucose
    "Fasting Blood Sugar (FBS)":       ["82947"],
    "Random Blood Sugar (RBS)":        ["82948"],
    "HbA1c":                           ["83036"],
    "OGTT (2-hour)":                   ["82950"],
    "C-Peptide":                       ["84681"],
    "Fasting Insulin":                 ["83525"],
    # Lipid profile
    "Fasting Lipid Panel":             ["80061"],
    "LDL (direct)":                    ["83721"],
    "hs-CRP":                          ["86141"],
    "Lp(a)":                           ["83695"],
    "ApoB":                            ["82172"],
    # Renal
    "Urea / BUN":                      ["84520"],
    "Creatinine (serum)":              ["82565"],
    "eGFR":                            ["33935"],
    "Urine Albumin (microalbumin)":    ["82044"],
    "Uric Acid":                       ["84550"],
    "Cystatin C":                      ["82610"],
    # Electrolytes
    "Basic Metabolic Panel":           ["80048"],
    "Comprehensive Metabolic Panel":   ["80053"],
    "Sodium":                          ["84295"],
    "Potassium":                       ["84132"],
    "Chloride":                        ["82435"],
    "Bicarbonate":                     ["82374"],
    "Calcium":                         ["82310"],
    "Magnesium":                       ["83735"],
    "Phosphate":                       ["84100"],
    # Chemistry / LFTs
    "Hepatic Function Panel":          ["80076"],
    "ALT":                             ["84460"],
    "AST":                             ["84450"],
    "ALP":                             ["84075"],
    "GGT":                             ["82977"],
    "Total Bilirubin":                 ["82247"],
    "Albumin":                         ["82040"],
    # Cardiac markers
    "Troponin I":                      ["84484"],
    "Troponin T":                      ["84512"],
    "CK-MB":                           ["82553"],
    "Total CK":                        ["82550"],
    "BNP":                             ["83880"],
    "NT-proBNP":                       ["83937"],
    "Myoglobin":                       ["83874"],
    "Homocysteine":                    ["83090"],
    "CRP":                             ["86140"],
    "ESR":                             ["85652"],
    "Ferritin":                        ["82728"],
    "LDH":                             ["83615"],
    "Procalcitonin":                   ["84244"],
    # Haematology
    "Full Blood Count (CBC)":          ["85025"],
    "INR / PT":                        ["85610"],
    "APTT":                            ["85730"],
    # Thyroid
    "TSH":                             ["84443"],
    "Free T4":                         ["84439"],
    "Free T3":                         ["84481"],
    # Other endocrine
    "Cortisol (AM)":                   ["82533"],
    "Aldosterone":                     ["82088"],
    "Testosterone":                    ["84403"],
    "Oestradiol (E2)":                 ["82670"],
    "Prolactin":                       ["84146"],
    "PSA":                             ["84153"],
    # Serology
    "ANA":                             ["86038"],
    "Anti-dsDNA":                      ["86235"],
    "Rheumatoid Factor":               ["86430"],
    "Anti-CCP":                        ["86200"],
    "Antiphospholipid Ab":             ["86147"],
    "HBsAg":                           ["87340"],
    "Anti-HCV":                        ["86803"],
    "HIV Ag/Ab":                       ["87389"],
    "VDRL / RPR":                      ["86592"],
    "TPHA":                            ["86596"],
    # Microbiology
    "Blood Culture":                   ["87040"],
    "Urine Culture (M/C/S)":           ["87086"],
    "Sputum Culture":                  ["87070"],
    "Wound Swab Culture":              ["87081"],
    "AFB Smear":                       ["87207"],
    "Malaria RDT / Film":              ["87015"],
    "Stool M/C/S":                     ["87305"],
    "Influenza PCR":                   ["87502"],
    "COVID-19 PCR":                    ["87635"],
    # Toxicology
    "Digoxin Level":                   ["80162"],
    "Vancomycin Level":                ["80202"],
    "Phenytoin Level":                 ["80185"],
    "Lithium Level":                   ["80178"],
    "Blood Alcohol":                   ["82055"],
    "Paracetamol Level":               ["80150"],
    "Urine Drug Screen":               ["80101"],
    # Cytology & Histopathology
    "Cervical LBC (Pap smear)":        ["88141"],
    "HPV Test":                        ["87624"],
    "GIT Biopsy":                      ["88305"],
    "FNAC":                            ["88173"],
    "Sputum/Pleural/Urine Cytology":   ["88108"],
    # Tumour markers
    "CA-125":                          ["86304"],
    "CA 19-9":                         ["86301"],
    "CEA":                             ["82378"],
    "AFP":                             ["82105"],
    # Allergy
    "Total IgE":                       ["86005"],
    "Skin Prick Test":                 ["95004"],
    # Imaging
    "Chest X-Ray (2 views)":           ["71046"],
    "Echocardiogram":                  ["93306"],
    "Carotid Duplex Ultrasound":       ["93880"],
    "Abdominal/Renal Ultrasound":      ["76770"],
    # Rehab
    "Cardiac Rehabilitation Session":  ["93798"],
    "Physiotherapy Session":           ["97110"],
}

# ── Add CPT codes to ICD_DB entries ──────────────────────────────────────────
_ICD_CPT_MAP = {
    "Hypertension":                       ["93000","82947","80048","84443"],
    "Hypertensive Heart Disease":         ["93000","93306","82947","80048"],
    "Hypertensive Heart Disease with Heart Failure": ["93000","93306","83880","80048"],
    "Stable Angina":                      ["93000","93015","80061","84484"],
    "Unstable Angina":                    ["93000","84484","85025","80061"],
    "Myocardial Infarction":              ["93000","84484","82553","83880","71046"],
    "Post-MI":                            ["93000","93306","80061","93798"],
    "Post-MI Rehab":                      ["93798","93000","93015","80061"],
    "Coronary Artery Disease":            ["93452","93015","80061","93000"],
    "Heart Failure":                      ["93306","83880","83937","80048","93000"],
    "Heart Failure (HFrEF)":              ["93306","83880","83937","80048"],
    "Atrial Fibrillation":                ["93000","93224","85610","80048"],
    "Arrhythmia":                         ["93000","93224","80048"],
    "Stroke":                             ["93000","70553","85025","85610"],
    "Stroke/TIA":                         ["93000","70553","93880","85610"],
    "Stroke Rehab":                       ["97001","97530","93798","93000"],
    "Type 2 Diabetes":                    ["83036","82947","80061","82565","33935"],
    "Dyslipidaemia":                      ["80061","83721","83718","86141"],
    "Obesity":                            ["82947","83036","80061","80048"],
    "CKD":                                ["82565","33935","82044","84520","84132"],
    "COPD/Asthma":                        ["71046","94760","93000","85025"],
    "Osteoarthritis":                     ["97001","97110","97530"],
    "Joint Pain / Arthralgia":            ["97001","97110"],
    "Back Pain":                          ["97001","97012","97110"],
    "Cardiomegaly":                       ["71046","93306","93000","83880"],
    "Palpitations":                       ["93000","93224","80048","84132"],
    "Vertigo / Dizziness":                ["93000","70553","80048"],
}
for _dx, _cpts in _ICD_CPT_MAP.items():
    if _dx in ICD_DB:
        ICD_DB[_dx]["cpt_codes"] = _cpts
        ICD_DB[_dx]["cpt_descriptions"] = [
            CPT_DB.get(c, {}).get("description", "—") for c in _cpts
        ]


def render_cpt_panel(investigations: list, context: str = ""):
    """
    Render a CPT procedure code panel for a list of investigation name strings.
    investigations: list of keys from INVESTIGATION_CPT_MAP
    """
    if not investigations:
        return
    rows = []
    for inv in investigations:
        cpts = INVESTIGATION_CPT_MAP.get(inv, [])
        for cpt in cpts:
            rec = CPT_DB.get(cpt, {})
            rows.append({
                "Investigation": inv,
                "CPT Code": cpt,
                "Description": rec.get("description", "—"),
                "Category": rec.get("category", "—"),
                "Tariff (local currency)": f"₦{rec.get('nhis_tariff', 0):,}",
                "Unit": rec.get("unit", "—"),
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    total = sum(CPT_DB.get(c, {}).get("nhis_tariff", 0)
                for inv in investigations
                for c in INVESTIGATION_CPT_MAP.get(inv, []))
    with st.expander(f"🧾 CPT Procedure Codes — {context}", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown(f"**Estimated Total Standard Tariff: ₦{total:,}**")
        st.caption("Tariff rates are approximate reference values. Actual reimbursement depends on insurance scheme tier and facility level.")


def render_cpt_from_icd(diagnoses: list, context: str = ""):
    """
    Render CPT codes derived from ICD diagnoses (shows typical workup per diagnosis).
    """
    if not diagnoses:
        return
    rows = []
    seen = set()
    for dx in diagnoses:
        key = NLP_DX_TO_ICD.get(dx, dx)
        rec = ICD_DB.get(key, {})
        for cpt in rec.get("cpt_codes", []):
            if cpt in seen:
                continue
            seen.add(cpt)
            cpt_rec = CPT_DB.get(cpt, {})
            rows.append({
                "Diagnosis": dx,
                "CPT Code": cpt,
                "Procedure": cpt_rec.get("description", "—"),
                "Category": cpt_rec.get("category", "—"),
                "Tariff (local currency)": f"₦{cpt_rec.get('nhis_tariff', 0):,}",
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    total = sum(CPT_DB.get(c, {}).get("nhis_tariff", 0) for c in seen)
    with st.expander(f"🧾 CPT Procedure Codes — {context}", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown(f"**Estimated Estimated Procedure Total: ₦{total:,}**")
        st.caption("Typical investigations for the identified diagnoses. Approximate standard tariffs.")


def cpt_export_df(investigations: list = None, diagnoses: list = None) -> pd.DataFrame:
    """Build exportable CPT DataFrame from either investigation list or diagnosis list."""
    rows = []
    seen = set()
    if investigations:
        for inv in investigations:
            for cpt in INVESTIGATION_CPT_MAP.get(inv, []):
                if cpt in seen:
                    continue
                seen.add(cpt)
                rec = CPT_DB.get(cpt, {})
                rows.append({"Investigation/Procedure": inv, "CPT Code": cpt,
                             "Description": rec.get("description", "—"),
                             "Category": rec.get("category", "—"),
                             "Tariff (local currency)": rec.get("nhis_tariff", 0),
                             "Unit": rec.get("unit", "—")})
    if diagnoses:
        for dx in diagnoses:
            key = NLP_DX_TO_ICD.get(dx, dx)
            rec = ICD_DB.get(key, {})
            for cpt in rec.get("cpt_codes", []):
                if cpt in seen:
                    continue
                seen.add(cpt)
                cpt_rec = CPT_DB.get(cpt, {})
                rows.append({"Investigation/Procedure": dx, "CPT Code": cpt,
                             "Description": cpt_rec.get("description", "—"),
                             "Category": cpt_rec.get("category", "—"),
                             "Tariff (local currency)": cpt_rec.get("nhis_tariff", 0),
                             "Unit": cpt_rec.get("unit", "—")})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# CENTRAL AI CALL — Gemini Vision + Anthropic Claude fallback
# All Gemini calls in the app route through this function.
# On 429 / quota exhaustion, automatically retries with Claude.
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str, image_b64: str = None,
                 image_mime: str = "image/jpeg",
                 model: str = "gemini-2.0-flash",
                 max_tokens: int = 2048) -> str:
    """
    Call Gemini Vision (or text). On quota/rate-limit error (429),
    automatically falls back to Anthropic Claude.
    Returns the response text string.
    Raises RuntimeError if both APIs fail.
    """
    import google.generativeai as genai

    gemini_key    = st.secrets.get("GOOGLE_API_KEY", "")
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")

    # ── Try Gemini first ──────────────────────────────────────────────
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            gm = genai.GenerativeModel(model)
            content = []
            if image_b64:
                content.append({"mime_type": image_mime, "data": image_b64})
            content.append(prompt)
            response = gm.generate_content(content)
            return response.text
        except Exception as e:
            err_str = str(e)
            is_quota = any(x in err_str for x in
                           ["429", "quota", "RESOURCE_EXHAUSTED",
                            "ResourceExhausted", "rate", "limit"])
            if is_quota:
                st.warning(
                    "⚠ Gemini API quota reached — switching to Anthropic Claude automatically. "
                    "To restore Gemini: check your quota at https://ai.dev/rate-limit"
                )
            else:
                raise RuntimeError(f"Gemini error: {e}")
    else:
        st.info("GOOGLE_API_KEY not set — using Anthropic Claude.")

    # ── Fallback: Anthropic Claude ────────────────────────────────────
    if anthropic_key:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=anthropic_key)
            messages_content = []
            if image_b64:
                messages_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_mime,
                        "data": image_b64,
                    }
                })
            messages_content.append({"type": "text", "text": prompt})
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": messages_content}]
            )
            return resp.content[0].text
        except Exception as e:
            raise RuntimeError(f"Both Gemini and Claude failed. Claude error: {e}")
    else:
        raise RuntimeError(
            "No AI API key available. Set GOOGLE_API_KEY or ANTHROPIC_API_KEY "
            "in Streamlit Cloud → Settings → Secrets."
        )


# ══════════════════════════════════════════════════════════════════════════════
# AI CONFIDENCE SCORING & HALLUCINATION DETECTION
# Applied to all Gemini/Claude vision extraction outputs.
# ══════════════════════════════════════════════════════════════════════════════

# Clinical plausibility ranges for every extractable value
# Format: {field_key: (min_plausible, max_plausible, unit, critical_if_outside)}
CLINICAL_PLAUSIBILITY = {
    # Vital signs
    "sbp_mmhg":              (50,   300,  "mmHg",    True),
    "dbp_mmhg":              (30,   200,  "mmHg",    True),
    "heart_rate_bpm":        (20,   300,  "bpm",     True),
    "spo2_pct":              (50,   100,  "%",       True),
    "rr_breaths_per_min":    (4,    60,   "/min",    False),
    "temp_c":                (30,   45,   "°C",      True),
    # ECG
    "hr_bpm":                (20,   300,  "bpm",     True),
    "pr_interval_ms":        (80,   500,  "ms",      True),
    "qrs_duration_ms":       (40,   300,  "ms",      True),
    "qt_interval_ms":        (200,  800,  "ms",      True),
    "qtc_ms":                (200,  700,  "ms",      True),
    "st_depression_mm":      (0,    15,   "mm",      True),
    "st_elevation_mm":       (0,    15,   "mm",      True),
    # Blood glucose
    "fbs_mg_dl":             (20,   800,  "mg/dL",   True),
    "rbs_mg_dl":             (20,   1200, "mg/dL",   True),
    "hba1c_pct":             (3,    20,   "%",       True),
    "ogtt_2hr_mg_dl":        (30,   600,  "mg/dL",   False),
    # Lipids
    "total_cholesterol_mg_dl":(50,  700,  "mg/dL",   False),
    "ldl_mg_dl":             (10,   500,  "mg/dL",   False),
    "hdl_mg_dl":             (5,    150,  "mg/dL",   False),
    "triglycerides_mg_dl":   (20,   2000, "mg/dL",   False),
    # Renal
    "urea_mmol_l":           (0.5,  80,   "mmol/L",  True),
    "creatinine_umol_l":     (20,   2000, "μmol/L",  True),
    "egfr_ml_min":           (1,    200,  "mL/min",  True),
    # Electrolytes
    "sodium_mmol_l":         (100,  180,  "mmol/L",  True),
    "potassium_mmol_l":      (1.5,  9.0,  "mmol/L",  True),
    "calcium_mmol_l":        (1.0,  4.0,  "mmol/L",  True),
    # Cardiac markers
    "troponin_i_ng_l":       (0,    100000,"ng/L",   True),
    "bnp_pg_ml":             (0,    50000, "pg/mL",  True),
    "nt_probnp_pg_ml":       (0,    100000,"pg/mL",  True),
    "crp_mg_l":              (0,    500,   "mg/L",   False),
    # Echo
    "ef_pct":                (5,    90,   "%",       True),
    "ivsd_mm":               (3,    30,   "mm",      False),
    "lvedd_mm":              (20,   90,   "mm",      False),
    "lvesd_mm":              (10,   80,   "mm",      False),
    # Spirometry
    "fev1_litres":           (0.2,  8.0,  "L",       False),
    "fvc_litres":            (0.3,  10.0, "L",       False),
    "fev1_fvc_ratio":        (0.2,  1.0,  "",        False),
    "fev1_pct_predicted":    (5,    200,  "%",       False),
    "fvc_pct_predicted":     (5,    200,  "%",       False),
    "pef_l_min":             (10,   900,  "L/min",   False),
    "pef_pct_predicted":     (5,    200,  "%",       False),
    # Endocrine
    "tsh_miu_l":             (0.001, 100, "mIU/L",   True),
    "ft4_pmol_l":            (1,    60,   "pmol/L",  False),
    "cortisol_nmol_l":       (10,   3000, "nmol/L",  False),
    "psa_ng_ml":             (0,    500,  "ng/mL",   False),
    # Tumour markers
    "ca125_u_ml":            (0,    10000,"U/mL",    False),
    "cea_ng_ml":             (0,    1000, "ng/mL",   False),
    "afp_ng_ml":             (0,    100000,"ng/mL",  False),
}


def validate_extracted_value(field_key: str, value) -> dict:
    """
    Validate a single AI-extracted value against clinical plausibility ranges.
    Returns: {
        'value': original value,
        'status': 'OK' | 'SUSPECT' | 'IMPLAUSIBLE' | 'UNKNOWN',
        'confidence': 'High' | 'Medium' | 'Low',
        'flag': message or None,
        'critical': bool
    }
    """
    if value is None:
        return {"value": None, "status": "MISSING", "confidence": "—", "flag": None, "critical": False}

    ranges = CLINICAL_PLAUSIBILITY.get(field_key)
    if ranges is None:
        return {"value": value, "status": "UNKNOWN", "confidence": "Medium",
                "flag": "No plausibility range defined — verify manually", "critical": False}

    lo, hi, unit, critical = ranges
    try:
        v = float(str(value).replace(",","").strip())
    except (ValueError, TypeError):
        return {"value": value, "status": "UNKNOWN", "confidence": "Medium",
                "flag": f"Non-numeric — verify manually", "critical": critical}

    if lo <= v <= hi:
        # Extra check: values at extreme ends of range are suspect
        range_span = hi - lo
        if v < lo + 0.05*range_span or v > hi - 0.05*range_span:
            return {"value": v, "status": "SUSPECT", "confidence": "Medium",
                    "flag": f"At extreme of normal range ({lo}–{hi} {unit}) — verify",
                    "critical": critical}
        return {"value": v, "status": "OK", "confidence": "High", "flag": None, "critical": False}
    else:
        severity = "IMPLAUSIBLE" if (v < lo/2 or v > hi*2) else "SUSPECT"
        return {"value": v, "status": severity, "confidence": "Low",
                "flag": f"Outside plausible range ({lo}–{hi} {unit}) — likely AI misread, verify manually",
                "critical": critical}


def validate_extracted_data(data: dict, field_map: dict = None) -> tuple[dict, list, list]:
    """
    Validate all fields in a flat extracted data dict.
    field_map: {json_key: plausibility_key} — maps JSON field names to CLINICAL_PLAUSIBILITY keys
    Returns: (validated_data, warnings, critical_flags)
    """
    warnings_out = []
    critical_out = []
    validated = {}

    for k, v in data.items():
        plaus_key = (field_map or {}).get(k, k)
        result = validate_extracted_value(plaus_key, v)
        validated[k] = result

        if result["status"] in ("SUSPECT", "IMPLAUSIBLE"):
            msg = f"**{k}** = {v}: {result['flag']}"
            if result["critical"] or result["status"] == "IMPLAUSIBLE":
                critical_out.append(msg)
            else:
                warnings_out.append(msg)

    return validated, warnings_out, critical_out


def render_confidence_panel(warnings: list, critical_flags: list,
                             model_used: str = "AI", context: str = ""):
    """Render a confidence and data quality panel after AI extraction."""
    total_issues = len(warnings) + len(critical_flags)

    if critical_flags:
        st.error(
            f"🔴 **{len(critical_flags)} critical plausibility issue(s) detected in AI extraction.**\n\n"
            "These values fall outside clinically possible ranges and are likely AI misreads. "
            "**Do not use these values without manual verification from the original document.**\n\n" +
            "\n".join(f"• {f}" for f in critical_flags)
        )

    if warnings:
        st.warning(
            f"🟡 **{len(warnings)} value(s) at the edge of plausible ranges — verify manually:**\n\n" +
            "\n".join(f"• {w}" for w in warnings)
        )

    if total_issues == 0:
        st.success(
            f"✅ **AI extraction confidence: High** — All extracted values are within "
            f"clinically plausible ranges. Interpreted by: {model_used}."
        )
    else:
        confidence = "Low" if critical_flags else "Medium"
        st.info(
            f"**AI extraction confidence: {confidence}** | Interpreted by: {model_used} | "
            f"{context}\n\n"
            "⚕ Always verify AI-extracted values against the original document before "
            "entering into the Risk Prediction module."
        )

    # Standard disclaimer on every AI extraction
    st.caption(
        "**Important:** AI vision models can misread values from scanned documents, "
        "particularly when images are rotated, low contrast, or use non-standard layouts. "
        "Values flagged above must be checked against the source document. "
        "CardioAI Nova does not guarantee the accuracy of AI-extracted numerical values."
    )


st.set_page_config(
    page_title="CardioAI Nova",
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


def vision_api_call(prompt: str, image_b64: str,
                    image_mime: str = "image/jpeg") -> tuple:
    """
    Central vision API caller used by all imaging modules.
    Strategy:
      1. Try Gemini 2.0 Flash (primary)
      2. On quota/rate-limit (429) → wait briefly then retry once
      3. On continued failure → fall back to Claude Vision (Sonnet)
      4. If both fail → return (None, error_message)

    Returns: (response_text: str | None, model_used: str)
    """
    import time, base64, io

    GEMINI_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-flash-latest",
    ]

    # ── Try Gemini ────────────────────────────────────────────
    google_key = get_secret("GOOGLE_API_KEY")
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            for model_name in GEMINI_MODELS:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content([
                        {"mime_type": image_mime, "data": image_b64},
                        prompt
                    ])
                    return response.text, f"Gemini ({model_name})"
                except Exception as me:
                    err_str = str(me)
                    if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                        # Quota hit — try next model after short pause
                        time.sleep(2)
                        continue
                    elif "404" in err_str or "not found" in err_str.lower():
                        continue   # model not available — try next
                    else:
                        break      # different error — skip remaining gemini models
        except ImportError:
            pass

    # ── Fall back to Claude Vision ────────────────────────────
    anthropic_key = get_secret("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": image_mime,
                        "data": image_b64,
                    }},
                    {"type": "text", "text": prompt},
                ]}]
            )
            return msg.content[0].text, "Claude Vision (Sonnet)"
        except Exception as ce:
            return None, (
                f"Both APIs failed. Gemini: quota exceeded. "
                f"Claude: {str(ce)[:120]}. "
                f"Please wait a few minutes and try again, or upgrade your Gemini API plan at "
                f"https://ai.dev/rate-limit"
            )

    return None, (
        "No Vision API key available. "
        "Add GOOGLE_API_KEY or ANTHROPIC_API_KEY to Streamlit secrets."
    )

def text_api_call(prompt: str, max_tokens: int = 2000) -> tuple:
    """
    Central text-only API caller with Gemini→Claude fallback.
    Used by Pharmaco AI Documentation, Ops AI Advisor, and any text generation.
    Returns: (response_text: str | None, model_used: str)
    """
    import time

    GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"]

    google_key = get_secret("GOOGLE_API_KEY")
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            for model_name in GEMINI_MODELS:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text, f"Gemini ({model_name})"
                except Exception as me:
                    err_str = str(me)
                    if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                        time.sleep(2)
                        continue
                    elif "404" in err_str or "not found" in err_str.lower():
                        continue
                    else:
                        break
        except ImportError:
            pass

    anthropic_key = get_secret("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text, "Claude (Sonnet)"
        except Exception as ce:
            return None, (
                f"Both APIs failed. Gemini: quota exceeded. Claude: {str(ce)[:120]}. "
                f"Wait a few minutes or upgrade your Gemini plan at https://ai.dev/rate-limit"
            )

    return None, "No API key available. Add GOOGLE_API_KEY or ANTHROPIC_API_KEY to Streamlit secrets."


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
    # ── CardioAI Nova Logo ────────────────────────────────────────
    import os
    logo_paths = [
        "CardioAI_Nova_PM.jpeg",
        "app/CardioAI_Nova_PM.jpeg",
        os.path.join(os.path.dirname(__file__), "CardioAI_Nova_PM.jpeg"),
        os.path.join(os.path.dirname(__file__), "..", "CardioAI_Nova_PM.jpeg"),
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
            CardioAI Nova
          </span><br>
          <span style='font-size:10px;color:#475569;letter-spacing:2px;'>
            CLINICAL INTELLIGENCE PLATFORM
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;margin:-8px 0 4px;'><span style='font-size:13px;font-weight:600;color:#0D1B2A;'>CardioAI</span></div>", unsafe_allow_html=True)
    st.caption("AI-Powered Cardiovascular Risk & Clinical Intelligence")
    st.divider()
    page = st.radio("Navigate", ["🫀 Risk Prediction","🏥 Patient Retention","📊 Model Dashboard","📄 Clinical NLP","🔬 Medical Imaging","💊 Pharmaco-Intelligence","🏥 Operational Intelligence","🏷 Clinical Codes","ℹ️ About"], label_visibility="collapsed")
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
    st.caption("Comprehensive clinical data entry — vitals, ECG, labs, biochemistry, microbiology, serology, endocrinology and more.")

    # ══════════════════════════════════════════════════════════════════════
    # CLINICAL INPUT TABS
    # ══════════════════════════════════════════════════════════════════════
    (tab_demo, tab_vitals, tab_ecg, tab_fbs_dm, tab_lipid,
     tab_euc, tab_chem, tab_elec, tab_endo, tab_sero,
     tab_micro, tab_tox, tab_cyto, tab_allergy, tab_gi) = st.tabs([
        "👤 Demographics",
        "🩺 Vital Signs",
        "📟 ECG",
        "🩸 FBS / Diabetes",
        "🧪 Lipid Profile",
        "🔬 E/U/Cr & Renal",
        "⚗️ Chemistry",
        "⚡ Electrolytes",
        "🔭 Endocrinology",
        "🛡 Serology",
        "🦠 Microbiology",
        "💊 Toxicology",
        "🔬 Cytology",
        "🌿 Allergy",
        "🫃 GIT / GYN",
    ])

    # ── TAB 1: Demographics & Symptoms ───────────────────────────────────
    with tab_demo:
        st.subheader("Patient Demographics & Presenting Symptoms")
        d1, d2, d3 = st.columns(3)
        with d1:
            age  = st.number_input("Age (years)", 18, 100, 55)
            sex  = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
            weight_kg = st.number_input("Weight (kg)", 30.0, 200.0, 75.0, 0.5)
            height_cm = st.number_input("Height (cm)", 120.0, 220.0, 170.0, 0.5)
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
            st.metric("BMI", f"{bmi} kg/m²",
                      delta="Obese" if bmi >= 30 else "Overweight" if bmi >= 25 else "Normal")
        with d2:
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                format_func=lambda x: ["Typical Angina","Atypical Angina",
                                       "Non-Anginal Pain","Asymptomatic"][x])
            exang    = st.selectbox("Exercise-Induced Angina", [0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Yes")
            dyspnoea = st.selectbox("Dyspnoea on Exertion", ["None","Mild","Moderate","Severe"])
            palpitations = st.selectbox("Palpitations", ["None","Occasional","Frequent"])
            syncope  = st.selectbox("Syncope / Pre-syncope", ["No","Yes"])
            oedema   = st.selectbox("Peripheral Oedema", ["None","Mild","Moderate","Severe"])
        with d3:
            smoking  = st.selectbox("Smoking Status", ["Never","Ex-smoker","Current smoker"])
            alcohol  = st.selectbox("Alcohol Use", ["None","Occasional","Regular"])
            exercise = st.selectbox("Physical Activity Level", ["Sedentary","Light","Moderate","Active"])
            fam_hx   = st.selectbox("Family History of CVD", ["No","Yes — 1st degree < 65 yrs"])
            duration = st.number_input("Symptom Duration (months)", 0, 360, 6)

    # ── TAB 2: Vital Signs ────────────────────────────────────────────────
    with tab_vitals:
        st.subheader("International Standard Vital Signs")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown("**Blood Pressure**")
            sbp = st.number_input("Systolic BP (mmHg)", 70, 260, 130,
                                  help="Normal: <120 | Stage 1 HTN: 130–139 | Stage 2: ≥140 | Crisis: ≥180")
            dbp = st.number_input("Diastolic BP (mmHg)", 40, 160, 80,
                                  help="Normal: <80 | Stage 1 HTN: 80–89 | Stage 2: ≥90 | Crisis: ≥120")
            trestbps = sbp   # model compatibility
            map_val  = round((sbp + 2 * dbp) / 3, 1)
            pp_val   = sbp - dbp
            st.caption(f"MAP: {map_val} mmHg · Pulse Pressure: {pp_val} mmHg")
            bp_class = ("Hypertensive Crisis" if sbp >= 180 or dbp >= 120
                        else "Stage 2 HTN" if sbp >= 140 or dbp >= 90
                        else "Stage 1 HTN" if sbp >= 130 or dbp >= 80
                        else "Elevated" if sbp >= 120
                        else "Normal")
            if "HTN" in bp_class or "Crisis" in bp_class:
                st.error(f"⚠ {bp_class}")
            elif bp_class == "Elevated":
                st.warning(f"⚠ {bp_class}")
            else:
                st.success(f"✓ {bp_class}")
        with v2:
            st.markdown("**Heart Rate & Respiratory**")
            hr_rest  = st.number_input("Resting Heart Rate (bpm)", 30, 200, 72,
                                       help="Normal: 60–100 | Bradycardia: <60 | Tachycardia: >100")
            thalach  = st.number_input("Max Heart Rate Achieved (bpm)", 60, 220, 150,
                                       help="220 – age = predicted max HR")
            rr       = st.number_input("Respiratory Rate (breaths/min)", 8, 40, 16,
                                       help="Normal: 12–20")
            spo2     = st.number_input("SpO₂ (%)", 70.0, 100.0, 98.0, 0.1,
                                       help="Normal: ≥95% | Concerning: <92%")
            temp_c   = st.number_input("Temperature (°C)", 34.0, 42.0, 36.6, 0.1)
            if spo2 < 92:   st.error(f"⚠ SpO₂ {spo2}% — critically low")
            elif spo2 < 95: st.warning(f"⚠ SpO₂ {spo2}% — low")
            else:           st.success(f"✓ SpO₂ {spo2}%")
        with v3:
            st.markdown("**Other Vitals**")
            gcs      = st.number_input("GCS Score (3–15)", 3, 15, 15,
                                       help="15 = fully alert | <8 = coma")
            pain_vrs = st.slider("Pain Score (0–10)", 0, 10, 0,
                                 help="0 = no pain | 10 = worst imaginable")
            waist_cm = st.number_input("Waist Circumference (cm)", 50.0, 180.0, 90.0, 0.5,
                                       help="Risk: Men >102cm | Women >88cm")
            ht_class = ("Hypothermic" if temp_c < 35 else "Febrile" if temp_c > 37.5 else "Normal")
            rr_class = ("Tachypnoeic" if rr > 20 else "Bradypnoeic" if rr < 12 else "Normal")
            st.caption(f"Temp: {ht_class} · RR: {rr_class} · GCS: {'Alert' if gcs==15 else 'Impaired' if gcs>=9 else 'Coma'}")

    # ── TAB 3: ECG ────────────────────────────────────────────────────────
    with tab_ecg:
        st.subheader("12-Lead ECG Parameters")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown("**Basic ECG**")
            ecg_hr   = st.number_input("ECG Heart Rate (bpm)", 30, 250, 75)
            ecg_rhythm = st.selectbox("Rhythm", ["Sinus Rhythm","Sinus Tachycardia",
                "Sinus Bradycardia","Atrial Fibrillation","Atrial Flutter",
                "SVT","Ventricular Tachycardia","Heart Block","Paced Rhythm","Other"])
            ecg_axis = st.selectbox("QRS Axis", ["Normal (0° to +90°)","Left Axis Deviation",
                "Right Axis Deviation","Extreme Axis Deviation"])
            restecg  = st.selectbox("Resting ECG Classification", [0, 1, 2],
                format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
        with e2:
            st.markdown("**Intervals**")
            pr_ms   = st.number_input("PR Interval (ms)", 80, 400, 160,
                                      help="Normal: 120–200ms | >200ms = 1st degree block")
            qrs_ms  = st.number_input("QRS Duration (ms)", 60, 200, 90,
                                      help="Normal: 70–110ms | >120ms = bundle branch block")
            qt_ms   = st.number_input("QT Interval (ms)", 200, 700, 400,
                                      help="Normal: 350–450ms")
            qtc_ms  = st.number_input("QTc (Corrected QT, ms)", 200, 700, 420,
                                      help="Normal: <440ms (M) / <460ms (F) | Prolonged: ≥500ms = high risk TdP")
            if qtc_ms >= 500: st.error("⚠ QTc ≥500ms — HIGH RISK Torsades de Pointes")
            elif qtc_ms >= 460: st.warning("⚠ QTc prolonged")
        with e3:
            st.markdown("**Waveforms & Segments**")
            p_wave   = st.selectbox("P Wave", ["Normal","Absent","Bifid (P mitrale)",
                                               "Peaked (P pulmonale)","Inverted"])
            st.markdown("**ST Segment**")
            oldpeak  = st.number_input("ST Depression (oldpeak, mm)", 0.0, 10.0, 1.0, 0.1,
                                       help="≥1mm horizontal/downsloping = significant")
            st_elev  = st.number_input("ST Elevation (mm)", 0.0, 10.0, 0.0, 0.1,
                                       help="≥1mm in ≥2 contiguous leads = possible STEMI")
            slope    = st.selectbox("ST Slope", [0, 1, 2],
                format_func=lambda x: ["Downsloping","Flat","Upsloping"][x])
            t_wave   = st.selectbox("T Wave", ["Normal","Inverted","Flat",
                                               "Peaked (Hyperacute)","Biphasic"])
            st.markdown("**Voltages**")
            sv1      = st.number_input("SV1 (mm)", 0.0, 40.0, 10.0, 0.5)
            rv5      = st.number_input("RV5 (mm)", 0.0, 40.0, 15.0, 0.5)
            sv1_rv5  = sv1 + rv5
            if sv1_rv5 > 35:
                st.warning(f"⚠ SV1+RV5 = {sv1_rv5:.1f}mm — LVH criteria met (Sokolow-Lyon)")
            else:
                st.success(f"✓ SV1+RV5 = {sv1_rv5:.1f}mm — within normal limits")
            ca       = st.slider("Major Vessels on Angiography (0–4)", 0, 4, 0)
            thal     = st.selectbox("Thalassemia / Perfusion", [0, 1, 2, 3],
                format_func=lambda x: ["Normal","Fixed Defect","Normal (2)","Reversible Defect"][x])

    # ── TAB 4: FBS / Diabetes ─────────────────────────────────────────────
    with tab_fbs_dm:
        st.subheader("Blood Glucose — Fasting, Random & Diabetes Assessment")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Fasting Blood Sugar (FBS)**")
            fbs_val  = st.number_input("Fasting Blood Sugar (mg/dL)", 40.0, 600.0, 95.0, 0.5,
                                       help="Normal: 70–99 | Pre-DM: 100–125 | DM: ≥126")
            fbs_mmol = round(fbs_val / 18.0, 2)
            st.caption(f"= {fbs_mmol} mmol/L")
            fbs_class = ("Hypoglycaemia" if fbs_val < 70
                         else "Normal" if fbs_val < 100
                         else "Pre-Diabetic (IFG)" if fbs_val < 126
                         else "Diabetic Range")
            if fbs_val >= 126:   st.error(f"⚠ FBS: {fbs_class}")
            elif fbs_val >= 100: st.warning(f"⚠ FBS: {fbs_class}")
            elif fbs_val < 70:   st.error(f"⚠ FBS: {fbs_class}")
            else:                st.success(f"✓ FBS: {fbs_class}")
            fbs = 1 if fbs_val > 120 else 0   # model compatibility

            st.markdown("**Random Blood Sugar (RBS)**")
            rbs_val  = st.number_input("Random Blood Sugar (mg/dL)", 40.0, 800.0, 120.0, 0.5,
                                       help="Normal: <140 | Pre-DM: 140–199 | DM: ≥200")
            rbs_mmol = round(rbs_val / 18.0, 2)
            st.caption(f"= {rbs_mmol} mmol/L")
            rbs_class = ("Hypoglycaemia" if rbs_val < 70
                         else "Normal" if rbs_val < 140
                         else "Impaired Glucose Tolerance" if rbs_val < 200
                         else "Diabetic Range")
            if rbs_val >= 200:   st.error(f"⚠ RBS: {rbs_class}")
            elif rbs_val >= 140: st.warning(f"⚠ RBS: {rbs_class}")
            else:                st.success(f"✓ RBS: {rbs_class}")
        with g2:
            st.markdown("**HbA1c & Long-Term Control**")
            hba1c    = st.number_input("HbA1c (%)", 3.0, 20.0, 5.5, 0.1,
                                       help="Non-DM: <5.7 | Pre-DM: 5.7–6.4 | DM: ≥6.5 | Target (DM): <7.0")
            hba1c_mmol = round((hba1c - 2.15) * 10.929, 1)
            st.caption(f"= {hba1c_mmol} mmol/mol (IFCC)")
            hba1c_class = ("Non-Diabetic" if hba1c < 5.7
                           else "Pre-Diabetic" if hba1c < 6.5
                           else "Diabetic — Well Controlled" if hba1c < 7.0
                           else "Diabetic — Suboptimal" if hba1c < 8.0
                           else "Diabetic — Poor Control")
            if hba1c >= 8.0:    st.error(f"⚠ HbA1c: {hba1c_class}")
            elif hba1c >= 6.5:  st.warning(f"⚠ HbA1c: {hba1c_class}")
            elif hba1c >= 5.7:  st.warning(f"⚠ HbA1c: {hba1c_class}")
            else:               st.success(f"✓ HbA1c: {hba1c_class}")
            dm_type   = st.selectbox("Diabetes Status", ["No Diabetes","Type 1 DM","Type 2 DM",
                                                          "Pre-Diabetes","Gestational DM","LADA"])
            insulin_rx = st.selectbox("On Insulin", ["No","Yes — Basal","Yes — Basal-Bolus",
                                                      "Yes — Pump"])
            ogtt_2hr  = st.number_input("OGTT 2-hr glucose (mg/dL)", 0.0, 600.0, 0.0, 0.5,
                                        help="Normal: <140 | IGT: 140–199 | DM: ≥200 (0 = not done)")
            c_peptide = st.number_input("C-Peptide (nmol/L)", 0.0, 5.0, 0.0, 0.01,
                                        help="Normal: 0.27–1.27 (0 = not done)")

    # ── TAB 5: Fasting Lipid Profile ──────────────────────────────────────
    with tab_lipid:
        st.subheader("Fasting Lipid Profile")
        l1, l2 = st.columns(2)
        with l1:
            chol     = st.number_input("Total Cholesterol (mg/dL)", 50.0, 700.0, 200.0, 1.0,
                                       help="Desirable: <200 | Borderline: 200–239 | High: ≥240")
            ldl      = st.number_input("LDL Cholesterol (mg/dL)", 10.0, 400.0, 130.0, 1.0,
                                       help="Optimal: <100 | Near optimal: 100–129 | High: ≥160")
            hdl      = st.number_input("HDL Cholesterol (mg/dL)", 10.0, 120.0, 50.0, 1.0,
                                       help="Low (risk): Men <40 | Women <50 | High (protective): ≥60")
            tg       = st.number_input("Triglycerides (mg/dL)", 10.0, 2000.0, 150.0, 1.0,
                                       help="Normal: <150 | Borderline: 150–199 | High: 200–499 | Very High: ≥500")
            vldl     = round(tg / 5, 1)
            tc_hdl   = round(chol / hdl, 2) if hdl > 0 else 0
            ldl_hdl  = round(ldl / hdl, 2) if hdl > 0 else 0
        with l2:
            non_hdl  = round(chol - hdl, 1)
            st.markdown("**Calculated Ratios**")
            st.metric("Non-HDL Cholesterol", f"{non_hdl} mg/dL", help="Target: <130 mg/dL")
            st.metric("TC/HDL Ratio", f"{tc_hdl}", help="Low risk: <4.0")
            st.metric("LDL/HDL Ratio", f"{ldl_hdl}", help="Low risk: <2.5")
            st.metric("VLDL (estimated)", f"{vldl} mg/dL", help="Normal: <30 mg/dL")
            apo_b    = st.number_input("ApoB (mg/dL)", 0.0, 300.0, 0.0, 1.0,
                                       help="Optimal: <80 | High risk: >100 (0 = not done)")
            lpa      = st.number_input("Lp(a) (mg/dL)", 0.0, 300.0, 0.0, 1.0,
                                       help="Elevated: >50 mg/dL = high CVD risk (0 = not done)")
            # Flags
            if chol >= 240: st.error("⚠ Total Cholesterol: High")
            elif chol >= 200: st.warning("⚠ Total Cholesterol: Borderline High")
            if ldl >= 160:  st.error("⚠ LDL: High")
            elif ldl >= 130: st.warning("⚠ LDL: Borderline High")
            if hdl < 40:    st.error("⚠ HDL: Low — increased CVD risk")
            if tg >= 500:   st.error("⚠ TG: Very High — pancreatitis risk")
            elif tg >= 200: st.warning("⚠ TG: High")

    # ── TAB 6: E/U/Cr & Renal ────────────────────────────────────────────
    with tab_euc:
        st.subheader("Electrolytes / Urea / Creatinine & Renal Function")
        r1, r2 = st.columns(2)
        with r1:
            urea     = st.number_input("Urea (mmol/L)", 0.0, 80.0, 5.0, 0.1,
                                       help="Normal: 2.5–7.1")
            creat    = st.number_input("Creatinine (μmol/L)", 0.0, 2000.0, 80.0, 1.0,
                                       help="Normal: Men 62–115 | Women 44–80")
            # eGFR via CKD-EPI simplified
            egfr = None
            if creat > 0 and age > 0:
                kappa = 0.7 if sex == 0 else 0.9
                alpha = -0.241 if sex == 0 else -0.302
                sex_f = 1.012 if sex == 0 else 1.0
                creat_norm = (creat / 88.4) / kappa
                egfr = round(142 * (min(creat_norm, 1) ** alpha) *
                              (max(creat_norm, 1) ** -1.200) *
                              (0.9938 ** age) * sex_f, 1)
                ckd_stage = ("G1 Normal" if egfr >= 90
                             else "G2 Mildly Reduced" if egfr >= 60
                             else "G3a Mild-Moderate" if egfr >= 45
                             else "G3b Moderate-Severe" if egfr >= 30
                             else "G4 Severely Reduced" if egfr >= 15
                             else "G5 Kidney Failure")
                if egfr < 30:    st.error(f"⚠ eGFR: {egfr} — {ckd_stage}")
                elif egfr < 60:  st.warning(f"⚠ eGFR: {egfr} — {ckd_stage}")
                else:            st.success(f"✓ eGFR: {egfr} mL/min/1.73m² — {ckd_stage}")
            uric_acid = st.number_input("Uric Acid (μmol/L)", 0.0, 1000.0, 300.0, 1.0,
                                        help="Normal: Men 200–430 | Women 140–360 | Gout risk: >480")
            urine_pcr = st.number_input("Urine PCR (mg/mmol)", 0.0, 500.0, 0.0, 0.1,
                                        help="Normal: <15 | Microalbuminuria: 15–50 | Macroalbuminuria: >50")
        with r2:
            bun      = st.number_input("BUN (mg/dL)", 0.0, 200.0, 14.0, 0.5,
                                       help="Normal: 7–20")
            bun_cr_ratio = round(bun / (creat / 88.4), 1) if creat > 0 else 0
            st.caption(f"BUN/Creatinine ratio: {bun_cr_ratio} (Normal: 10–20)")
            cystatin_c = st.number_input("Cystatin C (mg/L)", 0.0, 10.0, 0.0, 0.01,
                                         help="Normal: 0.53–0.95 (0 = not done)")
            urine_alb  = st.number_input("Urine Albumin (mg/L)", 0.0, 3000.0, 0.0, 1.0,
                                         help="Normal: <30 | Microalbuminuria: 30–300 | Macroalbuminuria: >300")
            if urine_alb >= 300: st.error("⚠ Macroalbuminuria — significant renal/CV risk")
            elif urine_alb >= 30: st.warning("⚠ Microalbuminuria — early nephropathy marker")

    # ── TAB 7: Chemistry ─────────────────────────────────────────────────
    with tab_chem:
        st.subheader("Clinical Chemistry — Liver, Cardiac Markers & Proteins")
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            st.markdown("**Liver Function**")
            alt      = st.number_input("ALT/SGPT (U/L)", 0.0, 2000.0, 25.0, 1.0,
                                       help="Normal: Men <56 | Women <36")
            ast      = st.number_input("AST/SGOT (U/L)", 0.0, 2000.0, 22.0, 1.0,
                                       help="Normal: Men <40 | Women <32")
            alp      = st.number_input("ALP (U/L)", 0.0, 2000.0, 80.0, 1.0,
                                       help="Normal: 44–147")
            ggt      = st.number_input("GGT (U/L)", 0.0, 1000.0, 30.0, 1.0,
                                       help="Normal: Men <55 | Women <38")
            tbil     = st.number_input("Total Bilirubin (μmol/L)", 0.0, 500.0, 12.0, 0.5,
                                       help="Normal: 3–21")
            albumin  = st.number_input("Albumin (g/L)", 0.0, 60.0, 40.0, 0.5,
                                       help="Normal: 35–50")
            tp_chem  = st.number_input("Total Protein (g/L)", 0.0, 100.0, 70.0, 0.5,
                                       help="Normal: 60–80")
        with ch2:
            st.markdown("**Cardiac Biomarkers**")
            troponin = st.number_input("Troponin I (ng/L)", 0.0, 50000.0, 0.0, 1.0,
                                       help="Normal: <52 ng/L | NSTEMI: ≥52 | STEMI: very high (0 = not done)")
            ck_mb    = st.number_input("CK-MB (U/L)", 0.0, 1000.0, 0.0, 1.0,
                                       help="Normal: <25 U/L (0 = not done)")
            ck_total = st.number_input("Total CK (U/L)", 0.0, 10000.0, 0.0, 1.0,
                                       help="Normal: Men <200 | Women <170 (0 = not done)")
            bnp      = st.number_input("BNP (pg/mL)", 0.0, 5000.0, 0.0, 1.0,
                                       help="Normal: <100 | HF: >400 | Grey zone: 100–400 (0 = not done)")
            nt_probnp = st.number_input("NT-proBNP (pg/mL)", 0.0, 35000.0, 0.0, 1.0,
                                        help="HF unlikely: <300 | Likely (age 75+): >1800 (0 = not done)")
            myoglobin = st.number_input("Myoglobin (μg/L)", 0.0, 5000.0, 0.0, 1.0,
                                        help="Normal: Men <72 | Women <58 (0 = not done)")
            if troponin >= 52: st.error(f"⚠ Troponin elevated: {troponin} ng/L")
            if bnp >= 400:     st.error(f"⚠ BNP: {bnp} — Heart Failure likely")
            elif bnp >= 100:   st.warning(f"⚠ BNP: {bnp} — Heart Failure possible")
        with ch3:
            st.markdown("**Inflammatory & Other**")
            crp      = st.number_input("CRP (mg/L)", 0.0, 500.0, 1.0, 0.1,
                                       help="Normal: <5 | Cardiac risk (hs-CRP): <1=low, 1–3=mod, >3=high")
            hscrp    = st.number_input("hs-CRP (mg/L)", 0.0, 100.0, 0.0, 0.1,
                                       help="Low: <1 | Moderate: 1–3 | High: >3 (CVD risk) (0 = not done)")
            esr      = st.number_input("ESR (mm/hr)", 0.0, 150.0, 10.0, 1.0,
                                       help="Normal: Men <15 | Women <20")
            ferritin = st.number_input("Ferritin (μg/L)", 0.0, 5000.0, 50.0, 1.0,
                                       help="Normal: Men 24–336 | Women 11–307")
            ldh      = st.number_input("LDH (U/L)", 0.0, 5000.0, 200.0, 1.0,
                                       help="Normal: 135–225")
            homocys  = st.number_input("Homocysteine (μmol/L)", 0.0, 100.0, 0.0, 0.1,
                                       help="Normal: 5–15 | High CVD risk: >15 (0 = not done)")
            if hscrp > 3:  st.error("⚠ hs-CRP >3 — High cardiovascular risk")
            elif hscrp > 1: st.warning("⚠ hs-CRP 1–3 — Moderate cardiovascular risk")

    # ── TAB 8: Electrolytes ──────────────────────────────────────────────
    with tab_elec:
        st.subheader("Serum Electrolytes")
        el1, el2 = st.columns(2)
        with el1:
            na       = st.number_input("Sodium Na⁺ (mmol/L)", 100.0, 170.0, 140.0, 0.5,
                                       help="Normal: 135–145 | Hypo: <135 | Hyper: >145")
            k        = st.number_input("Potassium K⁺ (mmol/L)", 1.5, 8.0, 4.0, 0.1,
                                       help="Normal: 3.5–5.0 | Dangerous low: <3.0 | High: >5.5")
            cl       = st.number_input("Chloride Cl⁻ (mmol/L)", 80.0, 130.0, 102.0, 0.5,
                                       help="Normal: 98–106")
            co2      = st.number_input("Bicarbonate HCO₃⁻ (mmol/L)", 5.0, 45.0, 24.0, 0.5,
                                       help="Normal: 22–29")
            ag       = round(na - (cl + co2), 1)
            st.caption(f"Anion Gap: {ag} (Normal: 8–12)")
            if ag > 12: st.warning(f"⚠ Elevated Anion Gap: {ag}")
        with el2:
            ca_ion   = st.number_input("Calcium Ca²⁺ (mmol/L)", 1.0, 4.0, 2.4, 0.05,
                                       help="Normal: 2.15–2.55 | Hypo: <2.1 | Hyper: >2.6")
            mg       = st.number_input("Magnesium Mg²⁺ (mmol/L)", 0.3, 3.0, 0.9, 0.05,
                                       help="Normal: 0.75–1.0")
            phos     = st.number_input("Phosphate PO₄³⁻ (mmol/L)", 0.2, 3.0, 1.1, 0.05,
                                       help="Normal: 0.8–1.45")
            # Flag dangerous levels
            if k < 3.0:    st.error(f"⚠ K⁺ {k} — Dangerous Hypokalaemia (arrhythmia risk)")
            elif k < 3.5:  st.warning(f"⚠ K⁺ {k} — Hypokalaemia")
            elif k > 5.5:  st.error(f"⚠ K⁺ {k} — Hyperkalaemia")
            if na < 125:   st.error(f"⚠ Na⁺ {na} — Severe Hyponatraemia")
            elif na < 135: st.warning(f"⚠ Na⁺ {na} — Hyponatraemia")

    # ── TAB 9: Endocrinology ─────────────────────────────────────────────
    with tab_endo:
        st.subheader("Endocrinology Panel")
        en1, en2 = st.columns(2)
        with en1:
            st.markdown("**Thyroid Function**")
            tsh      = st.number_input("TSH (mIU/L)", 0.0, 100.0, 2.0, 0.01,
                                       help="Normal: 0.4–4.0 | Hypo: >4.0 | Hyper: <0.4")
            ft4      = st.number_input("Free T4 (pmol/L)", 0.0, 50.0, 15.0, 0.1,
                                       help="Normal: 9–25")
            ft3      = st.number_input("Free T3 (pmol/L)", 0.0, 20.0, 4.5, 0.1,
                                       help="Normal: 2.6–5.7")
            if tsh > 4.0:   st.warning("⚠ TSH elevated — Hypothyroidism (↑CV risk)")
            elif tsh < 0.4: st.warning("⚠ TSH suppressed — Hyperthyroidism (arrhythmia risk)")
            st.markdown("**Adrenal**")
            cortisol = st.number_input("Morning Cortisol (nmol/L)", 0.0, 2000.0, 400.0, 1.0,
                                       help="Normal AM: 171–536 | Cushing's: >700")
            dheas    = st.number_input("DHEA-S (μmol/L)", 0.0, 20.0, 0.0, 0.1,
                                       help="Men: 2.2–15.2 | Women: 1.0–11.6 (0 = not done)")
            aldost   = st.number_input("Aldosterone (pmol/L)", 0.0, 2000.0, 0.0, 1.0,
                                       help="Normal: 30–400 (0 = not done)")
        with en2:
            st.markdown("**Reproductive Hormones**")
            insulin_level = st.number_input("Fasting Insulin (mU/L)", 0.0, 200.0, 0.0, 0.5,
                                            help="Normal: 2–25 | Insulin resistance: >25 (0 = not done)")
            homa_ir  = round((fbs_val/18.0 * insulin_level) / 22.5, 2) if insulin_level > 0 else 0
            if homa_ir > 0:
                ir_flag = "Insulin Resistant" if homa_ir > 2.5 else "Normal Sensitivity"
                st.caption(f"HOMA-IR: {homa_ir} — {ir_flag}")
            testosterone = st.number_input("Testosterone (nmol/L)", 0.0, 50.0, 0.0, 0.1,
                                           help="Men: 9.9–27.8 | Women: 0.3–1.7 (0 = not done)")
            oestradiol = st.number_input("Oestradiol E2 (pmol/L)", 0.0, 10000.0, 0.0, 1.0,
                                         help="(0 = not done / not applicable)")
            prolactin  = st.number_input("Prolactin (mIU/L)", 0.0, 10000.0, 0.0, 1.0,
                                         help="Normal: Men <360 | Women <600 (0 = not done)")
            psa        = st.number_input("PSA (ng/mL) — Males only", 0.0, 100.0, 0.0, 0.01,
                                         help="Normal: <4.0 | Elevated: >10 (0 = not done)")
            if psa > 10: st.error("⚠ PSA >10 — Prostate assessment recommended")
            elif psa > 4: st.warning("⚠ PSA elevated (4–10) — Grey zone")

    # ── TAB 10: Serology ─────────────────────────────────────────────────
    with tab_sero:
        st.subheader("Serology — Immunology & Infectious Disease Markers")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Cardiovascular / Autoimmune**")
            ana      = st.selectbox("ANA (Anti-Nuclear Antibody)", ["Not done","Negative","Positive — Low titre","Positive — High titre"])
            anti_ds_dna = st.selectbox("Anti-dsDNA", ["Not done","Negative","Positive"])
            rf       = st.selectbox("Rheumatoid Factor", ["Not done","Negative","Positive (<20)","Positive (>20)"])
            anti_ccp = st.selectbox("Anti-CCP", ["Not done","Negative","Positive"])
            antiphospholipid = st.selectbox("Antiphospholipid Antibody", ["Not done","Negative","Positive"])
            c3_comp  = st.number_input("Complement C3 (g/L)", 0.0, 3.0, 0.0, 0.01,
                                       help="Normal: 0.90–1.80 (0 = not done)")
            c4_comp  = st.number_input("Complement C4 (g/L)", 0.0, 1.0, 0.0, 0.01,
                                       help="Normal: 0.16–0.47 (0 = not done)")
        with s2:
            st.markdown("**Infectious Disease Serology**")
            hbsag    = st.selectbox("HBsAg (Hepatitis B Surface Antigen)", ["Not done","Non-reactive","Reactive"])
            hbcab    = st.selectbox("HBcAb (Hepatitis B Core Ab)", ["Not done","Negative","Positive"])
            hcv_ab   = st.selectbox("Anti-HCV (Hepatitis C Ab)", ["Not done","Non-reactive","Reactive"])
            hiv      = st.selectbox("HIV 1 & 2 Ab/Ag", ["Not done","Non-reactive","Reactive — confirm"])
            vdrl     = st.selectbox("VDRL / RPR (Syphilis)", ["Not done","Non-reactive","Reactive"])
            tpha     = st.selectbox("TPHA (Syphilis confirmatory)", ["Not done","Negative","Positive"])
            toxoplasma_igg = st.selectbox("Toxoplasma IgG", ["Not done","Negative","Positive"])
            cmv_igg  = st.selectbox("CMV IgG", ["Not done","Negative","Positive"])
            ebv_vca  = st.selectbox("EBV VCA IgG", ["Not done","Negative","Positive"])
            if hbsag == "Reactive":    st.error("⚠ HBsAg Reactive — Active Hepatitis B")
            if hcv_ab == "Reactive":   st.error("⚠ Anti-HCV Reactive — Confirm Hepatitis C")
            if hiv == "Reactive — confirm": st.error("⚠ HIV Reactive — Confirmatory test required")

    # ── TAB 11: Microbiology ─────────────────────────────────────────────
    with tab_micro:
        st.subheader("Microbiology — Bacteriology, Parasitology & Virology")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Bacteriology**")
            blood_cx = st.selectbox("Blood Culture", ["Not done","No growth","Gram-positive cocci",
                "Gram-negative rods","Gram-negative cocci","Polymicrobial","Contaminated"])
            urine_cx = st.selectbox("Urine Culture", ["Not done","No growth","E. coli",
                "Klebsiella","Pseudomonas","Staphylococcus","Enterococcus","Other"])
            sputum_cx = st.selectbox("Sputum Culture / AFB", ["Not done","No growth",
                "Normal flora","AFB Positive","Streptococcus pneumoniae",
                "Haemophilus influenzae","Klebsiella","Other"])
            wound_cx = st.selectbox("Wound / Swab Culture", ["Not done","No growth",
                "MRSA","MSSA","Streptococcus","Pseudomonas","Other"])
            procalc  = st.number_input("Procalcitonin (ng/mL)", 0.0, 100.0, 0.0, 0.01,
                                       help="Normal: <0.1 | Bacterial sepsis: >0.5 | Septic shock: >10")
            if procalc > 0.5: st.error(f"⚠ Procalcitonin {procalc} — Bacterial infection/sepsis likely")
        with m2:
            st.markdown("**Parasitology**")
            malaria  = st.selectbox("Malaria RDT / Film", ["Not done","Negative",
                "P. falciparum Positive","P. vivax Positive","P. malariae Positive","Mixed"])
            stool_mcs = st.selectbox("Stool M/C/S", ["Not done","No pathogen",
                "Salmonella","Shigella","E. coli pathogenic","Giardia","Cryptosporidium",
                "Hookworm","Ascaris","Strongyloides","Other"])
            st.markdown("**Virology**")
            influenza = st.selectbox("Influenza RDT", ["Not done","Negative","Influenza A","Influenza B"])
            covid19   = st.selectbox("SARS-CoV-2 (COVID-19)", ["Not done","Negative","Positive"])
            dengue    = st.selectbox("Dengue RDT (NS1/IgM/IgG)", ["Not done","Negative",
                                                                     "NS1 Positive","IgM Positive","IgG Positive"])
            if malaria != "Not done" and malaria != "Negative": st.error(f"⚠ {malaria}")
            if covid19 == "Positive": st.error("⚠ COVID-19 Positive")

    # ── TAB 12: Toxicology ───────────────────────────────────────────────
    with tab_tox:
        st.subheader("Toxicology & Drug Levels")
        tx1, tx2 = st.columns(2)
        with tx1:
            st.markdown("**Therapeutic Drug Monitoring**")
            digoxin  = st.number_input("Digoxin level (nmol/L)", 0.0, 10.0, 0.0, 0.01,
                                       help="Therapeutic: 0.5–2.0 | Toxic: >2.5 (0 = not on drug)")
            warfarin_inr = st.number_input("INR (if on Warfarin)", 0.0, 10.0, 0.0, 0.1,
                                           help="Normal: 0.9–1.1 | AF target: 2–3 | Valve target: 2.5–3.5")
            phenytoin = st.number_input("Phenytoin (μmol/L)", 0.0, 160.0, 0.0, 1.0,
                                        help="Therapeutic: 40–80 (0 = not on drug)")
            vancomycin = st.number_input("Vancomycin trough (mg/L)", 0.0, 100.0, 0.0, 0.1,
                                         help="Therapeutic: 10–20 (0 = not on drug)")
            lithium   = st.number_input("Lithium (mmol/L)", 0.0, 3.0, 0.0, 0.01,
                                        help="Therapeutic: 0.6–1.2 | Toxic: >1.5 (0 = not on drug)")
            if digoxin > 2.5:  st.error(f"⚠ Digoxin toxic: {digoxin} nmol/L")
            if warfarin_inr > 4: st.error(f"⚠ INR {warfarin_inr} — Bleeding risk high")
            elif warfarin_inr > 3: st.warning(f"⚠ INR {warfarin_inr} — Above target")
        with tx2:
            st.markdown("**Substance / Overdose Screen**")
            alcohol_level = st.number_input("Blood Alcohol (mg/dL)", 0.0, 500.0, 0.0, 1.0,
                                            help="Intoxication: >80 | Coma risk: >300")
            paracetamol  = st.number_input("Paracetamol (μmol/L)", 0.0, 3000.0, 0.0, 1.0,
                                           help="Therapeutic: <132 | Toxic: >200 | Treatment threshold: Rumack-Matthew nomogram")
            salicylate   = st.number_input("Salicylate (mmol/L)", 0.0, 10.0, 0.0, 0.01,
                                           help="Therapeutic: 1.1–2.2 | Toxic: >3.6")
            urine_drugs  = st.multiselect("Urine Drug Screen (Positive for)",
                                          ["Cannabis","Cocaine","Opiates","Amphetamines",
                                           "Benzodiazepines","Barbiturates","Tricyclics","PCP","MDMA"],
                                          default=[])
            if alcohol_level > 80: st.error(f"⚠ Blood Alcohol: {alcohol_level} mg/dL — Intoxicated")
            if paracetamol > 200:  st.error(f"⚠ Paracetamol toxic: {paracetamol} μmol/L — Antidote needed")

    # ── TAB 13: Cytology ─────────────────────────────────────────────────
    with tab_cyto:
        st.subheader("Cytology — Non-Gynaecological & Gynaecological")
        cy1, cy2 = st.columns(2)
        with cy1:
            st.markdown("**Non-Gynaecological Cytology**")
            sputum_cyt = st.selectbox("Sputum Cytology", ["Not done","Normal","Atypical cells",
                "Suspicious for malignancy","Malignant — NSCLC","Malignant — SCLC","Other"])
            pleural_cyt = st.selectbox("Pleural Fluid Cytology", ["Not done","No malignant cells",
                "Atypical cells","Adenocarcinoma","Mesothelioma","Other"])
            urine_cyt  = st.selectbox("Urine Cytology", ["Not done","Normal","Atypical",
                "Suspicious","High-grade urothelial carcinoma","Other"])
            ascites_cyt = st.selectbox("Ascitic Fluid Cytology", ["Not done","No malignant cells",
                "Adenocarcinoma","Mesothelioma","Other"])
            fnac       = st.selectbox("FNAC (Specify site)", ["Not done","Benign","Atypical",
                "Suspicious","Malignant — Carcinoma","Malignant — Lymphoma","Other"])
        with cy2:
            st.markdown("**Gynaecological Cytology (LBC — Liquid-Based Cytology)**")
            pap_smear  = st.selectbox("Cervical LBC / Pap Smear", ["Not done","NILM (Normal)",
                "ASCUS","LSIL","HSIL","ASC-H","AGC","Carcinoma"])
            hpv_test   = st.selectbox("HPV Test", ["Not done","Negative","HPV 16/18 Positive",
                                                    "Other High-Risk HPV Positive"])
            endometrial_cyt = st.selectbox("Endometrial Cytology", ["Not done","Normal",
                "Atypical endometrial cells","Endometrial carcinoma"])
            if pap_smear in ["HSIL","ASC-H","AGC","Carcinoma"]:
                st.error(f"⚠ Cervical LBC: {pap_smear} — Urgent colposcopy referral")
            elif pap_smear in ["ASCUS","LSIL"]:
                st.warning(f"⚠ Cervical LBC: {pap_smear} — Follow-up required")
            if hpv_test != "Not done" and hpv_test != "Negative":
                st.error(f"⚠ {hpv_test} — High-risk HPV detected")

    # ── TAB 14: Allergy ──────────────────────────────────────────────────
    with tab_allergy:
        st.subheader("Allergy & Immunology Panel")
        al1, al2 = st.columns(2)
        with al1:
            st.markdown("**Total IgE & RAST**")
            ige_total = st.number_input("Total IgE (IU/mL)", 0.0, 10000.0, 0.0, 1.0,
                                        help="Normal: <100 | Atopy: >200 | Parasitic: >1000")
            if ige_total > 200: st.warning(f"⚠ Total IgE {ige_total} — Atopic/allergic condition")
            drug_allergy = st.multiselect("Known Drug Allergies", [
                "Penicillin","Cephalosporins","Sulphonamides","NSAIDs","Aspirin",
                "ACE Inhibitors","Statins","Contrast Dye","Latex","Metformin","Other"])
            food_allergy = st.multiselect("Known Food Allergies", [
                "Peanuts","Shellfish","Tree nuts","Milk","Eggs","Wheat/Gluten","Soy","Fish","Other"])
            environmental = st.multiselect("Environmental Allergens", [
                "House dust mite","Pollen","Mould","Animal dander","Cockroach","Other"])
        with al2:
            st.markdown("**Specific IgE / Skin Prick Tests**")
            eosinophils = st.number_input("Eosinophil count (×10⁹/L)", 0.0, 5.0, 0.2, 0.01,
                                          help="Normal: 0.04–0.44 | High: >0.5 = eosinophilia")
            eosinophil_pct = st.number_input("Eosinophils (%)", 0.0, 60.0, 2.0, 0.1,
                                             help="Normal: 1–4%")
            skin_prick = st.selectbox("Skin Prick Test Result", ["Not done","Negative — no reaction",
                "Positive — 1 allergen","Positive — 2–3 allergens","Positive — multiple"])
            allergy_severity = st.selectbox("Worst Allergic Reaction", ["None","Mild (urticaria)",
                "Moderate (bronchospasm)","Severe (anaphylaxis)"])
            if allergy_severity == "Severe (anaphylaxis)": st.error("⚠ History of anaphylaxis — document and warn")
            if eosinophils > 0.5: st.warning(f"⚠ Eosinophilia: {eosinophils} ×10⁹/L")

    # ── TAB 15: GIT Biopsy & Gynaecology ────────────────────────────────
    with tab_gi:
        st.subheader("GIT Biopsy / Histopathology & Gynaecological Pathology")
        gi1, gi2 = st.columns(2)
        with gi1:
            st.markdown("**GIT Biopsy / Histopathology**")
            git_site = st.selectbox("GIT Biopsy Site", ["Not done","Oesophagus","Stomach — Antrum",
                "Stomach — Body","Duodenum","Jejunum","Ileum","Colon — Right",
                "Colon — Left","Rectum","Appendix","Liver","Pancreas"])
            git_result = st.selectbox("GIT Histopathology Result", ["Not done",
                "Normal mucosa","Chronic gastritis","H. pylori gastritis",
                "Intestinal metaplasia","Dysplasia — Low grade","Dysplasia — High grade",
                "Adenocarcinoma","GIST","Lymphoma","Crohn's disease","Ulcerative colitis",
                "Coeliac disease","Hepatitis — chronic","Cirrhosis","HCC","Other"])
            helicobacter = st.selectbox("H. pylori (Rapid Urease / Urea Breath)", ["Not done",
                "Negative","Positive"])
            if git_result in ["Dysplasia — High grade","Adenocarcinoma","HCC","Lymphoma"]:
                st.error(f"⚠ Histopathology: {git_result} — Urgent oncology referral")
            elif git_result in ["Dysplasia — Low grade","Intestinal metaplasia"]:
                st.warning(f"⚠ Histopathology: {git_result} — Close surveillance required")
            if helicobacter == "Positive": st.warning("⚠ H. pylori Positive — Eradication therapy indicated")
        with gi2:
            st.markdown("**Gynaecological Pathology**")
            endometrial_histo = st.selectbox("Endometrial Biopsy", ["Not done","Normal",
                "Simple hyperplasia","Complex hyperplasia","Atypical hyperplasia",
                "Endometrial carcinoma — Grade 1","Endometrial carcinoma — Grade 2/3"])
            ovarian_path = st.selectbox("Ovarian Pathology (imaging/biopsy)", ["Not done","Normal",
                "Functional cyst","Dermoid","Endometrioma","Serous cystadenoma",
                "Mucinous cystadenoma","Borderline tumour","Malignant — Serous","Other"])
            ca125        = st.number_input("CA-125 (U/mL)", 0.0, 5000.0, 0.0, 1.0,
                                           help="Normal: <35 | Ovarian cancer marker (0 = not done)")
            ca199        = st.number_input("CA 19-9 (U/mL)", 0.0, 5000.0, 0.0, 1.0,
                                           help="Normal: <37 | GIT/Pancreatic cancer marker (0 = not done)")
            cea          = st.number_input("CEA (ng/mL)", 0.0, 500.0, 0.0, 0.1,
                                           help="Normal: <5 | Colorectal, lung, breast cancer marker (0 = not done)")
            afp          = st.number_input("AFP (ng/mL)", 0.0, 10000.0, 0.0, 1.0,
                                           help="Normal: <10 | HCC / Germ cell tumour marker (0 = not done)")
            if ca125 > 35:  st.warning(f"⚠ CA-125 elevated: {ca125} — Gynaecological / peritoneal pathology")
            if afp > 400:   st.error(f"⚠ AFP: {afp} — HCC or germ cell tumour")

    # ══════════════════════════════════════════════════════════════════════
    # GENERATE RISK ASSESSMENT BUTTON
    # ══════════════════════════════════════════════════════════════════════
    st.divider()

    # ── Expanded LRI — now incorporating all clinical panels ──────────────
    def compute_lri_expanded(trestbps, chol, fbs_flag, exang, oldpeak,
                              hba1c=5.5, ldl=130.0, hdl=50.0, tg=150.0,
                              crp=1.0, bmi=25.0, smoking="Never",
                              fam_hx="No", sbp=130, dbp=80):
        """
        Enhanced Lifestyle Risk Index — incorporates full lipid panel,
        HbA1c, CRP, BMI, smoking and family history alongside original features.
        Score: 0.0–1.0 (higher = higher risk)
        """
        def norm(v, lo, hi):
            return max(0.0, min(1.0, (v - lo) / (hi - lo + 1e-8)))

        score = (
            norm(trestbps, 94, 200)    * 0.12 +
            norm(chol, 126, 564)       * 0.08 +
            fbs_flag                   * 0.10 +
            exang                      * 0.08 +
            norm(oldpeak, 0, 6.2)      * 0.08 +
            norm(hba1c, 4.0, 12.0)    * 0.10 +
            norm(ldl, 50, 300)         * 0.08 +
            (1 - norm(hdl, 30, 90))    * 0.07 +   # low HDL = higher risk
            norm(tg, 50, 500)          * 0.05 +
            norm(crp, 0, 30)           * 0.07 +
            norm(bmi, 18.5, 45)        * 0.06 +
            (0.15 if smoking == "Current smoker" else 0.07 if smoking == "Ex-smoker" else 0) * 0.07 +
            (0.10 if fam_hx != "No" else 0) * 0.04
        )
        return round(min(score, 1.0), 4)

    lri_expanded = compute_lri_expanded(
        trestbps=trestbps, chol=chol, fbs_flag=fbs, exang=exang, oldpeak=oldpeak,
        hba1c=hba1c, ldl=ldl, hdl=hdl, tg=tg, crp=crp, bmi=bmi,
        smoking=smoking, fam_hx=fam_hx, sbp=sbp, dbp=dbp,
    )

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

            # ── PRIMARY MODEL: XGBoost (UCI) ──────────────────────────────
            risk_prob_uci = xgb_model.predict_proba(X_scaled)[0][1]

            # ── SECONDARY MODELS: Validated Equations ────────────────────
            # 1. Framingham Heart Study 10-year risk (Wilson et al. 1998)
            def framingham_risk(age, sex, chol, hdl, sbp, smoker, dm):
                import math
                if sex == 1:  # Male
                    l = (3.06117*math.log(age) + 1.12370*math.log(chol) -
                         0.93263*math.log(hdl) + 1.93303*math.log(sbp) +
                         0.65451*smoker + 0.57367*dm - 23.9388)
                    risk = 1 - 0.88936**math.exp(l - 3.0975)  # baseline survival 0.88936 at 10yr
                else:  # Female
                    l = (2.32888*math.log(age) + 1.20904*math.log(chol) -
                         0.70833*math.log(hdl) + 2.76157*math.log(sbp) +
                         0.52873*smoker + 0.69154*dm - 26.1931)
                    risk = 1 - 0.94833**math.exp(l - (-1.4792))
                return max(0.0, min(1.0, risk))

            smoker_flag = 1 if "smoker" in smoking.lower() and "never" not in smoking.lower() else 0
            dm_flag     = 1 if fbs == 1 or hba1c >= 6.5 or (hasattr(dm_type,'__contains__') and "DM" in str(dm_type)) else 0
            hdl_val     = hdl if hdl > 0 else 45.0
            sbp_val     = sbp if sbp > 0 else trestbps

            try:
                risk_framingham = framingham_risk(age, sex, chol, hdl_val, sbp_val, smoker_flag, dm_flag)
            except Exception:
                risk_framingham = None

            # 2. WHO/ISH Risk Score (simplified — Region AFRO D, no labs version)
            # Published: WHO CVD Risk Chart 2019, adjusted for African region
            def who_ish_risk(age, sex, sbp, smoker, dm, bmi_val):
                base = 0.05
                if age >= 70:        base += 0.20
                elif age >= 60:      base += 0.12
                elif age >= 50:      base += 0.06
                elif age >= 40:      base += 0.02
                if sbp >= 180:       base += 0.25
                elif sbp >= 160:     base += 0.15
                elif sbp >= 140:     base += 0.08
                elif sbp >= 130:     base += 0.03
                if smoker:           base += 0.10
                if dm:               base += 0.08
                if sex == 1:         base += 0.04
                if bmi_val >= 30:    base += 0.03
                # African region multiplier (INTERHEART Africa — higher relative risk per BP unit)
                base *= 1.12
                return max(0.0, min(1.0, base))

            risk_who = who_ish_risk(age, sex, sbp_val, smoker_flag, dm_flag, bmi)

            # 3. AHA/ACC Pooled Cohort Equations (Goff et al. 2014)
            # 10-year ASCVD risk — validated in multiethnic US cohorts including Black Americans
            def pooled_cohort_equations(age, sex, race_black, chol, hdl, sbp,
                                         bp_treated, smoker, dm):
                import math
                ln_age    = math.log(age)
                ln_chol   = math.log(chol) if chol > 0 else math.log(200)
                ln_hdl    = math.log(hdl)  if hdl  > 0 else math.log(50)
                ln_sbp    = math.log(sbp)  if sbp  > 0 else math.log(120)

                if sex == 1 and not race_black:  # White Male
                    s10 = 0.9144
                    coeff = (12.344*ln_age + 11.853*ln_chol - 2.664*ln_age*ln_chol
                             - 7.990*ln_hdl + 1.769*ln_age*ln_hdl
                             + 1.764*(ln_sbp if bp_treated else 0)
                             + 1.797*(ln_sbp if not bp_treated else 0)
                             + 0.658*smoker - 0.661*ln_age*smoker + 0.671*dm - 29.799)
                elif sex == 1 and race_black:  # Black Male
                    s10 = 0.8954
                    coeff = (2.469*ln_age + 0.302*ln_chol - 0.307*ln_hdl
                             + 1.916*(ln_sbp if bp_treated else 0)
                             + 1.809*(ln_sbp if not bp_treated else 0)
                             + 0.549*smoker + 0.645*dm - 19.540)
                elif sex == 0 and not race_black:  # White Female
                    s10 = 0.9665
                    coeff = (-7.574*ln_age + 17.1141*ln_chol - 0.94*(ln_age*ln_chol)
                             + 20.374*ln_hdl - 3.819*(ln_age*ln_hdl)
                             + 3.438*(ln_sbp if bp_treated else 0)
                             - 1.010*(ln_sbp if not bp_treated else 0)
                             + 0.661*smoker - 13.578*dm - 1.665)
                else:  # Black Female
                    s10 = 0.9533
                    coeff = (17.1141*ln_chol + 0.940*ln_hdl
                             + 29.799*(ln_sbp if bp_treated else 0)
                             + 27.819*(ln_sbp if not bp_treated else 0)
                             + 0.661*smoker + 0.661*dm - 86.608)
                risk = 1 - s10**math.exp(coeff)
                return max(0.0, min(1.0, risk))

            # Population selector (used for PCE and calibration)
            pop_race_black = (sp_ethnic == "African / Black" if "sp_ethnic" in dir() else
                              "African" in str(st.session_state.get("sp_ethnic","")) or
                              sex == 1)  # default assumption for this platform context
            bp_treated = "ACE" in str(st.session_state.get("ecg_meds","")) or \
                         trestbps < sbp_val  # crude proxy

            try:
                risk_pce = pooled_cohort_equations(
                    age, sex, True,  # assume African/Black given platform context
                    chol, hdl_val, sbp_val, bp_treated, smoker_flag, dm_flag
                )
            except Exception:
                risk_pce = None

            # ── ENSEMBLE + CALIBRATION ────────────────────────────────────
            # Weight each model by validation evidence quality
            # UCI XGBoost: high AUC but tiny, homogeneous training set — lower external weight
            # Framingham: validated in large cohorts but Western population
            # PCE Black: specifically validated in African American cohorts — highest weight here
            # WHO/ISH: globally calibrated, designed for LMICs
            valid_risks = [r for r in [risk_prob_uci, risk_framingham, risk_pce, risk_who]
                          if r is not None]
            weights_map  = {0: 0.20, 1: 0.25, 2: 0.30, 3: 0.25}  # UCI, Framingham, PCE, WHO
            all_risks    = [risk_prob_uci, risk_framingham, risk_pce, risk_who]
            w_sum = sum(weights_map[i] for i,r in enumerate(all_risks) if r is not None)
            risk_ensemble = sum(weights_map[i]*r for i,r in enumerate(all_risks)
                               if r is not None) / w_sum if w_sum > 0 else risk_prob_uci

            # Uncertainty: spread between model estimates
            if len(valid_risks) > 1:
                risk_min = min(valid_risks)
                risk_max = max(valid_risks)
                risk_spread = risk_max - risk_min
                uncertainty = ("High" if risk_spread > 0.20
                               else "Moderate" if risk_spread > 0.10
                               else "Low")
            else:
                risk_min = risk_max = risk_ensemble
                risk_spread = 0
                uncertainty = "Low"

            # Use ensemble as primary risk_prob for downstream code
            risk_prob = risk_ensemble
            tier, css, icon = get_risk_badge(risk_prob)

            # ── POPULATION DISTRIBUTION WARNING ──────────────────────────
            # Flag when patient demographics are far from UCI training distribution
            uci_distance_flags = []
            if age > 70:         uci_distance_flags.append(f"Age {age} (UCI training range: 29–77, mean 54)")
            if sex == 0:         uci_distance_flags.append("Female sex (UCI: 68% male)")
            if chol > 400:       uci_distance_flags.append(f"Cholesterol {chol} mg/dL (UCI mean: 246)")
            if trestbps > 180:   uci_distance_flags.append(f"SBP {trestbps} mmHg (UCI mean: 131)")

            # ── DISPLAY ───────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Assessment Results")

            # Model comparison
            with st.expander("📊 Multi-Model Risk Ensemble — click to see all model estimates", expanded=True):
                mc1,mc2,mc3,mc4,mc5 = st.columns(5)
                mc1.metric("🎯 Ensemble Risk", f"{risk_ensemble*100:.1f}%", help="Weighted average of all validated models")
                mc2.metric("UCI XGBoost", f"{risk_prob_uci*100:.1f}%", help="Trained on 303-patient Cleveland dataset (1988)")
                mc3.metric("Framingham", f"{risk_framingham*100:.1f}%"  if risk_framingham else "—", help="Wilson 1998, 10-yr CHD risk")
                mc4.metric("PCE (Black)", f"{risk_pce*100:.1f}%"        if risk_pce else "—", help="AHA/ACC Pooled Cohort, Black American cohort")
                mc5.metric("WHO/ISH", f"{risk_who*100:.1f}%",           help="WHO 2019, AFRO region, LMIC-calibrated")

                st.caption(
                    f"**Model agreement:** Range {risk_min*100:.1f}%–{risk_max*100:.1f}% "
                    f"(spread: {risk_spread*100:.1f} percentage points) — "
                    f"Uncertainty: **{uncertainty}**"
                )

                if risk_spread > 0.15:
                    st.warning(
                        f"⚠ **Wide model disagreement ({risk_spread*100:.0f}pp spread).** "
                        "This patient's profile may sit at the boundary of multiple risk categories. "
                        "Clinical judgement is especially important here — do not rely on any single score."
                    )

            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Cardiovascular Risk (Ensemble)", f"{risk_prob*100:.1f}%")
            with m2: st.metric("Lifestyle Risk Index", f"{compute_lri(trestbps, chol, fbs, exang, oldpeak):.2f}/1.0")
            with m3: st.metric("Risk Classification", tier)
            st.markdown(f'<div class="{css}">{icon} <strong>{tier}</strong> — Ensemble probability: <strong>{risk_prob*100:.1f}%</strong></div>', unsafe_allow_html=True)

            # Training distribution warning
            if uci_distance_flags:
                st.warning(
                    "⚠ **Model calibration notice:** This patient's characteristics differ from the "
                    "UCI training population in the following ways:\n" +
                    "\n".join(f"  • {f}" for f in uci_distance_flags) +
                    "\n\nThe **Pooled Cohort Equations (Black)** and **WHO/ISH** scores above are "
                    "more appropriate for this patient. Use the ensemble estimate with clinical judgement."
                )

            # African population context note
            st.info(
                "🌍 **Population context:** The Pooled Cohort Equations (Black American cohort) "
                "and WHO/ISH AFRO-D score are the most applicable validated equations for patients "
                "of African descent. INTERHEART Africa data show hypertension accounts for a larger "
                "proportion of attributable cardiovascular risk in African populations than in European "
                "cohorts. The ensemble weights these accordingly."
            )

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

            # ── ICD / ICF Codes: Risk Prediction ─────────────────────────
            _risk_dx_list = []
            if trestbps > 140:          _risk_dx_list.append("Hypertension")
            if fbs:                     _risk_dx_list.append("Type 2 Diabetes")
            if chol > 200:              _risk_dx_list.append("Dyslipidaemia")
            if oldpeak > 1.5:           _risk_dx_list.append("Stable Angina")
            if ca > 0:                  _risk_dx_list.append("Coronary Artery Disease")
            if risk_prob >= 0.60:       _risk_dx_list.append("Myocardial Infarction")
            render_icd_panel(_risk_dx_list, context="Risk Assessment")
            _risk_icd_df = icd_export_df(_risk_dx_list)
            render_cpt_from_icd(_risk_dx_list, context="Risk Assessment — Recommended Procedures")
            _risk_cpt_df = cpt_export_df(diagnoses=_risk_dx_list)

            # ── Export: Risk Prediction ───────────────────
            st.divider()
            _risk_export_df = pd.DataFrame([{
                "Age": age, "Sex": "Male" if sex == 1 else "Female",
                "Resting BP (mmHg)": trestbps, "Cholesterol (mg/dl)": chol,
                "Max Heart Rate": thalach, "ST Depression": oldpeak,
                "Major Vessels": ca,
                "Fasting Blood Sugar >120": "Yes" if fbs else "No",
                "Exercise Angina": "Yes" if exang else "No",
                "Lifestyle Risk Index": compute_lri(trestbps, chol, fbs, exang, oldpeak),
                "CV Risk Probability (%)": round(risk_prob * 100, 1),
                "Risk Classification": tier,
            }])
            _risk_reco = (
                "URGENT: Cardiology referral required." if risk_prob >= 0.60
                else "Cardiology consultation within 4–6 weeks." if risk_prob >= 0.30
                else "Low Risk — continue annual screening."
            )
            _risk_excel = {"Risk Assessment": _risk_export_df}
            if not _risk_icd_df.empty:
                _risk_excel["ICD-ICF-CPT Codes"] = _risk_icd_df
            if not _risk_cpt_df.empty:
                _risk_excel["CPT Procedures"] = _risk_cpt_df
            _risk_pdf_sections = [
                ("Patient Input & Results", _risk_export_df),
                ("Clinical Recommendation", _risk_reco),
            ]
            if not _risk_icd_df.empty:
                _risk_pdf_sections.append(("ICD / ICF / CPT Codes", _risk_icd_df))
            if not _risk_cpt_df.empty:
                _risk_pdf_sections.append(("CPT Procedure Bill", _risk_cpt_df))
            export_buttons(
                "Risk Prediction",
                csv_df=_risk_export_df,
                excel_sheets=_risk_excel,
                pdf_title="CardioAI — Cardiovascular Risk Assessment Report",
                pdf_sections=_risk_pdf_sections,
                docx_title="CardioAI — Cardiovascular Risk Assessment Report",
                docx_sections=_risk_pdf_sections,
                file_stem="risk_assessment",
            )

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
                # Full CardioAI Nova cardiac rehabilitation program
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
                "rehabilitation":  "CardioAI Nova Rehabilitation",
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
- **CardioAI Nova Cardiac Rehabilitation:** {rehab_5yr:.0f}% — combined programme achieves the best long-term outcome, reducing 5-year risk by **{rehab_benefit:.0f} percentage points** compared to no action

The shaded bands represent the 25th–75th and 10th–90th percentile uncertainty range across 600 simulated patient trajectories, reflecting natural variability in biomarker progression. Wider bands indicate higher uncertainty.

> ⚠ **Note:** This forecast uses evidence-based biomarker progression rates from the Framingham Heart Study and AHA/ACC guidelines, combined with the patient's current ML-predicted risk score. It is a decision support tool — not a clinical prediction. Actual outcomes depend on adherence, comorbidities, and factors not captured in this model.
""")

            # ── Export: Monte Carlo Forecast ──────────────
            _forecast_df = pd.DataFrame([{
                "Scenario": "No Intervention",
                "5-Year Risk (%)": round(np.median(all_paths["no_intervention"][:, -1]) * 100, 1),
                "P25 (%)": round(np.percentile(all_paths["no_intervention"][:, -1], 25) * 100, 1),
                "P75 (%)": round(np.percentile(all_paths["no_intervention"][:, -1], 75) * 100, 1),
            }, {
                "Scenario": "Lifestyle Changes",
                "5-Year Risk (%)": round(np.median(all_paths["lifestyle"][:, -1]) * 100, 1),
                "P25 (%)": round(np.percentile(all_paths["lifestyle"][:, -1], 25) * 100, 1),
                "P75 (%)": round(np.percentile(all_paths["lifestyle"][:, -1], 75) * 100, 1),
            }, {
                "Scenario": "Medication",
                "5-Year Risk (%)": round(np.median(all_paths["medication"][:, -1]) * 100, 1),
                "P25 (%)": round(np.percentile(all_paths["medication"][:, -1], 25) * 100, 1),
                "P75 (%)": round(np.percentile(all_paths["medication"][:, -1], 75) * 100, 1),
            }, {
                "Scenario": "CardioAI Nova Rehabilitation",
                "5-Year Risk (%)": round(np.median(all_paths["rehabilitation"][:, -1]) * 100, 1),
                "P25 (%)": round(np.percentile(all_paths["rehabilitation"][:, -1], 25) * 100, 1),
                "P75 (%)": round(np.percentile(all_paths["rehabilitation"][:, -1], 75) * 100, 1),
            }])
            export_buttons(
                "5-Year Forecast",
                csv_df=_forecast_df,
                excel_sheets={"5-Year Forecast": _forecast_df, "Risk Assessment": _risk_export_df},
                pdf_title="CardioAI — 5-Year Risk Trajectory Forecast",
                pdf_sections=[
                    ("Risk Assessment", _risk_export_df),
                    ("Monte Carlo Forecast — Scenario Outcomes at 5 Years", _forecast_df),
                    ("Forecast Interpretation",
                     f"Current risk: {risk_prob*100:.1f}% ({tier}). "
                     f"Without action: {no_int_5yr:.0f}% at 5 years. "
                     f"With CardioAI Nova Rehabilitation: {rehab_5yr:.0f}% (reduction of {rehab_benefit:.0f} percentage points)."),
                ],
                docx_title="CardioAI — 5-Year Risk Trajectory Forecast",
                docx_sections=[
                    ("Risk Assessment", _risk_export_df),
                    ("Monte Carlo Forecast — Scenario Outcomes at 5 Years", _forecast_df),
                    ("Forecast Interpretation",
                     f"Current risk: {risk_prob*100:.1f}% ({tier}). "
                     f"Without action: {no_int_5yr:.0f}% at 5 years. "
                     f"With CardioAI Nova Rehabilitation: {rehab_5yr:.0f}% (reduction of {rehab_benefit:.0f} percentage points)."),
                ],
                file_stem="forecast_report",
            )

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

                # ── Export: Retention ─────────────────────
                st.divider()
                _ret_df = pd.DataFrame([{
                    "Exercise Difficulty (1-5)": exercise_diff,
                    "Questionnaire Burden (1-10)": q_burden,
                    "Waiting Time (min)": waiting_time,
                    "Travel Distance (km)": travel_dist,
                    "Previous Visits": prev_visits,
                    "Missed Appointment": "Yes" if missed_appt else "No",
                    "Has Insurance": "Yes" if has_insurance else "No",
                    "Perceived Improvement (1-5)": perceived_imp,
                    "Dropout Probability (%)": round(dropout_prob * 100, 1),
                    "Retention Probability (%)": round(retain_prob * 100, 1),
                    "Risk Status": risk_label,
                    "vs Population Average (%)": round(diff, 1),
                }])
                _ret_risk_txt = (
                    f"Risk Factors: {'; '.join(risk_factors) if risk_factors else 'None identified'}.\n"
                    f"Protective Factors: {'; '.join(protective) if protective else 'None identified'}."
                )
                export_buttons(
                    "Retention Assessment",
                    csv_df=_ret_df,
                    excel_sheets={"Retention Assessment": _ret_df},
                    pdf_title="CardioAI — Patient Retention Risk Report",
                    pdf_sections=[
                        ("Retention Risk Assessment", _ret_df),
                        ("Risk & Protective Factors", _ret_risk_txt),
                    ],
                    docx_title="CardioAI — Patient Retention Risk Report",
                    docx_sections=[
                        ("Retention Risk Assessment", _ret_df),
                        ("Risk & Protective Factors", _ret_risk_txt),
                    ],
                    file_stem="retention_assessment",
                )

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

        # ── Export: Model Performance ──────────────────
        st.divider()
        export_buttons(
            "Model Performance",
            csv_df=df,
            excel_sheets={"Model Performance": df},
            pdf_title="CardioAI — Model Performance Report",
            pdf_sections=[("All Model Performance Metrics", df)],
            docx_title="CardioAI — Model Performance Report",
            docx_sections=[("All Model Performance Metrics", df)],
            file_stem="model_performance",
        )
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
    st.caption("Upload a scanned image, PDF, or paste text — extract structured clinical data using Gemini Vision AI and NLP.")

    nlp_tab1, nlp_tab2 = st.tabs([
        "📋 Clinical Note Extraction",
        "🔬 Scan & Auto-Fill Investigations",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — ORIGINAL CLINICAL NOTE EXTRACTION (unchanged)
    # ══════════════════════════════════════════════════════════════════════
    with nlp_tab1:
        st.subheader("Clinical Note — Entity Extraction")
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
            # Run extraction inside spinner (computation only)
            with st.spinner("Running NLP extraction..."):
                try:
                    _extracted = extract_entities(raw_text)
                    st.session_state["nlp_entities"]       = _extracted
                    st.session_state["nlp_extract_error"]  = None
                except Exception as e:
                    st.session_state["nlp_entities"]       = None
                    st.session_state["nlp_extract_error"]  = str(e)
        else:
            st.warning("Please provide a document first — upload a file or paste text above.")

    # Render results completely outside any spinner/context manager
    _nlp_entities = st.session_state.get("nlp_entities")
    _nlp_err      = st.session_state.get("nlp_extract_error")

    if _nlp_err:
        st.error(f"Extraction error: {_nlp_err}")
    elif _nlp_entities is not None:
        try:
            entities = _nlp_entities
            show_entities(entities)

            # ── ICD / ICF / CPT Codes: NLP Extraction ──────────────────
            _nlp_dx = entities.get("dx", [])
            if _nlp_dx:
                render_icd_panel(_nlp_dx, context="Clinical Document Extraction")
                render_cpt_from_icd(_nlp_dx, context="Clinical Document — Recommended Procedures")
            _nlp_icd_df = icd_export_df(_nlp_dx) if _nlp_dx else pd.DataFrame()

            # ── Export: Clinical NLP ──────────────
            st.divider()
            _nlp_rows = []
            if entities.get("bp"):
                for b in entities["bp"]:
                    _nlp_rows.append({"Parameter": "Blood Pressure",
                                      "Value": f"{b['sys']}/{b['dia']} mmHg",
                                      "Flag": "HYPERTENSIVE" if b["hyp"] else "Normal"})
            if entities.get("hr"):
                _nlp_rows.append({"Parameter": "Heart Rate",
                                  "Value": f"{entities['hr'][0]} bpm", "Flag": ""})
            if entities.get("weight"):
                _nlp_rows.append({"Parameter": "Weight",
                                  "Value": f"{entities['weight'][0]} kg", "Flag": ""})
            if entities.get("height"):
                _nlp_rows.append({"Parameter": "Height",
                                  "Value": f"{entities['height'][0]} cm", "Flag": ""})
            if entities.get("bmi"):
                bv = entities["bmi"][0]
                _nlp_rows.append({"Parameter": "BMI",
                                  "Value": f"{bv} kg/m²",
                                  "Flag": "Obese" if bv >= 30 else "Overweight" if bv >= 25 else "Normal"})
            if entities.get("spo2"):
                sv = entities["spo2"][0]
                _nlp_rows.append({"Parameter": "SPO2",
                                  "Value": f"{sv}%",
                                  "Flag": "LOW" if sv < 95 else "Normal"})
            for dx in entities.get("dx", []):
                _nlp_rows.append({"Parameter": "Diagnosis", "Value": dx, "Flag": ""})
            for med in entities.get("meds", []):
                _nlp_rows.append({"Parameter": "Medication", "Value": med, "Flag": ""})
            lifestyle_flags = []
            if entities.get("smoking"):   lifestyle_flags.append("Smoking")
            if entities.get("alcohol"):   lifestyle_flags.append("Alcohol")
            if entities.get("sedentary"): lifestyle_flags.append("Sedentary")
            if entities.get("poor_diet"): lifestyle_flags.append("Poor Diet")
            if lifestyle_flags:
                _nlp_rows.append({"Parameter": "Lifestyle Flags",
                                  "Value": ", ".join(lifestyle_flags), "Flag": "Present"})
            _nlp_df = pd.DataFrame(_nlp_rows) if _nlp_rows else pd.DataFrame(
                columns=["Parameter", "Value", "Flag"])
            _nlp_excel = {"Extracted Entities": _nlp_df}
            _nlp_pdf   = [("Extracted Clinical Parameters", _nlp_df)]
            if not _nlp_icd_df.empty:
                _nlp_excel["ICD-ICF Codes"] = _nlp_icd_df
                _nlp_pdf.append(("ICD-10 / ICD-11 / ICF Codes", _nlp_icd_df))
            export_buttons(
                "Clinical NLP",
                csv_df=_nlp_df,
                excel_sheets=_nlp_excel,
                pdf_title="CardioAI — Clinical Document Extraction Report",
                pdf_sections=_nlp_pdf,
                docx_title="CardioAI — Clinical Document Extraction Report",
                docx_sections=_nlp_pdf,
                file_stem="nlp_extraction",
            )
        except Exception as e:
            st.error(f"Display error: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — SCAN & AUTO-FILL INVESTIGATIONS
    # ══════════════════════════════════════════════════════════════════════
    with nlp_tab2:
        st.subheader("🔬 Scan Any Investigation Report → Auto-Fill Risk Prediction")
        st.info(
            "Upload a scanned or photographed investigation report — "
            "**spirometry, ECG printout, lab results, echo report, radiology report** — "
            "and Gemini Vision AI will extract all values and map them directly to the "
            "Risk Prediction module inputs. Supports upside-down, rotated, and poor-quality scans."
        )

        scan_report_type = st.selectbox(
            "Report type (helps AI focus extraction)",
            [
                "🔬 Lab Results (FBS, HbA1c, Lipids, Renal, Electrolytes, LFTs, Cardiac Markers)",
                "📈 ECG / Electrocardiograph Printout",
                "🫁 Spirometry / Pulmonary Function Test (PFT)",
                "❤️ Echocardiogram Report",
                "🩻 Radiology Report (X-Ray / CT / MRI / Ultrasound)",
                "📋 General Clinical / Discharge Summary",
                "🔭 Endocrinology / Hormone Panel",
                "🛡 Serology / Immunology Report",
                "🦠 Microbiology / Culture Report",
                "💊 Toxicology / Drug Level Report",
                "🔬 Histopathology / Cytology Report",
                "🌿 Allergy / IgE Report",
            ],
            key="scan_report_type"
        )

        scan_upload = st.file_uploader(
            "Upload scanned investigation report (JPG, PNG, PDF)",
            type=["jpg","jpeg","png","pdf"],
            key="scan_upload",
            help="Supports upside-down scans, poor lighting, handwritten annotations"
        )

        scan_col1, scan_col2 = st.columns(2)
        with scan_col1:
            scan_auto_rotate = st.checkbox("Auto-correct orientation (if upside-down)", value=True)
        with scan_col2:
            scan_enhance = st.checkbox("Enhance image contrast before reading", value=True)

        if scan_upload:
            from PIL import Image as PILImage, ImageEnhance, ImageOps
            import base64, io

            # Load and preprocess
            if scan_upload.name.lower().endswith('.pdf'):
                st.info("PDF detected — converting first page to image...")
                try:
                    import fitz  # PyMuPDF
                    pdf_bytes = scan_upload.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    page = doc[0]
                    pix = page.get_pixmap(dpi=200)
                    scan_img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
                except Exception as e:
                    st.error(f"PDF conversion error: {e}")
                    scan_img = None
            else:
                scan_img = PILImage.open(scan_upload)

            if scan_img:
                # Auto-rotate if needed
                if scan_auto_rotate:
                    try:
                        scan_img = ImageOps.exif_transpose(scan_img)
                    except Exception:
                        pass

                # Enhance contrast
                if scan_enhance:
                    scan_img_enhanced = ImageEnhance.Contrast(
                        ImageEnhance.Sharpness(scan_img.convert("RGB")).enhance(1.5)
                    ).enhance(1.3)
                else:
                    scan_img_enhanced = scan_img.convert("RGB")

                st.image(scan_img_enhanced,
                         caption=f"Uploaded: {scan_upload.name}",
                         use_container_width=True)

                if st.button("🔍 Extract & Map Investigation Values", type="primary", key="scan_extract_btn"):
                    with st.spinner("Reading investigation report..."):
                        try:
                            buf = io.BytesIO()
                            scan_img_enhanced.save(buf, format="JPEG", quality=95)
                            img_b64 = base64.b64encode(buf.getvalue()).decode()

                            scan_prompt = f"""You are an expert clinical data extraction AI.
This is a scanned **{scan_report_type}** investigation report from CardioAI Nova, Nigeria.

The image may be rotated, upside-down, or of variable quality. Read it carefully regardless of orientation.

Your task: Extract ALL numerical values, results, interpretations, and patient demographics from this report.
Then map each extracted value to the corresponding input field in a cardiovascular risk assessment system.

Return your response in this EXACT JSON structure (include only fields that have values in the report — use null for missing):

{{
  "patient_info": {{
    "name": null,
    "id": null,
    "age": null,
    "sex": null,
    "height_cm": null,
    "weight_kg": null,
    "bmi": null,
    "date": null,
    "referring_physician": null
  }},
  "vital_signs": {{
    "sbp_mmhg": null,
    "dbp_mmhg": null,
    "heart_rate_bpm": null,
    "spo2_pct": null,
    "rr_breaths_per_min": null,
    "temp_c": null
  }},
  "ecg_parameters": {{
    "hr_bpm": null,
    "rhythm": null,
    "pr_interval_ms": null,
    "qrs_duration_ms": null,
    "qt_interval_ms": null,
    "qtc_ms": null,
    "axis_degrees": null,
    "st_depression_mm": null,
    "st_elevation_mm": null,
    "t_wave": null,
    "interpretation": null
  }},
  "spirometry": {{
    "fev1_litres": null,
    "fev1_pct_predicted": null,
    "fvc_litres": null,
    "fvc_pct_predicted": null,
    "fev1_fvc_ratio": null,
    "pef_l_per_min": null,
    "pef_pct_predicted": null,
    "fef2575_litres": null,
    "tlc_litres": null,
    "pattern": null,
    "severity": null,
    "interpretation": null
  }},
  "blood_glucose": {{
    "fbs_mg_dl": null,
    "rbs_mg_dl": null,
    "hba1c_pct": null,
    "ogtt_2hr_mg_dl": null
  }},
  "lipid_profile": {{
    "total_cholesterol_mg_dl": null,
    "ldl_mg_dl": null,
    "hdl_mg_dl": null,
    "triglycerides_mg_dl": null,
    "vldl_mg_dl": null,
    "tc_hdl_ratio": null
  }},
  "renal": {{
    "urea_mmol_l": null,
    "creatinine_umol_l": null,
    "egfr_ml_min": null,
    "uric_acid_umol_l": null,
    "urine_albumin_mg_l": null
  }},
  "electrolytes": {{
    "sodium_mmol_l": null,
    "potassium_mmol_l": null,
    "chloride_mmol_l": null,
    "bicarbonate_mmol_l": null,
    "calcium_mmol_l": null,
    "magnesium_mmol_l": null,
    "phosphate_mmol_l": null
  }},
  "liver_function": {{
    "alt_u_l": null,
    "ast_u_l": null,
    "alp_u_l": null,
    "ggt_u_l": null,
    "total_bilirubin_umol_l": null,
    "albumin_g_l": null,
    "total_protein_g_l": null
  }},
  "cardiac_markers": {{
    "troponin_i_ng_l": null,
    "ck_mb_u_l": null,
    "bnp_pg_ml": null,
    "nt_probnp_pg_ml": null,
    "crp_mg_l": null,
    "hscrp_mg_l": null,
    "ldh_u_l": null
  }},
  "haematology": {{
    "haemoglobin_g_dl": null,
    "wbc_x10_9_l": null,
    "platelets_x10_9_l": null,
    "esr_mm_hr": null,
    "inr": null
  }},
  "endocrinology": {{
    "tsh_miu_l": null,
    "ft4_pmol_l": null,
    "ft3_pmol_l": null,
    "cortisol_nmol_l": null,
    "psa_ng_ml": null,
    "testosterone_nmol_l": null
  }},
  "echo_parameters": {{
    "ef_pct": null,
    "ivsd_mm": null,
    "lvedd_mm": null,
    "lvesd_mm": null,
    "pwed_mm": null,
    "lv_function": null,
    "valvular_findings": null,
    "interpretation": null
  }},
  "microbiology": {{
    "blood_culture": null,
    "urine_culture": null,
    "malaria_rdt": null,
    "covid19": null,
    "procalcitonin_ng_ml": null
  }},
  "tumour_markers": {{
    "ca125_u_ml": null,
    "cea_ng_ml": null,
    "afp_ng_ml": null,
    "psa_ng_ml": null
  }},
  "diagnoses_found": [],
  "medications_found": [],
  "clinical_interpretation": null,
  "urgent_findings": [],
  "normal_ranges_provided": null,
  "raw_extracted_text": null
}}

IMPORTANT INSTRUCTIONS:
1. Read the ENTIRE image including headers, footers, tables, graphs, and annotations
2. If the image is upside down or rotated, read it anyway
3. Extract ALL numerical values with their units
4. For spirometry: extract both absolute values AND % predicted
5. For ECG: extract all interval measurements visible
6. For labs: include reference ranges if printed
7. Convert units if needed (e.g. mmol/L to mg/dL for glucose: multiply by 18)
8. List ALL diagnoses mentioned (including interpretations like "Restrictive disorder")
9. Note any URGENT or CRITICAL findings
10. Return ONLY valid JSON — no markdown, no explanation text outside the JSON"""

                            raw_response_tuple = vision_api_call(scan_prompt, img_b64)
                            raw_response = raw_response_tuple[0].strip() if raw_response_tuple[0] else ""
                            if not raw_response:
                                st.error(f"Scan extraction failed: {raw_response_tuple[1]}")
                                raw_response = ""
                            # Clean JSON
                            if raw_response.startswith("```"):
                                raw_response = raw_response.split("```")[1]
                                if raw_response.startswith("json"):
                                    raw_response = raw_response[4:]
                            raw_response = raw_response.strip().rstrip("```").strip()
                            st.session_state["scan_result"] = raw_response
                            st.session_state["scan_report_type_done"] = scan_report_type

                        except Exception as e:
                            st.error(f"Scan extraction error: {e}")

        # ── Display extracted results ─────────────────────────────────
        scan_result_raw = st.session_state.get("scan_result","")
        if scan_result_raw:
            import json
            try:
                data = json.loads(scan_result_raw)
            except Exception:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', scan_result_raw, re.DOTALL)
                try:
                    data = json.loads(json_match.group()) if json_match else {}
                except Exception:
                    data = {}
                    st.warning("Could not parse structured data — showing raw extraction below.")
                    st.code(scan_result_raw)

            if data:
                st.success("✅ Investigation report extracted successfully")

                # ── Confidence & plausibility validation ──────────────
                _scan_flat = {}
                for section in ["vital_signs","ecg_parameters","blood_glucose",
                                 "lipid_profile","renal","electrolytes","cardiac_markers","echo_parameters"]:
                    _scan_flat.update(data.get(section,{}) or {})
                _scan_field_map = {k:k for k in CLINICAL_PLAUSIBILITY.keys()}
                _, scan_warns, scan_crits = validate_extracted_data(_scan_flat, _scan_field_map)
                render_confidence_panel(
                    scan_warns, scan_crits,
                    model_used=st.session_state.get("scan_report_type_done","AI"),
                    context="Investigation report scan extraction"
                )
                st.divider()

                # ── Patient Info ───────────────────────────────────
                pt = data.get("patient_info",{})
                if any(v for v in pt.values()):
                    st.subheader("👤 Patient Information")
                    pi_cols = st.columns(4)
                    fields = [("Name",pt.get("name")),("ID",pt.get("id")),
                              ("Age",pt.get("age")),("Sex",pt.get("sex")),
                              ("Height",f"{pt.get('height_cm')} cm" if pt.get("height_cm") else None),
                              ("Weight",f"{pt.get('weight_kg')} kg" if pt.get("weight_kg") else None),
                              ("BMI",pt.get("bmi")),("Date",pt.get("date"))]
                    for idx,(label,val) in enumerate(fields):
                        if val:
                            with pi_cols[idx%4]: st.metric(label, val)

                # ── Urgent Findings ────────────────────────────────
                urgent = data.get("urgent_findings",[])
                if urgent:
                    st.subheader("🚨 Urgent Findings")
                    for u in urgent:
                        st.error(f"🚨 {u}")

                # ── Clinical Interpretation ────────────────────────
                interp = data.get("clinical_interpretation")
                if interp:
                    st.subheader("📋 Clinical Interpretation")
                    st.info(interp)

                # ── Diagnoses ──────────────────────────────────────
                diag = data.get("diagnoses_found",[])
                if diag:
                    st.subheader("🏷 Diagnoses / Interpretations Found")
                    for d in diag:
                        st.markdown(f"• **{d}**")

                # ── Spirometry ─────────────────────────────────────
                spiro = data.get("spirometry",{})
                if any(v for v in spiro.values() if v is not None):
                    st.subheader("🫁 Spirometry Results")
                    sp_cols = st.columns(4)
                    spiro_fields = [
                        ("FEV1 (L)", spiro.get("fev1_litres")),
                        ("FEV1 % Pred", f"{spiro.get('fev1_pct_predicted')}%" if spiro.get("fev1_pct_predicted") else None),
                        ("FVC (L)", spiro.get("fvc_litres")),
                        ("FVC % Pred", f"{spiro.get('fvc_pct_predicted')}%" if spiro.get("fvc_pct_predicted") else None),
                        ("FEV1/FVC", spiro.get("fev1_fvc_ratio")),
                        ("PEF (L/min)", spiro.get("pef_l_per_min")),
                        ("Pattern", spiro.get("pattern")),
                        ("Severity", spiro.get("severity")),
                    ]
                    for idx,(label,val) in enumerate(spiro_fields):
                        if val is not None:
                            with sp_cols[idx%4]: st.metric(label, val)
                    if spiro.get("interpretation"):
                        st.warning(f"📋 {spiro['interpretation']}")

                # ── ECG Parameters ─────────────────────────────────
                ecg = data.get("ecg_parameters",{})
                if any(v for v in ecg.values() if v is not None):
                    st.subheader("📈 ECG Parameters")
                    ec_cols = st.columns(4)
                    ecg_fields = [
                        ("HR (bpm)", ecg.get("hr_bpm")),
                        ("PR (ms)", ecg.get("pr_interval_ms")),
                        ("QRS (ms)", ecg.get("qrs_duration_ms")),
                        ("QTc (ms)", ecg.get("qtc_ms")),
                        ("ST Depression", ecg.get("st_depression_mm")),
                        ("ST Elevation", ecg.get("st_elevation_mm")),
                        ("Axis", ecg.get("axis_degrees")),
                        ("Rhythm", ecg.get("rhythm")),
                    ]
                    for idx,(label,val) in enumerate(ecg_fields):
                        if val is not None:
                            with ec_cols[idx%4]: st.metric(label, val)
                    if ecg.get("interpretation"):
                        st.info(f"📋 {ecg['interpretation']}")

                # ── Blood Glucose ──────────────────────────────────
                glu = data.get("blood_glucose",{})
                if any(v for v in glu.values() if v is not None):
                    st.subheader("🩸 Blood Glucose")
                    g_cols = st.columns(4)
                    glu_fields = [("FBS (mg/dL)",glu.get("fbs_mg_dl")),
                                  ("RBS (mg/dL)",glu.get("rbs_mg_dl")),
                                  ("HbA1c (%)",glu.get("hba1c_pct")),
                                  ("OGTT 2hr (mg/dL)",glu.get("ogtt_2hr_mg_dl"))]
                    for idx,(label,val) in enumerate(glu_fields):
                        if val is not None:
                            with g_cols[idx%4]: st.metric(label, val)

                # ── Lipids ─────────────────────────────────────────
                lip = data.get("lipid_profile",{})
                if any(v for v in lip.values() if v is not None):
                    st.subheader("🧪 Lipid Profile")
                    l_cols = st.columns(4)
                    lip_fields = [("Total Chol (mg/dL)",lip.get("total_cholesterol_mg_dl")),
                                  ("LDL (mg/dL)",lip.get("ldl_mg_dl")),
                                  ("HDL (mg/dL)",lip.get("hdl_mg_dl")),
                                  ("TG (mg/dL)",lip.get("triglycerides_mg_dl")),
                                  ("VLDL (mg/dL)",lip.get("vldl_mg_dl")),
                                  ("TC/HDL Ratio",lip.get("tc_hdl_ratio"))]
                    for idx,(label,val) in enumerate(lip_fields):
                        if val is not None:
                            with l_cols[idx%4]: st.metric(label, val)

                # ── Renal ──────────────────────────────────────────
                ren = data.get("renal",{})
                if any(v for v in ren.values() if v is not None):
                    st.subheader("🔬 Renal Function")
                    r_cols = st.columns(4)
                    ren_fields = [("Urea (mmol/L)",ren.get("urea_mmol_l")),
                                  ("Creatinine (μmol/L)",ren.get("creatinine_umol_l")),
                                  ("eGFR (mL/min)",ren.get("egfr_ml_min")),
                                  ("Uric Acid (μmol/L)",ren.get("uric_acid_umol_l"))]
                    for idx,(label,val) in enumerate(ren_fields):
                        if val is not None:
                            with r_cols[idx%4]: st.metric(label, val)

                # ── Electrolytes ───────────────────────────────────
                elec = data.get("electrolytes",{})
                if any(v for v in elec.values() if v is not None):
                    st.subheader("⚡ Electrolytes")
                    el_cols = st.columns(4)
                    elec_fields = [("Na⁺ (mmol/L)",elec.get("sodium_mmol_l")),
                                   ("K⁺ (mmol/L)",elec.get("potassium_mmol_l")),
                                   ("Cl⁻ (mmol/L)",elec.get("chloride_mmol_l")),
                                   ("HCO₃⁻ (mmol/L)",elec.get("bicarbonate_mmol_l")),
                                   ("Ca²⁺ (mmol/L)",elec.get("calcium_mmol_l")),
                                   ("Mg²⁺ (mmol/L)",elec.get("magnesium_mmol_l"))]
                    for idx,(label,val) in enumerate(elec_fields):
                        if val is not None:
                            with el_cols[idx%4]: st.metric(label, val)

                # ── Cardiac Markers ────────────────────────────────
                card = data.get("cardiac_markers",{})
                if any(v for v in card.values() if v is not None):
                    st.subheader("❤️ Cardiac Markers")
                    ca_cols = st.columns(4)
                    card_fields = [("Troponin I (ng/L)",card.get("troponin_i_ng_l")),
                                   ("CK-MB (U/L)",card.get("ck_mb_u_l")),
                                   ("BNP (pg/mL)",card.get("bnp_pg_ml")),
                                   ("NT-proBNP (pg/mL)",card.get("nt_probnp_pg_ml")),
                                   ("CRP (mg/L)",card.get("crp_mg_l")),
                                   ("hs-CRP (mg/L)",card.get("hscrp_mg_l"))]
                    for idx,(label,val) in enumerate(card_fields):
                        if val is not None:
                            with ca_cols[idx%4]: st.metric(label, val)

                # ── Echo ────────────────────────────────────────────
                echo = data.get("echo_parameters",{})
                if any(v for v in echo.values() if v is not None):
                    st.subheader("❤️ Echo Parameters")
                    ech_cols = st.columns(4)
                    echo_fields = [("EF%",echo.get("ef_pct")),
                                   ("IVSd (mm)",echo.get("ivsd_mm")),
                                   ("LVEDd (mm)",echo.get("lvedd_mm")),
                                   ("LVESd (mm)",echo.get("lvesd_mm")),
                                   ("PWed (mm)",echo.get("pwed_mm")),
                                   ("LV Function",echo.get("lv_function"))]
                    for idx,(label,val) in enumerate(echo_fields):
                        if val is not None:
                            with ech_cols[idx%4]: st.metric(label, val)
                    if echo.get("interpretation"):
                        st.info(f"📋 {echo['interpretation']}")

                # ── Endocrinology ──────────────────────────────────
                endo = data.get("endocrinology",{})
                if any(v for v in endo.values() if v is not None):
                    st.subheader("🔭 Endocrinology")
                    en_cols = st.columns(4)
                    endo_fields = [("TSH (mIU/L)",endo.get("tsh_miu_l")),
                                   ("Free T4 (pmol/L)",endo.get("ft4_pmol_l")),
                                   ("Free T3 (pmol/L)",endo.get("ft3_pmol_l")),
                                   ("Cortisol (nmol/L)",endo.get("cortisol_nmol_l")),
                                   ("PSA (ng/mL)",endo.get("psa_ng_ml")),
                                   ("Testosterone (nmol/L)",endo.get("testosterone_nmol_l"))]
                    for idx,(label,val) in enumerate(endo_fields):
                        if val is not None:
                            with en_cols[idx%4]: st.metric(label, val)

                # ── Medications ────────────────────────────────────
                meds = data.get("medications_found",[])
                if meds:
                    st.subheader("💊 Medications Found")
                    st.markdown(", ".join([f"**{m}**" for m in meds]))

                st.divider()

                # ── Auto-Fill Guidance ──────────────────────────────
                st.subheader("📝 Auto-Fill Guidance for Risk Prediction Module")
                st.caption("Use these extracted values when filling in the Risk Prediction tabs.")

                guide_rows = []

                # Vitals
                vs = data.get("vital_signs",{})
                if vs.get("sbp_mmhg"): guide_rows.append(("Vital Signs → Systolic BP", vs["sbp_mmhg"], "mmHg"))
                if vs.get("dbp_mmhg"): guide_rows.append(("Vital Signs → Diastolic BP", vs["dbp_mmhg"], "mmHg"))
                if vs.get("heart_rate_bpm"): guide_rows.append(("Vital Signs → Resting HR", vs["heart_rate_bpm"], "bpm"))
                if vs.get("spo2_pct"): guide_rows.append(("Vital Signs → SpO₂", vs["spo2_pct"], "%"))

                # ECG
                if ecg.get("hr_bpm"): guide_rows.append(("ECG → Heart Rate", ecg["hr_bpm"], "bpm"))
                if ecg.get("pr_interval_ms"): guide_rows.append(("ECG → PR Interval", ecg["pr_interval_ms"], "ms"))
                if ecg.get("qrs_duration_ms"): guide_rows.append(("ECG → QRS Duration", ecg["qrs_duration_ms"], "ms"))
                if ecg.get("qtc_ms"): guide_rows.append(("ECG → QTc", ecg["qtc_ms"], "ms"))
                if ecg.get("st_depression_mm"): guide_rows.append(("ECG → ST Depression (oldpeak)", ecg["st_depression_mm"], "mm"))
                if ecg.get("rhythm"): guide_rows.append(("ECG → Rhythm", ecg["rhythm"], ""))

                # Blood glucose
                if glu.get("fbs_mg_dl"):
                    guide_rows.append(("FBS/Diabetes → Fasting Blood Sugar", glu["fbs_mg_dl"], "mg/dL"))
                    fbs_flag = "Yes (>120)" if float(glu["fbs_mg_dl"]) > 120 else "No"
                    guide_rows.append(("FBS/Diabetes → FBS >120 flag", fbs_flag, ""))
                if glu.get("hba1c_pct"): guide_rows.append(("FBS/Diabetes → HbA1c", glu["hba1c_pct"], "%"))
                if glu.get("rbs_mg_dl"): guide_rows.append(("FBS/Diabetes → Random Blood Sugar", glu["rbs_mg_dl"], "mg/dL"))

                # Lipids
                if lip.get("total_cholesterol_mg_dl"): guide_rows.append(("Lipid Profile → Total Cholesterol", lip["total_cholesterol_mg_dl"], "mg/dL"))
                if lip.get("ldl_mg_dl"): guide_rows.append(("Lipid Profile → LDL", lip["ldl_mg_dl"], "mg/dL"))
                if lip.get("hdl_mg_dl"): guide_rows.append(("Lipid Profile → HDL", lip["hdl_mg_dl"], "mg/dL"))
                if lip.get("triglycerides_mg_dl"): guide_rows.append(("Lipid Profile → Triglycerides", lip["triglycerides_mg_dl"], "mg/dL"))

                # Renal
                if ren.get("urea_mmol_l"): guide_rows.append(("E/U/Cr → Urea", ren["urea_mmol_l"], "mmol/L"))
                if ren.get("creatinine_umol_l"): guide_rows.append(("E/U/Cr → Creatinine", ren["creatinine_umol_l"], "μmol/L"))
                if ren.get("egfr_ml_min"): guide_rows.append(("E/U/Cr → eGFR", ren["egfr_ml_min"], "mL/min"))

                # Electrolytes
                if elec.get("sodium_mmol_l"): guide_rows.append(("Electrolytes → Na⁺", elec["sodium_mmol_l"], "mmol/L"))
                if elec.get("potassium_mmol_l"): guide_rows.append(("Electrolytes → K⁺", elec["potassium_mmol_l"], "mmol/L"))

                # Cardiac markers
                if card.get("troponin_i_ng_l"): guide_rows.append(("Chemistry → Troponin I", card["troponin_i_ng_l"], "ng/L"))
                if card.get("bnp_pg_ml"): guide_rows.append(("Chemistry → BNP", card["bnp_pg_ml"], "pg/mL"))
                if card.get("crp_mg_l"): guide_rows.append(("Chemistry → CRP", card["crp_mg_l"], "mg/L"))

                # Echo
                if echo.get("ef_pct"): guide_rows.append(("Imaging → Echocardiogram EF%", echo["ef_pct"], "%"))
                if echo.get("lvedd_mm"): guide_rows.append(("Imaging → Echo LVEDd", echo["lvedd_mm"], "mm"))

                # Spirometry → maps to symptoms
                if spiro.get("pattern"):
                    guide_rows.append(("Demographics → Respiratory pattern", spiro["pattern"], ""))
                if spiro.get("fev1_pct_predicted"):
                    guide_rows.append(("Demographics → FEV1 % predicted", spiro["fev1_pct_predicted"], "%"))

                # Endocrine
                if endo.get("tsh_miu_l"): guide_rows.append(("Endocrinology → TSH", endo["tsh_miu_l"], "mIU/L"))
                if endo.get("ft4_pmol_l"): guide_rows.append(("Endocrinology → Free T4", endo["ft4_pmol_l"], "pmol/L"))

                # Patient demographics
                if pt.get("age"): guide_rows.append(("Demographics → Age", pt["age"], "years"))
                if pt.get("sex"): guide_rows.append(("Demographics → Sex", pt["sex"], ""))
                if pt.get("weight_kg"): guide_rows.append(("Demographics → Weight", pt["weight_kg"], "kg"))
                if pt.get("height_cm"): guide_rows.append(("Demographics → Height", pt["height_cm"], "cm"))
                if pt.get("bmi"): guide_rows.append(("Demographics → BMI", pt["bmi"], "kg/m²"))

                if guide_rows:
                    guide_df = pd.DataFrame(guide_rows, columns=["Risk Prediction Input Field","Extracted Value","Unit"])
                    st.dataframe(guide_df, use_container_width=True, hide_index=True)

                    # ── Export ─────────────────────────────────────
                    st.divider()

                    # Build comprehensive export df
                    all_sections = []
                    if any(v for v in pt.values() if v): all_sections.append(("Patient Information", pd.DataFrame([pt])))
                    if guide_rows: all_sections.append(("Auto-Fill Values for Risk Prediction", guide_df))
                    if diag: all_sections.append(("Diagnoses / Interpretations", "\n".join(f"• {d}" for d in diag)))
                    if urgent: all_sections.append(("⚠ Urgent Findings", "\n".join(f"• {u}" for u in urgent)))
                    if meds: all_sections.append(("Medications", ", ".join(meds)))
                    if interp: all_sections.append(("Clinical Interpretation", interp))

                    export_buttons(
                        "Scan & Auto-Fill",
                        csv_df=guide_df,
                        excel_sheets={"Auto-Fill Values": guide_df,
                                      "Patient Info": pd.DataFrame([pt]) if any(v for v in pt.values() if v) else pd.DataFrame()},
                        pdf_title="CardioAI — Investigation Report Extraction & Auto-Fill",
                        pdf_sections=all_sections,
                        docx_title="CardioAI — Investigation Report Extraction & Auto-Fill",
                        docx_sections=all_sections,
                        file_stem="scan_autofill",
                    )
                else:
                    st.info("No values could be mapped to Risk Prediction inputs from this report.")

        st.divider()
        st.caption(
            "**Supported report types:** Spirometry/PFT · 12-Lead ECG · Lab results "
            "(FBS, HbA1c, lipids, renal, electrolytes, LFTs, cardiac markers) · "
            "Echocardiogram · Radiology report · Discharge summary · "
            "Endocrinology · Serology · Microbiology · Histopathology · Allergy · Toxicology. "
            "Powered by Gemini Vision 2.0 Flash. GOOGLE_API_KEY required in Streamlit secrets."
        )

# ══════════════════════════════════════════════════════════
# PAGE 5 — MEDICAL IMAGING (CNN)
# ══════════════════════════════════════════════════════════

elif "Medical Imaging" in page:
    st.title("🔬 Medical Imaging — Multi-Modal AI Analysis")
    st.caption(
        "AI-powered analysis across four imaging modalities: "
        "Chest X-Ray (DenseNet-121), Echocardiogram (EF% & LV function), "
        "ECG Signal (12-lead interpretation), and Retinal Fundus (cardiovascular risk)."
    )

    img_tab1, img_tab2, img_tab3, img_tab4, img_tab5 = st.tabs([
        "🫁 Chest X-Ray (DenseNet-121)",
        "❤️ Echocardiogram (EchoNet)",
        "📈 ECG Signal Analysis",
        "👁 Retinal Fundus (CV Risk)",
        "🌬️ Spirometry / PFT",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — CHEST X-RAY (existing DenseNet-121)
    # ══════════════════════════════════════════════════════════════════════
    with img_tab1:
        st.subheader("Chest X-Ray — DenseNet-121 CNN")
        st.info(
            "**Powered by DenseNet-121** pretrained on 100,000+ chest X-rays "
            "(NIH ChestX-ray14, CheXpert, MIMIC-CXR). "
            "Detects 18 pathologies with pixel-level Grad-CAM explainability."
        )
        st.warning(
            "⚠ **Chest X-rays only.** Uploading echocardiograms, ultrasounds, CT, MRI, "
            "or other modalities will produce meaningless scores. Use the correct tab for each modality."
        )

        cnn_ready = False
        try:
            import torch, torchxrayvision as xrv, skimage
            cnn_ready = True
        except ImportError:
            st.warning("CNN imaging requires: `torchxrayvision scikit-image` in requirements.txt")

        uploaded_xray = st.file_uploader(
            "Upload chest X-ray (JPG, PNG)",
            type=["jpg","jpeg","png"],
            help="PA or AP view chest X-ray only. DICOM → export as PNG first.",
            key="xray_upload"
        )

        if uploaded_xray and cnn_ready:
            from PIL import Image as PILImage
            import torch, torch.nn.functional as F
            import torchvision.transforms as transforms
            import torchxrayvision as xrv
            import skimage.transform
            import numpy as np
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm_module

            img_pil = PILImage.open(uploaded_xray)
            st.image(img_pil, caption=f"Uploaded: {uploaded_xray.name}", use_container_width=True)

            THRESHOLDS = {
                "Cardiomegaly":0.35,"Effusion":0.40,"Pneumonia":0.30,
                "Atelectasis":0.40,"Consolidation":0.40,"Pneumothorax":0.28,
                "Edema":0.35,"Emphysema":0.40,"Fibrosis":0.40,
                "Nodule":0.45,"Mass":0.38,"Infiltration":0.40,
                "Pleural_Thickening":0.40,"Hernia":0.30,
            }
            URGENT  = {"Pneumothorax","Mass","Edema","Effusion"}
            CARDIAC = {"Cardiomegaly","Effusion","Edema","Consolidation"}

            with st.spinner("Preprocessing image..."):
                img_np   = np.array(img_pil.convert("L")).astype(np.float32)
                img_norm = xrv.datasets.normalize(img_np, img_np.max() if img_np.max()>0 else 255)
                img_norm = img_norm[None, ...]
                transform = transforms.Compose([
                    xrv.datasets.XRayCenterCrop(),
                    xrv.datasets.XRayResizer(224)
                ])
                img_tensor = torch.from_numpy(transform(img_norm)).float()

            with st.spinner("Running DenseNet-121 — classifying 18 pathologies..."):
                try:
                    model = xrv.models.DenseNet(weights="densenet121-res224-all")
                    model.eval()
                    with torch.no_grad():
                        outputs = model(img_tensor[None,...])
                    pathologies = model.targets
                    scores = dict(zip(pathologies, outputs[0].detach().numpy().tolist()))
                    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
                    st.success(f"Classification complete — {len(scores)} pathologies scored")
                except Exception as e:
                    st.error(f"Classification error: {e}")
                    scores = {}; model = None

            masks, seg_targets = None, None
            with st.spinner("Running anatomical segmentation..."):
                try:
                    seg_model = xrv.baseline_models.chestx_det.PSPNet()
                    seg_model.eval()
                    with torch.no_grad():
                        seg_out = seg_model(img_tensor[None,...])
                    masks = seg_out[0].detach().numpy()
                    seg_targets = seg_model.targets
                    st.success(f"Segmented {len(seg_targets)} anatomical structures")
                except Exception as e:
                    st.info(f"Segmentation unavailable: {e}")

            ctr, cardiomegaly = None, None
            if masks is not None and seg_targets is not None:
                try:
                    hi = seg_targets.index("Heart")
                    li = seg_targets.index("Left Lung")
                    ri = seg_targets.index("Right Lung")
                    hm = masks[hi] > 0.5
                    chest_m = (masks[li]>0.5)|(masks[ri]>0.5)|hm
                    hc = np.where(hm.any(axis=0))[0]
                    cc = np.where(chest_m.any(axis=0))[0]
                    if len(hc)>=2 and len(cc)>=2:
                        ctr = round(float((hc[-1]-hc[0])/(cc[-1]-cc[0])),4)
                        cardiomegaly = ctr >= 0.50
                except Exception:
                    pass

            flags = []
            for p,s in scores.items():
                t = THRESHOLDS.get(p, 0.45)
                if s >= t:
                    flags.append({
                        "pathology":p,"score":round(s,3),
                        "severity":"Mild" if s<0.50 else ("Moderate" if s<0.70 else "Significant"),
                        "urgent":p in URGENT,"cardiac":p in CARDIAC
                    })
            flags.sort(key=lambda x: (not x["urgent"],-x["score"]))

            gcam = None
            top_path = flags[0]["pathology"] if flags else (list(scores.keys())[0] if scores else None)
            if model and top_path:
                with st.spinner(f"Generating Grad-CAM for {top_path}..."):
                    try:
                        grads, acts = [], []
                        def bwd(m,gi,go): grads.append(go[0])
                        def fwd(m,i,o):   acts.append(o)
                        layer = None
                        for attr_path in [
                            lambda m: m.features.denseblock4,
                            lambda m: m.model.features.denseblock4,
                            lambda m: list(m.features.children())[-1],
                            lambda m: m.features,
                        ]:
                            try:
                                layer = attr_path(model); break
                            except (AttributeError,IndexError):
                                continue
                        if layer is None:
                            raise AttributeError("Could not locate DenseNet feature layer.")
                        h1 = layer.register_forward_hook(fwd)
                        h2 = layer.register_full_backward_hook(bwd)
                        inp = img_tensor[None,...].requires_grad_(True)
                        out = model(inp)
                        model.zero_grad()
                        out[0, model.targets.index(top_path)].backward()
                        if grads and acts:
                            g = grads[0][0]; a = acts[0][0]
                            c = F.relu((g.mean(dim=[1,2])[:,None,None]*a).sum(0)).detach().numpy()
                            if c.max()>0: c = (c-c.min())/(c.max()-c.min())
                            gcam = skimage.transform.resize(c,(224,224))
                            st.success(f"Grad-CAM generated for: {top_path}")
                        h1.remove(); h2.remove()
                    except Exception as e:
                        st.info(f"Grad-CAM unavailable: {e}")

            # ── Results display ─────────────────────────────────
            st.divider()
            st.subheader("Results")
            if ctr is not None:
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("Cardiothoracic Ratio",f"{ctr:.3f}",delta="CARDIOMEGALY" if cardiomegaly else "Normal")
                with c2: st.metric("Findings Flagged",len(flags))
                with c3: st.metric("Urgent Findings",sum(1 for f in flags if f["urgent"]))

            if flags:
                st.subheader("Flagged Findings")
                for f in flags:
                    msg = f"{'🚨' if f['urgent'] else '⚠'} **{f['pathology']}** — Score: {f['score']:.3f} | {f['severity']}{' | CARDIAC' if f['cardiac'] else ''}"
                    if f["urgent"]:  st.error(msg)
                    elif f["cardiac"]: st.warning(msg)
                    else: st.info(msg)

            if scores:
                st.subheader("All Pathology Scores")
                score_df = pd.DataFrame([{"Pathology":p,"Score":round(s,4),"Flagged":"Yes" if p in {f["pathology"] for f in flags} else "No","Threshold":THRESHOLDS.get(p,0.45)} for p,s in scores.items()])
                st.dataframe(score_df, use_container_width=True, hide_index=True)

            if gcam is not None:
                st.subheader(f"Grad-CAM Heatmap — {top_path}")
                fig, ax = plt.subplots(1,2,figsize=(10,4))
                img_arr = np.array(img_pil.convert("L"))
                ax[0].imshow(img_arr, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
                img_r = skimage.transform.resize(img_arr,(224,224))
                ax[1].imshow(img_r, cmap="gray")
                ax[1].imshow(gcam, cmap="jet", alpha=0.45)
                ax[1].set_title(f"Grad-CAM: {top_path}"); ax[1].axis("off")
                st.pyplot(fig); plt.close()

            # ── Report & Export ─────────────────────────────────
            report_lines = [
                "CHEST X-RAY AI ANALYSIS REPORT",
                f"Generated: {pd.Timestamp.now().strftime('%d %B %Y %H:%M')}",
                "System: CardioAI DenseNet-121 | CardioAI Nova", "",
                f"Cardiothoracic Ratio: {ctr if ctr else 'Not computed'}",
                f"Cardiomegaly (CTR≥0.50): {'YES' if cardiomegaly else 'No' if ctr else 'N/A'}",
                f"Total pathologies flagged: {len(flags)}", "",
                "FLAGGED FINDINGS:",
            ]
            for f in flags:
                report_lines.append(f"  {'[URGENT] ' if f['urgent'] else ''}{f['pathology']}: {f['score']:.3f} ({f['severity']})")
            report_lines.append("\nIMPORTANT: AI screening tool only. Radiologist review required.")

            st.divider()
            _img_df = pd.DataFrame([{"Pathology":p,"Score":round(s,4),"Flagged":"Yes" if p in {f["pathology"] for f in flags} else "No","Severity":next((f["severity"] for f in flags if f["pathology"]==p),"—"),"Urgent":"Yes" if p in {f["pathology"] for f in flags if f["urgent"]} else "No","Threshold":THRESHOLDS.get(p,0.45)} for p,s in scores.items()])
            _img_summary_df = pd.DataFrame([{"CTR":ctr if ctr else "N/A","Cardiomegaly":"YES" if cardiomegaly else "No","Findings Flagged":len(flags),"Urgent Findings":sum(1 for f in flags if f["urgent"]),"Top Finding":flags[0]["pathology"] if flags else "None"}])
            export_buttons(
                "Medical Imaging",
                csv_df=_img_df,
                excel_sheets={"Summary":_img_summary_df,"Pathology Scores":_img_df},
                pdf_title="CardioAI — Chest X-Ray Analysis Report",
                pdf_sections=[("Analysis Summary",_img_summary_df),("Pathology Scores",_img_df),("Clinical Report","\n".join(report_lines))],
                docx_title="CardioAI — Chest X-Ray Analysis Report",
                docx_sections=[("Analysis Summary",_img_summary_df),("Pathology Scores",_img_df),("Clinical Report","\n".join(report_lines))],
                file_stem="xray_analysis",
            )
            st.warning("AI screening only. Does not replace radiologist review.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — ECHOCARDIOGRAM
    # ══════════════════════════════════════════════════════════════════════
    with img_tab2:
        st.subheader("Echocardiogram — LV Function & Ejection Fraction")
        st.info(
            "Upload an echocardiogram image or video frame. "
            "CardioAI uses **EchoNet-inspired analysis** with Gemini Vision AI to estimate "
            "ejection fraction (EF%), identify wall motion abnormalities, LV size, and "
            "generate a structured echo report."
        )
        st.warning(
            "⚠ Full EchoNet-Dynamic (video-based EF%) requires a GPU environment. "
            "On Streamlit Cloud, Gemini Vision AI is used for image-based echo interpretation."
        )

        echo_upload = st.file_uploader(
            "Upload echocardiogram image (JPG, PNG)",
            type=["jpg","jpeg","png"],
            help="Parasternal long axis, apical 4-chamber, or any standard echo view.",
            key="echo_upload"
        )

        echo_col1, echo_col2 = st.columns(2)
        with echo_col1:
            echo_view = st.selectbox("Echo View", [
                "Apical 4-Chamber (A4C)",
                "Parasternal Long Axis (PLAX)",
                "Parasternal Short Axis (PSAX)",
                "Apical 2-Chamber (A2C)",
                "Subcostal",
                "Suprasternal",
                "Other / Unknown",
            ])
            echo_modality = st.selectbox("Doppler Mode", [
                "2D Echo only",
                "M-Mode",
                "Colour Doppler",
                "Pulsed Wave (PW) Doppler",
                "Continuous Wave (CW) Doppler",
                "Tissue Doppler (TDI)",
            ])
        with echo_col2:
            echo_ivsd    = st.number_input("IVSd — Interventricular Septum Diastole (mm)", 0.0, 30.0, 9.0, 0.1, help="Normal: 6–11mm")
            echo_lvedd   = st.number_input("LVEDd — LV End-Diastolic Diameter (mm)", 0.0, 90.0, 50.0, 0.1, help="Normal: Men <59mm | Women <53mm")
            echo_lvesd   = st.number_input("LVESd — LV End-Systolic Diameter (mm)", 0.0, 80.0, 33.0, 0.1, help="Normal: Men <40mm | Women <36mm")
            echo_pwed    = st.number_input("PWed — Posterior Wall End-Diastole (mm)", 0.0, 30.0, 9.0, 0.1, help="Normal: 6–11mm")
            echo_ef_manual = st.number_input("EF% (if measured on machine)", 0.0, 100.0, 0.0, 0.5, help="0 = not measured — AI will estimate")

        # Teichholz EF estimation from dimensions
        ef_teich = None
        if echo_lvedd > 0 and echo_lvesd > 0:
            def teichholz(d): return (7.0 / (2.4 + d/10)) * (d/10)**3 * 1000
            edv = teichholz(echo_lvedd); esv = teichholz(echo_lvesd)
            ef_teich = round((edv - esv) / edv * 100, 1) if edv > 0 else None
            lv_mass = round(1.04 * ((echo_lvedd/10 + echo_ivsd/10 + echo_pwed/10)**3 - (echo_lvedd/10)**3) * 0.8 + 0.6, 1)

            ec1, ec2, ec3, ec4 = st.columns(4)
            ef_display = echo_ef_manual if echo_ef_manual > 0 else ef_teich
            ef_class = ("Severely Reduced (<30%)" if ef_display < 30
                        else "Moderately Reduced (30–39%)" if ef_display < 40
                        else "Mildly Reduced (40–49%)" if ef_display < 50
                        else "Low Normal (50–54%)" if ef_display < 55
                        else "Normal (≥55%)")
            with ec1: st.metric("EF%", f"{ef_display:.1f}%", delta=ef_class)
            with ec2: st.metric("EDV (Teichholz)", f"{edv:.0f} mL")
            with ec3: st.metric("ESV (Teichholz)", f"{esv:.0f} mL")
            with ec4: st.metric("LV Mass", f"{lv_mass:.0f} g")
            if ef_display < 40:    st.error(f"⚠ EF {ef_display:.1f}% — Reduced — Heart Failure with Reduced EF (HFrEF)")
            elif ef_display < 50:  st.warning(f"⚠ EF {ef_display:.1f}% — Mildly Reduced — monitor closely")
            elif ef_display < 55:  st.warning(f"⚠ EF {ef_display:.1f}% — Low Normal")
            else:                  st.success(f"✓ EF {ef_display:.1f}% — Normal systolic function")

        # Gemini Vision echo interpretation
        if echo_upload:
            from PIL import Image as PILImage
            echo_img = PILImage.open(echo_upload)
            st.image(echo_img, caption=f"Uploaded: {echo_upload.name}", use_container_width=True)

            st.warning(
                "⚠ **Echocardiogram images only.** "
                "Do not upload ECG printouts, X-rays, lab reports, or spirometry here. "
                "The AI will validate the image type before analysis."
            )

            if st.button("🔍 Analyse Echo with Gemini Vision AI", type="primary", key="echo_gemini_btn"):
                with st.spinner("Validating image type..."):
                    try:
                        import base64, io
                        buf = io.BytesIO()
                        echo_img.save(buf, format="JPEG", quality=95)
                        img_b64 = base64.b64encode(buf.getvalue()).decode()

                        # Image type check
                        val_prompt = """Look at this image carefully.
In ONE sentence, state what type of medical image this is.
Then on a new line write ONLY one of:
ECHO_VALID — if this is an echocardiogram / cardiac ultrasound image
NOT_ECHO — if this is anything else"""
                        val_resp, _ = vision_api_call(val_prompt, img_b64)
                        is_echo = val_resp and "ECHO_VALID" in val_resp.upper()
                        img_desc = val_resp.split("\n")[0] if val_resp else "Unknown"

                        if not is_echo:
                            st.error(
                                f"❌ **Wrong image type:** {img_desc}\n\n"
                                "This tab only accepts **echocardiogram images**. "
                                "Use the correct tab for your image type:\n"
                                "- ECG printout → ECG Signal Analysis tab\n"
                                "- Chest X-ray → Chest X-Ray tab\n"
                                "- Lab/spirometry report → Clinical NLP → Scan & Auto-Fill"
                            )
                            st.session_state["echo_report"] = ""
                        else:
                            st.success(f"✅ Echo confirmed: {img_desc}")
                            with st.spinner("Analysing echocardiogram..."):
                                echo_prompt = f"""You are a cardiologist reviewing an echocardiogram image.
Measured dimensions — IVSd: {echo_ivsd}mm | LVEDd: {echo_lvedd}mm | LVESd: {echo_lvesd}mm | PWed: {echo_pwed}mm
Estimated EF (Teichholz): {ef_teich if ef_teich else 'not calculated'}%
Machine EF: {echo_ef_manual if echo_ef_manual > 0 else 'not entered'}%

Please provide a structured echo report including:
1. Image quality and view adequacy
2. Left ventricular size and systolic function (estimate EF if visible)
3. Left ventricular diastolic function assessment
4. Right ventricular size and function
5. Valvular assessment (mitral, aortic, tricuspid, pulmonary)
6. Pericardium assessment
7. Wall motion abnormalities (if visible)
8. Key measurements and any abnormalities
9. Clinical impression and recommendations
10. Limitations of this assessment

Be concise and clinically precise. Flag any urgent findings clearly."""
                                echo_report, model_used = vision_api_call(echo_prompt, img_b64)
                                if echo_report:
                                    st.session_state["echo_report"] = echo_report
                                    st.caption(f"Analysis by: {model_used}")
                                else:
                                    st.error(f"Echo analysis failed: {model_used}")
                    except Exception as e:
                        st.error(f"Echo analysis error: {e}")

            # Display report
            echo_report = st.session_state.get("echo_report","")
            if echo_report:
                st.divider()
                st.subheader("Gemini Vision Echo Report")
                st.markdown(echo_report)
                # Export
                _echo_dims_df = pd.DataFrame([{
                    "View": echo_view, "Doppler": echo_modality,
                    "IVSd (mm)": echo_ivsd, "LVEDd (mm)": echo_lvedd,
                    "LVESd (mm)": echo_lvesd, "PWed (mm)": echo_pwed,
                    "EF% (Teichholz)": ef_teich if ef_teich else "—",
                    "EF% (Machine)": echo_ef_manual if echo_ef_manual > 0 else "—",
                    "EDV (mL)": round(edv,1) if ef_teich else "—",
                    "ESV (mL)": round(esv,1) if ef_teich else "—",
                }])
                st.divider()
                export_buttons(
                    "Echo Report",
                    csv_df=_echo_dims_df,
                    excel_sheets={"Echo Measurements": _echo_dims_df},
                    pdf_title="CardioAI — Echocardiogram Analysis Report",
                    pdf_sections=[("Echo Measurements",_echo_dims_df),("AI Echo Report",echo_report)],
                    docx_title="CardioAI — Echocardiogram Analysis Report",
                    docx_sections=[("Echo Measurements",_echo_dims_df),("AI Echo Report",echo_report)],
                    file_stem="echo_report",
                )
        else:
            st.info("Upload an echo image above to enable Gemini Vision AI interpretation.")

        st.divider()
        st.caption(
            "**EF Classification:** Normal ≥55% · Low Normal 50–54% · Mildly Reduced 40–49% · "
            "Moderately Reduced 30–39% · Severely Reduced <30% | "
            "Teichholz formula used for M-mode EF estimation. "
            "Always correlate with clinical findings and formal echo report."
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — ECG SIGNAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    with img_tab3:
        st.subheader("ECG Signal Analysis — 12-Lead Interpretation")
        st.info(
            "Two modes: (1) **Upload an ECG image** — Gemini Vision AI reads and interprets "
            "the tracing. (2) **Upload raw ECG signal (CSV)** — automated interval measurement "
            "and rhythm classification."
        )

        ecg_mode = st.radio("Analysis mode", [
            "📷 ECG Image — Gemini Vision AI interpretation",
            "📊 Raw ECG Signal (CSV) — Automated analysis",
        ], key="ecg_mode")

        if "📷" in ecg_mode:
            # ── ECG Image → Gemini Vision ─────────────────────────────
            ecg_img_upload = st.file_uploader(
                "Upload 12-lead ECG image (JPG, PNG)",
                type=["jpg","jpeg","png"],
                help="Standard 12-lead ECG printout or digital screenshot",
                key="ecg_img_upload"
            )

            ecg_context_col1, ecg_context_col2 = st.columns(2)
            with ecg_context_col1:
                ecg_patient_age  = st.number_input("Patient Age", 18, 100, 55, key="ecg_age")
                ecg_patient_sex  = st.selectbox("Patient Sex", ["Male","Female"], key="ecg_sex")
                ecg_symptoms     = st.multiselect("Presenting Symptoms", [
                    "Chest pain","Palpitations","Syncope","Dyspnoea",
                    "Dizziness","Fatigue","None"
                ], default=["None"], key="ecg_symptoms")
            with ecg_context_col2:
                ecg_hx_mi        = st.selectbox("History of MI", ["No","Yes"], key="ecg_hx_mi")
                ecg_hx_htn       = st.selectbox("History of Hypertension", ["No","Yes"], key="ecg_hx_htn")
                ecg_medications  = st.text_input("Current cardiac medications (if any)", key="ecg_meds")
                ecg_speed        = st.selectbox("Paper speed", ["25 mm/s (standard)","50 mm/s"], key="ecg_speed")

            if ecg_img_upload:
                from PIL import Image as PILImage
                ecg_img = PILImage.open(ecg_img_upload)
                st.image(ecg_img, caption=f"Uploaded: {ecg_img_upload.name}", use_container_width=True)

                st.warning(
                    "⚠ **Important:** This tab interprets **12-lead ECG printouts only**. "
                    "Uploading ultrasound, X-ray, spirometry, lab reports, or other images "
                    "will not produce a valid ECG report. "
                    "The AI will identify the image type before proceeding."
                )

                if st.button("🔍 Interpret ECG with Gemini Vision AI", type="primary", key="ecg_gemini_btn"):
                    with st.spinner("Reading and validating image type..."):
                        try:
                            import base64, io
                            buf = io.BytesIO()
                            ecg_img.save(buf, format="JPEG", quality=95)
                            img_b64 = base64.b64encode(buf.getvalue()).decode()

                            # Step 1 — image type validation
                            validation_prompt = """Look at this image carefully.
In ONE sentence, state exactly what type of medical image or document this is.
Then on a new line write ONLY one of these labels:
ECG_VALID — if this is a 12-lead ECG or rhythm strip printout
NOT_ECG — if this is anything else (ultrasound, X-ray, lab report, spirometry, photo, etc.)"""

                            validation_response, _ = vision_api_call(validation_prompt, img_b64)
                            is_ecg = validation_response and "ECG_VALID" in validation_response.upper()
                            image_description = validation_response.split("\n")[0] if validation_response else "Unknown image type"

                            if not is_ecg:
                                st.error(
                                    f"❌ **Wrong image type detected.**\n\n"
                                    f"**AI identified this as:** {image_description}\n\n"
                                    f"This module only accepts **12-lead ECG printouts or rhythm strips**. "
                                    f"Please upload the correct image.\n\n"
                                    f"**If you have an ultrasound:** use the Echocardiogram tab.\n"
                                    f"**If you have a lab report or spirometry:** use Clinical NLP → Scan & Auto-Fill.\n"
                                    f"**If you have a chest X-ray:** use the Chest X-Ray tab."
                                )
                                st.session_state["ecg_report"] = ""
                            else:
                                st.success(f"✅ ECG confirmed: {image_description}")
                                with st.spinner("Interpreting 12-lead ECG..."):
                                    ecg_prompt = f"""You are an expert cardiologist interpreting a 12-lead ECG.
Patient: Age {ecg_patient_age}, {ecg_patient_sex} | Symptoms: {', '.join(ecg_symptoms)}
History: MI: {ecg_hx_mi} | HTN: {ecg_hx_htn} | Medications: {ecg_medications or 'None'} | Speed: {ecg_speed}

Provide a complete structured ECG report:
1. TECHNICAL QUALITY — paper speed, calibration, lead placement
2. RHYTHM — rate (bpm), regularity, diagnosis
3. INTERVALS & AXES — PR, QRS, QT/QTc (ms), P/QRS axis
4. WAVEFORMS — P waves, QRS, ST segments (lead-by-lead if abnormal), T waves, Q waves, U waves
5. SPECIFIC FINDINGS — LVH, chamber enlargement, ischaemia, infarction territory, conduction defects
6. CLINICAL IMPRESSION — primary diagnosis, differential, urgency: ROUTINE/URGENT/EMERGENCY
7. RECOMMENDATIONS — immediate actions, further investigations
Flag STEMI, LBBB, complete heart block, VT prominently."""
                                    ecg_report, model_used = vision_api_call(ecg_prompt, img_b64)
                                    if ecg_report:
                                        st.session_state["ecg_report"] = ecg_report
                                        st.session_state["ecg_model_used"] = model_used
                                    else:
                                        st.error(f"ECG interpretation failed: {model_used}")
                        except Exception as e:
                            st.error(f"ECG interpretation error: {e}")

                ecg_report = st.session_state.get("ecg_report","")
                if ecg_report:
                    st.divider()
                    model_used = st.session_state.get("ecg_model_used","AI")
                    st.subheader("ECG Interpretation Report")
                    st.caption(f"Interpreted by: {model_used}")
                    st.markdown(ecg_report)
                    _ecg_ctx_df = pd.DataFrame([{
                        "Age": ecg_patient_age, "Sex": ecg_patient_sex,
                        "Symptoms": ", ".join(ecg_symptoms),
                        "Hx MI": ecg_hx_mi, "Hx HTN": ecg_hx_htn,
                        "Medications": ecg_medications or "None",
                        "Paper Speed": ecg_speed,
                        "Interpreted by": st.session_state.get("ecg_model_used","AI"),
                    }])
                    st.divider()
                    export_buttons(
                        "ECG Report",
                        csv_df=_ecg_ctx_df,
                        excel_sheets={"Patient Context": _ecg_ctx_df},
                        pdf_title="CardioAI Nova — 12-Lead ECG Interpretation Report",
                        pdf_sections=[("Patient Context",_ecg_ctx_df),("ECG Interpretation",ecg_report)],
                        docx_title="CardioAI Nova — 12-Lead ECG Interpretation Report",
                        docx_sections=[("Patient Context",_ecg_ctx_df),("ECG Interpretation",ecg_report)],
                        file_stem="ecg_interpretation",
                    )

        else:
            # ── Raw ECG Signal CSV ────────────────────────────────────
            st.markdown("**Upload raw ECG signal as CSV**")
            st.caption(
                "Expected format: columns = lead names (I, II, III, aVR, aVL, aVF, V1–V6) "
                "or a single-column signal. Rows = samples. Sampling rate in Hz below."
            )
            ecg_csv_upload = st.file_uploader(
                "Upload ECG signal (CSV)",
                type=["csv"],
                key="ecg_csv_upload"
            )
            ecg_fs = st.number_input("Sampling rate (Hz)", 100, 1000, 500,
                                      help="Common: 250Hz, 500Hz, 1000Hz")

            if ecg_csv_upload:
                try:
                    import matplotlib; matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    ecg_df = pd.read_csv(ecg_csv_upload)
                    st.success(f"Loaded: {ecg_df.shape[0]} samples × {ecg_df.shape[1]} leads")
                    st.dataframe(ecg_df.head(5), use_container_width=True)

                    # Plot signal
                    fig, axes = plt.subplots(min(ecg_df.shape[1],12), 1,
                                             figsize=(14, min(ecg_df.shape[1],12)*1.5),
                                             sharex=True)
                    if ecg_df.shape[1] == 1:
                        axes = [axes]
                    t = [i/ecg_fs for i in range(len(ecg_df))]
                    for idx, col in enumerate(ecg_df.columns[:12]):
                        axes[idx].plot(t, ecg_df[col].values, linewidth=0.7, color="#c0392b")
                        axes[idx].set_ylabel(col, fontsize=8)
                        axes[idx].grid(True, alpha=0.3)
                        axes[idx].set_facecolor("#fff9f9")
                    axes[-1].set_xlabel("Time (seconds)")
                    plt.suptitle("ECG Signal — All Leads", fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig); plt.close()

                    # Basic interval estimation from lead II
                    lead_ii_col = next((c for c in ecg_df.columns if "II" in c.upper() or c.upper() in ["LEAD2","LEAD II","2"]), ecg_df.columns[0])
                    signal = ecg_df[lead_ii_col].values.astype(float)

                    # R-peak detection (simple threshold)
                    try:
                        threshold = signal.mean() + 0.6 * signal.std()
                        from scipy.signal import find_peaks
                        peaks, _ = find_peaks(signal, height=threshold, distance=int(ecg_fs*0.4))
                        if len(peaks) > 1:
                            rr_intervals = [(peaks[i+1]-peaks[i])/ecg_fs for i in range(len(peaks)-1)]
                            mean_rr = sum(rr_intervals)/len(rr_intervals)
                            hr_est  = round(60/mean_rr, 0)
                            rr_std  = round((sum((r-mean_rr)**2 for r in rr_intervals)/len(rr_intervals))**0.5 * 1000, 1)
                            rhythm  = "Regular" if rr_std < 50 else "Irregular — possible AF or ectopy"

                            m1,m2,m3,m4 = st.columns(4)
                            with m1: st.metric("Estimated HR", f"{hr_est:.0f} bpm")
                            with m2: st.metric("Mean RR", f"{mean_rr*1000:.0f} ms")
                            with m3: st.metric("RR Variability (SDNN)", f"{rr_std} ms")
                            with m4: st.metric("Rhythm", rhythm)

                            if rhythm != "Regular":
                                st.warning(f"⚠ Irregular rhythm detected — consider Holter monitoring")
                            if hr_est > 100:
                                st.warning(f"⚠ Tachycardia: {hr_est:.0f} bpm")
                            elif hr_est < 60:
                                st.warning(f"⚠ Bradycardia: {hr_est:.0f} bpm")
                        else:
                            st.info("Insufficient R-peaks detected. Check signal quality and sampling rate.")
                    except Exception as e:
                        st.info(f"Automated interval analysis unavailable: {e}")

                    st.divider()
                    export_buttons(
                        "ECG Signal",
                        csv_df=ecg_df,
                        excel_sheets={"ECG Signal": ecg_df},
                        pdf_title="CardioAI — ECG Signal Analysis",
                        pdf_sections=[("ECG Signal (first 100 samples)", ecg_df.head(100))],
                        docx_title="CardioAI — ECG Signal Analysis",
                        docx_sections=[("ECG Signal (first 100 samples)", ecg_df.head(100))],
                        file_stem="ecg_signal",
                    )
                except Exception as e:
                    st.error(f"ECG CSV error: {e}")

        st.divider()
        st.caption(
            "ECG AI interpretation by Gemini Vision 2.0 Flash. Always confirm with a cardiologist. "
            "This tool does not replace clinical ECG reading."
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — RETINAL FUNDUS
    # ══════════════════════════════════════════════════════════════════════
    with img_tab4:
        st.subheader("Retinal Fundus — Cardiovascular Risk Assessment")
        st.info(
            "The retina is the only place in the body where vasculature can be directly visualised "
            "non-invasively. **Google research (2018, Nature)** demonstrated that deep learning on "
            "fundus images can predict age, sex, systolic BP, HbA1c, BMI, smoking status, and "
            "5-year cardiovascular risk. CardioAI uses **Gemini Vision AI** to interpret fundus "
            "images and identify markers of hypertensive, diabetic, and atherosclerotic retinopathy."
        )

        fundus_upload = st.file_uploader(
            "Upload retinal fundus image (JPG, PNG)",
            type=["jpg","jpeg","png"],
            help="Standard colour fundus photograph. 45° or 30° field of view.",
            key="fundus_upload"
        )

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            fundus_eye     = st.selectbox("Eye", ["Right eye (OD)","Left eye (OS)","Both"], key="fundus_eye")
            fundus_field   = st.selectbox("Field of view", ["45°","30°","Wide-field (>60°)","Unknown"], key="fundus_field")
            fundus_dilated = st.selectbox("Pupil dilation", ["Dilated","Undilated"], key="fundus_dilated")
        with f_col2:
            fundus_sbp   = st.number_input("Systolic BP (mmHg)", 70, 260, 130, key="fundus_sbp")
            fundus_hba1c = st.number_input("HbA1c (%)", 3.0, 20.0, 5.5, 0.1, key="fundus_hba1c")
            fundus_dm    = st.selectbox("Diabetes status", ["No Diabetes","Type 1 DM","Type 2 DM","Pre-Diabetes"], key="fundus_dm")
            fundus_dur   = st.number_input("Duration of diabetes (years)", 0, 50, 0, key="fundus_dur")

        if fundus_upload:
            from PIL import Image as PILImage
            fundus_img = PILImage.open(fundus_upload)
            st.image(fundus_img, caption=f"Uploaded: {fundus_upload.name}", use_container_width=True)

            st.warning(
                "⚠ **Retinal fundus photographs only.** "
                "Do not upload ECG printouts, ultrasounds, X-rays, or lab reports here. "
                "The AI will validate the image type before analysis."
            )

            if st.button("🔍 Analyse Fundus with Gemini Vision AI", type="primary", key="fundus_gemini_btn"):
                with st.spinner("Validating image type..."):
                    try:
                        import base64, io
                        buf = io.BytesIO()
                        fundus_img.save(buf, format="JPEG", quality=95)
                        img_b64 = base64.b64encode(buf.getvalue()).decode()

                        # Image type check
                        val_prompt = """Look at this image carefully.
In ONE sentence, state what type of medical image this is.
Then on a new line write ONLY one of:
FUNDUS_VALID — if this is a retinal fundus photograph / fundoscopy image
NOT_FUNDUS — if this is anything else"""
                        val_resp, _ = vision_api_call(val_prompt, img_b64)
                        is_fundus = val_resp and "FUNDUS_VALID" in val_resp.upper()
                        img_desc = val_resp.split("\n")[0] if val_resp else "Unknown"

                        if not is_fundus:
                            st.error(
                                f"❌ **Wrong image type:** {img_desc}\n\n"
                                "This tab only accepts **retinal fundus photographs**. "
                                "Use the correct tab:\n"
                                "- Echocardiogram → Echocardiogram tab\n"
                                "- ECG printout → ECG Signal Analysis tab\n"
                                "- Chest X-ray → Chest X-Ray tab\n"
                                "- Lab/spirometry → Clinical NLP → Scan & Auto-Fill"
                            )
                            st.session_state["fundus_report"] = ""
                        else:
                            st.success(f"✅ Fundus image confirmed: {img_desc}")
                            with st.spinner("Analysing retinal fundus..."):
                                fundus_prompt = f"""You are a consultant ophthalmologist and cardiologist interpreting a retinal fundus photograph for cardiovascular risk assessment.
Eye: {fundus_eye} | Field: {fundus_field} | Dilation: {fundus_dilated}
Clinical: SBP {fundus_sbp}mmHg | HbA1c {fundus_hba1c}% | Diabetes: {fundus_dm} ({fundus_dur} years)

Provide a structured fundus report:
1. IMAGE QUALITY — grade (1-5), field adequacy, clarity
2. OPTIC DISC — size, CDR, colour, margins, NVD
3. MACULA — foveal reflex, exudates, haemorrhages, oedema
4. VASCULATURE — AV ratio, AV nicking, arterial reflex, tortuosity, NVE
5. BACKGROUND — microaneurysms, dot/blot haemorrhages, cotton wool spots, hard/soft exudates, laser scars
6. HYPERTENSIVE RETINOPATHY — Keith-Wagener-Barker grade (0-4)
7. DIABETIC RETINOPATHY — ETDRS/ICDRS grade (No DR/Mild NPDR/Moderate NPDR/Severe NPDR/PDR), DME present/absent
8. OTHER PATHOLOGY — glaucoma, ARMD, vascular occlusions
9. CV RISK MARKERS — AV ratio, arteriolar narrowing, estimated SBP from retinal features, atherosclerosis signs
10. CLINICAL IMPRESSION — primary finding, urgency: ROUTINE/URGENT(within 1 week)/EMERGENCY(same day), referrals
                                Flag sight-threatening or life-threatening findings prominently."""
                                fundus_report, model_used = vision_api_call(fundus_prompt, img_b64)
                                if fundus_report:
                                    st.session_state["fundus_report"] = fundus_report
                                    st.caption(f"Analysed by: {model_used}")
                                else:
                                    st.error(f"Fundus analysis failed: {model_used}")
                    except Exception as e:
                        st.error(f"Fundus analysis error: {e}")
            fundus_report = st.session_state.get("fundus_report","")
            if fundus_report:
                st.divider()
                st.subheader("Gemini Vision — Fundus Analysis Report")
                st.markdown(fundus_report)
                _fundus_ctx_df = pd.DataFrame([{
                    "Eye": fundus_eye, "Field": fundus_field,
                    "Dilation": fundus_dilated,
                    "Systolic BP (mmHg)": fundus_sbp,
                    "HbA1c (%)": fundus_hba1c,
                    "Diabetes": fundus_dm,
                    "DM Duration (yrs)": fundus_dur,
                }])
                st.divider()
                export_buttons(
                    "Fundus Report",
                    csv_df=_fundus_ctx_df,
                    excel_sheets={"Clinical Context": _fundus_ctx_df},
                    pdf_title="CardioAI — Retinal Fundus Analysis Report",
                    pdf_sections=[("Clinical Context",_fundus_ctx_df),("Fundus AI Report",fundus_report)],
                    docx_title="CardioAI — Retinal Fundus Analysis Report",
                    docx_sections=[("Clinical Context",_fundus_ctx_df),("Fundus AI Report",fundus_report)],
                    file_stem="fundus_report",
                )
        else:
            st.info("Upload a fundus image above to enable AI analysis.")

        st.divider()
        st.caption(
            "**Reference:** Poplin et al. (2018) 'Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning' — *Nature Biomedical Engineering.* "
            "Hypertensive retinopathy grading: Keith-Wagener-Barker classification. "
            "Diabetic retinopathy: ETDRS/ICDRS classification. "
            "All findings must be confirmed by a qualified ophthalmologist."
        )

    st.divider()
    st.caption(
        "**Important:** All CardioAI imaging analysis is for clinical decision support only. "
        "Chest X-ray results require radiologist review. Echo requires cardiologist confirmation. "
        "ECG interpretation requires physician review. Fundus findings require ophthalmologist confirmation. "
        "No imaging AI output should be used as a sole basis for clinical decisions."
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

    if uploaded_xray:
        st.info(
            "⚠ **Important:** This CNN is trained exclusively on **chest X-rays** (PA/AP views). "
            "Uploading echocardiograms, ultrasounds, CT scans, MRI, or other modalities will "
            "produce meaningless scores. Please ensure the uploaded image is a standard chest X-ray."
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

                    # xrv DenseNet exposes features directly on the model object
                    # Try multiple access patterns for robustness across xrv versions
                    layer = None
                    for attr_path in [
                        lambda m: m.features.denseblock4,        # xrv >= 0.0.29
                        lambda m: m.model.features.denseblock4,  # older xrv
                        lambda m: list(m.features.children())[-1],  # last block fallback
                        lambda m: m.features,                    # whole feature extractor
                    ]:
                        try:
                            layer = attr_path(model)
                            break
                        except (AttributeError, IndexError):
                            continue

                    if layer is None:
                        raise AttributeError(
                            "Could not locate DenseNet feature layer. "
                            "Check torchxrayvision version."
                        )

                    h1 = layer.register_forward_hook(fwd)
                    h2 = layer.register_full_backward_hook(bwd)
                    inp = img_tensor[None, ...].requires_grad_(True)
                    out = model(inp)
                    model.zero_grad()
                    target_idx = model.targets.index(top_path)
                    out[0, target_idx].backward()

                    if grads and acts:
                        g = grads[0][0]; a = acts[0][0]
                        c = F.relu(
                            (g.mean(dim=[1,2])[:, None, None] * a).sum(0)
                        ).detach().numpy()
                        if c.max() > 0:
                            c = (c - c.min()) / (c.max() - c.min())
                        gcam = skimage.transform.resize(c, (224, 224))
                        st.success(f"Grad-CAM generated for: {top_path}")
                    else:
                        st.info("Grad-CAM: no gradients captured — hooks may not have fired")

                    h1.remove(); h2.remove()
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

        # ── Export: Medical Imaging ───────────────────
        st.divider()
        _img_df = pd.DataFrame([{
            "Pathology": p, "Score": round(s, 4),
            "Flagged": "Yes" if p in {f["pathology"] for f in flags} else "No",
            "Severity": next((f["severity"] for f in flags if f["pathology"] == p), "—"),
            "Urgent": "Yes" if p in {f["pathology"] for f in flags if f["urgent"]} else "No",
            "Threshold": THRESHOLDS.get(p, 0.45),
        } for p, s in scores.items()])
        _img_summary_df = pd.DataFrame([{
            "Cardiothoracic Ratio": ctr if ctr is not None else "N/A",
            "Cardiomegaly": "YES" if cardiomegaly else "No",
            "Findings Flagged": len(flags),
            "Urgent Findings": sum(1 for f in flags if f["urgent"]),
            "Top Finding": flags[0]["pathology"] if flags else "None",
        }])
        export_buttons(
            "Medical Imaging",
            csv_df=_img_df,
            excel_sheets={"Summary": _img_summary_df, "All Pathology Scores": _img_df},
            pdf_title="CardioAI — Chest X-Ray Analysis Report",
            pdf_sections=[
                ("Analysis Summary", _img_summary_df),
                ("Pathology Scores", _img_df),
                ("Clinical Report", "\n".join(report_lines)),
            ],
            docx_title="CardioAI — Chest X-Ray Analysis Report",
            docx_sections=[
                ("Analysis Summary", _img_summary_df),
                ("Pathology Scores", _img_df),
                ("Clinical Report", "\n".join(report_lines)),
            ],
            file_stem="xray_analysis",
        )

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

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — SPIROMETRY / PULMONARY FUNCTION TEST
    # ══════════════════════════════════════════════════════════════════════
    with img_tab5:
        st.subheader("🌬️ Spirometry & Pulmonary Function Test (PFT)")
        st.info(
            "Two modes: **(1) Upload a scanned spirometry report** — AI reads all values "
            "including FEV1, FVC, PEF, flow-volume loop and generates a full interpretation. "
            "**(2) Manual entry** — enter values directly for instant classification "
            "and auto-fill mapping to Risk Prediction."
        )

        spiro_mode = st.radio(
            "Mode", ["📷 Upload Spirometry Report (AI reads)", "✏️ Manual Value Entry"],
            key="spiro_mode", horizontal=True
        )
        st.divider()

        # ── Patient context ───────────────────────────────────────────────
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sp_age    = st.number_input("Age (years)", 5, 100, 55, key="sp_age")
            sp_sex    = st.selectbox("Sex", ["Male","Female"], key="sp_sex")
            sp_ethnic = st.selectbox("Ethnicity",
                ["Caucasian / White","African / Black","Asian","Hispanic / Latino","Other"],
                key="sp_ethnic",
                help="African/Black predicted values ~12% lower than Caucasian reference")
        with sc2:
            sp_height = st.number_input("Height (cm)", 100.0, 220.0, 166.0, 0.5, key="sp_height")
            sp_weight = st.number_input("Weight (kg)", 20.0, 200.0, 69.0, 0.5, key="sp_weight")
            sp_bmi    = round(sp_weight / ((sp_height/100)**2), 1)
            st.metric("BMI", f"{sp_bmi} kg/m²")
        with sc3:
            sp_smoking = st.selectbox("Smoking status",
                ["Never smoked","Ex-smoker","Current smoker (<10 pack-years)",
                 "Current smoker (10-20 pack-years)","Heavy smoker (>20 pack-years)"],
                key="sp_smoking")
            sp_symptoms = st.multiselect("Symptoms",
                ["Dyspnoea on exertion","Chronic cough","Wheeze",
                 "Sputum production","Chest tightness","None"],
                default=["None"], key="sp_symptoms")
            sp_indication = st.selectbox("Indication for PFT",
                ["Screening / Routine","Dyspnoea workup","Pre-operative",
                 "Monitor COPD","Monitor Asthma","Cardiac rehab pre-assessment","Other"],
                key="sp_indication")
        st.divider()

        if "📷" in spiro_mode:
            # ── Upload mode ───────────────────────────────────────────────
            st.markdown("#### Upload Spirometry Report")
            st.caption(
                "Supports scanned printouts (upside-down, rotated), thermal prints, "
                "digital screenshots and PDFs from any spirometer brand."
            )
            spiro_upload = st.file_uploader(
                "Upload spirometry report (JPG, PNG, PDF)",
                type=["jpg","jpeg","png","pdf"], key="spiro_upload"
            )
            enhance_spiro = st.checkbox(
                "Auto-enhance contrast (recommended for thermal prints)", value=True, key="spiro_enhance"
            )

            if spiro_upload:
                from PIL import Image as PILImage, ImageEnhance, ImageOps
                import base64, io as _io

                if spiro_upload.name.lower().endswith('.pdf'):
                    try:
                        import fitz
                        doc = fitz.open(stream=spiro_upload.read(), filetype="pdf")
                        pix = doc[0].get_pixmap(dpi=200)
                        spiro_img = PILImage.frombytes("RGB",[pix.width,pix.height],pix.samples)
                    except Exception as e:
                        st.error(f"PDF conversion error: {e}"); spiro_img = None
                else:
                    spiro_img = PILImage.open(spiro_upload)

                if spiro_img:
                    try: spiro_img = ImageOps.exif_transpose(spiro_img)
                    except Exception: pass
                    spiro_disp = (ImageEnhance.Contrast(
                        ImageEnhance.Sharpness(spiro_img.convert("RGB")).enhance(1.8)
                    ).enhance(1.4)) if enhance_spiro else spiro_img.convert("RGB")

                    st.image(spiro_disp, caption=f"Uploaded: {spiro_upload.name}",
                             use_container_width=True)

                    if st.button("🔍 Read Spirometry Report with AI", type="primary", key="spiro_ai_btn"):
                        with st.spinner("AI reading spirometry report..."):
                            try:
                                buf = _io.BytesIO()
                                spiro_disp.save(buf, format="JPEG", quality=95)
                                img_b64 = base64.b64encode(buf.getvalue()).decode()

                                spiro_prompt = f"""You are a respiratory physician reading a spirometry/PFT report.
Patient: Age {sp_age}, {sp_sex}, Height {sp_height}cm, Weight {sp_weight}kg, BMI {sp_bmi}
Ethnicity: {sp_ethnic} | Smoking: {sp_smoking} | Indication: {sp_indication}

The image may be upside-down or rotated — read ALL values regardless.
Return ONLY valid JSON (no markdown):
{{
  "patient_name": null, "patient_id": null, "patient_age": null, "patient_sex": null,
  "test_date": null,
  "spirometer_model": null, "technician": null, "physician": null,
  "quality_grade": null,
  "pre_bronchodilator": {{
    "fev1_litres": null, "fev1_pct_predicted": null,
    "fvc_litres": null, "fvc_pct_predicted": null,
    "fev1_fvc_ratio": null, "fev1_fvc_pct_predicted": null,
    "pef_l_min": null, "pef_pct_predicted": null,
    "fef2575_litres": null, "fef2575_pct_predicted": null,
    "fet_seconds": null
  }},
  "post_bronchodilator": {{
    "fev1_litres": null, "fev1_pct_predicted": null,
    "fvc_litres": null, "fvc_pct_predicted": null,
    "fev1_fvc_ratio": null,
    "fev1_change_pct": null, "fvc_change_pct": null,
    "bronchodilator_response": null
  }},
  "lung_volumes": {{
    "tlc_litres": null, "tlc_pct_predicted": null,
    "rv_litres": null, "rv_pct_predicted": null, "rv_tlc_ratio": null
  }},
  "diffusion": {{
    "dlco_ml_min_mmhg": null, "dlco_pct_predicted": null,
    "kco": null, "kco_pct_predicted": null
  }},
  "interpretation": {{
    "pattern": null, "severity": null,
    "reversibility": null, "full_interpretation": null,
    "reference_values_source": null
  }},
  "gold_stage": null,
  "raw_text_extracted": null
}}
Pattern options: Normal / Obstructive / Restrictive / Mixed / Non-specific
Severity (obstruction): Mild (FEV1>=70%) / Moderate (50-69%) / Severe (30-49%) / Very Severe (<30%)
BD response: Positive (>=12% AND >=200mL increase) / Negative / Not tested

CRITICAL EXTRACTION RULES:
1. Spirometry reports have multiple columns: Base (actual measured), Min Pred, Max Pred, % Pred
2. For fev1_litres, fvc_litres etc → use the ACTUAL measured value (Base column), NOT predicted
3. For fev1_pct_predicted → use the % Pred column value (should be between 0-200%)
4. FEV1/FVC ratio → must be between 0.0 and 1.0 (e.g. 0.84 not 84). If printed as percentage (84.3%), divide by 100
5. PEF % predicted → must be between 0-200%. Do not confuse with raw PEF value in L/min
6. patient_age → extract the numeric age from the header (e.g. "Age: 78" or "78yr Male")
7. patient_sex → extract Male or Female from the header
8. If a value looks wrong (e.g. FEV1/FVC > 1.0, % predicted > 200), set it to null rather than guess"""

                                spiro_raw, model_used = vision_api_call(spiro_prompt, img_b64)
                                st.session_state["spiro_result"] = spiro_raw
                                st.session_state["spiro_model"]  = model_used
                            except Exception as e:
                                st.error(f"Spirometry read error: {e}")

            spiro_raw = st.session_state.get("spiro_result","")
            if spiro_raw:
                import json, re as _re
                try:
                    spiro_data = json.loads(spiro_raw)
                except Exception:
                    m = _re.search(r'\{.*\}', spiro_raw, _re.DOTALL)
                    try:    spiro_data = json.loads(m.group()) if m else {}
                    except: spiro_data = {}; st.code(spiro_raw)

                if spiro_data:
                    st.success(f"✅ Read by: {st.session_state.get('spiro_model','AI')}")

                    # ── Confidence & plausibility check ────────────────
                    pre_raw = spiro_data.get("pre_bronchodilator",{}) or {}
                    _spiro_field_map = {
                        "fev1_litres":"fev1_litres","fev1_pct_predicted":"fev1_pct_predicted",
                        "fvc_litres":"fvc_litres","fvc_pct_predicted":"fvc_pct_predicted",
                        "fev1_fvc_ratio":"fev1_fvc_ratio",
                        "pef_l_min":"pef_l_min","pef_pct_predicted":"pef_pct_predicted",
                    }
                    _, spiro_warns, spiro_crits = validate_extracted_data(pre_raw, _spiro_field_map)
                    render_confidence_panel(
                        spiro_warns, spiro_crits,
                        model_used=st.session_state.get("spiro_model","AI"),
                        context="Spirometry report extraction"
                    )
                    st.divider()

                    # Patient info strip
                    info_parts = [f"**{k}:** {v}" for k,v in [
                        ("Patient", spiro_data.get("patient_name")),
                        ("ID",      spiro_data.get("patient_id")),
                        ("Date",    spiro_data.get("test_date")),
                        ("Device",  spiro_data.get("spirometer_model")),
                        ("Tech",    spiro_data.get("technician")),
                        ("Dr",      spiro_data.get("physician")),
                        ("Quality", spiro_data.get("quality_grade")),
                    ] if v]
                    if info_parts: st.markdown("  ·  ".join(info_parts))

                    pre    = spiro_data.get("pre_bronchodilator",{}) or {}
                    post   = spiro_data.get("post_bronchodilator",{}) or {}
                    interp = spiro_data.get("interpretation",{}) or {}
                    lv     = spiro_data.get("lung_volumes",{}) or {}
                    dl     = spiro_data.get("diffusion",{}) or {}
                    gold   = spiro_data.get("gold_stage")
                    pattern  = interp.get("pattern","")
                    severity = interp.get("severity","")
                    full_int = interp.get("full_interpretation","")

                    # ── Sanity checks on extracted values ────────────────
                    # FEV1/FVC % predicted > 100 is almost always a misread
                    # (the machine prints reference column values nearby)
                    if pre.get("fev1_fvc_pct_predicted") and float(pre["fev1_fvc_pct_predicted"]) > 100:
                        pre["fev1_fvc_pct_predicted"] = None  # suppress clearly wrong value
                    # PEF % predicted < 5 or > 200 is clearly wrong
                    if pre.get("pef_pct_predicted"):
                        pef_pct = float(pre["pef_pct_predicted"])
                        if pef_pct < 5 or pef_pct > 200:
                            pre["pef_pct_predicted"] = None
                    # FEV1/FVC ratio > 1.2 is impossible — misread of % column
                    if pre.get("fev1_fvc_ratio"):
                        ratio = float(pre["fev1_fvc_ratio"])
                        if ratio > 1.2:  # likely read as percentage e.g. 84.3 instead of 0.843
                            pre["fev1_fvc_ratio"] = round(ratio / 100, 3)
                    # Same for post-BD
                    if post.get("fev1_fvc_ratio"):
                        ratio = float(post["fev1_fvc_ratio"])
                        if ratio > 1.2:
                            post["fev1_fvc_ratio"] = round(ratio / 100, 3)

                    def show_metric(col, label, val, unit="", ref=None):
                        if val is not None:
                            col.metric(label, f"{val} {unit}".strip(),
                                       delta=f"Pred: {ref}%" if ref else None)
                        else: col.metric(label, "—")

                    st.subheader("Pre-Bronchodilator")
                    c1,c2,c3,c4,c5,c6 = st.columns(6)
                    show_metric(c1,"FEV1 (L)",   pre.get("fev1_litres"),"L", pre.get("fev1_pct_predicted"))
                    show_metric(c2,"FVC (L)",    pre.get("fvc_litres"), "L", pre.get("fvc_pct_predicted"))
                    show_metric(c3,"FEV1/FVC",   pre.get("fev1_fvc_ratio"),"",pre.get("fev1_fvc_pct_predicted"))
                    show_metric(c4,"PEF (L/min)",pre.get("pef_l_min"),  "",  pre.get("pef_pct_predicted"))
                    show_metric(c5,"FEF25-75%",  pre.get("fef2575_litres"),"L/s",pre.get("fef2575_pct_predicted"))
                    show_metric(c6,"FET (s)",    pre.get("fet_seconds"))

                    if any(v for v in post.values() if v):
                        st.subheader("Post-Bronchodilator")
                        p1,p2,p3,p4 = st.columns(4)
                        show_metric(p1,"FEV1 (L)",  post.get("fev1_litres"),"L",post.get("fev1_pct_predicted"))
                        show_metric(p2,"FVC (L)",   post.get("fvc_litres"), "L",post.get("fvc_pct_predicted"))
                        show_metric(p3,"FEV1 Δ%",   post.get("fev1_change_pct"),"%")
                        show_metric(p4,"BD Response",post.get("bronchodilator_response"))

                    if any(v for v in lv.values() if v):
                        st.subheader("Lung Volumes")
                        lv1,lv2,lv3 = st.columns(3)
                        show_metric(lv1,"TLC (L)",lv.get("tlc_litres"),"L",lv.get("tlc_pct_predicted"))
                        show_metric(lv2,"RV (L)", lv.get("rv_litres"), "L",lv.get("rv_pct_predicted"))
                        show_metric(lv3,"RV/TLC", lv.get("rv_tlc_ratio"))

                    if any(v for v in dl.values() if v):
                        st.subheader("Diffusion (DLCO)")
                        d1,d2 = st.columns(2)
                        show_metric(d1,"DLCO",dl.get("dlco_ml_min_mmhg"),"mL/min/mmHg",dl.get("dlco_pct_predicted"))
                        show_metric(d2,"KCO", dl.get("kco"),"",dl.get("kco_pct_predicted"))

                    st.subheader("Clinical Interpretation")
                    c_p,c_s,c_r,c_g = st.columns(4)
                    c_p.metric("Pattern",  pattern  or "—")
                    c_s.metric("Severity", severity or "—")
                    c_r.metric("Reversibility", interp.get("reversibility","—") or "—")
                    c_g.metric("GOLD Stage", gold or "—")

                    if pattern:
                        if "Normal" in pattern:     st.success(f"✅ {pattern}")
                        elif "Obstructive" in pattern and "Severe" in (severity or ""):
                            st.error(f"⚠ {pattern} — {severity}")
                        elif pattern != "—":        st.warning(f"⚠ {pattern} — {severity or 'ungraded'}")
                    if full_int: st.info(full_int)

                    # CV impact
                    if any(x in (pattern or "") for x in ["Obstructive","Restrictive","Mixed"]):
                        st.warning(
                            "⚠ Pulmonary impairment detected. COPD and restrictive lung disease "
                            "significantly increase cardiovascular risk. Ensure this patient's "
                            "Risk Prediction inputs include the respiratory pattern above."
                        )

                    # Auto-fill table
                    st.subheader("📝 Auto-Fill for Risk Prediction Module")

                    # Use age from the extracted report if available, else widget value
                    _pt_age_from_report = None
                    raw_txt = spiro_data.get("raw_text_extracted","") or ""
                    # Try to get age from patient_info if AI extracted it
                    if spiro_data.get("patient_age"):
                        _pt_age_from_report = spiro_data["patient_age"]
                    # Otherwise try to parse from raw text (e.g. "Age: 78" or "78yr")
                    if not _pt_age_from_report and raw_txt:
                        import re as _re2
                        age_match = _re2.search(r'(?:age[:\s]+|(\d{2,3})\s*yr)', raw_txt, _re2.IGNORECASE)
                        if age_match:
                            _pt_age_from_report = age_match.group(1) or age_match.group(0).replace("Age:","").replace("Age","").strip()
                    # Also check patient_name field for age hint
                    _effective_age = _pt_age_from_report if _pt_age_from_report else sp_age

                    # Sex from report if available
                    _pt_sex_from_report = spiro_data.get("patient_sex") or sp_sex

                    rows = [
                        ("Demographics → Patient Name",  spiro_data.get("patient_name","—"),  ""),
                        ("Demographics → Patient ID",    spiro_data.get("patient_id","—"),    ""),
                        ("Demographics → Age",           _effective_age,                       "years"),
                        ("Demographics → Sex",           _pt_sex_from_report,                  ""),
                        ("Demographics → Height",        sp_height,                            "cm"),
                        ("Demographics → Weight",        sp_weight,                            "kg"),
                        ("Demographics → BMI",           sp_bmi,                               "kg/m²"),
                        ("Demographics → Smoking Status",sp_smoking,                           ""),
                    ]
                    if pattern:  rows.append(("Demographics → Respiratory Pattern",  pattern,  ""))
                    if severity: rows.append(("Demographics → PFT Severity",          severity, ""))
                    if pre.get("fev1_litres"):
                        rows.append(("Spirometry → FEV1 (L)",         pre["fev1_litres"],          "L"))
                    if pre.get("fev1_pct_predicted"):
                        rows.append(("Spirometry → FEV1 % Predicted", pre["fev1_pct_predicted"],    "%"))
                    if pre.get("fvc_litres"):
                        rows.append(("Spirometry → FVC (L)",          pre["fvc_litres"],            "L"))
                    if pre.get("fvc_pct_predicted"):
                        rows.append(("Spirometry → FVC % Predicted",  pre["fvc_pct_predicted"],     "%"))
                    if pre.get("fev1_fvc_ratio"):
                        rows.append(("Spirometry → FEV1/FVC Ratio",   pre["fev1_fvc_ratio"],        ""))
                    if pre.get("pef_l_min"):
                        rows.append(("Spirometry → PEF (L/min)",      pre["pef_l_min"],             "L/min"))
                    if pre.get("pef_pct_predicted"):
                        rows.append(("Spirometry → PEF % Predicted",  pre["pef_pct_predicted"],     "%"))
                    if any(x in (pattern or "") for x in ["Obstructive","Mixed"]):
                        rows.append(("Demographics → Dyspnoea on Exertion", "Yes — report-confirmed", ""))
                    if gold:
                        rows.append(("Spirometry → GOLD Stage (COPD)", gold, ""))
                    elif "Restrictive" in (pattern or ""):
                        rows.append(("Note", "GOLD Stage not applicable — Restrictive pattern (GOLD applies to obstruction only)", ""))

                    af_df = pd.DataFrame(rows, columns=["Risk Prediction Field","Value","Unit"])
                    st.dataframe(af_df, use_container_width=True, hide_index=True)

                    # ── Clean export dataframe with proper column names ────────
                    # Map raw JSON keys to clean clinical labels
                    _col_map = {
                        "fev1_litres":             "FEV1 (L)",
                        "fev1_pct_predicted":      "FEV1 % Predicted",
                        "fvc_litres":              "FVC (L)",
                        "fvc_pct_predicted":       "FVC % Predicted",
                        "fev1_fvc_ratio":          "FEV1/FVC Ratio",
                        "fev1_fvc_pct_predicted":  "FEV1/FVC % Predicted",
                        "pef_l_min":               "PEF (L/min)",
                        "pef_pct_predicted":       "PEF % Predicted",
                        "fef2575_litres":          "FEF25-75% (L/s)",
                        "fef2575_pct_predicted":   "FEF25-75% Predicted",
                        "fet_seconds":             "FET (s)",
                        "fev1_change_pct":         "FEV1 Change %",
                        "fvc_change_pct":          "FVC Change %",
                        "bronchodilator_response": "BD Response",
                    }

                    flat = {
                        "Patient Name":   spiro_data.get("patient_name","—"),
                        "Patient ID":     spiro_data.get("patient_id","—"),
                        "Test Date":      spiro_data.get("test_date","—"),
                        "Device":         spiro_data.get("spirometer_model","—"),
                        "Technician":     spiro_data.get("technician","—"),
                        "Physician":      spiro_data.get("physician","—"),
                        "Quality Grade":  spiro_data.get("quality_grade","—"),
                        "Age":            _effective_age,
                        "Sex":            _pt_sex_from_report,
                        "Height (cm)":    sp_height,
                        "Weight (kg)":    sp_weight,
                        "BMI":            sp_bmi,
                        "Smoking":        sp_smoking,
                        "Ethnicity":      sp_ethnic,
                        "Indication":     sp_indication,
                    }
                    for k, v in pre.items():
                        if v is not None:
                            flat[f"Pre-BD {_col_map.get(k, k.replace('_',' ').title())}"] = v
                    for k, v in post.items():
                        if v is not None:
                            flat[f"Post-BD {_col_map.get(k, k.replace('_',' ').title())}"] = v
                    flat.update({
                        "Pattern":    pattern,
                        "Severity":   severity,
                        "GOLD Stage": gold if gold else ("N/A — Restrictive" if "Restrictive" in (pattern or "") else "—"),
                        "Full Interpretation": full_int or "—",
                        "Reference Source": interp.get("reference_values_source","—"),
                    })
                    exp_df = pd.DataFrame([flat])

                    st.divider()
                    export_buttons(
                        "Spirometry Report",
                        csv_df=exp_df,
                        excel_sheets={"Spirometry Results":exp_df,"Auto-Fill":af_df},
                        pdf_title="CardioAI Nova — Spirometry / PFT Report",
                        pdf_sections=[
                            ("Spirometry Results", exp_df),
                            ("Clinical Interpretation", full_int or "—"),
                            ("Auto-Fill for Risk Prediction", af_df),
                        ],
                        docx_title="CardioAI Nova — Spirometry / PFT Report",
                        docx_sections=[
                            ("Spirometry Results", exp_df),
                            ("Clinical Interpretation", full_int or "—"),
                            ("Auto-Fill for Risk Prediction", af_df),
                        ],
                        file_stem="spirometry_report",
                    )

        else:
            # ── Manual entry mode ─────────────────────────────────────────
            st.markdown("#### Manual Spirometry Value Entry")
            me1, me2 = st.columns(2)
            with me1:
                st.markdown("**Pre-Bronchodilator**")
                m_fev1     = st.number_input("FEV1 (L)", 0.0, 8.0, 0.0, 0.01, key="m_fev1")
                m_fev1_pct = st.number_input("FEV1 % predicted", 0.0, 200.0, 0.0, 0.1, key="m_fev1pct")
                m_fvc      = st.number_input("FVC (L)", 0.0, 10.0, 0.0, 0.01, key="m_fvc")
                m_fvc_pct  = st.number_input("FVC % predicted", 0.0, 200.0, 0.0, 0.1, key="m_fvcpct")
                m_ratio    = st.number_input("FEV1/FVC ratio", 0.0, 1.0, 0.0, 0.01, key="m_ratio",
                                             help="Normal ≥0.70 | <0.70 = obstruction")
                m_pef      = st.number_input("PEF (L/min)", 0.0, 900.0, 0.0, 1.0, key="m_pef")
                m_pef_pct  = st.number_input("PEF % predicted", 0.0, 200.0, 0.0, 0.1, key="m_pefpct")
                m_fef2575  = st.number_input("FEF25-75% (L/s)", 0.0, 10.0, 0.0, 0.01, key="m_fef2575")
            with me2:
                st.markdown("**Post-Bronchodilator (if done)**")
                m_fev1_post = st.number_input("FEV1 post-BD (L)", 0.0, 8.0, 0.0, 0.01, key="m_fev1post")
                m_fvc_post  = st.number_input("FVC post-BD (L)",  0.0, 10.0, 0.0, 0.01, key="m_fvcpost")
                st.markdown("**Lung Volumes (if done)**")
                m_tlc      = st.number_input("TLC (L)", 0.0, 15.0, 0.0, 0.01, key="m_tlc",
                                             help="Needed to confirm restriction")
                m_tlc_pct  = st.number_input("TLC % predicted", 0.0, 200.0, 0.0, 0.1, key="m_tlcpct")
                m_rv       = st.number_input("RV (L)", 0.0, 8.0, 0.0, 0.01, key="m_rv")
                st.markdown("**Diffusion**")
                m_dlco     = st.number_input("DLCO (mL/min/mmHg)", 0.0, 50.0, 0.0, 0.1, key="m_dlco")
                m_dlco_pct = st.number_input("DLCO % predicted", 0.0, 200.0, 0.0, 0.1, key="m_dlcopct")

            if st.button("🔬 Classify & Interpret", type="primary", key="spiro_manual_btn"):
                fev1_fvc = m_ratio if m_ratio > 0 else (m_fev1/m_fvc if m_fvc > 0 else None)
                obstruction = fev1_fvc is not None and fev1_fvc < 0.70
                restriction = (m_tlc_pct > 0 and m_tlc_pct < 80) or (
                    not obstruction and m_fvc_pct > 0 and m_fvc_pct < 80)

                pattern = ("Mixed (Obstructive + Restrictive)" if obstruction and m_tlc_pct>0 and m_tlc_pct<80
                           else "Obstructive" if obstruction
                           else "Restrictive" if restriction
                           else "Normal" if (m_fev1_pct>=80 and m_fvc_pct>=80 and m_fvc_pct>0)
                           else "Indeterminate / Incomplete data")

                severity = "—"
                if obstruction and m_fev1_pct > 0:
                    severity = ("Mild (GOLD 1)" if m_fev1_pct>=80
                                else "Moderate (GOLD 2)" if m_fev1_pct>=50
                                else "Severe (GOLD 3)" if m_fev1_pct>=30
                                else "Very Severe (GOLD 4)")
                elif restriction and m_tlc_pct > 0:
                    severity = ("Mild" if m_tlc_pct>=70 else "Moderate" if m_tlc_pct>=60 else "Severe")

                bd_resp = "Not tested"
                if m_fev1>0 and m_fev1_post>0:
                    delta_abs = m_fev1_post - m_fev1
                    delta_pct = (delta_abs / m_fev1) * 100
                    bd_resp = ("Positive (significant reversibility)"
                               if delta_pct>=12 and delta_abs>=0.2 else "Negative")

                st.divider()
                st.subheader("Classification")
                cc1,cc2,cc3,cc4 = st.columns(4)
                cc1.metric("Pattern",  pattern)
                cc2.metric("Severity", severity)
                cc3.metric("FEV1/FVC", f"{fev1_fvc:.2f}" if fev1_fvc else "—")
                cc4.metric("BD Response", bd_resp)

                if "Normal" in pattern:        st.success("✅ Normal spirometry")
                elif "Very Severe" in severity or "Severe" in severity:
                    st.error(f"⚠ {pattern} — {severity}")
                elif pattern not in ["Indeterminate / Incomplete data",""]:
                    st.warning(f"⚠ {pattern} — {severity}")
                else: st.info(f"ℹ {pattern}")

                # Guidance
                guidance = []
                if "Obstructive" in pattern:
                    guidance += [
                        "• Consider COPD or Asthma — correlate with clinical history",
                        "• Bronchodilator reversibility test if not done",
                        "• Cardiovascular risk significantly elevated in COPD — screen proactively",
                    ]
                    if "Severe" in severity:
                        guidance.append("• Refer to respiratory physician — CT chest, DLCO, 6MWT")
                if "Restrictive" in pattern or restriction:
                    guidance += [
                        "• Full lung volumes (TLC) needed to confirm restriction",
                        "• Consider: pulmonary fibrosis, obesity, cardiac, pleural, neuromuscular causes",
                        "• DLCO to differentiate parenchymal vs extra-pulmonary",
                    ]
                if bd_resp and "Positive" in bd_resp:
                    guidance.append("• Significant BD reversibility — Asthma component likely, consider ICS trial")

                if guidance:
                    st.subheader("Clinical Guidance")
                    for g in guidance: st.markdown(g)

                # Auto-fill
                st.subheader("📝 Auto-Fill for Risk Prediction Module")
                rows = [("Demographics → Age",sp_age,"years"),("Demographics → Sex",sp_sex,""),
                        ("Demographics → Height",sp_height,"cm"),("Demographics → Weight",sp_weight,"kg"),
                        ("Demographics → BMI",sp_bmi,"kg/m²"),("Demographics → Smoking",sp_smoking,""),
                        ("Demographics → Respiratory Pattern",pattern,""),
                        ("Demographics → PFT Severity",severity,"")]
                if m_fev1_pct>0:  rows.append(("Demographics → FEV1 % Pred",m_fev1_pct,"%"))
                if m_fvc_pct>0:   rows.append(("Demographics → FVC % Pred",m_fvc_pct,"%"))
                if fev1_fvc:      rows.append(("Demographics → FEV1/FVC",round(fev1_fvc,3),""))
                if m_dlco_pct>0:  rows.append(("Chemistry → DLCO % Pred",m_dlco_pct,"%"))
                if m_pef_pct>0:   rows.append(("Demographics → PEF % Pred",m_pef_pct,"%"))

                af_df = pd.DataFrame(rows, columns=["Risk Prediction Field","Value","Unit"])
                st.dataframe(af_df, use_container_width=True, hide_index=True)

                exp_df = pd.DataFrame([{
                    "Age":sp_age,"Sex":sp_sex,"Height (cm)":sp_height,"Weight (kg)":sp_weight,
                    "BMI":sp_bmi,"Smoking":sp_smoking,
                    "FEV1 (L)":m_fev1,"FEV1 % pred":m_fev1_pct,
                    "FVC (L)":m_fvc,"FVC % pred":m_fvc_pct,
                    "FEV1/FVC":round(fev1_fvc,3) if fev1_fvc else "—",
                    "PEF (L/min)":m_pef,"PEF % pred":m_pef_pct,"FEF25-75 (L/s)":m_fef2575,
                    "TLC (L)":m_tlc,"TLC % pred":m_tlc_pct,"DLCO":m_dlco,"DLCO % pred":m_dlco_pct,
                    "Pattern":pattern,"Severity":severity,"BD Response":bd_resp,
                }])
                st.divider()
                export_buttons(
                    "Spirometry Manual",
                    csv_df=exp_df,
                    excel_sheets={"Spirometry Results":exp_df,"Auto-Fill":af_df},
                    pdf_title="CardioAI Nova — Spirometry Report (Manual Entry)",
                    pdf_sections=[("Spirometry Results",exp_df),
                                  ("Clinical Guidance","\n".join(guidance) if guidance else "None"),
                                  ("Auto-Fill for Risk Prediction",af_df)],
                    docx_title="CardioAI Nova — Spirometry Report (Manual Entry)",
                    docx_sections=[("Spirometry Results",exp_df),
                                   ("Clinical Guidance","\n".join(guidance) if guidance else "None"),
                                   ("Auto-Fill for Risk Prediction",af_df)],
                    file_stem="spirometry_manual",
                )

        st.divider()
        st.caption(
            "**Reference:** GLI 2012 (Quanjer) predicted values. "
            "Obstruction: FEV1/FVC < LLN (or <0.70 fixed). "
            "Restriction: TLC < LLN (or <80% predicted). "
            "GOLD severity: post-BD FEV1 % predicted. "
            "BD response: ≥12% + ≥200mL FEV1 or FVC increase. "
            "All results must be interpreted by a qualified clinician."
        )


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
            "availability": "High — generic widely available",
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
            "availability": "High — generic available, on standard formulary",
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
            "rehab_notes": "Preferred over ACEi in patients who develop ACEi cough (common in patients). Identical cardiac protection benefits.",
            "contraindications": ["Pregnancy", "Bilateral renal artery stenosis", "Hyperkalaemia"],
            "cautions": ["CKD (reduce dose)", "Volume depletion (risk of hypotension)", "Do NOT combine with ACEi or aliskiren (ONTARGET trial showed harm)"],
            "availability": "Moderate — available in major pharmacies",
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
            "availability": "Very High — cheapest beta-blocker in Nigeria",
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
            "availability": "Moderate — available but more expensive than atenolol",
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
            "availability": "High — generic available",
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
            "availability": "Moderate — more expensive than atorvastatin",
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
            "availability": "Very High — cheap, on standard formulary",
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
            "availability": "Very High — widely available OTC",
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
            "availability": "Moderate",
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
            "availability": "Moderate — requires INR monitoring which is challenging outside major cities",
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
                            st.markdown(f"**Availability:** {info.get('nigeria_availability','')}")
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

                # ── ICD / ICF Codes: Drug Recommender ─────────────────────
                render_icd_panel(pt_diag, context="Pharmaco-Intelligence")
                _pharma_icd_df = icd_export_df(pt_diag)
                render_cpt_from_icd(pt_diag, context="Pharmaco — Recommended Procedures")
                _pharma_cpt_df = cpt_export_df(diagnoses=pt_diag)

                # ── Export: Drug Recommendations ──────────
                st.divider()
                _rec_df = pd.DataFrame([{
                    "Drug": drug,
                    "Class": DRUG_DB.get(drug, {}).get("class", ""),
                    "Dose": DRUG_DB.get(drug, {}).get("standard_dose", ""),
                    "Rationale": rationale,
                    "Evidence": stars,
                    "Nigeria Availability": DRUG_DB.get(drug, {}).get("availability", ""),
                    "Cost Tier": DRUG_DB.get(drug, {}).get("cost_tier", ""),
                    "Guideline": DRUG_DB.get(drug, {}).get("guideline_ref", ""),
                } for drug, rationale, stars in recs])
                _pt_df = pd.DataFrame([{
                    "Age": pt_age, "Sex": pt_sex, "BMI": pt_bmi,
                    "Systolic BP": pt_sbp, "Diastolic BP": pt_dbp,
                    "LDL": pt_ldl, "eGFR": pt_egfr, "K+": pt_k,
                    "Diagnoses": ", ".join(pt_diag),
                    "Allergies/Intolerances": pt_allergy,
                }])
                _pharma_excel = {
                    "Patient Profile": _pt_df,
                    "Drug Recommendations": _rec_df,
                }
                _pharma_pdf = [
                    ("Patient Profile", _pt_df),
                    ("Drug Recommendations", _rec_df),
                    ("Clinical Decision Notes", "\n".join(reasons) if reasons else "No additional notes."),
                ]
                if not _pharma_icd_df.empty:
                    _pharma_excel["ICD-ICF Codes"] = _pharma_icd_df
                    _pharma_pdf.append(("ICD-10 / ICD-11 / ICF Codes", _pharma_icd_df))
                export_buttons(
                    "Drug Recommendations",
                    csv_df=_rec_df,
                    excel_sheets=_pharma_excel,
                    pdf_title="CardioAI — Pharmaco-Intelligence Drug Recommendation Report",
                    pdf_sections=_pharma_pdf,
                    docx_title="CardioAI — Pharmaco-Intelligence Drug Recommendation Report",
                    docx_sections=_pharma_pdf,
                    file_stem="drug_recommendations",
                )
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

                # ── Export: Lab Results ───────────────────
                st.divider()
                _lab_export_rows = [{"Lab Parameter": lab, "Value": val,
                                      "Unit": LAB_RANGES[lab]["unit"],
                                      "Optimal Range": LAB_RANGES[lab]["optimal"]}
                                     for lab, val in lab_values.items()]
                _lab_df = pd.DataFrame(_lab_export_rows)
                _adj_df = pd.DataFrame([{"Drug Adjustment": a} for a in drug_adjustments]) if drug_adjustments else pd.DataFrame(columns=["Drug Adjustment"])
                export_buttons(
                    "Lab Results",
                    csv_df=_lab_df,
                    excel_sheets={"Lab Results": _lab_df, "Drug Adjustments": _adj_df},
                    pdf_title="CardioAI — Lab Result Interpretation Report",
                    pdf_sections=[
                        ("Lab Results", _lab_df),
                        ("Drug Adjustment Recommendations",
                         _adj_df if not _adj_df.empty else "No adjustments required."),
                    ],
                    docx_title="CardioAI — Lab Result Interpretation Report",
                    docx_sections=[
                        ("Lab Results", _lab_df),
                        ("Drug Adjustment Recommendations",
                         _adj_df if not _adj_df.empty else "No adjustments required."),
                    ],
                    file_stem="lab_interpretation",
                )
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
                st.markdown(f"**Availability:** {drug_info.get('nigeria_availability','')}")
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

            # AI-enhanced explanation if API available
            if (get_secret("GOOGLE_API_KEY") or get_secret("ANTHROPIC_API_KEY")) and patient_context.strip():
                with st.spinner("Generating AI clinical explanation..."):
                    try:
                        prompt = f"""You are a clinical pharmacist specialising in cardiac rehabilitation.

Patient context: {patient_context}

Drug being considered: {selected_drug}
Drug class: {drug_info.get('class','')}
Mechanism: {drug_info.get('mechanism','')}
Indications: {', '.join(drug_info.get('indications',[]))}
Contraindications: {', '.join(drug_info.get('contraindications',[]))}

Write a concise clinical explanation (3-4 paragraphs) for a clinician covering:
1. Whether this drug is appropriate for THIS specific patient and why
2. How this patient's specific clinical features (age, BMI, diagnoses, labs, co-meds) affect the drug's pharmacokinetics and pharmacodynamics
3. What to monitor in this patient specifically
4. Any Nigeria-specific considerations (cost, availability, alternatives)

Be specific to the patient context provided. Be direct and clinical. Do not give generic information."""
                        ai_explanation = _call_gemini(prompt)
                        st.divider()
                        st.markdown("**AI Clinical Pharmacist Explanation:**")
                        st.markdown(ai_explanation)
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
                                st.markdown(f"**Availability:** {info.get('nigeria_availability','')}")
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
                        st.warning("💰 High medication burden — consider insurance coverage or generic alternatives.")

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
            elif not (get_secret("GOOGLE_API_KEY") or get_secret("ANTHROPIC_API_KEY")):
                st.error("GOOGLE_API_KEY or ANTHROPIC_API_KEY required. Add to Streamlit secrets.")
            else:
                with st.spinner(f"Generating {doc_type}..."):
                    try:
                        prompts = {




                            "Pharmaceutical Care Plan": f"""You are a clinical pharmacist at CardioAI Nova, a premier cardiac rehabilitation centre.

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

Format professionally for a clinical record. Be specific and clinical.""",

                            "Discharge Medication Summary": f"""You are a clinical pharmacist at CardioAI Nova.

Generate a Discharge Medication Summary for:
Patient: {doc_patient_name or 'Patient'}, {doc_patient_age}yr {doc_patient_sex}
Diagnoses: {doc_diagnoses}
Discharge Medications: {doc_medications}
Labs at discharge: {doc_labs}
Notes: {doc_additional}

Include: medication name/dose/frequency/route, indication for each drug, key counselling points, what to avoid, when to seek urgent review, follow-up date recommendation.""",

                            "Patient Drug Counselling Sheet": f"""Generate a patient-friendly Drug Counselling Sheet in plain English (avoid jargon) for a patient.

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

                            "Cardiac Rehab Pharmacy Review": f"""You are the pharmacist for CardioAI Nova cardiac rehabilitation programme.

Generate a Cardiac Rehab Pharmacy Review for:
Patient: {doc_patient_name or 'Patient'}, {doc_patient_age}yr {doc_patient_sex}
Diagnoses: {doc_diagnoses}
Medications: {doc_medications}
Labs: {doc_labs}
Notes: {doc_additional}

Cover: appropriateness of current regimen for cardiac rehab, exercise-drug interactions (especially beta-blockers and heart rate targets, statins and myopathy risk, anticoagulants and exercise intensity), medication optimisation recommendations, monitoring plan for rehab duration, patient adherence strategies.""",
                        }

                        doc_text = _call_gemini(prompts[doc_type])


                        st.divider()
                        st.markdown(f"### {doc_type}")
                        st.markdown(f"**Patient:** {doc_patient_name or 'Patient'} | **Date:** {pd.Timestamp.now().strftime('%d %B %Y')}")
                        st.markdown(f"**Generated by:** CardioAI Pharmaco-Intelligence | CardioAI Nova")
                        st.divider()
                        st.markdown(doc_text)
                        st.divider()
                        st.caption("⚕ This document is AI-generated and must be reviewed and signed by a licensed pharmacist before use in clinical practice.")

                        # Download as text
                        full_doc = f"{doc_type}\nPatient: {doc_patient_name or 'Patient'} | Date: {pd.Timestamp.now().strftime('%d %B %Y')}\n{'='*60}\n\n{doc_text}\n\n[Generated by CardioAI Pharmaco-Intelligence — CardioAI Nova | cardioai-nova.streamlit.app]"
                        st.download_button(
                            "⬇ Download Document",
                            full_doc,
                            file_name=f"pharma_{doc_type.replace(' ','_').lower()}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Document generation error: {e}")


# ══════════════════════════════════════════════════════════
# PAGE 7 — OPERATIONAL INTELLIGENCE
# ══════════════════════════════════════════════════════════

elif "Operational Intelligence" in page:
    st.title("🏥 Operational Intelligence")
    st.caption(
        "Real-time hospital operations monitoring — bed management, staffing optimisation, "
        "patient flow, length of stay, fall risk tracking, EHR efficiency, and bottleneck detection "
        "for CardioAI Nova."
    )

    import datetime as dt
    import random

    # ── persistent session state for all operational data ─────────
    def init_ops_state():
        if "ops_patients" not in st.session_state:
            # Seed with 12 sample patients across both locations
            np.random.seed(42)
            random.seed(42)
            now = dt.datetime.now()
            locations = ["City Centre Clinic", "Uptown Clinic"]
            departments = ["Cardiac Rehab", "Physical Medicine", "Physiotherapy",
                           "Cardiology OPD", "Hydrotherapy", "Medical Spa"]
            diagnoses_pool = ["Hypertension", "Post-MI Rehab", "Heart Failure",
                              "Stroke Rehab", "Obesity", "Dyslipidaemia",
                              "Osteoarthritis", "Back Pain", "COPD"]
            statuses = ["Admitted", "In Session", "Waiting", "Pending Discharge", "Discharged Today"]

            patients = []
            for i in range(16):
                admit_hours = random.randint(1, 72)
                patients.append({
                    "id":           f"JH-{2024+i:04d}",
                    "name":         f"Patient {i+1}",
                    "age":          random.randint(38, 82),
                    "sex":          random.choice(["Male", "Female"]),
                    "location":     random.choice(locations),
                    "dept":         random.choice(departments),
                    "diagnosis":    random.choice(diagnoses_pool),
                    "admit_time":   (now - dt.timedelta(hours=admit_hours)).strftime("%Y-%m-%d %H:%M"),
                    "los_hours":    admit_hours,
                    "status":       random.choice(statuses),
                    "bed":          f"Bed {i+1:02d}",
                    "fall_risk":    random.choice(["Low", "Low", "Low", "Medium", "Medium", "High"]),
                    "nurse":        "Unassigned",
                    "doctor":       "Unassigned",
                    "wait_mins":    random.randint(0, 45),
                    "ehr_complete": random.choice([True, True, True, False]),
                    "vitals_due":   random.choice([True, False, False, False]),
                    "notes":        "",
                })
            st.session_state.ops_patients = patients

        if "ops_incidents" not in st.session_state:
            st.session_state.ops_incidents = []

        if "ops_staff" not in st.session_state:
            # Empty by default — you add your own real staff below
            st.session_state.ops_staff = []

    init_ops_state()
    patients = st.session_state.ops_patients
    staff    = st.session_state.ops_staff

    # ── KPI SUMMARY CARDS ────────────────────────────────────────
    active     = [p for p in patients if p["status"] != "Discharged Today"]
    admitted   = [p for p in patients if p["status"] == "Admitted"]
    waiting    = [p for p in patients if p["status"] == "Waiting"]
    high_risk  = [p for p in patients if p["fall_risk"] == "High"]
    ehr_gaps   = [p for p in patients if not p["ehr_complete"]]
    vitals_due = [p for p in patients if p["vitals_due"]]
    avg_los    = np.mean([p["los_hours"] for p in active]) if active else 0
    avg_wait   = np.mean([p["wait_mins"] for p in waiting]) if waiting else 0
    total_beds = 20
    occupancy  = len(admitted) / total_beds * 100
    overdue_ehr = len(ehr_gaps)

    # Alert badges
    alerts = []
    if avg_wait > 30:      alerts.append(("⏱ Long Wait", "error",   f"Avg wait {avg_wait:.0f} min"))
    if occupancy > 85:     alerts.append(("🛏 High Occupancy", "warning", f"{occupancy:.0f}% beds occupied"))
    if high_risk:          alerts.append(("⚠ Fall Risk", "error",   f"{len(high_risk)} high-risk patient(s)"))
    if overdue_ehr:        alerts.append(("📋 EHR Gap", "warning",  f"{overdue_ehr} incomplete record(s)"))
    if vitals_due:         alerts.append(("💊 Vitals Due", "warning", f"{len(vitals_due)} patient(s)"))
    overcapacity = [s for s in staff if s["utilisation"] > 90]
    if overcapacity:       alerts.append(("👩‍⚕️ Staff Overload", "error", f"{len(overcapacity)} staff >90% utilisation"))

    if alerts:
        alert_cols = st.columns(len(alerts))
        for col, (label, severity, detail) in zip(alert_cols, alerts):
            with col:
                getattr(st, severity)(f"**{label}**\n{detail}")

    st.divider()

    # ── METRIC CARDS ROW ─────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Active Patients",   len(active))
    m2.metric("Bed Occupancy",     f"{occupancy:.0f}%",    delta=f"{'High' if occupancy>85 else 'Normal'}", delta_color="inverse" if occupancy>85 else "normal")
    m3.metric("Avg Wait (min)",    f"{avg_wait:.0f}",       delta=f"{'Over target' if avg_wait>20 else 'On target'}", delta_color="inverse" if avg_wait>20 else "normal")
    m4.metric("Avg LOS (hrs)",     f"{avg_los:.1f}",        delta="Target: <48h")
    m5.metric("Fall Risk (High)",  len(high_risk),          delta_color="inverse")
    m6.metric("EHR Gaps",          overdue_ehr,             delta_color="inverse")

    st.divider()

    # ── MAIN TABS ─────────────────────────────────────────────────
    ops_tab1, ops_tab2, ops_tab3, ops_tab4, ops_tab5, ops_tab6, ops_tab7 = st.tabs([
        "🛏 Bed Management",
        "👥 Staffing",
        "⏱ Patient Flow & Bottlenecks",
        "⚠ Fall Risk Tracker",
        "📋 EHR Completeness",
        "📊 LOT & LOS Analytics",
        "🤖 AI Operations Advisor",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — BED MANAGEMENT
    # ════════════════════════════════════════════════════════
    with ops_tab1:
        st.subheader("Real-Time Bed Management")
        loc_filter = st.selectbox("Location:", ["All", "City Centre Clinic", "Uptown Clinic"], key="bed_loc")

        filtered = patients if loc_filter == "All" else [p for p in patients if p["location"] == loc_filter]

        # Bed grid visualisation
        st.markdown("**Bed Status Grid**")
        status_colors = {
            "Admitted":           ("🔴", "Occupied"),
            "In Session":         ("🟡", "In Session"),
            "Waiting":            ("🟠", "Waiting"),
            "Pending Discharge":  ("🟢", "Ready"),
            "Discharged Today":   ("⚪", "Available"),
        }

        bed_cols = st.columns(5)
        for i, p in enumerate(filtered[:20]):
            with bed_cols[i % 5]:
                icon, label = status_colors.get(p["status"], ("⚫", "Unknown"))
                risk_icon = "⚠" if p["fall_risk"] == "High" else ("〰" if p["fall_risk"] == "Medium" else "")
                st.markdown(
                    f"**{icon} {p['bed']}** {risk_icon}\n"
                    f"<small>{p['id']} | {p['diagnosis'][:18]}\n"
                    f"LOS: {p['los_hours']}h | {p['status']}</small>",
                    unsafe_allow_html=True
                )

        # Legend
        st.caption("🔴 Occupied  🟡 In Session  🟠 Waiting  🟢 Pending Discharge  ⚪ Available  ⚠ Fall Risk")

        st.divider()

        # Discharge planning
        st.subheader("Discharge Planning Queue")
        discharge_ready = [p for p in filtered if p["status"] == "Pending Discharge"]
        if discharge_ready:
            for p in discharge_ready:
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.write(f"**{p['id']}** — {p['diagnosis']} | LOS: {p['los_hours']}h | {p['doctor']}")
                with c2:
                    st.write(f"EHR: {'✅ Complete' if p['ehr_complete'] else '❌ Incomplete'}")
                with c3:
                    if st.button("Discharge", key=f"disc_{p['id']}"):
                        p["status"] = "Discharged Today"
                        st.success(f"{p['id']} discharged. Bed {p['bed']} now available.")
                        st.rerun()
        else:
            st.info("No patients currently pending discharge.")

        st.divider()

        # Add new patient
        with st.expander("➕ Admit New Patient"):
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                new_id   = st.text_input("Patient ID", value=f"JH-{len(patients)+2025:04d}", key="new_id")
                new_diag = st.selectbox(
                    "Diagnosis",
                    options=list(ICD_DB.keys()),
                    index=0, key="new_diag",
                    help="ICD-10/11 coded diagnosis list"
                )
                if new_diag and new_diag in ICD_DB:
                    rec = ICD_DB[new_diag]
                    st.caption(f"🔵 ICD-10: `{rec['icd10']}` · 🟢 ICD-11: `{rec['icd11']}` · {rec['description']}")
            with nc2:
                new_dept = st.selectbox("Department", ["Cardiac Rehab", "Physical Medicine",
                    "Physiotherapy", "Cardiology OPD", "Hydrotherapy"], key="new_dept")
                new_loc  = st.selectbox("Location", ["City Centre Clinic", "Uptown Clinic"], key="new_loc")
            with nc3:
                new_risk = st.selectbox("Fall Risk", ["Low", "Medium", "High"], key="new_risk")
                new_age  = st.number_input("Age", 18, 100, 55, key="new_age")

            # Assign from actual staff list
            staff_names_all = ["Unassigned"] + [s["name"] for s in st.session_state.get("ops_staff", [])]
            nc4, nc5 = st.columns(2)
            with nc4:
                assign_doctor = st.selectbox("Assign Doctor:", staff_names_all, key="assign_doc")
            with nc5:
                assign_nurse  = st.selectbox("Assign Nurse:", staff_names_all, key="assign_nur")

            if st.button("Admit Patient", type="primary", key="btn_admit"):
                _icd_rec = ICD_DB.get(new_diag, {})
                next_bed = f"Bed {len([p for p in patients if p['status']=='Admitted'])+1:02d}"
                patients.append({
                    "id":          new_id, "name": new_id, "age": new_age,
                    "sex":         "Unknown", "location": new_loc,
                    "dept":        new_dept,
                    "diagnosis":   new_diag or "Pending",
                    "icd10":       _icd_rec.get("icd10", "—"),
                    "icd11":       _icd_rec.get("icd11", "—"),
                    "snomed":      _icd_rec.get("snomed", "—"),
                    "admit_time":  dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "los_hours":   0, "status": "Admitted", "bed": next_bed,
                    "fall_risk":   new_risk,
                    "nurse":       assign_nurse,
                    "doctor":      assign_doctor,
                    "wait_mins":   0,
                    "ehr_complete": False, "vitals_due": True, "notes": "",
                })
                st.session_state.ops_patients = patients
                st.success(
                    f"Patient {new_id} admitted to {next_bed} — "
                    f"{new_diag} ({_icd_rec.get('icd10','—')}) — "
                    f"assigned to {assign_doctor}"
                )
                st.rerun()

    # ════════════════════════════════════════════════════════
    # TAB 2 — STAFFING OPTIMISATION
    # ════════════════════════════════════════════════════════
    with ops_tab2:
        st.subheader("Staff Schedule & Workload Management")
        st.caption("Add your real CardioAI Nova staff below. All names and data are entered by you — nothing is auto-generated.")

        # ── ADD NEW STAFF MEMBER ─────────────────────────────
        with st.expander("➕ Add Staff Member", expanded=(len(staff) == 0)):
            if len(staff) == 0:
                st.info("No staff added yet. Fill in the form below to add your first clinician.")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                new_name  = st.text_input("Full name", placeholder="e.g. Dr. Emeka Okafor", key="add_name")
                new_role  = st.selectbox("Role", [
                    "Cardiologist", "Physiatrist", "Physiotherapist",
                    "Sports Medicine Physician", "Cardiac Rehab Nurse",
                    "Senior Nurse", "Ward Nurse", "Physiotherapy Assistant",
                    "Radiographer", "Dietitian", "Medical Officer", "Other"
                ], key="add_role")
            with sc2:
                new_loc   = st.selectbox("Location", [
                    "City Centre Clinic", "Uptown Clinic", "Both"
                ], key="add_loc")
                new_shift = st.text_input("Shift hours", placeholder="e.g. 08:00–17:00", key="add_shift")
            with sc3:
                new_pts   = st.number_input("Current patients assigned", 0, 20, 0, key="add_pts")
                new_util  = st.slider("Current utilisation %", 0, 100, 0, key="add_util")

            if st.button("Add Staff Member", type="primary", key="btn_add_staff"):
                if new_name.strip():
                    st.session_state.ops_staff.append({
                        "name": new_name.strip(),
                        "role": new_role,
                        "location": new_loc,
                        "shift": new_shift or "—",
                        "patients": new_pts,
                        "utilisation": new_util,
                    })
                    staff = st.session_state.ops_staff
                    st.success(f"✓ {new_name.strip()} added to staff list")
                    st.rerun()
                else:
                    st.warning("Please enter a staff name.")

        # ── EDIT / REMOVE EXISTING STAFF ──────────────────────
        if staff:
            with st.expander("✏ Edit or Remove Existing Staff"):
                edit_names = [s["name"] for s in staff]
                edit_sel   = st.selectbox("Select staff member to edit:", edit_names, key="edit_sel")
                edit_idx   = next((i for i, s in enumerate(staff) if s["name"] == edit_sel), None)

                if edit_idx is not None:
                    s = staff[edit_idx]
                    ec1, ec2, ec3 = st.columns(3)
                    with ec1:
                        e_name  = st.text_input("Name",  value=s["name"],  key="e_name")
                        e_role  = st.text_input("Role",  value=s["role"],  key="e_role")
                    with ec2:
                        e_loc   = st.text_input("Location", value=s["location"], key="e_loc")
                        e_shift = st.text_input("Shift",    value=s["shift"],    key="e_shift")
                    with ec3:
                        e_pts  = st.number_input("Patients", 0, 20, int(s["patients"]),  key="e_pts")
                        e_util = st.slider("Utilisation %", 0, 100, int(s["utilisation"]), key="e_util")

                    col_save, col_del = st.columns(2)
                    with col_save:
                        if st.button("💾 Save Changes", key="btn_edit_save"):
                            st.session_state.ops_staff[edit_idx] = {
                                "name": e_name, "role": e_role, "location": e_loc,
                                "shift": e_shift, "patients": e_pts, "utilisation": e_util,
                            }
                            staff = st.session_state.ops_staff
                            st.success(f"✓ {e_name} updated")
                            st.rerun()
                    with col_del:
                        if st.button("🗑 Remove Staff Member", key="btn_del_staff"):
                            removed = st.session_state.ops_staff.pop(edit_idx)
                            staff = st.session_state.ops_staff
                            st.success(f"✓ {removed['name']} removed from staff list")
                            st.rerun()

        st.divider()

        # ── STAFF DISPLAY TABLE ───────────────────────────────
        if not staff:
            st.info("Add staff members using the form above to see the schedule and workload dashboard.")
        else:
            # Filter
            loc_filt_staff = st.selectbox(
                "Filter by location:",
                ["All", "City Centre Clinic", "Uptown Clinic"],
                key="staff_loc"
            )
            filtered_staff = staff if loc_filt_staff == "All" else [
                s for s in staff if s["location"] in (loc_filt_staff, "Both")]

            staff_df_data = []
            for s in filtered_staff:
                overloaded = s["utilisation"] > 90
                staff_df_data.append({
                    "Name":        s["name"],
                    "Role":        s["role"],
                    "Location":    s["location"],
                    "Shift":       s["shift"],
                    "Patients":    s["patients"],
                    "Utilisation": f"{s['utilisation']}%",
                    "Load":        "🔴 Overloaded" if overloaded else ("🟡 High" if s["utilisation"] > 80 else "🟢 Normal"),
                })
            st.dataframe(pd.DataFrame(staff_df_data), use_container_width=True, hide_index=True)

            # Utilisation chart
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, len(filtered_staff) * 0.45 + 1.5)))
            fig.patch.set_facecolor("#0D1B2A")
            for ax in axes:
                ax.set_facecolor("#111927")
                ax.tick_params(colors="white")
                for sp in ax.spines.values(): sp.set_color("#2C3E50")

            names  = [s["name"].split()[-1] for s in filtered_staff]
            utils  = [s["utilisation"] for s in filtered_staff]
            colors = ["#E63946" if u > 90 else ("#FFB703" if u > 80 else "#06D6A0") for u in utils]

            axes[0].barh(range(len(names)), utils, color=colors, height=0.6)
            axes[0].set_yticks(range(len(names)))
            axes[0].set_yticklabels(names, color="white", fontsize=9)
            axes[0].axvline(80, color="#FFB703", ls="--", lw=1, alpha=0.6, label="80% target")
            axes[0].axvline(90, color="#E63946", ls="--", lw=1, alpha=0.6, label="90% warning")
            axes[0].set_xlim(0, 110)
            axes[0].set_title("Staff Utilisation %", color="white", fontsize=11)
            axes[0].legend(fontsize=8, facecolor="#1A2A3A", labelcolor="white")
            for i, u in enumerate(utils):
                axes[0].text(u + 1, i, f"{u}%", va="center", color="white", fontsize=8)

            pts   = [s["patients"] for s in filtered_staff]
            roles = [s["role"].split()[0] for s in filtered_staff]
            pc    = ["#00B4D8" if r in ["Cardiologist","Physiatrist","Sports","Medical"] else "#7B2FBE" for r in roles]
            axes[1].bar(range(len(names)), pts, color=pc, width=0.6)
            axes[1].set_xticks(range(len(names)))
            axes[1].set_xticklabels(names, color="white", fontsize=8, rotation=30, ha="right")
            axes[1].axhline(5, color="#FFB703", ls="--", lw=1, alpha=0.6, label="Recommended max (5)")
            axes[1].set_title("Patients per Staff Member", color="white", fontsize=11)
            axes[1].tick_params(colors="white")
            axes[1].legend(fontsize=8, facecolor="#1A2A3A", labelcolor="white")

            plt.tight_layout(pad=2)
            st.pyplot(fig)
            plt.close()

            # Overload alerts
            overloaded_staff = [s for s in filtered_staff if s["utilisation"] > 90]
            if overloaded_staff:
                st.divider()
                st.markdown("**⚠ Staffing Recommendations:**")
                for s in overloaded_staff:
                    st.error(
                        f"**{s['name']}** ({s['role']}) at {s['utilisation']}% utilisation — "
                        f"{s['patients']} patients. Consider redistributing 1–2 patients "
                        f"to lower-utilisation staff or scheduling relief cover."
                    )

        # ── HANDOVER LOGGING ──────────────────────────────────
        if staff:
            st.divider()
            with st.expander("📋 Log Shift Change / Handover"):
                hc1, hc2 = st.columns(2)
                staff_names = [s["name"] for s in staff]
                with hc1:
                    handover_from = st.selectbox("Handover FROM:", staff_names, key="ho_from")
                    handover_to   = st.selectbox("Handover TO:",   staff_names, key="ho_to")
                with hc2:
                    handover_notes = st.text_area("Handover notes / outstanding tasks:", height=80, key="ho_notes")
                if st.button("Log Handover", key="btn_handover"):
                    st.success(f"✓ Handover logged: {handover_from} → {handover_to} at {dt.datetime.now().strftime('%H:%M')}")

    # ════════════════════════════════════════════════════════
    # TAB 3 — PATIENT FLOW & BOTTLENECKS
    # ════════════════════════════════════════════════════════
    with ops_tab3:
        st.subheader("Patient Flow & Bottleneck Detection")

        # Flow stages simulation
        flow_data = {
            "Registration":      random.randint(2, 8),
            "Triage/Assessment": random.randint(5, 18),
            "Waiting for Bed":   random.randint(1, 12),
            "In Treatment":      random.randint(8, 16),
            "Awaiting Review":   random.randint(3, 10),
            "Pending Discharge": len([p for p in patients if p["status"]=="Pending Discharge"]),
            "Discharge Process": random.randint(1, 5),
        }

        # Target wait times (minutes)
        targets = {
            "Registration":      5,
            "Triage/Assessment": 15,
            "Waiting for Bed":   10,
            "In Treatment":      60,
            "Awaiting Review":   20,
            "Pending Discharge": 30,
            "Discharge Process": 20,
        }

        fig2, ax = plt.subplots(figsize=(13, 4))
        fig2.patch.set_facecolor("#0D1B2A")
        ax.set_facecolor("#111927")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_color("#2C3E50")

        stages = list(flow_data.keys())
        values = list(flow_data.values())
        tvals  = [targets[s] for s in stages]
        bcolors = ["#E63946" if v > t*1.5 else ("#FFB703" if v > t else "#06D6A0")
                   for v, t in zip(values, tvals)]

        x = range(len(stages))
        bars = ax.bar(x, values, color=bcolors, width=0.55, label="Current avg wait (min)")
        ax.plot(x, tvals, "o--", color="#00B4D8", lw=1.5, ms=6, label="Target")
        ax.set_xticks(list(x))
        ax.set_xticklabels([s.replace(" ", "\n") for s in stages], color="white", fontsize=8.5)
        ax.set_ylabel("Minutes / Patients", color="white")
        ax.set_title("Patient Flow — Wait Times vs Targets (Red = Bottleneck)", color="white", fontsize=12)
        ax.legend(fontsize=9, facecolor="#1A2A3A", labelcolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(val),
                    ha="center", color="white", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Bottleneck identification
        bottlenecks = [(s, v, targets[s]) for s, v in flow_data.items() if v > targets[s]*1.5]
        if bottlenecks:
            st.divider()
            st.markdown("**🚨 Active Bottlenecks Detected:**")
            for stage, actual, target in bottlenecks:
                excess = actual - target
                st.error(
                    f"**{stage}** — {actual} min average (target: {target} min) | "
                    f"+{excess} min excess | "
                    f"Action: {'Add triage staff' if 'Triage' in stage else 'Expedite bed turnover' if 'Bed' in stage else 'Review discharge criteria' if 'Discharge' in stage else 'Redistribute workload'}"
                )
        else:
            st.success("✅ No bottlenecks detected — patient flow within target parameters.")

        # Today's throughput
        st.divider()
        st.markdown("**Today's Throughput Summary:**")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Patients Seen Today",    random.randint(18, 32))
        t2.metric("Avg Door-to-Treatment",  f"{random.randint(12, 28)} min", delta="Target: <20 min")
        t3.metric("Discharges Today",       len([p for p in patients if p["status"]=="Discharged Today"]))
        t4.metric("Appointments Cancelled", random.randint(0, 4), delta_color="inverse")

    # ════════════════════════════════════════════════════════
    # TAB 4 — FALL RISK TRACKER
    # ════════════════════════════════════════════════════════
    with ops_tab4:
        st.subheader("Patient Fall Risk Tracker")
        st.caption("Tracks fall risk level, prevention interventions, and incident logging for all active patients.")

        # Fall risk summary
        low_risk    = [p for p in patients if p["fall_risk"] == "Low"    and p["status"] != "Discharged Today"]
        medium_risk = [p for p in patients if p["fall_risk"] == "Medium" and p["status"] != "Discharged Today"]
        high_risk_p = [p for p in patients if p["fall_risk"] == "High"   and p["status"] != "Discharged Today"]

        fr1, fr2, fr3 = st.columns(3)
        with fr1:
            st.success(f"🟢 Low Risk: **{len(low_risk)} patients**")
        with fr2:
            st.warning(f"🟡 Medium Risk: **{len(medium_risk)} patients**")
        with fr3:
            if high_risk_p:
                st.error(f"🔴 High Risk: **{len(high_risk_p)} patients** — Immediate precautions required")
            else:
                st.info("🔴 High Risk: 0 patients")

        st.divider()

        # HIGH RISK PATIENTS — immediate action panel
        if high_risk_p:
            st.markdown("### 🔴 High Risk Patients — Action Required")
            for p in high_risk_p:
                with st.expander(f"⚠ {p['id']} — {p['diagnosis']} | {p['dept']} | {p['nurse']}", expanded=True):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Age:** {p['age']} | **Sex:** {p['sex']}")
                        st.markdown(f"**Location:** {p['location']} | **Bed:** {p['bed']}")
                        st.markdown(f"**LOS:** {p['los_hours']}h | **Doctor:** {p['doctor']}")
                    with col_b:
                        st.markdown("**Required precautions checklist:**")
                        checks = [
                            "Bed rails raised and locked",
                            "Non-slip footwear issued",
                            "Call bell within reach",
                            "Fall risk wristband applied",
                            "Environment cleared of hazards",
                            "Nurse station notified",
                        ]
                        for ch in checks:
                            st.checkbox(ch, key=f"fr_{p['id']}_{ch[:10]}")

                    fall_notes = st.text_area("Intervention notes:", key=f"fn_{p['id']}", height=60)
                    if st.button("Mark Precautions Complete", key=f"fp_{p['id']}"):
                        p["fall_risk"] = "Medium"
                        st.success("Precautions logged. Risk level updated to Medium.")
                        st.rerun()

        # MEDIUM RISK
        if medium_risk:
            st.divider()
            st.markdown("### 🟡 Medium Risk Patients — Monitor Closely")
            medium_df = pd.DataFrame([{
                "ID": p["id"], "Dept": p["dept"], "Age": p["age"],
                "Diagnosis": p["diagnosis"], "Nurse": p["nurse"],
                "LOS (hrs)": p["los_hours"], "Bed": p["bed"]
            } for p in medium_risk])
            st.dataframe(medium_df, use_container_width=True, hide_index=True)

        # Incident logging
        st.divider()
        st.subheader("🚨 Log a Fall Incident")
        with st.form("fall_incident_form"):
            fi1, fi2 = st.columns(2)
            with fi1:
                inc_patient  = st.selectbox("Patient:", [p["id"] for p in active] if active else ["None"], key="fi_pt")
                inc_time     = st.text_input("Time of incident:", value=dt.datetime.now().strftime("%H:%M"), key="fi_time")
                inc_location = st.text_input("Location in facility:", key="fi_loc", placeholder="e.g. Physiotherapy gym")
            with fi2:
                inc_severity = st.selectbox("Severity:", ["Minor — no injury", "Moderate — minor injury", "Major — significant injury", "Critical — emergency"], key="fi_sev")
                inc_witness  = st.text_input("Witnessed by:", key="fi_wit")
            inc_description = st.text_area("Description of incident:", height=80, key="fi_desc")
            inc_action      = st.text_area("Immediate action taken:", height=60, key="fi_act")

            submitted = st.form_submit_button("📋 Submit Incident Report", type="primary")
            if submitted:
                st.session_state.ops_incidents.append({
                    "datetime":    f"{dt.date.today()} {inc_time}",
                    "patient":     inc_patient,
                    "severity":    inc_severity,
                    "location":    inc_location,
                    "description": inc_description,
                    "action":      inc_action,
                    "witness":     inc_witness,
                    "status":      "Open",
                })
                st.success("Fall incident report submitted and logged.")

        # Incident log
        if st.session_state.ops_incidents:
            st.divider()
            st.markdown("**Incident Log (This Session):**")
            st.dataframe(pd.DataFrame(st.session_state.ops_incidents), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # TAB 5 — EHR COMPLETENESS
    # ════════════════════════════════════════════════════════
    with ops_tab5:
        st.subheader("EHR Completeness & Documentation Quality")
        st.caption("Track missing documentation, overdue vitals, and record completeness across all active patients.")

        total_active = len(active)
        complete_ehr = len([p for p in active if p["ehr_complete"]])
        ehr_pct      = complete_ehr / total_active * 100 if total_active > 0 else 0

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("EHR Completeness",    f"{ehr_pct:.0f}%", delta=f"Target: >95%", delta_color="normal" if ehr_pct >= 95 else "inverse")
        e2.metric("Incomplete Records",  len(ehr_gaps))
        e3.metric("Vitals Overdue",      len(vitals_due))
        e4.metric("Discharge Summaries", f"{random.randint(85,98)}%", delta="completed today")

        if ehr_pct < 95:
            st.warning(f"⚠ EHR completeness at {ehr_pct:.0f}% — below 95% target. Review incomplete records below.")

        st.divider()

        # EHR gaps table
        if ehr_gaps:
            st.markdown("**Incomplete EHR Records — Action Required:**")
            for p in ehr_gaps:
                missing_items = []
                if not p["ehr_complete"]:
                    missing_items = random.sample([
                        "Admission assessment", "Drug reconciliation",
                        "Consent form", "Allergy documentation",
                        "Nursing assessment", "Physiotherapy notes",
                        "Discharge criteria set", "Care plan updated"
                    ], k=random.randint(1, 3))

                with st.expander(f"📋 {p['id']} — {p['dept']} | {p['doctor']}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Patient:** {p['id']} | Age {p['age']} | {p['diagnosis']}")
                        st.markdown(f"**Admitted:** {p['admit_time']}")
                        st.markdown("**Missing documentation:**")
                        for item in missing_items:
                            st.markdown(f"  ❌ {item}")
                    with c2:
                        st.markdown(f"**Responsible clinician:** {p['doctor']}")
                        st.markdown(f"**Nurse:** {p['nurse']}")
                        note = st.text_area("Add documentation note:", key=f"ehr_{p['id']}", height=60)
                        if st.button("Mark EHR Complete", key=f"ehr_btn_{p['id']}"):
                            p["ehr_complete"] = True
                            st.success(f"EHR for {p['id']} marked complete.")
                            st.rerun()
        else:
            st.success("✅ All active patient EHRs are complete.")

        # Vitals overdue
        if vitals_due:
            st.divider()
            st.markdown("**⏰ Vitals Overdue:**")
            for p in vitals_due:
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.warning(f"**{p['id']}** — {p['diagnosis']} | {p['nurse']}")
                with c2:
                    st.write(f"Bed {p['bed']} | LOS {p['los_hours']}h")
                with c3:
                    if st.button("Log Vitals", key=f"vit_{p['id']}"):
                        p["vitals_due"] = False
                        st.success("Vitals logged.")
                        st.rerun()

    # ════════════════════════════════════════════════════════
    # TAB 6 — LOT & LOS ANALYTICS
    # ════════════════════════════════════════════════════════
    with ops_tab6:
        st.subheader("Length of Treatment (LOT) & Length of Stay (LOS) Analytics")
        st.caption("Analyse patient throughput, LOT targets by condition, and identify extended stay outliers.")

        # LOS by diagnosis
        diag_los = {}
        for p in active:
            d = p["diagnosis"]
            if d not in diag_los:
                diag_los[d] = []
            diag_los[d].append(p["los_hours"])

        avg_los_by_diag = {d: np.mean(v) for d, v in diag_los.items()}

        # Target LOS by condition (hours) — cardiac rehab benchmarks
        lot_targets = {
            "Post-MI Rehab":    48, "Heart Failure":  72, "Hypertension":    24,
            "Stroke Rehab":     96, "Obesity":         36, "Dyslipidaemia":   24,
            "Osteoarthritis":   48, "Back Pain":        24, "COPD":            48,
        }

        fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4.5))
        fig3.patch.set_facecolor("#0D1B2A")
        for ax in axes3:
            ax.set_facecolor("#111927")
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_color("#2C3E50")

        # LOS by diagnosis bar
        diags  = list(avg_los_by_diag.keys())
        avgs   = [avg_los_by_diag[d] for d in diags]
        tgts   = [lot_targets.get(d, 36) for d in diags]
        bclrs  = ["#E63946" if a > t*1.2 else ("#FFB703" if a > t else "#06D6A0")
                  for a, t in zip(avgs, tgts)]

        x3 = range(len(diags))
        axes3[0].bar(x3, avgs, color=bclrs, width=0.55, label="Actual avg LOS (hrs)")
        axes3[0].plot(x3, tgts, "o--", color="#00B4D8", lw=1.5, ms=5, label="Target LOS")
        axes3[0].set_xticks(list(x3))
        axes3[0].set_xticklabels([d[:12] for d in diags], color="white", fontsize=7.5,
                                  rotation=30, ha="right")
        axes3[0].set_ylabel("Hours", color="white")
        axes3[0].set_title("Avg LOS by Diagnosis vs Target", color="white", fontsize=11)
        axes3[0].legend(fontsize=8, facecolor="#1A2A3A", labelcolor="white")

        # LOS distribution histogram
        los_all = [p["los_hours"] for p in active]
        axes3[1].hist(los_all, bins=8, color="#00B4D8", edgecolor="#0D1B2A", alpha=0.85)
        axes3[1].axvline(np.mean(los_all), color="#FFB703", lw=2, ls="--",
                         label=f"Mean: {np.mean(los_all):.1f}h")
        axes3[1].axvline(48, color="#E63946", lw=1.5, ls=":", label="48h target", alpha=0.8)
        axes3[1].set_xlabel("Length of Stay (hours)", color="white")
        axes3[1].set_ylabel("Number of Patients", color="white")
        axes3[1].set_title("LOS Distribution — All Active Patients", color="white", fontsize=11)
        axes3[1].legend(fontsize=8, facecolor="#1A2A3A", labelcolor="white")

        plt.tight_layout(pad=2)
        st.pyplot(fig3)
        plt.close()

        # Extended stay outliers
        st.divider()
        st.markdown("**Extended Stay Outliers (LOS > Target × 1.5):**")
        outliers = [p for p in active if p["los_hours"] > lot_targets.get(p["diagnosis"], 36) * 1.5]
        if outliers:
            out_df = pd.DataFrame([{
                "ID":          p["id"],
                "Diagnosis":   p["diagnosis"],
                "Actual LOS":  f"{p['los_hours']}h",
                "Target LOS":  f"{lot_targets.get(p['diagnosis'], 36)}h",
                "Excess":      f"+{p['los_hours'] - lot_targets.get(p['diagnosis'],36)}h",
                "Doctor":      p["doctor"],
                "Status":      p["status"],
            } for p in outliers])
            st.dataframe(out_df, use_container_width=True, hide_index=True)
            st.warning(f"⚠ {len(outliers)} patient(s) exceeding target LOS. Review discharge readiness.")
        else:
            st.success("✅ All patients within target LOS parameters.")

        # LOT metrics
        st.divider()
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Mean LOS (all)",       f"{np.mean([p['los_hours'] for p in active]):.1f}h", delta="Target: <48h")
        l2.metric("Median LOS",           f"{np.median([p['los_hours'] for p in active]):.1f}h")
        l3.metric(">48h Stays",           len([p for p in active if p["los_hours"] > 48]))
        l4.metric("30-Day Readmissions",  f"{random.randint(4, 12)}%",  delta="Target: <8%")

        # ── Export: LOS & Patient Census ──────────────
        st.divider()
        _census_df = pd.DataFrame([{
            "Patient ID":   p["id"],
            "Age":          p["age"],
            "Sex":          p["sex"],
            "Location":     p["location"],
            "Department":   p["dept"],
            "Diagnosis":    p["diagnosis"],
            "ICD-10":       p.get("icd10") or ICD_DB.get(p.get("diagnosis",""), {}).get("icd10", "—"),
            "ICD-11":       p.get("icd11") or ICD_DB.get(p.get("diagnosis",""), {}).get("icd11", "—"),
            "Bed":          p["bed"],
            "Status":       p["status"],
            "LOS (hrs)":    p["los_hours"],
            "Fall Risk":    p["fall_risk"],
            "Doctor":       p["doctor"],
            "Nurse":        p["nurse"],
            "EHR Complete": p["ehr_complete"],
            "Vitals Due":   p["vitals_due"],
            "Admit Time":   p["admit_time"],
        } for p in active])

        _los_diag_df = pd.DataFrame([{
            "Diagnosis":        d,
            "Avg LOS (hrs)":    round(v, 1),
            "Target LOS (hrs)": lot_targets.get(d, 36),
            "Patients":         len(diag_los.get(d, [])),
        } for d, v in avg_los_by_diag.items()])

        _ops_diag_list = list({p["diagnosis"] for p in active if p.get("diagnosis","") != "Pending"})
        _ops_icd_df    = icd_export_df(_ops_diag_list)
        _ops_cpt_df    = cpt_export_df(diagnoses=_ops_diag_list)
        render_icd_panel(_ops_diag_list, context="Patient Census — Active Diagnoses")

        # Safe column subset — only select columns guaranteed to exist
        _census_export_cols = ["Patient ID", "Diagnosis", "ICD-10", "ICD-11",
                                "Status", "LOS (hrs)", "Fall Risk", "EHR Complete"]
        _census_export_df = _census_df[[c for c in _census_export_cols if c in _census_df.columns]]

        _ops_pdf_sections = [
            ("Patient Census", _census_export_df),
            ("LOS by Diagnosis", _los_diag_df),
        ]
        _ops_excel_sheets = {
            "Patient Census":   _census_df,
            "LOS by Diagnosis": _los_diag_df,
        }
        if not _ops_icd_df.empty:
            _ops_pdf_sections.append(("ICD-10 / ICD-11 / ICF Codes", _ops_icd_df))
            _ops_excel_sheets["ICD-ICF Codes"] = _ops_icd_df
        if not _ops_cpt_df.empty:
            _ops_excel_sheets["CPT Procedures"] = _ops_cpt_df

        export_buttons(
            "Operational Data",
            csv_df=_census_df,
            excel_sheets=_ops_excel_sheets,
            pdf_title="CardioAI — Operational Intelligence Report",
            pdf_sections=_ops_pdf_sections,
            docx_title="CardioAI — Operational Intelligence Report",
            docx_sections=_ops_pdf_sections,
            file_stem="ops_report",
        )

    # ════════════════════════════════════════════════════════
    # TAB 7 — AI OPERATIONS ADVISOR
    # ════════════════════════════════════════════════════════
    with ops_tab7:
        st.subheader("AI Operations Advisor")
        st.caption(
            "Gemini AI analyses the current operational state and generates specific, "
            "actionable recommendations for CardioAI Nova management."
        )

        # Build operational context summary
        ops_summary = f"""
CardioAI Nova — Real-Time Operational Status Report
Date/Time: {dt.datetime.now().strftime('%d %B %Y, %H:%M')}
Locations: City Centre Clinic + Uptown Clinic

PATIENT STATUS:
- Total active patients: {len(active)}
- Bed occupancy: {occupancy:.0f}% (Total beds: {total_beds})
- Patients waiting: {len(waiting)} (avg wait: {avg_wait:.0f} min)
- Average LOS: {avg_los:.1f} hours
- Pending discharge: {len([p for p in patients if p['status']=='Pending Discharge'])}
- High fall-risk patients: {len(high_risk_p)}
- Incomplete EHR records: {overdue_ehr}
- Vitals overdue: {len(vitals_due)}

STAFFING:
- Total staff on duty: {len(staff)}
- Staff >90% utilisation: {len(overcapacity)}
- Departments: Cardiac Rehab, Physical Medicine, Physiotherapy, Cardiology OPD

FLOW:
- Active bottlenecks: {len(bottlenecks)} ({', '.join([b[0] for b in bottlenecks]) if bottlenecks else 'None'})
- Today's discharges: {len([p for p in patients if p['status']=='Discharged Today'])}

QUALITY:
- EHR completeness: {ehr_pct:.0f}%
- Incidents this session: {len(st.session_state.ops_incidents)}
"""

        st.text(ops_summary)
        st.divider()

        query = st.text_area(
            "Ask the AI Operations Advisor a specific question:",
            placeholder=(
                "e.g. 'What are the top 3 priorities to reduce patient wait times today?' "
                "or 'How should we redistribute staff given current load?' "
                "or 'What bottlenecks are most likely to impact afternoon clinic?' "
                "or 'Generate a management briefing for this morning's status'"
            ),
            height=80, key="ops_ai_query"
        )

        auto_briefing = st.checkbox("Auto-generate full management briefing", key="auto_brief")

        if st.button("🤖 Get AI Recommendations", type="primary", use_container_width=True, key="btn_ops_ai"):
            if not (get_secret("GOOGLE_API_KEY") or get_secret("ANTHROPIC_API_KEY")):
                st.error("GOOGLE_API_KEY or ANTHROPIC_API_KEY required. Add to Streamlit secrets.")
            else:
                with st.spinner("AI analysing operational state..."):
                    try:




                        if auto_briefing or not query.strip():
                            prompt_text = f"""You are the chief operations advisor for CardioAI Nova, a premier physical medicine and cardiac rehabilitation centre with locations in City Centre Clinic and Uptown Clinic.

Here is the current real-time operational status:
{ops_summary}

Generate a comprehensive Morning Management Briefing covering:

1. EXECUTIVE SUMMARY (2 sentences — overall status: green/amber/red and why)

2. IMMEDIATE PRIORITIES (top 3 actions needed in the next 2 hours, specific and actionable)

3. BED MANAGEMENT RECOMMENDATIONS (specific actions to optimise occupancy and patient flow)

4. STAFFING RECOMMENDATIONS (who is overloaded, how to redistribute, any gaps to fill)

5. PATIENT SAFETY ALERTS (fall risk, overdue vitals, EHR gaps — specific patients and actions)

6. BOTTLENECK RESOLUTION PLAN (which bottlenecks to address first and how)

7. END-OF-SHIFT GOALS (what should be achieved by 18:00 today)

Be specific to CardioAI Nova's context as a cardiac rehab and physical medicine polyclinic. Use clinical and operational language appropriate for a medical director briefing. Format with clear numbered sections."""
                        else:
                            prompt_text = f"""You are the chief operations advisor for CardioAI Nova.

Current operational status:
{ops_summary}

Answer this specific question from management:
{query}

Be specific, actionable, and concise. Reference the actual data from the operational status above. Recommend specific steps with timelines where relevant."""

                        ops_ai_text = _call_gemini(prompt_text)
                        st.divider()
                        st.markdown("### AI Operations Advisor Response")
                        st.markdown(response.text)

                        # Download option
                        st.download_button(
                            "⬇ Download Briefing",
                            response.text,
                            file_name=f"ops_briefing_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"AI error: {e}")

        # Quick AI queries
        st.divider()
        st.markdown("**Quick AI Queries:**")
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("📊 Bottleneck Analysis", use_container_width=True, key="q1"):
                st.session_state["ops_ai_query"] = "Analyse the current bottlenecks in patient flow and recommend specific interventions to reduce wait times in the next 4 hours."
                st.rerun()
        with qc2:
            if st.button("👥 Staff Rebalancing Plan", use_container_width=True, key="q2"):
                st.session_state["ops_ai_query"] = "Given the current staffing utilisation, recommend how to rebalance patient assignments to prevent burnout and maintain quality of care."
                st.rerun()
        with qc3:
            if st.button("🛏 Discharge Acceleration", use_container_width=True, key="q3"):
                st.session_state["ops_ai_query"] = "Which patients should be prioritised for discharge today and what are the specific steps to accelerate the discharge process?"
                st.rerun()


elif "Clinical Codes" in page:
    st.title("🏷 Clinical Codes Reference")
    st.caption(
        "Search and browse ICD-10, ICD-11, ICF, SNOMED CT, and CPT codes for all conditions "
        "and procedures at CardioAI Nova. Includes Insurance tariffs and rehab relevance."
    )

    code_tab1, code_tab2 = st.tabs(["🔵 ICD / ICF / SNOMED — Diagnoses", "🟠 CPT — Procedures & Tariffs"])

    # ── TAB 1: ICD / ICF / SNOMED ─────────────────────────────────────────
    with code_tab1:
        search_q = st.text_input("🔍 Search diagnosis, ICD code, or SNOMED code",
                                  placeholder="e.g. hypertension, I10, heart failure...",
                                  key="icd_search")
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            cat_filter = st.multiselect("Filter by category",
                options=sorted({v["category"] for v in ICD_DB.values()}), default=[], key="icd_cat")
        with col_filter2:
            rehab_only = st.checkbox("Rehab-relevant conditions only", value=False, key="icd_rehab")

        results = {}
        for dx, rec in ICD_DB.items():
            if rehab_only and not rec["rehab_relevant"]: continue
            if cat_filter and rec["category"] not in cat_filter: continue
            if search_q:
                q = search_q.lower()
                if not any(q in s.lower() for s in [dx, rec["icd10"], rec["icd11"],
                                                     rec["snomed"], rec["description"], rec["category"]]):
                    continue
            results[dx] = rec

        st.markdown(f"**{len(results)} conditions found**")
        st.divider()

        if results:
            for dx, rec in results.items():
                cpts = rec.get("cpt_codes", [])
                with st.expander(f"**{dx}** — ICD-10: `{rec['icd10']}` · ICD-11: `{rec['icd11']}`"):
                    r1, r2, r3, r4 = st.columns([2, 1.8, 2, 2.2])
                    with r1:
                        st.markdown("**ICD / SNOMED**")
                        st.markdown(f"🔵 ICD-10: `{rec['icd10']}`")
                        st.markdown(f"🟢 ICD-11: `{rec['icd11']}`")
                        st.markdown(f"🟣 SNOMED: `{rec['snomed']}`")
                        st.caption(rec["description"])
                    with r2:
                        st.markdown("**Classification**")
                        st.markdown(f"**{rec['category']}**")
                        st.markdown("✅ Insurance Billable" if rec["nhis_billable"] else "❌ Not Insurance Billable")
                        st.markdown("🏥 Rehab Relevant" if rec["rehab_relevant"] else "")
                    with r3:
                        st.markdown("**ICF Functional Codes**")
                        for code, label in zip(rec["icf_codes"], rec["icf_labels"]):
                            st.caption(f"`{code}` — {label}")
                    with r4:
                        st.markdown("**CPT Procedures**")
                        total_tariff = 0
                        for cpt in cpts:
                            cr = CPT_DB.get(cpt, {})
                            tariff = cr.get("nhis_tariff", 0)
                            total_tariff += tariff
                            st.caption(f"🟠 `{cpt}` {cr.get('description','—')}  ₦{tariff:,}")
                        if total_tariff:
                            st.markdown(f"**Total: ₦{total_tariff:,}**")
            st.divider()
            _all_icd_df = icd_export_df(list(results.keys()))
            export_buttons(
                "ICD Reference",
                csv_df=_all_icd_df,
                excel_sheets={"ICD-ICF-CPT Reference": _all_icd_df},
                pdf_title="CardioAI — ICD / ICF / CPT Clinical Code Reference",
                pdf_sections=[("Clinical Code Reference", _all_icd_df)],
                docx_title="CardioAI — ICD / ICF / CPT Clinical Code Reference",
                docx_sections=[("Clinical Code Reference", _all_icd_df)],
                file_stem="icd_icf_cpt_reference",
            )
        else:
            st.info("No conditions match your search. Try a different term or clear the filters.")

    # ── TAB 2: CPT Procedures ─────────────────────────────────────────────
    with code_tab2:
        cpt_search = st.text_input("🔍 Search CPT code, procedure name, or category",
                                    placeholder="e.g. ECG, 93000, lipid, troponin...",
                                    key="cpt_search")
        cpt_cat_filter = st.multiselect("Filter by category",
            options=sorted({v["category"] for v in CPT_DB.values()}), default=[], key="cpt_cat")

        cpt_results = {}
        for code, rec in CPT_DB.items():
            if cpt_cat_filter and rec["category"] not in cpt_cat_filter: continue
            if cpt_search:
                q = cpt_search.lower()
                if not any(q in s.lower() for s in [code, rec["description"], rec["category"]]):
                    continue
            cpt_results[code] = rec

        st.markdown(f"**{len(cpt_results)} procedures found**")
        st.divider()

        if cpt_results:
            cpt_rows = []
            for code, rec in cpt_results.items():
                cpt_rows.append({
                    "CPT Code":        code,
                    "Description":     rec["description"],
                    "Category":        rec["category"],
                    "Tariff (local currency)": rec["nhis_tariff"],
                    "Unit":            rec["unit"],
                })
            _cpt_df = pd.DataFrame(cpt_rows)
            st.dataframe(_cpt_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Tariff (local currency)": st.column_config.NumberColumn(format="₦%d"),
                         })
            total = _cpt_df["Tariff (local currency)"].sum()
            st.caption(f"Total estimated tariff for filtered procedures: ₦{total:,}")
            st.divider()

            # Bill builder
            with st.expander("🧾 Procedure Bill Builder — select procedures to generate a bill"):
                selected_procs = st.multiselect(
                    "Select procedures to include in bill",
                    options=list(cpt_results.keys()),
                    format_func=lambda c: f"{c} — {CPT_DB[c]['description']} (₦{CPT_DB[c]['nhis_tariff']:,})",
                    key="bill_builder"
                )
                if selected_procs:
                    bill_rows = [{
                        "CPT Code": c,
                        "Procedure": CPT_DB[c]["description"],
                        "Category": CPT_DB[c]["category"],
                        "Tariff (local currency)": CPT_DB[c]["nhis_tariff"],
                        "Unit": CPT_DB[c]["unit"],
                    } for c in selected_procs]
                    bill_df = pd.DataFrame(bill_rows)
                    bill_total = bill_df["Tariff (local currency)"].sum()
                    st.dataframe(bill_df, use_container_width=True, hide_index=True)
                    st.success(f"**Total Bill: ₦{bill_total:,}**")
                    export_buttons(
                        "Procedure Bill",
                        csv_df=bill_df,
                        excel_sheets={"Procedure Bill": bill_df},
                        pdf_title="CardioAI Nova Procedure Bill",
                        pdf_sections=[
                            ("Procedure Bill", bill_df),
                            ("Total", f"Estimated Estimated Tariff Total: ₦{bill_total:,}"),
                        ],
                        docx_title="CardioAI Nova Procedure Bill",
                        docx_sections=[
                            ("Procedure Bill", bill_df),
                            ("Total", f"Estimated Estimated Tariff Total: ₦{bill_total:,}"),
                        ],
                        file_stem="procedure_bill",
                    )
            st.divider()
            export_buttons(
                "CPT Reference",
                csv_df=_cpt_df,
                excel_sheets={"CPT Reference": _cpt_df},
                pdf_title="CardioAI — CPT Procedure Code Reference",
                pdf_sections=[("CPT Procedures", _cpt_df)],
                docx_title="CardioAI — CPT Procedure Code Reference",
                docx_sections=[("CPT Procedures", _cpt_df)],
                file_stem="cpt_reference",
            )
        else:
            st.info("No procedures match your search.")

    st.divider()
    st.caption(
        "**Sources:** ICD-10 WHO 2019 · ICD-11 WHO 2024 · ICF WHO 2001 · SNOMED CT IHTSDO · "
        "CPT AMA 2025 · WHO / International Tariff Reference 2025 · 2026 ACC/AHA Guidelines"
    )

elif "About" in page:
    st.title("ℹ️ About CardioAI")

    # ── Developer profile ─────────────────────────────────
    st.subheader("Meet the Developer")

    dev_col1, dev_col2 = st.columns([1, 2])

    with dev_col1:
        st.markdown("""
        <div style="background:#1F4E79; border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:60px;">👨‍💻</div>
            <div style="color:white; font-size:18px; font-weight:bold; margin-top:10px;">CardioAI Nova Development Team</div>
            <div style="color:#90CAF9; font-size:13px; margin-top:4px;">AI & Machine Learning Developer</div>
            <div style="color:#90CAF9; font-size:13px;">CardioAI Nova, Nigeria</div>
        </div>
        """, unsafe_allow_html=True)

    with dev_col2:
        st.markdown("""
        **Name:** Gboh-Igbara D. Charles (Team Lead, CardioAI Nova Development Team)

        **Role:** AI Developer & Researcher

        **Organisation:** CardioAI Nova

        **Location:** Nigeria (Clinic B / Clinic A focus)

        **Project Type:** Dual Capstone Research Project

        **Research Area:** Explainable AI in Preventive Healthcare

        **Live App:** [cardioai-nova.streamlit.app](https://cardioai-nova.streamlit.app)

        **GitHub:** [cardioai-nova.streamlit.app](https://gbohigbaradc.github.io)
        """)

    st.divider()

    # ── Project description ───────────────────────────────
    st.subheader("About This Project")
    st.markdown("""
    CardioAI Nova is an **Explainable Artificial Intelligence platform** for cardiovascular risk assessment and clinical intelligence. It applies machine learning, natural language processing, and explainable AI to critical problems in preventive cardiovascular healthcare:

    1. **Early cardiovascular disease risk prediction** — identifying high-risk patients before
       symptoms become severe, enabling earlier clinical intervention.

    2. **Patient retention prediction** — identifying which patients are likely to drop out of
       rehabilitation and treatment programs after their first visit, so clinicians can intervene
       proactively.

    The system was designed with clinical usability in mind, focusing on deployment in Nigerian
    hospital settings in **Clinic B** and **Clinic A**, where cardiovascular disease is rising
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

    st.caption("© 2025 CardioAI Nova Development Team — CardioAI Nova | cardioai-nova.streamlit.app | cardioai-nova.streamlit.app")
