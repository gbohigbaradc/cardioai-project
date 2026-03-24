# ============================================================
# MODULE 6: LLM INTERPRETATION LAYER
# File: 06_llm_interpretation.py
#
# WHAT THIS SCRIPT DOES:
# Converts raw ML model outputs (numbers, SHAP values) into
# natural language explanations a clinician can actually read.
#
# TWO MODES:
# Mode A — Rule-based (no API key needed):
#   Uses templates and logic to generate explanations.
#   Works offline, fully deterministic, no cost.
#   Best for: prototypes, demos, offline environments.
#
# Mode B — LLM-based (requires API key):
#   Sends prediction context to an LLM (Claude / GPT-4) and
#   gets a richly worded clinical interpretation back.
#   Best for: production deployments, richer text.
#
# OUTPUT:
# - A structured patient risk report (text + JSON)
# - outputs/patient_risk_report_sample.txt
# ============================================================

import json
import os
import joblib
import pandas as pd
import numpy as np

os.makedirs("outputs", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# PART A: RULE-BASED EXPLANATION ENGINE
# Works without any API — uses conditional logic + templates
# ══════════════════════════════════════════════════════════════════════════

class RuleBasedExplainer:
    """
    Generates clinical explanations from model predictions using
    threshold-based rules and template strings.

    This is the fallback mode — always available, no cost, deterministic.
    """

    # Clinical thresholds based on WHO / AHA guidelines
    THRESHOLDS = {
        "trestbps": {"normal": 120, "elevated": 130, "high": 140},
        "chol":     {"normal": 200, "borderline": 240, "high": 240},
        "thalach":  {"low_risk_min": 150},  # Low max HR can indicate poor fitness
        "oldpeak":  {"concerning": 2.0},
    }

    # Risk tier definitions
    RISK_TIERS = {
        (0.0, 0.30): ("LOW",      "green",  "✓"),
        (0.30, 0.60): ("MODERATE","amber",  "⚠"),
        (0.60, 0.80): ("HIGH",    "orange", "⚠⚠"),
        (0.80, 1.0):  ("VERY HIGH","red",   "✗✗"),
    }

    def get_risk_tier(self, probability):
        """Returns (tier_name, color, icon) for a given probability"""
        for (low, high), (tier, color, icon) in self.RISK_TIERS.items():
            if low <= probability < high:
                return tier, color, icon
        return "VERY HIGH", "red", "✗✗"

    def explain_features(self, patient_data, shap_values, feature_names):
        """
        Generates bullet-point explanations for the top SHAP contributors.

        Parameters:
        - patient_data: dict of feature values for this patient
        - shap_values: array of SHAP values for this patient
        - feature_names: list of feature names
        """
        explanations = {"risk_increasing": [], "risk_reducing": [], "neutral": []}

        # Clinical explanations for each feature based on value + SHAP direction
        feature_explanations = {
            "age": lambda v, s: (
                f"Age {int(v)} years — {'older age is a non-modifiable risk factor' if v > 55 else 'younger age reduces baseline risk'}",
                "risk_increasing" if s > 0 else "risk_reducing"
            ),
            "trestbps": lambda v, s: (
                f"Resting blood pressure {int(v)} mmHg — {'elevated, suggesting arterial stress' if v >= 140 else 'within normal range'}",
                "risk_increasing" if s > 0 else "risk_reducing"
            ),
            "chol": lambda v, s: (
                f"Cholesterol {int(v)} mg/dl — {'high, associated with plaque formation' if v >= 240 else 'borderline' if v >= 200 else 'normal range'}",
                "risk_increasing" if s > 0 else "risk_reducing"
            ),
            "thalach": lambda v, s: (
                f"Maximum heart rate achieved: {int(v)} bpm — {'reduced exercise capacity may indicate cardiac stress' if v < 140 else 'adequate exercise capacity'}",
                "risk_increasing" if s > 0 else "risk_reducing"
            ),
            "exang": lambda v, s: (
                f"Exercise-induced angina: {'Present — chest pain during exertion is a significant cardiac symptom' if v == 1 else 'Absent — no chest pain on exertion'}",
                "risk_increasing" if s > 0 else "risk_reducing"
            ),
            "oldpeak": lambda v, s: (
                f"ST depression: {v:.1f} — {'significant ST changes may indicate myocardial ischemia' if v >= 2.0 else 'mild ST changes' if v > 0 else 'no ST depression'}",
                "risk_increasing" if s > 0 else "risk_reducing"
            ),
            "lifestyle_risk_index": lambda v, s: (
                f"Lifestyle Risk Index: {v:.2f}/1.0 — {'multiple lifestyle risk factors contributing to elevated risk' if v > 0.6 else 'moderate lifestyle risk' if v > 0.4 else 'relatively low lifestyle risk burden'}",
                "risk_increasing" if s > 0 else "risk_reducing"
            ),
        }

        # Sort features by |SHAP| to get most important first
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]

        for idx in sorted_idx[:6]:  # Top 6 contributors
            feat = feature_names[idx]
            shap_val = shap_values[idx]
            feat_val = patient_data.get(feat, 0)

            if feat in feature_explanations:
                try:
                    text, category = feature_explanations[feat](feat_val, shap_val)
                    explanations[category].append({
                        "feature": feat,
                        "explanation": text,
                        "shap_value": round(float(shap_val), 4)
                    })
                except Exception:
                    pass

        return explanations

    def generate_report(self, patient_id, risk_probability, patient_data,
                        shap_values, feature_names, retention_risk=None):
        """
        Generates a full patient risk report.

        Parameters:
        - patient_id: string identifier
        - risk_probability: float, cardiovascular risk probability
        - patient_data: dict of patient feature values
        - shap_values: SHAP values array for this patient
        - feature_names: list of feature names
        - retention_risk: optional float, dropout probability

        Returns:
        - dict containing structured report
        - formatted string for display
        """
        tier, color, icon = self.get_risk_tier(risk_probability)
        feature_explanations = self.explain_features(patient_data, shap_values, feature_names)

        # Build recommendation based on risk tier
        recommendations = self._get_recommendations(tier, patient_data)

        report = {
            "patient_id": patient_id,
            "cardiovascular_risk": {
                "probability": round(risk_probability, 3),
                "percentage": f"{risk_probability * 100:.1f}%",
                "risk_tier": tier,
                "color_code": color,
            },
            "key_risk_factors": feature_explanations["risk_increasing"],
            "protective_factors": feature_explanations["risk_reducing"],
            "recommendations": recommendations,
            "retention_risk": {
                "dropout_probability": round(retention_risk, 3) if retention_risk else None,
                "retention_flag": "HIGH DROPOUT RISK" if retention_risk and retention_risk > 0.6 else "STABLE"
            }
        }

        # Format as readable text report
        formatted = self._format_report(report, icon)
        return report, formatted

    def _get_recommendations(self, tier, patient_data):
        """Returns actionable clinical recommendations based on risk tier"""
        base_recommendations = [
            "Schedule follow-up cardiovascular risk assessment in 3 months.",
            "Encourage patient to complete lifestyle modification questionnaire.",
        ]
        if tier in ["HIGH", "VERY HIGH"]:
            return [
                "URGENT: Refer to cardiologist for comprehensive evaluation.",
                "Order: Full lipid panel, HbA1c, ECG, and echocardiogram.",
                "Initiate or review antihypertensive medication if BP ≥ 140/90.",
                "Prescribe structured cardiac rehabilitation program.",
                "Counsel on immediate lifestyle modifications: smoking cessation, diet, exercise.",
            ] + base_recommendations
        elif tier == "MODERATE":
            return [
                "Schedule cardiology consultation within 4-6 weeks.",
                "Repeat lipid panel and blood pressure monitoring in 4 weeks.",
                "Lifestyle counselling: diet review, supervised exercise program.",
                "Consider preventive aspirin therapy after physician review.",
            ] + base_recommendations
        else:
            return [
                "Continue annual cardiovascular risk screening.",
                "Reinforce healthy lifestyle habits.",
                "Monitor blood pressure and cholesterol annually.",
            ]

    def _format_report(self, report, icon):
        """Formats the report dict into a readable clinical text"""
        cv = report["cardiovascular_risk"]
        ret = report["retention_risk"]

        lines = [
            "=" * 65,
            f"  CARDIOVASCULAR RISK REPORT — Patient: {report['patient_id']}",
            "=" * 65,
            "",
            f"  {icon} RISK LEVEL: {cv['risk_tier']}",
            f"  Predicted Cardiovascular Risk Probability: {cv['percentage']}",
            "",
            "── KEY RISK FACTORS (driving risk UP) ──────────────────────",
        ]
        for item in report["key_risk_factors"]:
            lines.append(f"  ↑  {item['explanation']}")
            lines.append(f"     (SHAP contribution: +{item['shap_value']:.4f})")

        lines += ["", "── PROTECTIVE FACTORS (driving risk DOWN) ───────────────────"]
        for item in report["protective_factors"]:
            lines.append(f"  ↓  {item['explanation']}")
            lines.append(f"     (SHAP contribution: {item['shap_value']:.4f})")

        lines += ["", "── CLINICAL RECOMMENDATIONS ──────────────────────────────────"]
        for i, rec in enumerate(report["recommendations"], 1):
            lines.append(f"  {i}. {rec}")

        if ret["dropout_probability"] is not None:
            lines += [
                "",
                "── PATIENT RETENTION RISK ────────────────────────────────────",
                f"  Dropout Probability: {ret['dropout_probability']*100:.1f}%",
                f"  Status: {ret['retention_flag']}",
                "  Action: Contact patient proactively if dropout risk > 60%."
            ]

        lines += ["", "=" * 65,
                  "  ⚠ This report is a clinical DECISION SUPPORT TOOL only.",
                  "  Final clinical decisions remain with the treating physician.",
                  "=" * 65]

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# PART B: LLM-BASED EXPLANATION (requires API key)
# ══════════════════════════════════════════════════════════════════════════

def generate_llm_explanation(risk_probability, top_risk_factors, patient_summary,
                              api_provider="anthropic"):
    """
    Generates a natural language clinical explanation using an LLM.

    Parameters:
    - risk_probability: float (0-1)
    - top_risk_factors: list of dicts with feature and explanation
    - patient_summary: dict of key patient values
    - api_provider: "anthropic" or "openai"

    Returns:
    - string with LLM-generated explanation
    """

    # Build the prompt — this is what gets sent to the LLM
    # A good medical AI prompt:
    # 1. Sets the role/context
    # 2. Provides specific data
    # 3. Specifies the output format
    # 4. Includes safety disclaimers

    risk_factors_text = "\n".join([
        f"- {item['explanation']} (SHAP: {item['shap_value']:+.4f})"
        for item in top_risk_factors[:4]
    ])

    prompt = f"""You are a clinical decision support AI assisting a cardiologist.

A cardiovascular risk prediction model has assessed a patient with the following findings:

PREDICTED CARDIOVASCULAR RISK: {risk_probability * 100:.1f}%
RISK TIER: {"HIGH" if risk_probability > 0.6 else "MODERATE" if risk_probability > 0.3 else "LOW"}

TOP CONTRIBUTING FACTORS (from SHAP explainability analysis):
{risk_factors_text}

PATIENT SUMMARY:
- Age: {patient_summary.get("age", "unknown")}
- Resting BP: {patient_summary.get("trestbps", "unknown")} mmHg
- Cholesterol: {patient_summary.get("chol", "unknown")} mg/dl
- Max Heart Rate: {patient_summary.get("thalach", "unknown")} bpm

Please provide:
1. A 2-3 sentence plain-language summary of the risk assessment
2. The top 3 modifiable risk factors the clinician should address
3. Suggested next steps in plain clinical language

Keep the response concise (under 200 words) and use language suitable for
a clinician briefing. End with the standard disclaimer that this is a
decision support tool, not a diagnosis.
"""

    # ── Try Anthropic (Claude) first ──────────────────────────────────────
    if api_provider == "anthropic":
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        except Exception as e:
            print(f"  ⚠ Anthropic API unavailable: {e}")
            return None

    # ── Try OpenAI (GPT-4) as alternative ────────────────────────────────
    elif api_provider == "openai":
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a clinical decision support AI."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"  ⚠ OpenAI API unavailable: {e}")
            return None

    return None


# ══════════════════════════════════════════════════════════════════════════
# PART C: DEMONSTRATION RUN
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("LLM INTERPRETATION LAYER — DEMONSTRATION")
print("=" * 60)

# Create a sample patient prediction scenario
sample_patient = {
    "age": 62,
    "sex": 1,
    "trestbps": 148,
    "chol": 267,
    "fbs": 1,
    "thalach": 112,
    "exang": 1,
    "oldpeak": 2.4,
    "lifestyle_risk_index": 0.72
}

# Simulate SHAP values (in real run these come from 04_explainability.py)
sample_shap_values = np.array([0.12, 0.08, 0.21, 0.15, 0.09, -0.18, 0.17, 0.22, 0.25, 0.0])
sample_feature_names = list(sample_patient.keys()) + ["ca"]
sample_risk_prob = 0.78

# ── Generate rule-based report ────────────────────────────────────────────
print("\nGenerating rule-based report...")
explainer = RuleBasedExplainer()
report_dict, report_text = explainer.generate_report(
    patient_id="PT-00482",
    risk_probability=sample_risk_prob,
    patient_data=sample_patient,
    shap_values=sample_shap_values,
    feature_names=sample_feature_names,
    retention_risk=0.43
)

print("\n" + report_text)

# Save report
with open("outputs/patient_risk_report_sample.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
with open("outputs/patient_risk_report_sample.json", "w", encoding="utf-8") as f:
    json.dump(report_dict, f, indent=2, ensure_ascii=False)

print("\n✓ Report saved to outputs/patient_risk_report_sample.txt")

# ── Try LLM enhancement (optional) ───────────────────────────────────────
print("\nAttempting LLM enhancement (requires ANTHROPIC_API_KEY env variable)...")
llm_text = generate_llm_explanation(
    risk_probability=sample_risk_prob,
    top_risk_factors=report_dict["key_risk_factors"],
    patient_summary=sample_patient
)

if llm_text:
    print("\n── LLM-Enhanced Explanation ──")
    print(llm_text)
    with open("outputs/llm_enhanced_explanation.txt", "w", encoding="utf-8") as f:
        f.write(llm_text)
else:
    print("  LLM not available — rule-based report is the output.")
    print("  Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable LLM explanations.")

print("\n" + "=" * 60)
print("LLM INTERPRETATION LAYER COMPLETE")
print("=" * 60)
