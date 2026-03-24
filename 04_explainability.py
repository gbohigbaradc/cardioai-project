# ============================================================
# MODULE 4: EXPLAINABLE AI — SHAP ANALYSIS
# File: 04_explainability.py
#
# WHAT THIS SCRIPT DOES:
# SHAP = SHapley Additive exPlanations.
# It answers: "WHY did the model predict X for THIS patient?"
#
# HOW SHAP WORKS (simplified):
# Imagine a game where each feature is a "player" contributing to a
# prediction. SHAP calculates each player's fair contribution by
# averaging their marginal contribution across all possible subsets
# of other players. This comes from game theory (Shapley values).
#
# SHAP gives us:
# - Global importance: which features matter MOST across ALL patients
# - Local explanation: for ONE patient, which features pushed the
#   prediction up or down
#
# OUTPUT:
# - outputs/shap_summary_bar.png     (global feature importance)
# - outputs/shap_summary_beeswarm.png (direction + magnitude)
# - outputs/shap_waterfall_patient.png (single patient explanation)
# - outputs/shap_dependence_plots.png  (feature interaction effects)
# - outputs/shap_retention_summary.png (retention model)
# ============================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ── Load data and models ──────────────────────────────────────────────────
print("Loading models and data...")
X_test  = pd.read_csv("data/X_test.csv")
y_test  = pd.read_csv("data/y_test.csv").squeeze()

# Load best cardiovascular model (we'll use XGBoost as primary — usually best)
best_xgb   = joblib.load("models/cardio_xgb.pkl")
best_rf    = joblib.load("models/cardio_rf.pkl")
scaler     = joblib.load("models/scaler.pkl")

# Friendly column names for charts (makes SHAP plots readable)
feature_name_map = {
    "age":                  "Age",
    "sex":                  "Sex (Male=1)",
    "trestbps":             "Resting BP (mmHg)",
    "chol":                 "Cholesterol (mg/dl)",
    "fbs":                  "Fasting Blood Sugar",
    "thalach":              "Max Heart Rate",
    "exang":                "Exercise Angina",
    "oldpeak":              "ST Depression",
    "ca":                   "Major Vessels",
    "lifestyle_risk_index": "Lifestyle Risk Index",
}

# Rename columns where mapping exists
display_cols = [feature_name_map.get(c, c) for c in X_test.columns]


# ══════════════════════════════════════════════════════════════════════════
# PART A: SHAP FOR CARDIOVASCULAR RISK (XGBoost)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("COMPUTING SHAP VALUES — Cardiovascular Risk (XGBoost)")
print("=" * 60)

# TreeExplainer is optimized for tree-based models (RF, XGBoost, GBM).
# It computes exact SHAP values much faster than the general KernelExplainer.
explainer_xgb = shap.TreeExplainer(best_xgb)

# shap_values is a matrix:
# Rows = patients, Columns = features
# Each cell = SHAP value for that feature for that patient
# Positive SHAP = pushed prediction toward class 1 (disease)
# Negative SHAP = pushed prediction toward class 0 (no disease)
print("Computing SHAP values (this may take 30-60 seconds)...")
shap_values_xgb = explainer_xgb.shap_values(X_test)

# For binary classification in XGBoost, shap_values is a 2D array
# where positive values = evidence FOR disease
if isinstance(shap_values_xgb, list):
    # Some models return [class0_shap, class1_shap] — take class 1
    shap_values_xgb = shap_values_xgb[1]

print(f"✓ SHAP values computed: {shap_values_xgb.shape}")
print(f"  Rows = patients ({X_test.shape[0]})")
print(f"  Cols = features ({X_test.shape[1]})")


# ── Chart 1: SHAP Summary Bar Plot ───────────────────────────────────────
# Shows mean absolute SHAP value per feature.
# Mean |SHAP| = average impact on model output magnitude.
# This is the most common way to show "global feature importance" in XAI.

print("\nGenerating SHAP summary bar plot...")
fig, ax = plt.subplots(figsize=(10, 7))

# Compute mean absolute SHAP per feature
mean_abs_shap = np.abs(shap_values_xgb).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=X_test.columns)
shap_importance = shap_importance.sort_values(ascending=True)

# Rename for display
shap_importance.index = [feature_name_map.get(c, c) for c in shap_importance.index]

colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(shap_importance)))
bars = ax.barh(shap_importance.index, shap_importance.values,
               color=colors, edgecolor="white", height=0.7)

for bar, val in zip(bars, shap_importance.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

ax.set_xlabel("Mean |SHAP Value| (Average Impact on Prediction)")
ax.set_title("Global Feature Importance — XGBoost Cardiovascular Model\n(SHAP-based)",
             fontsize=13, fontweight="bold")
ax.axvline(shap_importance.values.mean(), color="red", linestyle="--",
           alpha=0.5, label="Mean importance")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/shap_summary_bar.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_summary_bar.png")


# ── Chart 2: SHAP Beeswarm Plot ───────────────────────────────────────────
# Better than the bar chart — shows DIRECTION and MAGNITUDE.
# Each dot = one patient.
# Color = feature value (red=high, blue=low).
# X position = SHAP value (right = pushes toward disease, left = away)
# This reveals: "High cholesterol INCREASES disease risk" vs
#               "High max heart rate DECREASES disease risk"

print("Generating SHAP beeswarm plot...")
X_test_display = X_test.copy()
X_test_display.columns = display_cols

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values_xgb,
    X_test_display,
    plot_type="dot",       # beeswarm style
    max_display=15,        # show top 15 features
    show=False,
    color_bar_label="Feature Value (Low → High)"
)
plt.title("SHAP Beeswarm Plot — Direction and Magnitude of Feature Impact",
          fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("outputs/shap_summary_beeswarm.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_summary_beeswarm.png")


# ── Chart 3: Waterfall Plot — Single Patient Explanation ─────────────────
# The waterfall plot is the CLINICAL tool — it explains one individual
# patient's prediction. Perfect for showing a clinician WHY the model
# flagged this patient as high risk.
#
# It shows: base value (average prediction) + each feature's SHAP
# contribution stacked up to reach the final prediction.

print("Generating waterfall plot for sample patients...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("SHAP Waterfall — Individual Patient Explanations", fontsize=14, fontweight="bold")

# Find one high-risk patient and one low-risk patient to compare
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
high_risk_idx = np.argmax(y_pred_proba)      # patient with highest predicted risk
low_risk_idx  = np.argmin(y_pred_proba)       # patient with lowest predicted risk

for ax_idx, (patient_idx, title) in enumerate([
    (high_risk_idx, f"High-Risk Patient (Predicted Prob: {y_pred_proba[high_risk_idx]:.2f})"),
    (low_risk_idx,  f"Low-Risk Patient (Predicted Prob: {y_pred_proba[low_risk_idx]:.2f})")
]):
    patient_shap = shap_values_xgb[patient_idx]
    patient_features = X_test.iloc[patient_idx]
    feature_names_display = [feature_name_map.get(c, c) for c in X_test.columns]

    # Sort features by absolute SHAP value for this patient
    sorted_idx = np.argsort(np.abs(patient_shap))[::-1][:10]  # top 10
    top_shap   = patient_shap[sorted_idx]
    top_names  = [feature_names_display[i] for i in sorted_idx]
    top_values = patient_features.iloc[sorted_idx].values

    # Color bars: positive SHAP = red (increases risk), negative = blue (decreases risk)
    colors_wf = ["#E53935" if v > 0 else "#1E88E5" for v in top_shap]

    ax = axes[ax_idx]
    bars = ax.barh(range(len(top_shap)), top_shap[::-1],
                   color=colors_wf[::-1], edgecolor="white", height=0.7)
    ax.set_yticks(range(len(top_shap)))
    ax.set_yticklabels([f"{n}\n(val={v:.2f})" for n, v in
                        zip(top_names[::-1], top_values[::-1])], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (← reduces risk | increases risk →)")
    ax.set_title(title, fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/shap_waterfall_patient.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_waterfall_patient.png")


# ── Chart 4: SHAP Dependence Plots ────────────────────────────────────────
# Shows how SHAP value for one feature changes as its value changes.
# Reveals: "At what cholesterol level does risk start rising sharply?"
# Color = interaction feature (the feature that SHAP auto-detects
# as most interacting with the primary feature)

print("Generating SHAP dependence plots...")
top_features_idx = np.argsort(mean_abs_shap)[::-1][:4]  # top 4 features
top_feature_names = [X_test.columns[i] for i in top_features_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle("SHAP Dependence Plots — How Feature Value Affects Risk",
             fontsize=13, fontweight="bold")

for i, feature in enumerate(top_feature_names):
    feature_idx = list(X_test.columns).index(feature)
    display_name = feature_name_map.get(feature, feature)

    # Find the feature most correlated with this one (interaction proxy)
    # shap.approximate_interactions() was removed in SHAP 0.40+
    # We use Pearson correlation as a reliable, version-safe alternative
    correlations = np.abs([
        np.corrcoef(X_test.iloc[:, feature_idx].values, X_test.iloc[:, j].values)[0, 1]
        if j != feature_idx else 0
        for j in range(X_test.shape[1])
    ])
    interact_idx = int(np.argmax(correlations))
    interact_name = feature_name_map.get(X_test.columns[interact_idx], X_test.columns[interact_idx])

    x_vals = X_test.iloc[:, feature_idx].values
    y_vals = shap_values_xgb[:, feature_idx]
    c_vals = X_test.iloc[:, interact_idx].values

    sc = axes[i].scatter(x_vals, y_vals, c=c_vals, cmap="coolwarm",
                         alpha=0.6, s=20, edgecolors="none")
    plt.colorbar(sc, ax=axes[i], label=interact_name)
    axes[i].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[i].set_xlabel(display_name)
    axes[i].set_ylabel("SHAP Value")
    axes[i].set_title(f"{display_name} (colored by {interact_name})")

plt.tight_layout()
plt.savefig("outputs/shap_dependence_plots.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_dependence_plots.png")


# ══════════════════════════════════════════════════════════════════════════
# PART B: SHAP FOR PATIENT RETENTION (Random Forest)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("COMPUTING SHAP VALUES — Patient Retention (Random Forest)")
print("=" * 60)

X_ret_test = pd.read_csv("data/X_ret_test.csv")
y_ret_test = pd.read_csv("data/y_ret_test.csv").squeeze()
rf_ret = joblib.load("models/retention_rf.pkl")

explainer_ret = shap.TreeExplainer(rf_ret)
shap_values_ret = explainer_ret.shap_values(X_ret_test)

# Handle all SHAP output formats across different library versions:
# Older SHAP (<0.41): returns list [class0_array, class1_array]
# Newer SHAP (>=0.41): returns single 3D array (n_samples, n_features, n_classes)
# We always want class 1 (retained=1) as a clean 2D array (n_samples, n_features)
if isinstance(shap_values_ret, list):
    shap_values_ret = shap_values_ret[1]
elif hasattr(shap_values_ret, 'shape') and len(shap_values_ret.shape) == 3:
    shap_values_ret = shap_values_ret[:, :, 1]

print("Generating retention SHAP summary...")
retention_feature_map = {
    "exercise_difficulty":    "Exercise Difficulty",
    "questionnaire_burden":   "Questionnaire Burden",
    "waiting_time_minutes":   "Waiting Time (min)",
    "travel_distance_km":     "Travel Distance (km)",
    "previous_visits":        "Previous Visits",
    "missed_appointment":     "Missed Appointment",
    "has_insurance":          "Has Insurance",
    "perceived_improvement":  "Perceived Improvement",
    "age_group_encoded":      "Age Group",
    "visit_reason_encoded":   "Visit Reason"
}

X_ret_display = X_ret_test.copy()
X_ret_display.columns = [retention_feature_map.get(c, c) for c in X_ret_test.columns]

plt.figure(figsize=(12, 7))
shap.summary_plot(
    shap_values_ret,
    X_ret_display,
    plot_type="dot",
    max_display=10,
    show=False
)
plt.title("SHAP — Patient Retention Factors (Red=High Value, Blue=Low Value)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/shap_retention_summary.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/shap_retention_summary.png")

# ── Generate a clinical interpretation text ───────────────────────────────
# This is what the LLM module will later elaborate into natural language
mean_abs_ret = np.abs(shap_values_ret).mean(axis=0)
ret_importance = pd.Series(mean_abs_ret, index=X_ret_test.columns)
ret_importance = ret_importance.sort_values(ascending=False)

print("\n── Top Retention Risk Factors (Global SHAP Importance) ──")
for feat, val in ret_importance.head(5).items():
    display = retention_feature_map.get(feat, feat)
    print(f"  {display:30s}: {val:.4f}")

print("\n" + "=" * 60)
print("EXPLAINABILITY COMPLETE — All SHAP charts saved to outputs/")
print("=" * 60)
