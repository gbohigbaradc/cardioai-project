# ============================================================
# MODULE 2: EXPLORATORY DATA ANALYSIS (EDA)
# File: 02_eda.py
#
# WHAT THIS SCRIPT DOES:
# Before building any model, we need to UNDERSTAND our data.
# EDA helps us answer:
#   - What does the distribution of each feature look like?
#   - Are there patterns that separate disease from no-disease?
#   - Which features correlate with the target?
#   - Are there outliers or data quality issues we missed?
#
# Every chart and table in this script should be saved and
# included in your research report as evidence of data understanding.
#
# OUTPUT:
# - outputs/eda_target_distribution.png
# - outputs/eda_feature_distributions.png
# - outputs/eda_correlation_heatmap.png
# - outputs/eda_boxplots_by_target.png
# - outputs/eda_lifestyle_risk_by_target.png
# - outputs/eda_retention_factors.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# ── Plotting style settings ───────────────────────────────────────────────
# These settings make all charts look consistent and professional
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

COLORS = {
    "no_disease": "#2196F3",   # Blue for class 0
    "disease":    "#E53935",   # Red for class 1
    "neutral":    "#546E7A",   # Dark grey for neutral
    "accent":     "#FF6F00",   # Orange for highlights
}


# ══════════════════════════════════════════════════════════════════════════
# PART A: CARDIOVASCULAR DATASET EDA
# ══════════════════════════════════════════════════════════════════════════

print("Loading processed dataset...")
df = pd.read_csv("data/heart.csv")  # Use original for EDA (readable column values)

# Add the lifestyle risk index we created in preprocessing
def compute_lifestyle_risk(df):
    """Recompute the Lifestyle Risk Index for EDA purposes"""
    def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-8)
    return (
        norm(df["chol"]) * 0.25 +
        norm(df["trestbps"]) * 0.25 +
        df["fbs"] * 0.20 +
        df["exang"] * 0.15 +
        norm(df["oldpeak"]) * 0.15
    )

df["lifestyle_risk_index"] = compute_lifestyle_risk(df)


# ── Chart 1: Target Class Distribution ───────────────────────────────────
# Why: Confirms class balance. Important because imbalanced datasets
# require special handling (SMOTE, weighted loss functions).

print("Generating Chart 1: Target distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Cardiovascular Disease — Target Variable Distribution", fontsize=15, fontweight="bold")

# Bar chart
counts = df["target"].value_counts()
labels = ["No Disease (0)", "Disease Present (1)"]
colors = [COLORS["no_disease"], COLORS["disease"]]
bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
axes[0].set_title("Class Count")
axes[0].set_ylabel("Number of Patients")
for bar, count in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"n={count}", ha="center", fontsize=11, fontweight="bold")

# Pie chart
axes[1].pie(counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Class Proportion")

plt.tight_layout()
plt.savefig("outputs/eda_target_distribution.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/eda_target_distribution.png")


# ── Chart 2: Feature Distributions ───────────────────────────────────────
# Why: Reveals the shape of each variable (normal? skewed? bimodal?).
# Side-by-side histograms split by class show which features separate
# disease from no-disease patients visually.

print("Generating Chart 2: Feature distributions by class...")
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak", "lifestyle_risk_index"]
feature_labels = {
    "age": "Age (years)",
    "trestbps": "Resting Blood Pressure (mmHg)",
    "chol": "Cholesterol (mg/dl)",
    "thalach": "Max Heart Rate (bpm)",
    "oldpeak": "ST Depression",
    "lifestyle_risk_index": "Lifestyle Risk Index"
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()  # Convert 2D array of axes to 1D for easier looping

for i, col in enumerate(continuous_features):
    # Plot distribution for class 0 (no disease)
    axes[i].hist(
        df[df["target"] == 0][col],
        bins=30, alpha=0.6, color=COLORS["no_disease"],
        label="No Disease", edgecolor="white", linewidth=0.5
    )
    # Plot distribution for class 1 (disease) on same axes
    axes[i].hist(
        df[df["target"] == 1][col],
        bins=30, alpha=0.6, color=COLORS["disease"],
        label="Disease", edgecolor="white", linewidth=0.5
    )
    axes[i].set_title(feature_labels.get(col, col))
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    axes[i].legend(fontsize=9)

    # Add vertical lines for mean of each class
    mean_0 = df[df["target"] == 0][col].mean()
    mean_1 = df[df["target"] == 1][col].mean()
    axes[i].axvline(mean_0, color=COLORS["no_disease"], linestyle="--", linewidth=1.5, alpha=0.8)
    axes[i].axvline(mean_1, color=COLORS["disease"], linestyle="--", linewidth=1.5, alpha=0.8)

plt.suptitle("Feature Distributions by Target Class (dashed lines = class means)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/eda_feature_distributions.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/eda_feature_distributions.png")


# ── Chart 3: Correlation Heatmap ──────────────────────────────────────────
# Why: Reveals relationships between features.
# High correlation between two features = they carry redundant information.
# The "target" row shows which features are most linearly correlated
# with the outcome — useful for feature selection.

print("Generating Chart 3: Correlation heatmap...")
numeric_df = df.select_dtypes(include=[np.number])

# Compute Pearson correlation coefficients
# corr() returns a matrix where cell (i,j) is correlation between col i and col j
# Range: -1 (perfect negative) to +1 (perfect positive), 0 = no linear relationship
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # mask upper triangle (redundant)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,          # Show correlation numbers in each cell
    fmt=".2f",           # Round to 2 decimal places
    cmap="coolwarm",     # Red = positive, Blue = negative correlation
    vmin=-1, vmax=1,     # Fix scale so colors are consistent
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 8}
)
ax.set_title("Pearson Correlation Matrix — Cardiovascular Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/eda_correlation_heatmap.png")


# ── Chart 4: Boxplots by Target Class ────────────────────────────────────
# Why: Boxplots show median, spread, and outliers per class.
# If the two boxes don't overlap much = the feature is discriminative
# (helps distinguish disease from no-disease).
# Box = 25th to 75th percentile, Line = median, Whiskers = 1.5x IQR

print("Generating Chart 4: Boxplots by class...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(continuous_features):
    data_0 = df[df["target"] == 0][col]
    data_1 = df[df["target"] == 1][col]

    bp = axes[i].boxplot(
        [data_0, data_1],
        tick_labels=["No Disease", "Disease"],
        patch_artist=True,        # Fill boxes with color
        notch=False,
        widths=0.5
    )
    # Set colors
    bp["boxes"][0].set_facecolor(COLORS["no_disease"] + "80")   # 80 = 50% opacity in hex
    bp["boxes"][1].set_facecolor(COLORS["disease"] + "80")
    bp["medians"][0].set_color("navy")
    bp["medians"][1].set_color("darkred")

    axes[i].set_title(feature_labels.get(col, col))
    axes[i].set_ylabel("Value")

plt.suptitle("Feature Distribution by Target Class (Boxplots)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_boxplots_by_target.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/eda_boxplots_by_target.png")


# ── Chart 5: Lifestyle Risk Index Deep Dive ───────────────────────────────
# Why: This is our custom-engineered feature. We need to validate that
# it actually separates the two classes — if it doesn't, the index
# formula needs revisiting.

print("Generating Chart 5: Lifestyle Risk Index analysis...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Lifestyle Risk Index — Distribution and Discriminative Power",
             fontsize=14, fontweight="bold")

# Density plot split by class
for cls, label, color in [(0, "No Disease", COLORS["no_disease"]),
                           (1, "Disease", COLORS["disease"])]:
    values = df[df["target"] == cls]["lifestyle_risk_index"]
    axes[0].hist(values, bins=30, alpha=0.6, color=color, label=label,
                density=True, edgecolor="white")  # density=True normalizes to probability
axes[0].set_title("Distribution by Class")
axes[0].set_xlabel("Lifestyle Risk Index Score")
axes[0].set_ylabel("Density")
axes[0].legend()

# Mean index by target class
means = df.groupby("target")["lifestyle_risk_index"].mean()
axes[1].bar(["No Disease (0)", "Disease (1)"], means.values,
            color=[COLORS["no_disease"], COLORS["disease"]], width=0.5)
axes[1].set_title("Mean Index by Class")
axes[1].set_ylabel("Mean Lifestyle Risk Index")
for j, v in enumerate(means.values):
    axes[1].text(j, v + 0.005, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")

# Scatter: age vs lifestyle risk, colored by class
scatter_colors = df["target"].map({0: COLORS["no_disease"], 1: COLORS["disease"]})
axes[2].scatter(df["age"], df["lifestyle_risk_index"],
                c=scatter_colors, alpha=0.4, s=20)
axes[2].set_xlabel("Age")
axes[2].set_ylabel("Lifestyle Risk Index")
axes[2].set_title("Age vs Lifestyle Risk (Red=Disease)")

plt.tight_layout()
plt.savefig("outputs/eda_lifestyle_risk_by_target.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/eda_lifestyle_risk_by_target.png")


# ── Chart 6: Categorical Features vs Target ───────────────────────────────
# Why: For binary/categorical features, bar charts showing the proportion
# with disease by category value reveal which categories are high-risk.

print("Generating Chart 6: Categorical features vs target...")
categorical_features = ["sex", "fbs", "exang", "cp", "slope"]
cat_labels = {
    "sex": ("Female", "Male"),
    "fbs": ("Normal Sugar", "High Sugar"),
    "exang": ("No Exercise Angina", "Exercise Angina"),
    "cp": ("Typical Angina", "Atypical", "Non-Anginal", "Asymptomatic"),
    "slope": ("Downsloping", "Flat", "Upsloping")
}

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, col in enumerate(categorical_features):
    # Calculate % with disease for each category value
    disease_rate = df.groupby(col)["target"].mean() * 100
    axes[i].bar(range(len(disease_rate)), disease_rate.values,
                color=COLORS["disease"], alpha=0.8, width=0.6, edgecolor="white")
    axes[i].set_xticks(range(len(disease_rate)))
    labels = cat_labels.get(col, [str(x) for x in disease_rate.index])
    axes[i].set_xticklabels(labels[:len(disease_rate)], rotation=30, ha="right", fontsize=8)
    axes[i].set_title(col.upper())
    axes[i].set_ylabel("Disease Rate (%)")
    axes[i].set_ylim(0, 100)
    for j, v in enumerate(disease_rate.values):
        axes[i].text(j, v + 1, f"{v:.0f}%", ha="center", fontsize=9)

plt.suptitle("Disease Rate by Categorical Feature Value", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_categorical_vs_target.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/eda_categorical_vs_target.png")


# ══════════════════════════════════════════════════════════════════════════
# PART B: RETENTION DATASET EDA
# ══════════════════════════════════════════════════════════════════════════

print("\nLoading retention dataset...")
ret_df = pd.read_csv("data/retention_dataset.csv")

print("Generating Chart 7: Retention factors analysis...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
fig.suptitle("Patient Retention — Factor Analysis", fontsize=14, fontweight="bold")

retention_factors = [
    ("exercise_difficulty",   "Exercise Difficulty (1-5)"),
    ("questionnaire_burden",  "Questionnaire Burden (1-10)"),
    ("waiting_time_minutes",  "Waiting Time (minutes)"),
    ("travel_distance_km",    "Travel Distance (km)"),
    ("perceived_improvement", "Perceived Improvement (1-5)"),
    ("has_insurance",         "Has Insurance (0=No, 1=Yes)"),
]

for i, (col, label) in enumerate(retention_factors):
    for cls, clabel, color in [(0, "Dropped Out", "#E53935"),
                                (1, "Retained",   "#2196F3")]:
        axes[i].hist(ret_df[ret_df["retained"] == cls][col],
                    bins=20, alpha=0.6, color=color, label=clabel,
                    edgecolor="white", linewidth=0.5)
    axes[i].set_title(label, fontsize=10)
    axes[i].set_ylabel("Count")
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/eda_retention_factors.png", bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/eda_retention_factors.png")


# ── Print Statistical Summary ─────────────────────────────────────────────
# Numerical summary of key differences between disease/no-disease groups
print("\n── Statistical Summary: Cardiovascular Dataset ──")
summary = df.groupby("target")[continuous_features].mean().round(3)
summary.index = ["No Disease", "Disease"]
print(summary.to_string())

print("\n── Statistical Summary: Retention Dataset ──")
ret_summary = ret_df.groupby("retained")[
    [c for c, _ in retention_factors]
].mean().round(3)
ret_summary.index = ["Dropped Out", "Retained"]
print(ret_summary.to_string())

print("\n" + "=" * 60)
print("EDA COMPLETE — All charts saved to outputs/")
print("=" * 60)
