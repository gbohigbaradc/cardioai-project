# ============================================================
# MODULE 1: DATA PREPROCESSING
# File: 01_preprocessing.py
#
# WHAT THIS SCRIPT DOES:
# This is the foundation of the entire project. Raw datasets
# are almost never clean — they have missing values, wrong
# data types, unbalanced classes, and features on different
# scales. This script fixes all of that before any ML model
# sees the data.
#
# DATASETS USED:
# - heart.csv (cardiovascular risk — 1025 rows, 14 features)
# - retention_dataset.csv (patient dropout — synthetic, created here)
#
# OUTPUT:
# - data/heart_processed.csv
# - data/retention_processed.csv
# - data/X_train.csv, X_test.csv, y_train.csv, y_test.csv
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
# SMOTE is applied inside 03_model_training.py on training folds only
# (not here, to avoid contaminating the saved X_train.csv with synthetic rows)
import joblib
import os

# ── Create output folder if it doesn't exist ──────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# PART A: CARDIOVASCULAR RISK DATASET (heart.csv)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART A: Loading and Preprocessing Cardiovascular Dataset")
print("=" * 60)

# ── Step 1: Load the data ─────────────────────────────────────────────────
# pd.read_csv reads a comma-separated file and turns it into a DataFrame
# (think of it like an Excel table in Python)
df = pd.read_csv("data/heart.csv")

print(f"\n✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names and data types:")
print(df.dtypes)

# ── Step 2: Understand what each column means ─────────────────────────────
# This is critical — you cannot preprocess what you don't understand.
# Here is the medical meaning of each UCI Heart Disease feature:
FEATURE_DESCRIPTIONS = {
    "age":      "Patient age in years",
    "sex":      "Sex: 1 = male, 0 = female",
    "cp":       "Chest pain type: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic",
    "trestbps": "Resting blood pressure in mmHg (normal: 120/80)",
    "chol":     "Serum cholesterol in mg/dl (normal: <200)",
    "fbs":      "Fasting blood sugar >120 mg/dl: 1=true, 0=false",
    "restecg":  "Resting ECG results: 0=normal, 1=ST abnormality, 2=left ventricular hypertrophy",
    "thalach":  "Maximum heart rate achieved during exercise",
    "exang":    "Exercise-induced angina: 1=yes, 0=no",
    "oldpeak":  "ST depression induced by exercise relative to rest",
    "slope":    "Slope of peak exercise ST segment: 0=downsloping, 1=flat, 2=upsloping",
    "ca":       "Number of major vessels (0-4) coloured by fluoroscopy",
    "thal":     "Thalassemia: 0=normal, 1=fixed defect, 2=normal, 3=reversible defect",
    "target":   "Heart disease: 1=disease present, 0=no disease (THIS IS WHAT WE PREDICT)"
}

for col, desc in FEATURE_DESCRIPTIONS.items():
    print(f"  {col:12s}: {desc}")


# ── Step 3: Check for missing values ─────────────────────────────────────
# Missing values appear as NaN (Not a Number). We need to know how many
# exist per column before deciding how to handle them.
print("\n── Missing Value Check ──")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
})
print(missing_report[missing_report["Missing Count"] > 0])

if missing.sum() == 0:
    print("✓ No missing values found — no imputation needed!")
else:
    # If there ARE missing values, we fill them:
    # - Numeric columns: fill with the median (middle value, robust to outliers)
    # - Categorical columns: fill with the most frequent value (mode)
    print("Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "target"]

    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    joblib.dump(imputer, "models/imputer.pkl")
    print("✓ Missing values filled with median")


# ── Step 4: Detect and handle outliers ───────────────────────────────────
# Outliers are extreme values that can confuse ML models.
# We use IQR (Interquartile Range) method:
#   - IQR = Q3 (75th percentile) - Q1 (25th percentile)
#   - Anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier
# We CLIP rather than delete — replace extreme values with the boundary value

print("\n── Outlier Detection and Clipping ──")
continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)  # clip replaces, not deletes
    print(f"  {col:12s}: {outlier_count} outliers clipped to [{lower:.1f}, {upper:.1f}]")


# ── Step 5: Create the Lifestyle Risk Index ───────────────────────────────
# This is one of the project's unique contributions. We combine multiple
# lifestyle/behavioral features into a single composite score.
# Formula: normalized weighted sum of risk factors.
# Higher score = higher lifestyle-driven cardiovascular risk.

print("\n── Creating Lifestyle Risk Index ──")

# Normalize each component to 0-1 range first
# (so that a column with values 0-200 doesn't dominate one with 0-1)
def normalize_col(series):
    """Scales a column to range [0, 1]"""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0  # avoid division by zero
    return (series - min_val) / (max_val - min_val)

# Each factor and its weight — higher weight = more important risk contributor
# These weights are based on clinical literature (AHA guidelines)
lifestyle_components = {
    "chol":     0.25,   # High cholesterol is a major risk factor
    "trestbps": 0.25,   # High blood pressure is the #1 modifiable risk factor
    "fbs":      0.20,   # Elevated fasting blood sugar → diabetes risk
    "exang":    0.15,   # Exercise-induced angina indicates heart stress
    "oldpeak":  0.15,   # ST depression indicates ischemia
}

# Compute the weighted index
lifestyle_risk_index = sum(
    normalize_col(df[col]) * weight
    for col, weight in lifestyle_components.items()
)

df["lifestyle_risk_index"] = lifestyle_risk_index.round(4)

print("✓ Lifestyle Risk Index created (range 0-1, higher = more risk)")
print(f"  Mean: {df['lifestyle_risk_index'].mean():.3f}")
print(f"  Std:  {df['lifestyle_risk_index'].std():.3f}")


# ── Step 6: Encode categorical variables ─────────────────────────────────
# ML models work with numbers, not text or categories.
# For columns like 'cp' (chest pain type) with values 0,1,2,3 that have
# NO natural order, we use one-hot encoding: create a new binary column
# for each category value.

print("\n── Encoding Categorical Variables ──")

# Columns that are categories even though stored as integers
categorical_cols = ["cp", "restecg", "slope", "thal"]

# pd.get_dummies creates one binary column per category
# drop_first=True removes one column to avoid multicollinearity
#   (if cp_1, cp_2, cp_3 are all 0, we know it's cp_0 — no need for cp_0 col)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"✓ One-hot encoding applied to: {categorical_cols}")
print(f"  New shape: {df.shape}")


# ── Step 7: Separate features (X) from target (y) ────────────────────────
# X = everything the model uses as input (features)
# y = what the model is trying to predict (target)

X = df.drop("target", axis=1)   # axis=1 means drop a column (not a row)
y = df["target"]

print(f"\n✓ Features (X): {X.shape[1]} columns")
print(f"✓ Target (y): {y.value_counts().to_dict()}")


# ── Step 8: Train/test split ──────────────────────────────────────────────
# We split data into training (80%) and testing (20%) sets.
# The model NEVER sees the test set during training — it's used only for
# final evaluation, like a mock exam before the real exam.
# stratify=y ensures both splits have the same class ratio

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% for testing
    random_state=42,      # random_state makes the split reproducible
    stratify=y            # maintain class balance in both splits
)

print(f"\n── Train/Test Split ──")
print(f"  Training set: {X_train.shape[0]} rows")
print(f"  Test set:     {X_test.shape[0]} rows")
print(f"  Train class ratio: {y_train.value_counts(normalize=True).round(3).to_dict()}")
print(f"  Test class ratio:  {y_test.value_counts(normalize=True).round(3).to_dict()}")


# ── Step 9: Feature Scaling ───────────────────────────────────────────────
# Many ML algorithms (Logistic Regression, SVM, Neural Networks) are
# sensitive to feature scale. Age (29-77) and cholesterol (126-564) are
# on very different scales — scaling puts them on the same footing.
#
# StandardScaler transforms each feature to: (value - mean) / std_deviation
# Result: mean=0, std=1 for every feature.
#
# CRITICAL RULE: Fit the scaler ONLY on training data.
# Then use it to transform BOTH train and test.
# Fitting on test data would "leak" information from the test set.

scaler = StandardScaler()

# FIX: Fit scaler on DataFrame (not numpy) so feature_names_in_ attribute
# is set — this is required by streamlit_app.py's scaler.feature_names_in_
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),   # fit + transform training only
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),         # ONLY transform test (no refit)
    columns=X_test.columns
)

# Save the scaler so we can use it on new patient data at prediction time
joblib.dump(scaler, "models/scaler.pkl")
print(f"\n✓ StandardScaler fitted on DataFrame — feature_names_in_ preserved")
print(f"  Scaler feature names: {list(scaler.feature_names_in_)}")


# ── Step 10: Handle class imbalance with SMOTE ────────────────────────────
# SMOTE = Synthetic Minority Oversampling Technique.
# Creates NEW synthetic samples of the minority class by interpolating
# between existing minority examples.
#
# IMPORTANT RULES:
# 1. Apply SMOTE ONLY to training data — never to test data
# 2. Apply SMOTE AFTER scaling (so synthetic points are in scaled space)
# 3. This dataset is nearly balanced (51/49) so SMOTE effect is small,
#    but we include it for demonstration and future imbalanced datasets.
#
# NOTE: We save the NON-SMOTE scaled data as X_train.csv so that
# cross-validation in 03_model_training.py is not contaminated.
# SMOTE is applied inside the training script on the training fold only.

print("\n── Class Balance Check ──")
print(f"  Class distribution: {y_train.value_counts().to_dict()}")
print(f"  Dataset is {'balanced' if y_train.value_counts().min() / y_train.value_counts().max() > 0.8 else 'imbalanced'}")
print(f"  SMOTE will be applied inside 03_model_training.py on training folds only")


# ── Step 11: Save processed data ─────────────────────────────────────────
# Save clean scaled data (no SMOTE contamination)
X_train_scaled.to_csv("data/X_train.csv", index=False)
X_test_scaled.to_csv("data/X_test.csv", index=False)
pd.Series(y_train.values, name="target").to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
df.to_csv("data/heart_processed.csv", index=False)

print("\n✓ All processed files saved to data/")


# ══════════════════════════════════════════════════════════════════════════
# PART B: PATIENT RETENTION DATASET (Synthetic)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART B: Creating Synthetic Patient Retention Dataset")
print("=" * 60)

# Since we don't have a real clinic dropout dataset, we generate a
# realistic synthetic one based on documented factors from healthcare
# operations research.
# np.random.seed ensures reproducibility — same numbers every run.

np.random.seed(42)
n = 800  # 800 simulated patient records

# ── Generate features ─────────────────────────────────────────────────────
# Each feature is drawn from a realistic distribution:

retention_data = pd.DataFrame({

    # How hard is the prescribed exercise program? (1=easy, 5=very hard)
    # Higher difficulty → patient more likely to drop out
    "exercise_difficulty":  np.random.randint(1, 6, n),

    # How many questionnaires/forms must the patient fill each visit? (1-10)
    # Higher burden → frustration → dropout
    "questionnaire_burden": np.random.randint(1, 11, n),

    # How long does the patient wait at the clinic? (minutes)
    # Drawn from a right-skewed distribution (most wait 10-30min, some much longer)
    "waiting_time_minutes": np.random.exponential(scale=20, size=n).clip(5, 90).astype(int),

    # Distance from home to clinic (km)
    "travel_distance_km":   np.random.exponential(scale=15, size=n).clip(1, 80).astype(int),

    # Number of previous visits before current assessment
    "previous_visits":      np.random.randint(0, 20, n),

    # Did the patient miss any appointment in the last 3 months? 1=yes, 0=no
    "missed_appointment":   np.random.choice([0, 1], n, p=[0.65, 0.35]),

    # Does the patient have insurance? 1=yes, 0=no
    # Uninsured patients more likely to drop out due to cost
    "has_insurance":        np.random.choice([0, 1], n, p=[0.30, 0.70]),

    # Patient age group bucket
    "age_group":            np.random.choice(
        ["18-30", "31-45", "46-60", "61+"], n,
        p=[0.15, 0.30, 0.35, 0.20]
    ),

    # Patient's self-reported perceived improvement (1=none, 5=significant)
    # Lower perceived improvement → dropout
    "perceived_improvement": np.random.randint(1, 6, n),

    # Primary reason for visiting clinic
    "visit_reason":         np.random.choice(
        ["cardiac_rehab", "hypertension", "diabetes", "weight_management", "general"],
        n, p=[0.25, 0.30, 0.20, 0.15, 0.10]
    ),
})

# ── Generate target: retained (1) or dropped out (0) ─────────────────────
# We use a realistic formula where:
# - High exercise difficulty increases dropout probability
# - High questionnaire burden increases dropout
# - Long waiting times increase dropout
# - Perceived improvement DECREASES dropout (patients who feel better stay)
# - Insurance DECREASES dropout (financial barrier removed)

# Compute a raw "dropout score" for each patient
dropout_score = (
    0.25 * (retention_data["exercise_difficulty"] / 5) +
    0.20 * (retention_data["questionnaire_burden"] / 10) +
    0.20 * (retention_data["waiting_time_minutes"] / 90) +
    0.15 * (retention_data["travel_distance_km"] / 80) +
    0.10 * retention_data["missed_appointment"] -
    0.15 * (retention_data["perceived_improvement"] / 5) -
    0.10 * retention_data["has_insurance"] +
    np.random.normal(0, 0.05, n)  # small random noise for realism
)

# Convert score to probability using sigmoid function
# sigmoid(x) = 1 / (1 + e^(-x)) — maps any value to range [0, 1]
dropout_prob = 1 / (1 + np.exp(-5 * (dropout_score - 0.5)))

# Assign dropout (1) or retained (0) based on probability
retention_data["retained"] = (np.random.random(n) > dropout_prob).astype(int)

print(f"✓ Synthetic dataset created: {retention_data.shape}")
print(f"  Retained (1): {retention_data['retained'].sum()} patients")
print(f"  Dropped out (0): {(retention_data['retained'] == 0).sum()} patients")

# ── Encode categorical columns ────────────────────────────────────────────
# age_group and visit_reason are text categories — convert to numbers
le_age = LabelEncoder()
le_visit = LabelEncoder()

retention_data["age_group_encoded"] = le_age.fit_transform(retention_data["age_group"])
retention_data["visit_reason_encoded"] = le_visit.fit_transform(retention_data["visit_reason"])

# Save encoders for use during prediction on new data
joblib.dump(le_age, "models/le_age.pkl")
joblib.dump(le_visit, "models/le_visit.pkl")

# Drop original text columns (model only works with numbers)
retention_data_encoded = retention_data.drop(["age_group", "visit_reason"], axis=1)

# ── Save retention dataset ────────────────────────────────────────────────
retention_data.to_csv("data/retention_dataset.csv", index=False)
retention_data_encoded.to_csv("data/retention_processed.csv", index=False)

print("✓ Retention dataset saved")
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE — All outputs saved to data/ and models/")
print("=" * 60)
