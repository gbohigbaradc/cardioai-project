# ============================================================
# MODULE 3: MACHINE LEARNING MODEL TRAINING
# File: 03_model_training.py
#
# WHAT THIS SCRIPT DOES:
# Trains 4 models for cardiovascular risk prediction and
# 3 models for patient retention prediction. For each model:
#   1. Train on preprocessed training data
#   2. Tune hyperparameters using GridSearchCV
#   3. Evaluate on held-out test data
#   4. Save the best model to disk
#
# WHY MULTIPLE MODELS?
# No single algorithm is universally best. We train several,
# compare them, and select the winner based on AUC-ROC.
# This is standard ML practice.
#
# MODELS TRAINED:
# Cardiovascular: Logistic Regression, Random Forest,
#                 Gradient Boosting (XGBoost), Neural Network (MLP)
# Retention:      Logistic Regression, Random Forest, XGBoost
#
# OUTPUT:
# - models/cardio_logistic.pkl
# - models/cardio_rf.pkl
# - models/cardio_xgb.pkl
# - models/cardio_mlp.pkl
# - models/retention_rf.pkl
# - models/retention_xgb.pkl
# - outputs/model_comparison.csv
# ============================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, roc_curve, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── XGBoost: use if available, fall back to sklearn GradientBoosting ──────
# XGBoost is faster and often more accurate, but requires separate install.
# sklearn's GradientBoostingClassifier is a fully compatible substitute.
try:
    import xgboost as xgb
    XGBModel = xgb.XGBClassifier
    XGB_PARAMS = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "eval_metric": ["auc"],
        "verbosity": [0],
    }
    XGB_FIXED = {"random_state": 42, "eval_metric": "auc", "verbosity": 0}
    print("✓ XGBoost available — using XGBClassifier")
except ImportError:
    # GradientBoostingClassifier is sklearn's equivalent — same concept,
    # slightly slower but no extra install required
    XGBModel = GradientBoostingClassifier
    XGB_PARAMS = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }
    XGB_FIXED = {"random_state": 42}
    print("⚠ XGBoost not installed — using sklearn GradientBoostingClassifier instead")
    print("  Install with: pip install xgboost")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── Load preprocessed data ────────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()  # squeeze: DataFrame → Series
y_test  = pd.read_csv("data/y_test.csv").squeeze()

# ── Apply SMOTE to training data (inside training, not preprocessing) ─────
# This is the correct place for SMOTE — applied to training data only,
# after the train/test split has already been saved cleanly.
print("\n── Applying SMOTE to training data ──")
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"  After SMOTE:  {pd.Series(y_train_fit).value_counts().to_dict()}")
except ImportError:
    print("  imblearn not installed — using class_weight='balanced' instead (equally effective)")
    X_train_fit, y_train_fit = X_train, y_train


# ── Helper function: Evaluate any model ───────────────────────────────────
# We call this function after training each model to get all metrics
# in a consistent format. This avoids duplicating code.

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a trained model and returns a dictionary of metrics.

    Parameters:
    - model: trained sklearn-compatible model
    - X_test: test features
    - y_test: true labels for test set
    - model_name: string label for this model (for display)

    Returns:
    - dict with all evaluation metrics
    """
    # predict_proba returns probabilities for both classes
    # [:, 1] selects the probability of the POSITIVE class (disease=1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # predict returns the final class decision (0 or 1) using default 0.5 threshold
    y_pred = model.predict(X_test)

    metrics = {
        "Model":        model_name,
        # AUC-ROC: Area Under the Receiver Operating Characteristic Curve
        # Measures how well the model separates classes across all thresholds
        # 1.0 = perfect, 0.5 = random guessing
        "AUC-ROC":      round(roc_auc_score(y_test, y_pred_proba), 4),

        # F1-score: harmonic mean of precision and recall
        # Best metric when you care about BOTH false positives AND false negatives
        "F1":           round(f1_score(y_test, y_pred), 4),

        # Accuracy: % of predictions that were correct (can be misleading on imbalanced data)
        "Accuracy":     round(accuracy_score(y_test, y_pred), 4),

        # Precision: of all predicted positives, what % were actually positive?
        # High precision = few false alarms
        "Precision":    round(precision_score(y_test, y_pred), 4),

        # Recall (Sensitivity): of all actual positives, what % did we catch?
        # High recall = few missed cases (critical in medical screening)
        "Recall":       round(recall_score(y_test, y_pred), 4),

        # Brier Score: mean squared error between predicted probabilities and actual outcomes
        # 0.0 = perfect calibration, 0.25 = no skill (always predicts 0.5)
        "Brier":        round(brier_score_loss(y_test, y_pred_proba), 4),
    }

    print(f"\n── {model_name} ──")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:12s}: {v}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# CARDIOVASCULAR RISK MODELS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CARDIOVASCULAR RISK — MODEL TRAINING")
print("=" * 60)

cardio_results = []

# ── Model 1: Logistic Regression (Baseline) ───────────────────────────────
# WHY: Simple, fast, interpretable. Coefficients directly show feature impact.
# Used as a performance baseline — if complex models can't beat this,
# they're not worth the added complexity.
# C = regularization strength (1/lambda). Smaller C = stronger regularization.

print("\nTraining Logistic Regression...")

# GridSearchCV: tries all combinations of hyperparameters and picks the best
# using cross-validation. This is "hyperparameter tuning".
# cv=5 means 5-fold cross-validation: data split into 5 parts,
# each part used as validation once while model trains on the other 4.
lr_params = {
    "C": [0.01, 0.1, 1.0, 10.0],         # regularization strength
    "solver": ["liblinear", "lbfgs"],      # optimization algorithm
    "max_iter": [200, 500]                 # max iterations for convergence
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_params,
    cv=5,
    scoring="roc_auc",    # optimize for AUC-ROC
    n_jobs=-1             # use all available CPU cores
)
lr_grid.fit(X_train_fit, y_train_fit)

best_lr = lr_grid.best_estimator_   # extract the best model
print(f"  Best params: {lr_grid.best_params_}")
print(f"  Best CV AUC: {lr_grid.best_score_:.4f}")

metrics = evaluate_model(best_lr, X_test, y_test, "Logistic Regression")
cardio_results.append(metrics)
joblib.dump(best_lr, "models/cardio_logistic.pkl")


# ── Model 2: Random Forest ─────────────────────────────────────────────────
# WHY: Ensemble of decision trees. Handles non-linear relationships,
# categorical features, and is robust to outliers.
# Each tree is trained on a random subset of features AND a random
# bootstrap sample of the data (bagging).
# Final prediction = majority vote across all trees.

print("\nTraining Random Forest...")

rf_params = {
    "n_estimators": [100, 200],         # number of trees
    "max_depth": [5, 10, 15],           # FIX: removed None (unlimited) — prevents overfitting
    "min_samples_split": [2, 5],        # min samples to split a node
    "max_features": ["sqrt", "log2"]    # features considered per split
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced"),
    rf_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
rf_grid.fit(X_train_fit, y_train_fit)

best_rf = rf_grid.best_estimator_
print(f"  Best params: {rf_grid.best_params_}")
print(f"  Best CV AUC: {rf_grid.best_score_:.4f}")

metrics = evaluate_model(best_rf, X_test, y_test, "Random Forest")
cardio_results.append(metrics)
joblib.dump(best_rf, "models/cardio_rf.pkl")


# ── Model 3: XGBoost (Gradient Boosting) ─────────────────────────────────
# WHY: Usually the best-performing model on tabular data.
# Builds trees SEQUENTIALLY — each tree corrects the errors of the previous.
# Gradient Boosting minimizes a loss function by adding trees that
# represent the gradient of the loss.
# XGBoost adds regularization (L1 + L2) to prevent overfitting.

print("\nTraining Gradient Boosting (XGBoost or sklearn GBM)...")

xgb_grid = GridSearchCV(
    XGBModel(**XGB_FIXED),
    XGB_PARAMS,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
xgb_grid.fit(X_train_fit, y_train_fit)

best_xgb = xgb_grid.best_estimator_
print(f"  Best params: {xgb_grid.best_params_}")
print(f"  Best CV AUC: {xgb_grid.best_score_:.4f}")

metrics = evaluate_model(best_xgb, X_test, y_test, "XGBoost")
cardio_results.append(metrics)
joblib.dump(best_xgb, "models/cardio_xgb.pkl")


# ── Model 4: Neural Network (MLP) ────────────────────────────────────────
# WHY: Can learn complex non-linear patterns that tree models may miss.
# MLP = Multi-Layer Perceptron. Layers of interconnected neurons.
# Each neuron applies: output = activation(weighted_sum(inputs) + bias)
# 'relu' activation: max(0, x) — adds non-linearity
# Requires scaled features — already done in preprocessing.

print("\nTraining Neural Network (MLP)...")

mlp_params = {
    "hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16)],  # architecture
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001],          # L2 regularization term
    "learning_rate_init": [0.001, 0.01]
}

mlp_grid = GridSearchCV(
    MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
    mlp_params,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
mlp_grid.fit(X_train_fit, y_train_fit)

best_mlp = mlp_grid.best_estimator_
print(f"  Best params: {mlp_grid.best_params_}")
print(f"  Best CV AUC: {mlp_grid.best_score_:.4f}")

metrics = evaluate_model(best_mlp, X_test, y_test, "Neural Network (MLP)")
cardio_results.append(metrics)
joblib.dump(best_mlp, "models/cardio_mlp.pkl")


# ── Identify and save the best cardiovascular model ───────────────────────
cardio_df = pd.DataFrame(cardio_results)
best_cardio_row = cardio_df.loc[cardio_df["AUC-ROC"].idxmax()]
best_cardio_name = best_cardio_row["Model"]
print(f"\n★ Best Cardiovascular Model: {best_cardio_name}")
print(f"  AUC-ROC: {best_cardio_row['AUC-ROC']}")


# ══════════════════════════════════════════════════════════════════════════
# PATIENT RETENTION MODELS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PATIENT RETENTION — MODEL TRAINING")
print("=" * 60)

# Load retention data
ret_df = pd.read_csv("data/retention_processed.csv")
X_ret = ret_df.drop("retained", axis=1)
y_ret = ret_df["retained"]

# Split (same 80/20 strategy)
from sklearn.model_selection import train_test_split
X_ret_train, X_ret_test, y_ret_train, y_ret_test = train_test_split(
    X_ret, y_ret, test_size=0.20, random_state=42, stratify=y_ret
)

# Scale retention features
from sklearn.preprocessing import StandardScaler
scaler_ret = StandardScaler()
X_ret_train_s = scaler_ret.fit_transform(X_ret_train)
X_ret_test_s  = scaler_ret.transform(X_ret_test)
X_ret_train_s = pd.DataFrame(X_ret_train_s, columns=X_ret_train.columns)
X_ret_test_s  = pd.DataFrame(X_ret_test_s, columns=X_ret_test.columns)
joblib.dump(scaler_ret, "models/scaler_retention.pkl")

retention_results = []

print("\nTraining Retention — Logistic Regression...")
lr_ret = LogisticRegression(C=1.0, max_iter=500, random_state=42, class_weight="balanced")
lr_ret.fit(X_ret_train_s, y_ret_train)
metrics = evaluate_model(lr_ret, X_ret_test_s, y_ret_test, "Retention — Logistic Regression")
retention_results.append(metrics)
joblib.dump(lr_ret, "models/retention_logistic.pkl")

print("\nTraining Retention — Random Forest...")
rf_ret = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced", random_state=42)
rf_ret.fit(X_ret_train_s, y_ret_train)
metrics = evaluate_model(rf_ret, X_ret_test_s, y_ret_test, "Retention — Random Forest")
retention_results.append(metrics)
joblib.dump(rf_ret, "models/retention_rf.pkl")

print("\nTraining Retention — Gradient Boosting (XGBoost or sklearn GBM)...")
xgb_ret = XGBModel(n_estimators=200, max_depth=5, learning_rate=0.1, **XGB_FIXED)
xgb_ret.fit(X_ret_train_s, y_ret_train)
metrics = evaluate_model(xgb_ret, X_ret_test_s, y_ret_test, "Retention — XGBoost")
retention_results.append(metrics)
joblib.dump(xgb_ret, "models/retention_xgb.pkl")

# Save retention test sets for evaluation script
X_ret_test_s.to_csv("data/X_ret_test.csv", index=False)
y_ret_test.to_csv("data/y_ret_test.csv", index=False)


# ── Model Comparison Chart ────────────────────────────────────────────────
# A bar chart comparing all models side-by-side on AUC-ROC, F1, and Recall.
# This goes directly into your research report.

print("\nGenerating model comparison chart...")
all_results = cardio_results + retention_results
all_df = pd.DataFrame(all_results)
all_df.to_csv("outputs/model_comparison.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Performance Comparison — All Models", fontsize=15, fontweight="bold")

metrics_to_plot = ["AUC-ROC", "F1", "Recall"]
palette = sns.color_palette("Set2", len(all_df))

for i, metric in enumerate(metrics_to_plot):
    bars = axes[i].barh(all_df["Model"], all_df[metric],
                        color=palette, edgecolor="white", height=0.6)
    axes[i].set_xlim(0, 1.05)
    axes[i].set_title(metric)
    axes[i].set_xlabel("Score")
    axes[i].axvline(0.8, color="red", linestyle="--", alpha=0.5, linewidth=1)
    for bar, val in zip(bars, all_df[metric]):
        axes[i].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", bbox_inches="tight")
plt.close()

# ── ROC Curve Plot ────────────────────────────────────────────────────────
# ROC curve plots True Positive Rate (recall) vs False Positive Rate
# at every possible threshold. AUC is the area under this curve.
# A model hugging the top-left corner is best.

print("Generating ROC curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ROC Curves", fontsize=14, fontweight="bold")

cardio_models = [
    (best_lr,  "Logistic Regression"),
    (best_rf,  "Random Forest"),
    (best_xgb, "XGBoost"),
    (best_mlp, "Neural Network")
]

for model, name in cardio_models:
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)  # fpr=false pos rate, tpr=true pos rate
    auc = roc_auc_score(y_test, proba)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

axes[0].plot([0, 1], [0, 1], "k--", label="Random Guess", alpha=0.5)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("Cardiovascular Risk Models")
axes[0].legend(fontsize=8)
axes[0].fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

ret_models = [
    (lr_ret,   "Logistic Regression"),
    (rf_ret,   "Random Forest"),
    (xgb_ret,  "XGBoost")
]

for model, name in ret_models:
    proba = model.predict_proba(X_ret_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_ret_test, proba)
    auc = roc_auc_score(y_ret_test, proba)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

axes[1].plot([0, 1], [0, 1], "k--", label="Random Guess", alpha=0.5)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate (Recall)")
axes[1].set_title("Patient Retention Models")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("outputs/roc_curves.png", bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print(f"  Best Cardiovascular Model: {best_cardio_name}")
print(f"  All models saved to models/")
print(f"  Comparison charts saved to outputs/")
print("=" * 60)
