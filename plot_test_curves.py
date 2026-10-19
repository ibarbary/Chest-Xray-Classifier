
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    ROOT / "reports" / "predictions" / "test_predictions.npz"
)
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load the saved test results. No model inference or retraining needed.
data = np.load(PREDICTIONS_PATH, allow_pickle=False)

probs = data["probabilities"]
targets = data["targets"].astype(int)
labels = data["labels"].tolist()

if probs.shape != targets.shape:
    raise ValueError(
        f"Shape mismatch: probabilities={probs.shape}, "
        f"targets={targets.shape}"
    )

if probs.shape[1] != len(labels):
    raise ValueError("Number of labels does not match prediction columns.")

print(f"Loaded {len(targets):,} test images and {len(labels)} labels.")


# --------------------------------------------------
# 1. ROC curves
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 9))

aucs = []

for i, label in enumerate(labels):
    y_true = targets[:, i]
    y_score = probs[:, i]

    # ROC-AUC requires both positive and negative examples.
    if np.unique(y_true).size < 2:
        print(f"Skipping ROC for {label}: only one target class.")
        continue

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    aucs.append(auc)

    ax.plot(fpr, tpr, linewidth=1.5, label=f"{label} (AUC={auc:.3f})")

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    linewidth=1,
    label="Random classifier",
)

ax.set(
    xlim=(0, 1),
    ylim=(0, 1.02),
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Test-Set ROC Curves — 14-Disease Chest X-ray Classification",
)
ax.grid(alpha=0.25)
ax.legend(loc="lower right", fontsize=8, ncol=2)
fig.tight_layout()

roc_path = FIGURES_DIR / "test_roc_curves.png"
fig.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {roc_path}")


# --------------------------------------------------
# 2. Precision–Recall curves
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 9))

aps = []

for i, label in enumerate(labels):
    y_true = targets[:, i]
    y_score = probs[:, i]

    if y_true.sum() == 0:
        print(f"Skipping PR curve for {label}: no positive examples.")
        continue

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    aps.append(ap)

    ax.plot(
        recall,
        precision,
        linewidth=1.5,
        label=f"{label} (AP={ap:.3f})",
    )

ax.set(
    xlim=(0, 1),
    ylim=(0, 1.02),
    xlabel="Recall",
    ylabel="Precision",
    title="Test-Set Precision–Recall Curves — 14-Disease Classification",
)
ax.grid(alpha=0.25)
ax.legend(loc="upper right", fontsize=8, ncol=2)
fig.tight_layout()

pr_path = FIGURES_DIR / "test_precision_recall_curves.png"
fig.savefig(pr_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {pr_path}")


# --------------------------------------------------
# 3. Per-disease summary CSV
# --------------------------------------------------
rows = []

for i, label in enumerate(labels):
    y_true = targets[:, i]
    y_score = probs[:, i]
    positives = int(y_true.sum())

    auc = (
        roc_auc_score(y_true, y_score)
        if np.unique(y_true).size == 2
        else np.nan
    )

    ap = (
        average_precision_score(y_true, y_score)
        if positives > 0
        else np.nan
    )

    rows.append(
        {
            "disease": label,
            "roc_auc": auc,
            "average_precision": ap,
            "positive_cases": positives,
            "negative_cases": int(len(y_true) - positives),
            "prevalence": float(y_true.mean()),
        }
    )

summary = pd.DataFrame(rows).sort_values(
    "roc_auc", ascending=False
)

csv_path = ROOT / "reports" / "test_per_disease_summary.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(csv_path, index=False, float_format="%.6f")

print(f"Saved: {csv_path}")
print("\nPer-disease summary (sorted by ROC-AUC):")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nAll test-curve plots and summary files generated successfully.")
