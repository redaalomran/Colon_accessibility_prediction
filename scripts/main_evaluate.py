import numpy as np
import pandas as pd
import pyBigWig
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import shap
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    auc
)

all_probs = np.load("outputs/probs.npy")
all_preds = np.load("outputs/preds.npy")
all_labels = np.load("outputs/labels.npy")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# ROC AUC and Curve
roc_auc = roc_auc_score(all_labels, all_probs)
print(f"\nROC AUC: {roc_auc:.4f}")
fpr, tpr, _ = roc_curve(all_labels, all_probs)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("outputs/roc_curve.png", dpi=300)
plt.close()

# Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("outputs/precision_recall_curve.png", dpi=300)
plt.close()

# Probability Distribution
plt.figure(figsize=(6, 4))
plt.hist(all_probs, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
plt.xlabel("Predicted Probability")
plt.ylabel("Number of Bins")
plt.title("Distribution of Predicted Probabilities")
plt.tight_layout()
plt.show()
plt.savefig("outputs/probability_distribution.png", dpi=300)
plt.close()
