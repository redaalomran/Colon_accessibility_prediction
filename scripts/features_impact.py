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

X_test = np.load("outputs/X_test.npy")
y_test = np.load("outputs/y_test.npy")

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load("outputs/simple_nn_model.pth"))
model.eval()

# Define a wrapper for the model's forward pass (numpy -> torch -> numpy)
def model_forward(x_np):
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
    return logits.numpy().flatten()

# Select background and test data for SHAP explanation
background = X_test[np.random.choice(X_test.shape[0], 100, replace=False)]
X_display = X_test[:100]

# Initialize SHAP KernelExplainer and compute SHAP values
explainer = shap.KernelExplainer(model_forward, background)
shap_values = explainer.shap_values(X_display, nsamples=100)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

feature_names = ["H3K27ac", "H3K4me3", "H3K27me3", "h3k9me3", "H3K4me1", "H3K36me3", "CTCF", "RAD21"]

# SHAP Summary Plot – per-feature contributions per sample
plt.figure()
shap.summary_plot(shap_values, X_display, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()
plt.savefig("outputs/shap_summary_plot.png", dpi=300)
plt.close()

# SHAP Bar Plot – global importance (mean |SHAP value| per feature)
shap.summary_plot(shap_values, X_display, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.show()
plt.savefig("outputs/shap_summary_barplot.png", dpi=300)
plt.close()

# SHAP Waterfall Plot – individual breakdown for a specific sample
sample_index = 0
shap.plots._waterfall.waterfall_legacy(
    expected_value=explainer.expected_value,
    shap_values=shap_values[sample_index],
    features=X_display[sample_index],
    feature_names=feature_names
)
plt.tight_layout()
plt.show()
plt.savefig("outputs/shap_waterfall_sample0.png", dpi=300)
plt.close()
