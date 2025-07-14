import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main_train import SimpleNN

X_test = np.load("outputs/X_test.npy")
y_test = np.load("outputs/y_test.npy")

model = SimpleNN()
model.load_state_dict(torch.load("outputs/simple_nn_model.pth"))
model.eval()

def model_forward(x_np):
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
    return logits.numpy().flatten()

background = X_test[np.random.choice(X_test.shape[0], 100, replace=False)]
X_display = X_test[:100]

explainer = shap.KernelExplainer(model_forward, background)
shap_values = explainer.shap_values(X_display, nsamples=100)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

feature_names = ["H3K27ac", "H3K4me3", "H3K27me3", "h3k9me3", "H3K9ac", "CTCF", "ATF3", "CBX3", "CEBPB", "EGR1", "RAD21"]

plt.figure()
shap.summary_plot(shap_values, X_display, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary_plot.png", dpi=300)
plt.close()

for i, name in enumerate(feature_names):
    plt.figure()
    shap.dependence_plot(i, shap_values, X_display, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"outputs/shap_dependence_{name}.png", dpi=300)
    plt.close()
