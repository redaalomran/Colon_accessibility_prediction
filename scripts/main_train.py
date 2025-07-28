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

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Define a simple 3-layer feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# Train model and save outputs
def train_and_save():
    # Load labeled bins and drop the ATAC reference column
    bins_df = pd.read_csv("data/labeled_colon_bins.csv", sep="\t")
    bins_df = bins_df.drop(columns=['ATAC'])

    
    # Define feature columns (exclude coordinates and labels)
    non_feature_cols = ["chrom", "start", "end", "label"]
    feature_names = [col for col in bins_df.columns if col not in non_feature_cols]

    # Hold out chromosome 1 for testing, use others for training
    held_out_chrom = "chr1"
    train_df = bins_df[bins_df["chrom"] != held_out_chrom].copy()
    test_df = bins_df[bins_df["chrom"] == held_out_chrom].copy()

    # Convert to NumPy arrays for PyTorch
    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["label"].astype(np.float32).values

    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df["label"].astype(np.float32).values

    # Wrap data in PyTorch Dataset + DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Handle class imbalance by computing positive class weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    # Initialize model, loss, and optimizer
    model = SimpleNN()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Loop over training batches
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # Track training accuracy
            total_loss += loss.item() * xb.size(0)
            preds = torch.sigmoid(logits) >= 0.5  # Temporary for tracking accuracy
            correct += (preds == yb.bool()).sum().item()
            total += yb.size(0)

        train_acc = correct / total

        # Evaluate on test set after each epoch
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb)
                preds = torch.sigmoid(logits) >= 0.5
                correct_test += (preds == yb.bool()).sum().item()
                total_test += yb.size(0)

        test_acc = correct_test / total_test

        # Print progress for each epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

    # Save model and test data
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/simple_nn_model.pth")
    np.save("outputs/X_test.npy", X_test)
    np.save("outputs/y_test.npy", y_test)

    # Generate predictions and probabilities on test set
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.squeeze().numpy())
            all_labels.extend(yb.squeeze().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Compute Youden’s J statistic for optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    print(f"\nBest dynamic threshold (Youden’s J): {best_thresh:.4f}")

    # Apply best threshold to generate final binary predictions
    best_preds = (all_probs >= best_thresh).astype(int)

    # Save predictions and true labels for evaluation
    np.save("outputs/probs.npy", all_probs)
    np.save("outputs/preds.npy", best_preds)
    np.save("outputs/labels.npy", all_labels)

    print("\nTraining complete. Model and predictions saved.")

# Run training pipeline
if __name__ == "__main__":
    train_and_save()
