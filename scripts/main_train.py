import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

bins_df = pd.read_csv("labeled_colon_bins.csv")

held_out_chrom = "chr1"

train_df = bins_df[bins_df["chrom"] != held_out_chrom].copy()
test_df = bins_df[bins_df["chrom"] == held_out_chrom].copy()

X_train = train_df[["H3K27ac", "H3K4me3", "H3K27me3", "CTCF"]].values.astype(np.float32)
y_train = train_df["label"].astype(np.float32).values

X_test = test_df[["H3K27ac", "H3K4me3", "H3K27me3", "CTCF"]].values.astype(np.float32)
y_test = test_df["label"].astype(np.float32).values

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()

pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

model = SimpleNN()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = torch.sigmoid(logits) > 0.5
        correct += (preds == yb.bool()).sum().item()
        total += yb.size(0)

    train_acc = correct / total

    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            preds = torch.sigmoid(logits) > 0.5
            correct_test += (preds == yb.bool()).sum().item()
            total_test += yb.size(0)
    test_acc = correct_test / total_test

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "simple_nn_model.pth")
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

all_probs = []
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_probs.extend(probs.squeeze().numpy())
        all_preds.extend(preds.squeeze().numpy())
        all_labels.extend(yb.squeeze().numpy())

np.save("probs.npy", np.array(all_probs))
np.save("preds.npy", np.array(all_preds))
np.save("labels.npy", np.array(all_labels))

print("\nTraining complete. Model and outputs saved.")
