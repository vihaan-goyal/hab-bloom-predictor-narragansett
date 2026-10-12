import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import time
import os

# Use all available CPU cores
torch.set_num_threads(os.cpu_count())
print(f"Using {os.cpu_count()} CPU threads")


# ------------------------------------------------------------------
# ConvLSTM Cell
# ------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # Single conv computes all 4 gates at once
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


# ------------------------------------------------------------------
# Full ConvLSTM HAB Predictor
# ------------------------------------------------------------------
class ConvLSTMHAB(nn.Module):
    """
    Two-layer ConvLSTM followed by global average pool + MLP classifier.

    Input shape:  (batch, time, 1, H, W)
    Output shape: (batch,)  -- raw logits, no sigmoid (use BCEWithLogitsLoss)
    """
    def __init__(self, input_channels=1, hidden_channels=16, kernel_size=3, patch_size=8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size

        self.convlstm1 = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.convlstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)

        # No Sigmoid here -- BCEWithLogitsLoss handles it (numerically safer)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (batch, time, channels, H, W)
        batch_size, seq_len, C, H, W = x.shape

        h1 = torch.zeros(batch_size, self.hidden_channels, H, W)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        for t in range(seq_len):
            h1, c1 = self.convlstm1(x[:, t], h1, c1)
            h2, c2 = self.convlstm2(h1, h2, c2)

        return self.classifier(h2).squeeze(1)  # (batch,)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class ConvHABDataset(Dataset):
    def __init__(self, X, y):
        # X: (samples, time, H, W) -> add channel dim -> (samples, time, 1, H, W)
        self.X = torch.FloatTensor(X).unsqueeze(2)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            all_logits.extend(logits.numpy())
            all_labels.extend(y_batch.numpy())

    logits = np.array(all_logits)
    labels = np.array(all_labels)
    probs  = torch.sigmoid(torch.FloatTensor(logits)).numpy()
    preds  = (probs >= threshold).astype(int)

    auc  = roc_auc_score(labels, probs)
    ap   = average_precision_score(labels, probs)
    report = classification_report(labels, preds, target_names=["No Bloom", "Bloom"], digits=3)
    cm   = confusion_matrix(labels, preds)

    return auc, ap, report, cm, probs


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":

    # --- Load data ---
    print("Loading sequences...")
    X = np.load("data/X_conv_sequences.npy")
    y = np.load("data/y_conv_labels.npy")
    meta = pd.read_csv("data/conv_meta.csv")
    meta['date'] = pd.to_datetime(meta['date'])

    print(f"X shape: {X.shape}")   # (samples, time, H, W)
    print(f"y shape: {y.shape}")
    print(f"Bloom rate: {y.mean()*100:.1f}%")

    # Infer patch size dynamically from data
    patch_size = X.shape[2]
    print(f"Patch size inferred from data: {patch_size}x{patch_size}")

    # --- Preprocessing ---
    X = np.nan_to_num(X, nan=0.0)
    X_mean = X.mean()
    X_std  = X.std() + 1e-8
    X = (X - X_mean) / X_std
    print(f"Normalized: mean={X_mean:.4f}, std={X_std:.4f}")

    # --- Temporal train/val/test split ---
    train_mask = meta['date'].dt.year <= 2019
    val_mask   = (meta['date'].dt.year >= 2020) & (meta['date'].dt.year <= 2022)
    test_mask  = meta['date'].dt.year >= 2023

    train_idx = meta[train_mask].index
    val_idx   = meta[val_mask].index
    test_idx  = meta[test_mask].index

    print(f"\nSplit -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # --- Class imbalance weight ---
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight_val = n_neg / (n_pos + 1e-8)
    print(f"Class imbalance -> neg: {n_neg}, pos: {n_pos}, pos_weight: {pos_weight_val:.2f}")

    # --- DataLoaders (small batch for CPU memory) ---
    # batch_size=16 is gentler on CPU RAM than 32
    train_loader = DataLoader(ConvHABDataset(X_train, y_train), batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ConvHABDataset(X_val,   y_val),   batch_size=16, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ConvHABDataset(X_test,  y_test),  batch_size=16, shuffle=False, num_workers=0)

    # --- Model ---
    # hidden_channels=16 instead of 32 -- cuts compute ~4x on CPU with minimal accuracy loss
    model = ConvLSTMHAB(patch_size=patch_size, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # BCEWithLogitsLoss with pos_weight handles class imbalance properly
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # --- Training loop ---
    EPOCHS          = 30
    PATIENCE        = 5
    best_auc        = 0.0
    patience_counter = 0
    train_losses    = []
    val_aucs        = []

    print("\nStarting training (CPU)...")
    print("-" * 55)

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        val_auc, val_ap, _, _, _ = evaluate(model, val_loader)
        val_aucs.append(val_auc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val AP: {val_ap:.4f} | "
              f"{elapsed:.1f}s")

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "data/convlstm_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    print(f"\nBest Val AUC: {best_auc:.4f}")

    # --- Test set evaluation ---
    print("\n" + "=" * 55)
    print("TEST SET EVALUATION")
    print("=" * 55)

    if len(test_idx) > 0:
        model.load_state_dict(torch.load("data/convlstm_model.pt"))
        test_auc, test_ap, test_report, test_cm, test_probs = evaluate(model, test_loader)

        print(f"Test AUC-ROC:  {test_auc:.4f}")
        print(f"Test Avg Prec: {test_ap:.4f}")
        print(f"\nClassification Report:\n{test_report}")
        print(f"Confusion Matrix:\n{test_cm}")
        print(f"\nXGBoost baseline AUC: 0.936")
        print(f"LSTM baseline AUC:     0.926")
        print(f"ConvLSTM AUC:          {test_auc:.4f}")

        np.save("data/convlstm_test_probs.npy", test_probs)
        np.save("data/convlstm_test_labels.npy", y_test)
    else:
        print("No test samples (years >= 2023) in this dataset.")

    # --- Training curve plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(val_aucs, label="Val AUC", color="orange")
    ax2.axhline(y=0.936, linestyle="--", color="gray", label="XGBoost baseline (0.936)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Validation AUC")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("figures/model/convlstm_training_curve.png", dpi=150)
    print("\nTraining curve saved to figures/model/convlstm_training_curve.png")
    plt.show()