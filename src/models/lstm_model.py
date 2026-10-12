import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import matplotlib.pyplot as plt

# Load data
print("Loading sequences...")
X = np.load("data/X_sequences.npy")
y = np.load("data/y_labels.npy")
meta = pd.read_csv("data/sequence_meta.csv")
meta['date'] = pd.to_datetime(meta['date'])

# Subsample for local testing - comment out for full run on Colab
SUBSET = False
if SUBSET:
    idx = np.random.choice(len(X), size=50000, replace=False)
    X = X[idx]
    y = y[idx]
    meta = meta.iloc[idx].reset_index(drop=True)
    print("Running on 50k subset for testing")

# Spatiotemporal split by year
train_idx = meta[meta['date'].dt.year <= 2019].index
val_idx = meta[(meta['date'].dt.year >= 2020) & (meta['date'].dt.year <= 2022)].index
test_idx = meta[meta['date'].dt.year >= 2023].index

print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Normalize using train stats
mean = X_train.mean(axis=(0,1))
std = X_train.std(axis=(0,1)) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Dataset
class HABDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(HABDataset(X_train, y_train), batch_size=512, shuffle=True)
val_loader = DataLoader(HABDataset(X_val, y_val), batch_size=512, shuffle=False)

# LSTM Model
class HABPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.5): #0.3 -> 0.5
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze()

# Class imbalance weight
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = HABPredictor(input_size=15).to(device)
# Add weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCELoss()

# Training loop
EPOCHS = 50
best_auc = 0
patience = 5
patience_counter = 0
best_model_state = None
train_losses, val_aucs = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    val_preds = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            val_preds.extend(preds)
    
    val_auc = roc_auc_score(y_val, val_preds)
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    val_aucs.append(val_auc)
    
    if val_auc > best_auc:
        best_auc = val_auc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Best: {best_auc:.4f}")
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Restore best model
model.load_state_dict(best_model_state)
print(f"\nBest Val AUC: {best_auc:.4f}")    

# Final evaluation
print("\nFinal evaluation on validation set:")
val_preds_binary = (np.array(val_preds) > 0.5).astype(int)
print(classification_report(y_val, val_preds_binary))
print(f"AUC-ROC: {roc_auc_score(y_val, val_preds):.4f}")
print(f"Avg Precision: {average_precision_score(y_val, val_preds):.4f}")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax2.plot(val_aucs, color='green')
ax2.axhline(y=0.928, color='red', linestyle='--', label='XGBoost baseline')
ax2.set_title('Validation AUC-ROC')
ax2.set_xlabel('Epoch')
ax2.legend()
plt.tight_layout()
plt.savefig('figures/model/lstm_training.png', dpi=150)
plt.show()

torch.save(model.state_dict(), 'data/lstm_model.pt')
print("Model saved.")