import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

SEQUENCE_LEN = 6    # ~12 weeks of biweekly observations
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.4
BATCH_SIZE   = 64
MAX_EPOCHS   = 50
PATIENCE     = 7
LR           = 1e-3
WEIGHT_DECAY = 1e-4

features = [
    'Chlorophyll', 'chl_roll3_mean', 'chl_roll6_mean', 'chl_anomaly',
    'chl_climatology', 'sea_water_temperature', 'sea_water_salinity',
    'oxygen_concentration_in_sea_water', 'month', 'latitude_x', 'longitude_x',
]

# ── Load and prep ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/hab_features_daily.csv")
df['date'] = pd.to_datetime(df['date'])

for n, min_p in [(3,2),(6,3)]:
    df[f'chl_roll{n}_mean'] = (df.groupby('station_name')['Chlorophyll']
                                 .transform(lambda x: x.rolling(n, min_periods=min_p).mean()))

df['bloom_28d'] = 0
for station, grp in df.groupby('station_name'):
    idx = grp.index
    dates = grp['date'].values
    chl = grp['Chlorophyll'].values
    labels = np.zeros(len(grp), dtype=int)
    for i in range(len(grp)):
        mask = ((dates > dates[i]) &
                (dates <= dates[i] + np.timedelta64(28, 'D')))
        if mask.any() and (chl[mask] > 10).any():
            labels[i] = 1
    df.loc[idx, 'bloom_28d'] = labels

features = [f for f in features if f in df.columns]
df = df.sort_values(['station_name', 'date']).reset_index(drop=True)

# Fill NaNs with station median
for col in features:
    df[col] = df.groupby('station_name')[col].transform(
        lambda x: x.fillna(x.median()))
df[features] = df[features].fillna(df[features].median())

# ── Build sequences ───────────────────────────────────────────────────────────
print("Building sequences...")

train_mask = df['date'].dt.year <= 2019
val_mask   = (df['date'].dt.year >= 2020) & (df['date'].dt.year <= 2022)
test_mask  = df['date'].dt.year >= 2023

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

def build_sequences(mask):
    X_seqs, y_labels = [], []
    sub = df[mask].copy()
    for station, grp in sub.groupby('station_name'):
        grp = grp.sort_values('date').reset_index(drop=True)
        vals = grp[features].values
        labels = grp['bloom_28d'].values
        for i in range(SEQUENCE_LEN - 1, len(grp)):
            seq = vals[i - SEQUENCE_LEN + 1 : i + 1]
            if np.isnan(seq).sum() == 0:
                X_seqs.append(seq)
                y_labels.append(labels[i])
    return np.array(X_seqs, dtype=np.float32), np.array(y_labels, dtype=np.float32)

X_train, y_train = build_sequences(train_mask)
X_val,   y_val   = build_sequences(val_mask)
X_test,  y_test  = build_sequences(test_mask)

print(f"Train sequences: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"Bloom rates: train={y_train.mean():.1%} val={y_val.mean():.1%} test={y_test.mean():.1%}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class HABDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
train_loader = DataLoader(HABDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(HABDataset(X_val,   y_val),   batch_size=256)
test_loader  = DataLoader(HABDataset(X_test,  y_test),  batch_size=256)

# ── Model ─────────────────────────────────────────────────────────────────────
class HABLstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out).squeeze(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = HABLstm(len(features), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

# ── Training ──────────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_b.numpy())
    return np.array(all_probs), np.array(all_labels)

best_val_auc = 0
patience_count = 0
train_losses = []
val_aucs = []
print("\nTraining...")
for epoch in range(MAX_EPOCHS):
    model.train()
    total_loss = 0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        logits = model(X_b.to(device))
        loss = criterion(logits, y_b.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_probs, val_labels = evaluate(val_loader)
    val_auc = roc_auc_score(val_labels, val_probs)

    train_losses.append(total_loss / len(train_loader))
    val_aucs.append(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "data/lstm_best.pt")
        patience_count = 0
    else:
        patience_count += 1

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} | val_auc={val_auc:.3f} | best={best_val_auc:.3f}")

    if patience_count >= PATIENCE:
        print(f"  Early stopping at epoch {epoch+1}")
        break

# ── Final evaluation ──────────────────────────────────────────────────────────
model.load_state_dict(torch.load("data/lstm_best.pt"))
val_probs,  val_labels  = evaluate(val_loader)
test_probs, test_labels = evaluate(test_loader)

print(f"\nLSTM Val AUC:  {roc_auc_score(val_labels,  val_probs):.3f}")
print(f"LSTM Test AUC: {roc_auc_score(test_labels, test_probs):.3f}")
print(f"LSTM Val AP:   {average_precision_score(val_labels,  val_probs):.3f}")
print(f"LSTM Test AP:  {average_precision_score(test_labels, test_probs):.3f}")

# Save probabilities for ensemble
np.save("data/lstm_val_probs.npy",  val_probs)
np.save("data/lstm_test_probs.npy", test_probs)
np.save("data/lstm_val_labels.npy",  val_labels)
np.save("data/lstm_test_labels.npy", test_labels)
print("Saved LSTM probabilities for ensemble.")

# ── Training curve figure ─────────────────────────────────────────────────────
print("\nGenerating training curve figure...")
epochs_range = range(1, len(train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training loss')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('BCE Loss', fontsize=11)
ax1.set_title('Training Loss per Epoch', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.plot(epochs_range, val_aucs, 'g-', linewidth=2, label='Val AUC')
ax2.axhline(y=0.827, color='#E63946', linestyle='--', linewidth=1.5,
            label='Ensemble test AUC baseline (0.827)')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Validation AUC', fontsize=11)
ax2.set_title('Validation AUC per Epoch', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle(
    'LSTM Training Curves — HAB Bloom Predictor (28-day forecast horizon)',
    fontsize=13, fontweight='bold',
)
plt.tight_layout()
plt.savefig('figures/model/fig9_lstm_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/model/fig9_lstm_training.png")