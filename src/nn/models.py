"""Model definitions for findings 26-27. All small (< 100k params), CPU-friendly."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Tabular control: Linear-GELU-Dropout-Linear-GELU-Linear."""

    def __init__(self, n_in, hidden=(64, 32), dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.GELU(),
            nn.Linear(hidden[1], 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class SeqEncoder(nn.Module):
    """Strided 1-D CNN over (B, C, 672) -> 128-d embedding (global mean + max)."""

    def __init__(self, n_ch=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, 32, kernel_size=9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2), nn.GELU())
        self.out_dim = 128

    def forward(self, x):
        h = self.conv(x)
        return torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)


class SeqCNN(nn.Module):
    """The hypothesis cell: window + static (sin/cos DOY) -> head."""

    def __init__(self, n_ch=8, n_static=2, dropout=0.3):
        super().__init__()
        self.enc = SeqEncoder(n_ch)
        self.head = nn.Sequential(
            nn.Linear(self.enc.out_dim + n_static, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x_seq, x_static):
        return self.head(torch.cat([self.enc(x_seq), x_static], dim=1)).squeeze(1)


class HybridCNN(nn.Module):
    """CNN embedding concatenated with standardised tier-A features + DOY."""

    def __init__(self, n_ch=8, n_static=2, n_tab=23, dropout=0.3):
        super().__init__()
        self.enc = SeqEncoder(n_ch)
        self.head = nn.Sequential(
            nn.Linear(self.enc.out_dim + n_static + n_tab, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x_seq, x_static, x_tab):
        return self.head(torch.cat([self.enc(x_seq), x_static, x_tab], dim=1)).squeeze(1)


class SiteMLP(nn.Module):
    """Pooled multi-site MLP with an optional site embedding (index n_sites = UNK)."""

    def __init__(self, n_in, n_sites, emb_dim=4, use_emb=True, hidden=(64, 32), dropout=0.3):
        super().__init__()
        self.use_emb = use_emb
        self.unk = n_sites
        self.emb = nn.Embedding(n_sites + 1, emb_dim) if use_emb else None
        self.mlp = MLP(n_in + (emb_dim if use_emb else 0), hidden, dropout)

    def forward(self, x, site, emb_vec=None):
        """emb_vec: optional (emb_dim,) tensor used for every row instead of the table
        (the mean-of-site-embeddings sensitivity in findings 27)."""
        if self.use_emb:
            e = self.emb(site) if emb_vec is None else emb_vec.expand(len(x), -1)
            x = torch.cat([x, e], dim=1)
        return self.mlp(x)
