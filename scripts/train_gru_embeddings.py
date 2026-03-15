"""Train a GRU to extract temporal embeddings for each market snapshot.

The GRU processes per-market sequences and outputs an embedding per timestep.
These embeddings are then appended to the XGBoost feature set for stacking.

Usage:
    uv run python scripts/train_gru_embeddings.py

Produces:
    data/prepared/gru_embeddings_train.npy
    data/prepared/gru_embeddings_val.npy
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PREPARED_DIR = Path(__file__).resolve().parent.parent / "data" / "prepared"

# GRU hyperparameters — optimized for speed
HIDDEN_DIM = 32
N_LAYERS = 1
DROPOUT = 0.1
BATCH_SIZE = 512
LR = 3e-3
MAX_EPOCHS = 20
PATIENCE = 5
MAX_SEQ_LEN = 128  # truncate long sequences for speed


class MarketGRU(nn.Module):
    """Bidirectional GRU that predicts binary outcome from market sequences."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, n_layers: int = N_LAYERS):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=DROPOUT if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (batch_size,)."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)  # hidden: (n_layers, batch, hidden_dim)
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        return self.head(last_hidden).squeeze(1)

    def embed(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Extract per-timestep embeddings (batch, max_len, hidden_dim)."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        unpacked, _ = pad_packed_sequence(output, batch_first=True)
        return unpacked


def group_by_market(X: np.ndarray, y: np.ndarray, market_ids: np.ndarray):
    """Group samples by market_id into sequences, truncated to MAX_SEQ_LEN."""
    markets: dict[str, list[int]] = {}
    for i, mid in enumerate(market_ids):
        mid_str = str(mid)
        if mid_str not in markets:
            markets[mid_str] = []
        markets[mid_str].append(i)
    # Truncate long sequences (keep most recent)
    for mid_str in markets:
        if len(markets[mid_str]) > MAX_SEQ_LEN:
            markets[mid_str] = markets[mid_str][-MAX_SEQ_LEN:]
    return markets


def create_batches(X, y, markets, batch_size, device):
    """Create padded batches of market sequences."""
    market_list = list(markets.items())
    np.random.shuffle(market_list)

    for i in range(0, len(market_list), batch_size):
        batch_markets = market_list[i : i + batch_size]
        sequences = []
        labels = []
        lengths = []

        for mid, indices in batch_markets:
            seq = torch.tensor(X[indices], dtype=torch.float32)
            sequences.append(seq)
            labels.append(float(y[indices[0]]))  # same label for all snapshots of a market
            lengths.append(len(indices))

        # Pad sequences
        padded = pad_sequence(sequences, batch_first=True)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        lengths_t = torch.tensor(lengths, dtype=torch.long)

        yield padded.to(device), labels_t.to(device), lengths_t


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load data
    meta = json.load(open(PREPARED_DIR / "meta.json"))
    feature_names = meta["feature_names"]
    n_features = len(feature_names)

    train_data = np.load(PREPARED_DIR / "train.npz", allow_pickle=True)
    val_data = np.load(PREPARED_DIR / "val.npz", allow_pickle=True)

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    train_mids = train_data["market_ids"]
    val_mids = val_data["market_ids"]

    print(f"Train: {X_train.shape[0]} samples, {len(set(train_mids))} markets")
    print(f"Val:   {X_val.shape[0]} samples, {len(set(val_mids))} markets")

    # Standardize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Group by market
    train_markets = group_by_market(X_train_scaled, y_train, train_mids)
    val_markets = group_by_market(X_val_scaled, y_val, val_mids)
    print(f"Train markets: {len(train_markets)}, Val markets: {len(val_markets)}")

    # Train GRU
    model = MarketGRU(n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    start = time.monotonic()
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for padded, labels, lengths in create_batches(X_train_scaled, y_train, train_markets, BATCH_SIZE, device):
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for padded, labels, lengths in create_batches(X_val_scaled, y_val, val_markets, BATCH_SIZE, device):
                logits = model(padded, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        elapsed = time.monotonic() - start
        print(
            f"Epoch {epoch + 1}/{MAX_EPOCHS} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | elapsed={elapsed:.1f}s",
            flush=True,
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.cpu()
    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    print(f"Best val_loss: {best_val_loss:.4f}")

    # Extract per-snapshot embeddings
    print("Extracting embeddings...", flush=True)
    embedding_dim = HIDDEN_DIM  # unidirectional

    for split_name, X_scaled, mids, markets_dict in [
        ("train", X_train_scaled, train_mids, train_markets),
        ("val", X_val_scaled, val_mids, val_markets),
    ]:
        embeddings = np.zeros((len(X_scaled), embedding_dim), dtype=np.float32)

        with torch.no_grad():
            for mid_str, indices in markets_dict.items():
                seq = torch.tensor(X_scaled[indices], dtype=torch.float32).unsqueeze(0)
                length = torch.tensor([len(indices)], dtype=torch.long)
                emb = model.embed(seq, length)  # (1, seq_len, hidden*2)
                emb_np = emb.squeeze(0).numpy()[: len(indices)]
                for j, idx in enumerate(indices):
                    embeddings[idx] = emb_np[j]

        out_path = PREPARED_DIR / f"gru_embeddings_{split_name}.npy"
        np.save(out_path, embeddings)
        print(f"  {split_name}: saved {embeddings.shape} to {out_path}")

    # Save model
    torch.save(
        {"state_dict": best_state, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
        PREPARED_DIR / "gru_model.pt",
    )
    print(f"Done in {time.monotonic() - start:.1f}s")


if __name__ == "__main__":
    main()
