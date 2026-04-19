"""
LSTM Autoencoder — Deep Anomaly Detection on Topic Velocity Time Series.

Trains on pre-2020 topic proportion patterns (normal baseline).
Flags time bins where reconstruction error is abnormally high — indicating
narrative patterns the model has never seen before.

Advantage over Z-score: captures temporal autocorrelation and slow-building
crises (e.g. early COVID signal Jan 2020) that single-point Z-scores miss.

Usage:
    python src/lstm_anomaly_detector.py
    python src/lstm_anomaly_detector.py --top-topics 15 --seq-len 8 --epochs 30

Requirements:
    pip install torch
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
TM      = REPORTS / "topic_modeling" / "11_lda_sbert"
SPIKES  = REPORTS / "topic_modeling" / "12_spikes_anchors"

TOPICS_CSV = TM / "topics_over_time.csv"
SPIKE_CSV  = SPIKES / "spike_events.csv"
OUT_CSV    = REPORTS / "lstm_anomalies.csv"
OUT_MODEL  = REPORTS / "lstm_autoencoder.pt"


# ── LSTM Autoencoder ───────────────────────────────────────────────────────
def build_model(input_size, hidden_size=32, num_layers=2):
    """Returns encoder, decoder as a single nn.Module."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: PyTorch not installed.  pip install torch", file=sys.stderr)
        sys.exit(1)

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True, dropout=0.1 if num_layers > 1 else 0)
            self.decoder = nn.LSTM(hidden_size, input_size, num_layers,
                                   batch_first=True, dropout=0.1 if num_layers > 1 else 0)

        def forward(self, x):
            _, (h, c) = self.encoder(x)
            # broadcast latent vector across sequence length
            latent  = h[-1].unsqueeze(1).expand(-1, x.size(1), -1)
            out, _  = self.decoder(latent)
            return out

    return LSTMAutoencoder()


# ── data preparation ────────────────────────────────────────────────────────
def load_topic_matrix(top_n=20):
    """
    Build a (time_bins × top_n_topics) matrix of proportions from topics_over_time.csv.
    Returns: matrix (np.array), dates (list of timestamps), topic_ids (list).
    """
    if not TOPICS_CSV.exists():
        print(f"ERROR: {TOPICS_CSV} not found. Run notebook 11 first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(TOPICS_CSV)
    df["bin_mid"] = pd.to_datetime(df["bin_mid"], errors="coerce")
    df = df.dropna(subset=["bin_mid"])
    df = df[df["topic_id"] >= 0]   # drop noise label -1

    # pick top_n topics by total proportion
    top_ids = (
        df.groupby("topic_id")["proportion"].sum()
        .nlargest(top_n).index.tolist()
    )
    df = df[df["topic_id"].isin(top_ids)]

    # pivot: rows = time bins (sorted), columns = topic_id
    pivot = (
        df.pivot_table(index="bin_mid", columns="topic_id",
                       values="proportion", aggfunc="mean")
        .sort_index()
        .fillna(0.0)
    )

    dates     = pivot.index.tolist()
    topic_ids = pivot.columns.tolist()
    matrix    = pivot.values.astype(np.float32)

    print(f"Topic matrix: {matrix.shape[0]} time bins × {matrix.shape[1]} topics")
    print(f"Date range: {dates[0]} → {dates[-1]}")
    return matrix, dates, topic_ids


def normalise(matrix, train_mask):
    """Min-max normalise per column using training statistics."""
    mins = matrix[train_mask].min(axis=0)
    maxs = matrix[train_mask].max(axis=0)
    rang = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
    return (matrix - mins) / rang, mins, maxs


def make_sequences(matrix, seq_len):
    """Sliding window sequences of shape (N, seq_len, features)."""
    seqs = []
    for i in range(len(matrix) - seq_len):
        seqs.append(matrix[i: i + seq_len])
    return np.stack(seqs, axis=0).astype(np.float32)


# ── training ────────────────────────────────────────────────────────────────
def train(model, seqs_train, epochs=30, lr=1e-3, batch_size=32):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.tensor(seqs_train)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optim    = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in dl:
            optim.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(batch)
        if epoch % 5 == 0 or epoch == 1:
            avg = total_loss / len(seqs_train)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.6f}", flush=True)

    return model


# ── reconstruction error ────────────────────────────────────────────────────
def compute_errors(model, matrix, seq_len):
    """
    Return per-time-bin reconstruction error.
    Bins in the initial seq_len window get error=0 (no prior sequence).
    """
    import torch
    import torch.nn as nn

    criterion = nn.MSELoss(reduction="none")
    model.eval()
    errors = np.zeros(len(matrix))

    with torch.no_grad():
        for i in range(len(matrix) - seq_len):
            seq  = torch.tensor(matrix[i: i + seq_len]).unsqueeze(0)
            recon = model(seq)
            mse  = criterion(recon, seq).mean(dim=(1, 2)).item()
            errors[i + seq_len] = mse   # error assigned to the *last* bin in the window

    return errors


# ── compare with z-score baseline ───────────────────────────────────────────
def compare_with_zscore(anomaly_df, threshold):
    if not SPIKE_CSV.exists():
        return
    spike_df = pd.read_csv(SPIKE_CSV)
    spike_df["spike_date"] = pd.to_datetime(spike_df["spike_date"], errors="coerce")

    lstm_months   = set(anomaly_df[anomaly_df["is_anomaly"]]["date"].dt.to_period("M").astype(str))
    zscore_months = set(spike_df["spike_date"].dt.to_period("M").astype(str))

    print("\n── LSTM vs Z-Score Comparison (month-level) ────────────────────")
    print(f"  LSTM anomaly months   : {len(lstm_months)}")
    print(f"  Z-score spike months  : {len(zscore_months)}")
    print(f"  Detected by BOTH      : {len(lstm_months & zscore_months)}")
    print(f"  LSTM only             : {len(lstm_months - zscore_months)}")
    print(f"  Z-score only          : {len(zscore_months - lstm_months)}")

    only_lstm = sorted(lstm_months - zscore_months)
    if only_lstm:
        print(f"\n  Months caught by LSTM but NOT Z-score ({len(only_lstm)}):")
        for m in only_lstm[:10]:
            print(f"    {m}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LSTM Autoencoder anomaly detector")
    parser.add_argument("--top-topics", type=int, default=20,  help="top N topics to use")
    parser.add_argument("--seq-len",    type=int, default=10,  help="sliding window length")
    parser.add_argument("--hidden",     type=int, default=32,  help="LSTM hidden size")
    parser.add_argument("--layers",     type=int, default=2,   help="LSTM layers")
    parser.add_argument("--epochs",     type=int, default=30,  help="training epochs")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch",      type=int, default=32)
    parser.add_argument("--threshold-sigma", type=float, default=2.0,
                        help="anomaly = error > mean + N*std of training errors")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed.  pip install torch", file=sys.stderr)
        sys.exit(1)

    REPORTS.mkdir(exist_ok=True)

    # ── load data ──
    matrix, dates, topic_ids = load_topic_matrix(top_n=args.top_topics)

    # training mask: bins before 2020 (pre-COVID baseline)
    dates_arr  = np.array(dates)
    cutoff     = pd.Timestamp("2020-01-01")
    train_mask = np.array([d < cutoff for d in dates_arr])

    if train_mask.sum() < args.seq_len * 3:
        print("WARNING: very few pre-2020 bins — training on all data instead.")
        train_mask = np.ones(len(dates), dtype=bool)

    print(f"Training bins  : {train_mask.sum()} (before 2020)")
    print(f"Inference bins : {len(dates)}")

    # ── normalise ──
    matrix_norm, mins, maxs = normalise(matrix, train_mask)

    # ── sequences ──
    train_idx  = np.where(train_mask)[0]
    # build training sequences only from consecutive training bins
    train_seqs = make_sequences(matrix_norm[train_mask], args.seq_len)
    all_seqs   = make_sequences(matrix_norm, args.seq_len)

    print(f"Training sequences : {len(train_seqs)}")
    print(f"Total sequences    : {len(all_seqs)}")

    # ── model ──
    model = build_model(matrix.shape[1], hidden_size=args.hidden, num_layers=args.layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Training for {args.epochs} epochs…\n")

    model = train(model, train_seqs, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

    # ── save model ──
    torch.save(model.state_dict(), OUT_MODEL)
    print(f"\nModel saved: {OUT_MODEL}")

    # ── reconstruction errors ──
    errors = compute_errors(model, matrix_norm, args.seq_len)

    # threshold from training error distribution
    train_errors = errors[train_mask]
    train_errors = train_errors[train_errors > 0]  # exclude zero-padded
    err_mean  = train_errors.mean()
    err_std   = train_errors.std()
    threshold = err_mean + args.threshold_sigma * err_std

    print(f"\nTraining error  mean={err_mean:.6f}  std={err_std:.6f}")
    print(f"Anomaly threshold (μ + {args.threshold_sigma}σ) = {threshold:.6f}")

    # ── build results DataFrame ──
    results = pd.DataFrame({
        "date":                 dates_arr,
        "reconstruction_error": errors,
        "is_anomaly":           errors >= threshold,
        "in_training":          train_mask,
    })
    results["date"] = pd.to_datetime(results["date"])

    anomalies = results[results["is_anomaly"] & ~results["in_training"]]
    print(f"\nAnomalies detected (post-training period): {len(anomalies)}")

    # ── save ──
    results.to_csv(OUT_CSV, index=False)
    print(f"Results saved: {OUT_CSV}")

    # ── print anomaly report ──
    print("\n── Top Anomalies (highest reconstruction error) ───────────────")
    top_anom = results[results["is_anomaly"]].nlargest(15, "reconstruction_error")
    for _, r in top_anom.iterrows():
        marker = "(TRAIN)" if r["in_training"] else ""
        print(f"  {str(r['date'])[:10]}  error={r['reconstruction_error']:.6f}  {marker}")

    # ── compare with z-score ──
    compare_with_zscore(results, threshold)

    print("\nDone.")


if __name__ == "__main__":
    main()
