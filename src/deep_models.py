"""Recurrent forecasters (LSTM and GRU) for next-day AQI, in PyTorch.

Input for day d:
  * a window of the `window` days d - window .. d - 1 of the same city: AQI and the core
    pollutants, standardised, with missing values set to 0 (the training mean) and a
    flag for missing AQI. Day d's own readings are never in its window;
  * static features of day d that are known in advance: day-of-year sin/cos and city.

Protocol (same as the classical and ML models):
  * scalers are fitted on the rows the model is fitted on (train, or train + validation);
  * each configuration is trained on TRAIN with early stopping on VALIDATION loss, and the
    best configuration per cell type is chosen by validation MAE relative to persistence;
  * validation forecasts come from the train-only fit, test forecasts from a refit on
    train + validation for the number of epochs early stopping picked. Test data is never
    used for any choice.
One pooled model is fitted across cities (city is a static feature), like the ML models.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.baselines import check_continuous
from src.ml_models import pooled_mae, relative_mae

GRID = [{"cell": c, "hidden": h, "window": w}
        for c in ("lstm", "gru") for h in (32, 64) for w in (14, 30)]
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def fit_scaler(df: pd.DataFrame, cols, fit_splits) -> tuple:
    """Mean and std of each column over the rows of fit_splits only."""
    rows = df["split"].isin(fit_splits)
    mean = df.loc[rows, cols].mean().fillna(0.0)
    std = df.loc[rows, cols].std()
    std = std.where(std > 0, 1.0).fillna(1.0)
    return mean, std


def make_windows(df: pd.DataFrame, cols, scaler: tuple, window: int):
    """Sequence input, static input and a validity mask, aligned with df's rows.

    Row of day d gets days d - window .. d - 1 of its own city. The first `window` days of
    each city have no full history and are marked invalid. df must hold one row per
    consecutive day for each city, sorted by City and Date."""
    check_continuous(df)
    mean, std = scaler
    z = ((df[cols] - mean) / std).fillna(0.0).to_numpy(np.float32)
    aqi_missing = df[config.TARGET].isna().to_numpy(np.float32)[:, None]
    vals = np.hstack([z, aqi_missing])

    seq = np.zeros((len(df), window, vals.shape[1]), np.float32)
    valid = np.zeros(len(df), bool)
    for pos in df.groupby("City").indices.values():
        if len(pos) <= window:
            continue
        w = np.lib.stride_tricks.sliding_window_view(vals[pos], window, axis=0)  # (n, F, window)
        seq[pos[window:]] = w[:-1].transpose(0, 2, 1)       # window ending the day before
        valid[pos[window:]] = True

    doy = df["Date"].dt.dayofyear.to_numpy()
    static = [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
    static += [(df["City"] == c).to_numpy() for c in sorted(df["City"].unique())]
    return seq, np.column_stack(static).astype(np.float32), valid


class RecurrentForecaster(nn.Module):
    """LSTM or GRU over the window, then a small head on [last hidden state, static]."""

    def __init__(self, n_seq: int, n_static: int, cell: str, hidden: int):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_seq, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, seq, static):
        out, _ = self.rnn(seq)
        return self.head(torch.cat([out[:, -1], static], dim=1)).squeeze(1)


def _predict_scaled(model, seq, static) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(seq), torch.from_numpy(static)).numpy()


def train_model(seq, static, y, fit_idx, cfg: dict, epochs: int = MAX_EPOCHS,
                monitor_idx=None, patience: int = PATIENCE):
    """Train on rows fit_idx (scaled target y, L1 loss). With monitor_idx, stop early on
    their loss and return the best weights; otherwise train exactly `epochs` epochs.
    Returns (model, number of epochs of the returned weights)."""
    torch.manual_seed(config.RANDOM_STATE)
    model = RecurrentForecaster(seq.shape[2], static.shape[1], cfg["cell"], cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()
    X_seq, X_static = torch.from_numpy(seq[fit_idx]), torch.from_numpy(static[fit_idx])
    target = torch.from_numpy(y[fit_idx])
    gen = torch.Generator().manual_seed(config.RANDOM_STATE)

    best_loss, best_epoch, best_state = np.inf, epochs, None
    for epoch in range(1, epochs + 1):
        model.train()
        for b in torch.randperm(len(fit_idx), generator=gen).split(BATCH_SIZE):
            opt.zero_grad()
            loss_fn(model(X_seq[b], X_static[b]), target[b]).backward()
            opt.step()
        if monitor_idx is None:
            continue
        loss = float(np.mean(np.abs(_predict_scaled(model, seq[monitor_idx], static[monitor_idx]) - y[monitor_idx])))
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def fit_predict_deep(df: pd.DataFrame, cols, cfg: dict, train_splits, pred_split: str,
                     epochs: int = MAX_EPOCHS, early_stop: bool = False):
    """Fit on rows of train_splits (target known, full window) and predict pred_split.
    With early_stop, the loss on pred_split decides when to stop, so only use it when
    pred_split is the validation split. Returns (predictions on the AQI scale, epochs)."""
    scaler = fit_scaler(df, [*cols, config.TARGET], train_splits)
    seq, static, valid = make_windows(df, [*cols, config.TARGET], scaler, cfg["window"])
    mu, sd = scaler[0][config.TARGET], scaler[1][config.TARGET]
    y = ((df[config.TARGET] - mu) / sd).to_numpy(np.float32)

    has_y = df[config.TARGET].notna().to_numpy()
    fit_idx = np.flatnonzero(df["split"].isin(train_splits).to_numpy() & valid & has_y)
    pred_mask = (df["split"] == pred_split).to_numpy() & valid
    monitor_idx = np.flatnonzero(pred_mask & has_y) if early_stop else None
    model, n_epochs = train_model(seq, static, y, fit_idx, cfg, epochs, monitor_idx)

    pred = pd.Series(np.nan, index=df.index)
    pred[pred_mask] = _predict_scaled(model, seq[pred_mask], static[pred_mask]) * sd + mu
    return pred, n_epochs


def run_deep_models(df: pd.DataFrame, core, grid=GRID, max_epochs: int = MAX_EPOCHS):
    """Select on validation, then produce validation and test forecasts.

    Returns (predictions DataFrame with one pred_<cell> column, selection table, chosen
    configs with their epoch counts)."""
    cols = list(core)
    rows, val_preds = [], []
    for cfg in grid:
        pred_val, n_epochs = fit_predict_deep(df, cols, cfg, ["train"], "val", max_epochs, early_stop=True)
        rows.append({**cfg, "epochs": n_epochs, "val_MAE": pooled_mae(df, pred_val, "val"),
                     "val_rel_MAE": relative_mae(df, pred_val)})
        val_preds.append(pred_val)
    table = pd.DataFrame(rows)

    preds = pd.DataFrame(index=df.index)
    chosen, best_rows = {}, []
    for cell in table["cell"].unique():
        i = table.loc[table["cell"] == cell, "val_rel_MAE"].idxmin()
        cfg, n_epochs = grid[i], int(table.loc[i, "epochs"])
        pred_test, _ = fit_predict_deep(df, cols, cfg, ["train", "val"], "test", n_epochs)
        preds[f"pred_{cell}"] = val_preds[i].combine_first(pred_test)
        chosen[cell] = {**cfg, "epochs": n_epochs}
        best_rows.append(i)
    table["chosen"] = table.index.isin(best_rows)
    return preds, table, chosen
