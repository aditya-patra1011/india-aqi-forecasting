"""Chronological train / validation / test labelling"""
import pandas as pd
import numpy as np

from src import config

def assign_split(df: pd.DataFrame, train_end=config.TRAIN_END, val_end=config.VAL_END) -> pd.DataFrame:
    """Add a 'split' column. The full continous series is kept so lag and rolling
    features can be built across split boundaries using past data only"""
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if train_end >= val_end:
        raise ValueError("TRAIN_END must be earlier than VAL_END")
    dates = pd.to_datetime(df["Date"])
    out = df.copy()
    out["split"] = np.select([dates <= train_end, dates <= val_end], ["train", "val"], default="test")
    return out

def get_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {name}")
    return df[df["split"] == name].copy()
