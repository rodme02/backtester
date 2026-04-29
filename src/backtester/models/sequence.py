"""PyTorch sequence-classifier wrappers (LSTM and TCN).

Both models share the project's uniform ``fit / predict_proba``
interface so the eval harness drives them identically. Inputs are
``(N, T, F)`` tensors where ``T`` is the lookback window, ``F`` the
number of features per timestep. CPU-only; deterministic seed.

The wrappers expect a *flat* ``(N, T*F)`` DataFrame at fit/predict
time (column order = sequence × feature, time-major). Helper
``stack_sequences`` converts a tidy long DataFrame into that shape
without leakage.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# torch is imported lazily so the base install (without `[ml]`)
# can still import the rest of the package.


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # some ops lack det impls; we accept noise


def stack_sequences(
    feats: pd.DataFrame,
    *,
    lookback: int,
    feature_cols: list[str],
    group_col: str = "ticker",
    date_index_level: str = "datetime",
) -> tuple[pd.DataFrame, pd.MultiIndex]:
    """Build a (N, lookback × F) flat DataFrame from a tidy long panel.

    ``feats`` must have a ``(datetime, ticker)`` MultiIndex. For each
    (date t, ticker), we stack rows ``[t-lookback+1 .. t]`` into one
    row of length ``lookback * len(feature_cols)``. Rows where the
    full window isn't available are dropped.

    Returns the flattened DataFrame and the surviving MultiIndex.
    """
    out_index_pairs: list[tuple[pd.Timestamp, str]] = []
    out_rows: list[np.ndarray] = []
    n_features = len(feature_cols)
    flat_cols = [
        f"t-{lookback - 1 - tau}__{f}"
        for tau in range(lookback)
        for f in feature_cols
    ]

    for ticker, sub in feats.groupby(level=group_col, sort=False):
        sub = sub.sort_index().droplevel(group_col)
        arr = sub[feature_cols].to_numpy(dtype=np.float32)
        if arr.shape[0] < lookback:
            continue
        n_windows = arr.shape[0] - lookback + 1
        # sliding window: shape (n_windows, lookback, n_features)
        idx = np.arange(lookback)[None, :] + np.arange(n_windows)[:, None]
        windows = arr[idx]  # (n_windows, lookback, n_features)
        flat = windows.reshape(n_windows, lookback * n_features)
        out_rows.append(flat)
        # End date of each window is sub.index at position lookback-1, lookback, ...
        end_dates = sub.index.get_level_values(date_index_level)[lookback - 1 :]
        out_index_pairs.extend((d, ticker) for d in end_dates)

    if not out_rows:
        return pd.DataFrame(columns=flat_cols), pd.MultiIndex.from_tuples(
            [], names=[date_index_level, group_col]
        )

    flat_arr = np.concatenate(out_rows, axis=0)
    mi = pd.MultiIndex.from_tuples(out_index_pairs, names=[date_index_level, group_col])
    return pd.DataFrame(flat_arr, index=mi, columns=flat_cols), mi


@dataclass
class _BaseSequenceClassifier:
    lookback: int = 30
    n_features: int = 1
    hidden: int = 64
    epochs: int = 8
    batch_size: int = 1024
    learning_rate: float = 1e-3
    random_state: int = 17
    device: str = "cpu"

    _model: Any = field(default=None, init=False, repr=False)
    _flat_columns: list[str] = field(default_factory=list, init=False, repr=False)

    def _build_model(self):  # pragma: no cover - subclassed
        raise NotImplementedError

    def _reshape(self, X: pd.DataFrame):
        import torch

        n = X.shape[0]
        arr = X.to_numpy(dtype=np.float32).reshape(n, self.lookback, self.n_features)
        return torch.from_numpy(arr).to(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> _BaseSequenceClassifier:
        import torch
        from torch import nn, optim

        _seed_everything(self.random_state)
        self._flat_columns = list(X.columns)
        if X.shape[1] != self.lookback * self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} columns but expected {self.lookback}*"
                f"{self.n_features} = {self.lookback * self.n_features}"
            )

        x_tensor = self._reshape(X)
        y_arr = np.asarray(y, dtype=np.float32)
        y_tensor = torch.from_numpy(y_arr).to(self.device)

        self._model = self._build_model().to(self.device)
        loss_fn = nn.BCEWithLogitsLoss()
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate)

        n = x_tensor.shape[0]
        n_batches = math.ceil(n / self.batch_size)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for b in range(n_batches):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                xb = x_tensor[idx]
                yb = y_tensor[idx]
                logits = self._model(xb).squeeze(-1)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        X = X[self._flat_columns]
        x_tensor = self._reshape(X)
        with torch.no_grad():
            logits = self._model(x_tensor).squeeze(-1)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba.astype(float)


class _LSTMNet:
    """Small LSTM head: lstm -> last hidden -> linear -> logit."""

    def __init__(self, n_features: int, hidden: int):
        from torch import nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features, hidden_size=hidden, batch_first=True
                )
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        self.net = _Net()


class _TCNNet:
    """Small dilated-causal TCN: 3 blocks of (conv -> relu) with dilations 1,2,4."""

    def __init__(self, n_features: int, hidden: int):
        from torch import nn

        class _CausalConv1d(nn.Conv1d):
            def __init__(self, in_c, out_c, kernel_size, dilation):
                super().__init__(
                    in_c, out_c, kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation,
                )
                self._chomp = (kernel_size - 1) * dilation

            def forward(self, x):
                out = super().forward(x)
                return out[..., : -self._chomp] if self._chomp else out

        class _Block(nn.Module):
            def __init__(self, in_c, out_c, dilation):
                super().__init__()
                self.conv = _CausalConv1d(in_c, out_c, kernel_size=3, dilation=dilation)
                self.act = nn.ReLU()

            def forward(self, x):
                return self.act(self.conv(x))

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.b1 = _Block(n_features, hidden, dilation=1)
                self.b2 = _Block(hidden, hidden, dilation=2)
                self.b3 = _Block(hidden, hidden, dilation=4)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                # x: (N, T, F) -> (N, F, T) for conv
                x = x.transpose(1, 2)
                x = self.b1(x)
                x = self.b2(x)
                x = self.b3(x)
                # take last timestep
                x = x[..., -1]
                return self.head(x)

        self.net = _Net()


class _TransformerNet:
    """Small encoder-only Transformer: input proj → 2 self-attention layers → last-token head."""

    def __init__(self, n_features: int, hidden: int, n_heads: int = 2, n_layers: int = 2):
        from torch import nn

        head_dim = hidden // n_heads
        d_model = head_dim * n_heads  # ensure divisibility

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(n_features, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=2 * d_model,
                    dropout=0.0,
                    batch_first=True,
                    activation="relu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.head = nn.Linear(d_model, 1)

            def forward(self, x):
                # x: (N, T, F) -> (N, T, d_model)
                x = self.proj(x)
                x = self.encoder(x)
                return self.head(x[:, -1, :])

        self.net = _Net()


@dataclass
class LSTMClassifier(_BaseSequenceClassifier):
    def _build_model(self):
        return _LSTMNet(self.n_features, self.hidden).net


@dataclass
class TCNClassifier(_BaseSequenceClassifier):
    def _build_model(self):
        return _TCNNet(self.n_features, self.hidden).net


@dataclass
class TransformerClassifier(_BaseSequenceClassifier):
    n_heads: int = 2
    n_layers: int = 2

    def _build_model(self):
        return _TransformerNet(
            self.n_features, self.hidden,
            n_heads=self.n_heads, n_layers=self.n_layers,
        ).net
