"""
MLP (fully-connected) Autoencoder anomaly detector.

Trained exclusively on benign flows; anomaly = reconstruction error above
the p-percentile threshold computed on benign validation data.

Configuration is passed via constructor arguments so this module has no
dependency on the API config layer — it can be used standalone in tests or
batch scripts without importing the web framework.
"""
from __future__ import annotations
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.base import PreprocessingSpec


class _AENet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        enc, dec = [], []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        rev = list(reversed(dims))
        for i in range(len(rev) - 1):
            dec += [nn.Linear(rev[i], rev[i + 1])]
            if i < len(rev) - 2:
                dec.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class MLPAutoencoder:
    """
    Multi-Layer Perceptron autoencoder trained on benign flows.
    Anomaly = reconstruction error (mean MSE) above a percentile threshold.

    Fine-tune: load previous weights → freeze encoder N epochs → unfreeze all.
    Per-feature explainability: (x - x̂)² — no SHAP needed.
    """

    preprocessing = PreprocessingSpec(
        needs_scaling=True,
        preferred_scalers=("robust", "quantile"),
        supervised=False,
    )

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        anomaly_threshold_percentile: float = 95.0,
        freeze_epochs: int = 5,
    ):
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self.freeze_epochs = freeze_epochs
        self._net: _AENet | None = None
        self.threshold: float | None = None
        self.input_dim: int | None = None

    def fit(self, X_benign: np.ndarray) -> None:
        self.input_dim = X_benign.shape[1]
        self._net = _AENet(self.input_dim, self.hidden_dims)
        self._train(X_benign, freeze_encoder_epochs=0)
        self.threshold = float(
            np.percentile(self._reconstruction_errors(X_benign),
                          self.anomaly_threshold_percentile)
        )

    def fine_tune(self, X_incremental: np.ndarray,
                  X_replay: np.ndarray | None = None) -> None:
        if self._net is None:
            raise RuntimeError("Call fit() or load() before fine_tune().")
        X = np.vstack([X_incremental, X_replay]) if X_replay is not None else X_incremental
        self._train(X, freeze_encoder_epochs=self.freeze_epochs)
        self.threshold = float(
            np.percentile(self._reconstruction_errors(X_incremental),
                          self.anomaly_threshold_percentile)
        )

    def score(self, x: np.ndarray) -> float:
        """Total reconstruction error (mean MSE across all features)."""
        return float(self._reconstruction_errors(x.reshape(1, -1))[0])

    def feature_errors(self, x: np.ndarray) -> np.ndarray:
        """
        Per-feature squared reconstruction error (x - x̂)².
        High error on a feature = AE failed to reconstruct it = anomalous feature.
        Use this in the analyst alert panel instead of SHAP.
        """
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
            recon = self._net(t).numpy()
        return ((x.reshape(1, -1) - recon) ** 2).flatten()

    def is_anomaly(self, x: np.ndarray) -> tuple[bool, float]:
        s = self.score(x)
        return s > (self.threshold or 0.0), s

    def _train(self, X: np.ndarray, freeze_encoder_epochs: int) -> None:
        net = self._net
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size,
                            shuffle=True)
        for epoch in range(self.epochs):
            frozen = freeze_encoder_epochs and epoch < freeze_encoder_epochs
            for p in net.encoder.parameters():
                p.requires_grad = not frozen
            net.train()
            for (batch,) in loader:
                optimizer.zero_grad()
                loss = criterion(net(batch), batch)
                loss.backward()
                optimizer.step()

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            recon = self._net(t).numpy()
        return np.mean((X - recon) ** 2, axis=1)

    def save(self, path: str) -> None:
        joblib.dump({
            "state_dict":   self._net.state_dict() if self._net else None,
            "hidden_dims":  self.hidden_dims,
            "input_dim":    self.input_dim,
            "threshold":    self.threshold,
            "epochs":       self.epochs,
            "batch_size":   self.batch_size,
            "lr":           self.lr,
            "anomaly_threshold_percentile": self.anomaly_threshold_percentile,
            "freeze_epochs": self.freeze_epochs,
        }, path)

    @classmethod
    def load(cls, path: str) -> "MLPAutoencoder":
        d = joblib.load(path)
        obj = cls(
            hidden_dims=d["hidden_dims"],
            epochs=d["epochs"],
            batch_size=d["batch_size"],
            lr=d["lr"],
            anomaly_threshold_percentile=d.get("anomaly_threshold_percentile", 95.0),
            freeze_epochs=d.get("freeze_epochs", 5),
        )
        obj.input_dim = d["input_dim"]
        obj.threshold = d["threshold"]
        if d["state_dict"] and d["input_dim"]:
            obj._net = _AENet(d["input_dim"], d["hidden_dims"])
            obj._net.load_state_dict(d["state_dict"])
        return obj
