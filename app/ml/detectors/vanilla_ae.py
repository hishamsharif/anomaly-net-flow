from __future__ import annotations
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import settings


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


class VanillaAE:
    """
    MLP autoencoder trained on benign flows.
    Anomaly = reconstruction error above p95 threshold on benign validation split.

    Fine-tune: load previous weights → freeze encoder N epochs → unfreeze all.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
    ):
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._net: _AENet | None = None
        self.threshold: float | None = None
        self.input_dim: int | None = None

    # ------------------------------------------------------------------
    def fit(self, X_benign: np.ndarray) -> None:
        self.input_dim = X_benign.shape[1]
        self._net = _AENet(self.input_dim, self.hidden_dims)
        self._train(X_benign, freeze_encoder_epochs=0)
        self.threshold = float(
            np.percentile(self._reconstruction_errors(X_benign),
                          settings.ae_anomaly_threshold_percentile)
        )

    def fine_tune(self, X_incremental: np.ndarray, X_replay: np.ndarray | None = None) -> None:
        if self._net is None:
            raise RuntimeError("Call fit() or load() before fine_tune().")
        X = np.vstack([X_incremental, X_replay]) if X_replay is not None else X_incremental
        X_benign = X  # fine-tune set is already analyst-labelled benign + confirmed flows
        self._train(X_benign, freeze_encoder_epochs=settings.ae_freeze_epochs)
        # Update threshold on the incremental benign portion
        self.threshold = float(
            np.percentile(self._reconstruction_errors(X_incremental),
                          settings.ae_anomaly_threshold_percentile)
        )

    # ------------------------------------------------------------------
    def score(self, x: np.ndarray) -> float:
        return float(self._reconstruction_errors(x.reshape(1, -1))[0])

    def is_anomaly(self, x: np.ndarray) -> tuple[bool, float]:
        s = self.score(x)
        return s > (self.threshold or 0.0), s

    # ------------------------------------------------------------------
    def _train(self, X: np.ndarray, freeze_encoder_epochs: int) -> None:
        net = self._net
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            if freeze_encoder_epochs and epoch < freeze_encoder_epochs:
                for p in net.encoder.parameters():
                    p.requires_grad = False
            else:
                for p in net.encoder.parameters():
                    p.requires_grad = True

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

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump({
            "state_dict": self._net.state_dict() if self._net else None,
            "hidden_dims": self.hidden_dims,
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }, path)

    @classmethod
    def load(cls, path: str) -> "VanillaAE":
        data = joblib.load(path)
        obj = cls(hidden_dims=data["hidden_dims"], epochs=data["epochs"],
                  batch_size=data["batch_size"], lr=data["lr"])
        obj.input_dim = data["input_dim"]
        obj.threshold = data["threshold"]
        if data["state_dict"] and data["input_dim"]:
            obj._net = _AENet(data["input_dim"], data["hidden_dims"])
            obj._net.load_state_dict(data["state_dict"])
        return obj
