"""
TransferAutoencoder — pretrain on UNSW-NB15 benign flows, transfer to CIC-IDS2017.

Training protocol (4-experiment comparison, E3/E4 in architecture.md):
  E1 — IsolationForest          (baseline, no deep learning)
  E2 — MLPAutoencoder CIC-only  (no transfer)
  E3 — TransferAutoencoder, encoder frozen throughout  (pretrained, no adaptation)
  E4 — TransferAutoencoder, encoder unfrozen after freeze_epochs  (full fine-tune)

Architecture:
  Pretrain (source domain, UNSW-NB15):
    X_src (src_dim) → encoder(src_dim → bottleneck) → decoder_src(bottleneck → src_dim)

  Transfer (target domain, CIC-IDS2017):
    X_tgt (tgt_dim) → FeatureAlignment(tgt_dim → src_dim)
                     → encoder  [frozen for freeze_epochs, then unfrozen]
                     → decoder_tgt(bottleneck → tgt_dim)  [always trainable]

  Anomaly score: mean per-row MSE on X_tgt reconstruction.
  Threshold:     p-percentile on benign validation flows (same as MLPAutoencoder).

If pretrain() is never called, TransferAutoencoder falls back to plain MLP-AE
behaviour (no alignment, random encoder init).
"""
from __future__ import annotations
from typing import Callable
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.base import PreprocessingSpec


# ── Network building blocks ───────────────────────────────────────────────────

def _build_encoder(input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    return nn.Sequential(*layers)


def _build_decoder(hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    rev = list(reversed(hidden_dims))
    layers: list[nn.Module] = []
    for i in range(len(rev) - 1):
        layers += [nn.Linear(rev[i], rev[i + 1]), nn.ReLU()]
    layers.append(nn.Linear(rev[-1], output_dim))
    return nn.Sequential(*layers)


class _SourceAENet(nn.Module):
    """Plain AE for pretraining on the source domain (no alignment layer)."""
    def __init__(self, src_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.encoder = _build_encoder(src_dim, hidden_dims)
        self.decoder = _build_decoder(hidden_dims, src_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _TransferAENet(nn.Module):
    """
    Transfer AE: alignment → shared encoder (pretrained) → new decoder.
    The alignment projects target features into the source encoder's input space.
    """
    def __init__(self, tgt_dim: int, src_dim: int, hidden_dims: list[int]):
        super().__init__()
        # Alignment layer maps target feature space → source feature space
        self.align = nn.Linear(tgt_dim, src_dim)
        self.encoder = _build_encoder(src_dim, hidden_dims)
        self.decoder = _build_decoder(hidden_dims, tgt_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.align(x))
        return self.decoder(z)


# ── Public class ─────────────────────────────────────────────────────────────

class TransferAutoencoder:
    """
    Transfer learning autoencoder: pretrain on UNSW-NB15 → transfer to CIC-IDS2017.

    Usage (E4 experiment — pretrained + fine-tuned):
        det = TransferAutoencoder()
        det.pretrain(X_unsw_benign, input_dim_source=X_unsw_benign.shape[1])
        det.fit(X_cic_benign_train, X_benign_val=X_cic_benign_val)

    Usage (E2 baseline — same API, skip pretrain):
        det = TransferAutoencoder()
        det.fit(X_cic_benign_train, X_benign_val=X_cic_benign_val)
    """

    preprocessing = PreprocessingSpec(
        needs_scaling=True,
        preferred_scalers=("quantile", "robust"),
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

        self._src_net: _SourceAENet | None = None
        self._net: _SourceAENet | _TransferAENet | None = None
        self.threshold: float | None = None
        self.input_dim: int | None = None    # target input dim
        self._src_dim: int | None = None      # source input dim (None if no pretrain)
        self._pretrained: bool = False

    @staticmethod
    def _device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Pretrain on source domain ─────────────────────────────────────────────

    def pretrain(self, X_source: np.ndarray,
                 input_dim_source: int | None = None) -> None:
        """
        Train a full AE on source-domain benign flows (e.g. UNSW-NB15).
        Encoder weights are saved for transfer in fit().
        """
        src_dim = input_dim_source if input_dim_source is not None else X_source.shape[1]
        self._src_dim = src_dim
        self._src_net = _SourceAENet(src_dim, self.hidden_dims).to(self._device())
        self._train_net(self._src_net, X_source, freeze_encoder_epochs=0)
        self._pretrained = True

    # ── Fit on target domain (with or without pretrained init) ───────────────

    def fit(self, X_benign: np.ndarray,
            X_benign_val: np.ndarray | None = None,
            on_progress: Callable[[str], None] | None = None) -> None:
        """
        Fine-tune on target-domain benign flows.

        With pretrain(): builds _TransferAENet, copies encoder from source,
          trains alignment + decoder for freeze_epochs (encoder frozen),
          then trains all layers.

        Without pretrain(): behaves like plain MLPAutoencoder (E2 baseline).
        """
        tgt_dim = X_benign.shape[1]
        self.input_dim = tgt_dim

        if self._pretrained and self._src_net is not None:
            src_dim = self._src_dim
            net = _TransferAENet(tgt_dim, src_dim, self.hidden_dims).to(self._device())
            # Copy pretrained encoder weights into transfer network
            net.encoder.load_state_dict(self._src_net.encoder.state_dict())
            self._net = net
            self._train_net(net, X_benign, freeze_encoder_epochs=self.freeze_epochs,
                            on_progress=on_progress)
        else:
            # No pretraining — plain AE (E2 baseline path)
            self._net = _SourceAENet(tgt_dim, self.hidden_dims).to(self._device())
            self._train_net(self._net, X_benign, freeze_encoder_epochs=0,
                            on_progress=on_progress)

        threshold_data = X_benign_val if X_benign_val is not None else X_benign
        self.threshold = float(
            np.percentile(self._reconstruction_errors(threshold_data),
                          self.anomaly_threshold_percentile)
        )

    def fine_tune(self, X_incremental: np.ndarray,
                  X_replay: np.ndarray | None = None,
                  on_progress: Callable[[str], None] | None = None) -> None:
        """HITL fine-tune loop: update threshold on analyst-vetted flows."""
        if self._net is None:
            raise RuntimeError("Call fit() or load() before fine_tune().")
        self._net = self._net.to(self._device())
        X = np.vstack([X_incremental, X_replay]) if X_replay is not None else X_incremental
        self._train_net(self._net, X, freeze_encoder_epochs=self.freeze_epochs,
                        on_progress=on_progress)
        self.threshold = float(
            np.percentile(self._reconstruction_errors(X_incremental),
                          self.anomaly_threshold_percentile)
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(self, x: np.ndarray) -> float:
        return float(self._reconstruction_errors(x.reshape(1, -1))[0])

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        return self._reconstruction_errors(X)

    def is_anomaly(self, x: np.ndarray) -> tuple[bool, float]:
        s = self.score(x)
        return s > (self.threshold or 0.0), s

    def feature_errors(self, x: np.ndarray) -> np.ndarray:
        """Per-feature squared reconstruction error for analyst explanation panel."""
        device = self._device()
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor(x.reshape(1, -1), dtype=torch.float32).to(device)
            recon = self._net(t).cpu().numpy()
        return ((x.reshape(1, -1) - recon) ** 2).flatten()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        src_sd = (
            {k: v.cpu() for k, v in self._src_net.state_dict().items()}
            if self._src_net else None
        )
        net_sd = (
            {k: v.cpu() for k, v in self._net.state_dict().items()}
            if self._net else None
        )
        joblib.dump({
            "hidden_dims":   self.hidden_dims,
            "input_dim":     self.input_dim,
            "src_dim":       self._src_dim,
            "pretrained":    self._pretrained,
            "threshold":     self.threshold,
            "epochs":        self.epochs,
            "batch_size":    self.batch_size,
            "lr":            self.lr,
            "anomaly_threshold_percentile": self.anomaly_threshold_percentile,
            "freeze_epochs": self.freeze_epochs,
            "src_state_dict": src_sd,
            "net_state_dict": net_sd,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TransferAutoencoder":
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
        obj._src_dim = d.get("src_dim")
        obj._pretrained = d.get("pretrained", False)
        obj.threshold = d["threshold"]

        src_sd = d.get("src_state_dict")
        if src_sd and d.get("src_dim"):
            obj._src_net = _SourceAENet(d["src_dim"], d["hidden_dims"])
            obj._src_net.load_state_dict(src_sd)
            obj._src_net = obj._src_net.to(cls._device())

        net_sd = d.get("net_state_dict")
        if net_sd and d["input_dim"]:
            if d.get("pretrained") and d.get("src_dim") and d["src_dim"] != d["input_dim"]:
                net = _TransferAENet(d["input_dim"], d["src_dim"], d["hidden_dims"])
            else:
                net = _SourceAENet(d["input_dim"], d["hidden_dims"])
            net.load_state_dict(net_sd)
            obj._net = net.to(cls._device())

        return obj

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _train_net(
        self,
        net: nn.Module,
        X: np.ndarray,
        freeze_encoder_epochs: int,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        device = self._device()
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)
        report_every = max(1, self.epochs // 5)

        for epoch in range(self.epochs):
            frozen = freeze_encoder_epochs and epoch < freeze_encoder_epochs
            # Freeze/unfreeze encoder parameters only (alignment + decoder always train)
            encoder = getattr(net, "encoder", None)
            if encoder is not None:
                for p in encoder.parameters():
                    p.requires_grad = not frozen
            net.train()
            epoch_loss = 0.0
            n_batches = 0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = criterion(net(batch), batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if on_progress and (epoch + 1) % report_every == 0:
                avg_loss = epoch_loss / n_batches if n_batches else 0.0
                on_progress(
                    f"  AE epoch {epoch + 1}/{self.epochs}  "
                    f"loss={avg_loss:.6f}"
                    + ("  [encoder frozen]" if frozen else "")
                )

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        device = self._device()
        self._net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(device)
            recon = self._net(t).cpu().numpy()
        return np.mean((X - recon) ** 2, axis=1)
