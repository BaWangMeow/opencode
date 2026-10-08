"""Synthetic beam-diagnostics signal generation.

This module models a small set of *latent* physical beam parameters (energy,
current, transverse position, RF phase / amplitude) and maps them through a
linear mixing matrix into the *observed* diagnostic channels (BPMs, BCM, energy
monitor, RF stations, BLM, timing).  The latent factors follow slow
Ornstein--Uhlenbeck drifts plus a periodic ripple, so the resulting channels are
mutually correlated in a physically motivated way — which is exactly the
structure that association-based anomaly detectors (e.g. Anomaly Transformer)
are meant to exploit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class Channel:
    """Description of a single observed diagnostic channel."""

    name: str
    unit: str
    bias: float = 0.0
    mix: dict[str, float] = field(default_factory=dict)  # latent -> coefficient


# Latent physical factors.  ``theta`` is the mean-reversion rate (per sample);
# ``sigma`` is the innovation scale; ``period`` is the periodic ripple period
# (in samples).  Smaller theta -> longer correlation time.
LATENT_FACTORS: dict[str, dict[str, float]] = {
    "beam_energy":  {"theta": 0.001, "sigma": 0.05, "period": 2000.0},
    "beam_current": {"theta": 0.002, "sigma": 0.05, "period": 1500.0},
    "pos_x":        {"theta": 0.005, "sigma": 0.03, "period": 800.0},
    "pos_y":        {"theta": 0.005, "sigma": 0.03, "period": 900.0},
    "rf_phase":     {"theta": 0.001, "sigma": 0.04, "period": 1200.0},
    "rf_amp":       {"theta": 0.001, "sigma": 0.04, "period": 1000.0},
}

# Observed channels: bias + linear combination of latent factors.
CHANNELS: tuple[Channel, ...] = (
    Channel("BPM1.x", "mm", 0.0, {"pos_x": 1.0}),
    Channel("BPM1.y", "mm", 0.0, {"pos_y": 1.0}),
    Channel("BPM2.x", "mm", 0.0, {"pos_x": 0.8, "beam_energy": 0.2}),
    Channel("BPM2.y", "mm", 0.0, {"pos_y": 0.9, "beam_current": 0.05}),
    Channel("BCM.charge", "nC", 1.0, {"beam_current": 1.0}),
    Channel("EM.energy", "MeV", 0.0, {"beam_energy": 1.0}),
    Channel("RF1.phase", "deg", 0.0, {"rf_phase": 1.0}),
    Channel("RF1.amp", "a.u.", 1.0, {"rf_amp": 1.0, "beam_current": 0.3}),
    Channel("RF2.phase", "deg", 0.0, {"rf_phase": 0.9, "beam_energy": 0.1}),
    Channel("RF2.amp", "a.u.", 1.0, {"rf_amp": 0.9}),
    Channel("BLM.signal", "a.u.", 0.0, {}),  # near zero unless beam loss
    Channel("Timing.jitter", "ps", 0.0, {"beam_current": 0.02}),
)


def generate_latent_factors(
    n_samples: int, seed: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """Generate standardized latent factors.

    Returns
    -------
    latents : np.ndarray, shape (n_samples, n_latent)
        Each column is zero-mean / unit-std.
    names : list[str]
        Latent factor names in column order.
    """
    rng = np.random.default_rng(seed)
    names = list(LATENT_FACTORS.keys())
    cols: list[np.ndarray] = []
    for name in names:
        p = LATENT_FACTORS[name]
        a = float(np.exp(-p["theta"]))
        noise = rng.standard_normal(n_samples) * p["sigma"]
        drift = lfilter([1.0], [1.0, -a], noise)
        t = np.arange(n_samples)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        ripple = 0.1 * np.sin(2.0 * np.pi * t / p["period"] + phase)
        col = drift + ripple
        col = (col - col.mean()) / (col.std() + 1e-9)
        cols.append(col)
    return np.column_stack(cols), names


def mix_channels(
    latents: np.ndarray,
    latent_names: list[str],
    noise_level: float,
    seed: int | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Map latent factors to observed channels.

    Returns
    -------
    data : np.ndarray, shape (n_samples, n_channels)
    channel_names : list[str]
    channel_units : list[str]
    """
    rng = np.random.default_rng(seed)
    latent_idx = {name: i for i, name in enumerate(latent_names)}
    n_samples = latents.shape[0]
    out = np.zeros((n_samples, len(CHANNELS)))
    names: list[str] = []
    units: list[str] = []
    for j, ch in enumerate(CHANNELS):
        col = np.full(n_samples, ch.bias, dtype=float)
        for latent, coeff in ch.mix.items():
            col = col + coeff * latents[:, latent_idx[latent]]
        out[:, j] = col
        names.append(ch.name)
        units.append(ch.unit)
    out = out + noise_level * rng.standard_normal(out.shape)
    return out, names, units
