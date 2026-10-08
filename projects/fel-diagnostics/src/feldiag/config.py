"""Configuration dataclasses for the FEL diagnostics data generator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnomalyConfig:
    """Controls how anomalies are injected into the timeline.

    Probabilities are relative weights used to sample the anomaly type of each
    injected event (they are normalized internally, so they need not sum to 1).
    """

    # Number of anomalies to inject.  A positive value is used directly;
    # 0 derives a count from ``anomaly_density``.
    n_anomalies: int = 30

    # Relative weights for sampling anomaly type.
    point_prob: float = 0.20
    drift_prob: float = 0.25
    oscillation_prob: float = 0.20
    level_shift_prob: float = 0.20
    noise_prob: float = 0.15

    # Duration ranges (in samples).
    point_duration: tuple[int, int] = (3, 8)
    drift_duration: tuple[int, int] = (200, 800)
    oscillation_duration: tuple[int, int] = (200, 800)
    level_shift_duration: tuple[int, int] = (200, 600)
    noise_duration: tuple[int, int] = (200, 600)

    # Amplitude ranges (in units of the target channel's std).
    point_amplitude: tuple[float, float] = (8.0, 15.0)
    drift_amplitude: tuple[float, float] = (3.0, 8.0)
    oscillation_amplitude: tuple[float, float] = (2.0, 6.0)
    level_shift_amplitude: tuple[float, float] = (3.0, 8.0)
    noise_factor: tuple[float, float] = (3.0, 6.0)

    # Minimum gap (in samples) between consecutive anomalies.
    min_separation: int = 200

    # Fallback: target fraction of the timeline covered by anomalies
    # (only used when ``n_anomalies == 0``).
    anomaly_density: float = 0.10


@dataclass
class GeneratorConfig:
    """Top-level configuration for generating a diagnostics dataset."""

    n_samples: int = 50_000
    sampling_rate: float = 10.0  # Hz
    noise_level: float = 0.1     # additive measurement-noise scale
    seed: int = 42
    anomalies: AnomalyConfig = field(default_factory=AnomalyConfig)
