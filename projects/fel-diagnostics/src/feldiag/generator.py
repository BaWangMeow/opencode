"""Top-level dataset generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anomalies import AnomalyInjector, InjectedAnomaly
from .config import GeneratorConfig
from .signals import generate_latent_factors, mix_channels


@dataclass
class DiagnosticsDataset:
    """A generated diagnostics dataset with ground-truth labels."""

    data: np.ndarray              # (n_samples, n_channels)
    label: np.ndarray             # (n_samples,) binary global anomaly label
    type_label: np.ndarray        # (n_samples,) anomaly type code
    channel_label: np.ndarray     # (n_samples, n_channels) per-channel label
    channel_names: list[str]
    channel_units: list[str]
    sampling_rate: float
    events: list[InjectedAnomaly]
    seed: int


def generate(config: GeneratorConfig | None = None) -> DiagnosticsDataset:
    """Generate a synthetic diagnostics dataset.

    Parameters
    ----------
    config : GeneratorConfig, optional
        Generation parameters.  A default config is used when omitted.

    Returns
    -------
    DiagnosticsDataset
    """
    config = config or GeneratorConfig()

    latents, latent_names = generate_latent_factors(config.n_samples, seed=config.seed)
    clean, names, units = mix_channels(
        latents, latent_names, config.noise_level, seed=config.seed + 1
    )

    injector = AnomalyInjector(config.anomalies, names, seed=config.seed + 2)
    data, label, type_label, channel_label, events = injector.inject(clean)

    return DiagnosticsDataset(
        data=data,
        label=label,
        type_label=type_label,
        channel_label=channel_label,
        channel_names=names,
        channel_units=units,
        sampling_rate=config.sampling_rate,
        events=events,
        seed=config.seed,
    )
