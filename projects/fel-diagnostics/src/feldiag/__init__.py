"""feldiag — synthetic beam-diagnostics data generator for FEL anomaly detection.

This package generates realistic multi-channel beam-diagnostics time series
(BPM, beam current, energy, RF, beam-loss, timing) with injected anomalies of
several types, together with ground-truth labels.  It is the data foundation
for the online anomaly-detection subsystem of the thesis.
"""

from .config import AnomalyConfig, GeneratorConfig
from .generator import DiagnosticsDataset, generate
from .anomalies import AnomalyType, InjectedAnomaly
from .export import to_csv, to_hdf5, to_npz

__all__ = [
    "AnomalyConfig",
    "GeneratorConfig",
    "DiagnosticsDataset",
    "generate",
    "AnomalyType",
    "InjectedAnomaly",
    "to_hdf5",
    "to_npz",
    "to_csv",
]

__version__ = "0.1.0"
