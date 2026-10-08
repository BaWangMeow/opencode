"""Export a generated dataset to HDF5 / NPZ / CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .anomalies import AnomalyType
from .generator import DiagnosticsDataset

ANOMALY_NAMES = {int(t): t.name.lower() for t in AnomalyType}


def to_hdf5(dataset: DiagnosticsDataset, path: str | Path) -> Path:
    """Save the dataset as HDF5 (primary format, XFEL/NeXus-adjacent)."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=dataset.data, compression="gzip", shuffle=True)
        f.create_dataset("label", data=dataset.label, compression="gzip")
        f.create_dataset("type_label", data=dataset.type_label, compression="gzip")
        f.create_dataset("channel_label", data=dataset.channel_label, compression="gzip")
        f.attrs["sampling_rate"] = dataset.sampling_rate
        f.attrs["seed"] = dataset.seed
        f.attrs["channel_names"] = list(dataset.channel_names)
        f.attrs["channel_units"] = list(dataset.channel_units)
        f.attrs["anomaly_type_names"] = [ANOMALY_NAMES[i] for i in sorted(ANOMALY_NAMES)]

        if dataset.events:
            ev = np.array(
                [
                    (int(e.type), e.start, e.end, round(e.amplitude, 4),
                     ",".join(e.channels).encode())
                    for e in dataset.events
                ],
                dtype=[
                    ("type", "i4"), ("start", "i4"), ("end", "i4"),
                    ("amplitude", "f8"), ("channels", "S64"),
                ],
            )
            f.create_dataset("events", data=ev)
    return path


def to_npz(dataset: DiagnosticsDataset, path: str | Path) -> Path:
    """Save the dataset as a NumPy ``.npz`` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        data=dataset.data,
        label=dataset.label,
        type_label=dataset.type_label,
        channel_label=dataset.channel_label,
        channel_names=np.array(dataset.channel_names),
        channel_units=np.array(dataset.channel_units),
        sampling_rate=np.array(dataset.sampling_rate),
        seed=np.array(dataset.seed),
    )
    return path


def to_csv(dataset: DiagnosticsDataset, path: str | Path) -> Path:
    """Save the dataset as CSV (for quick inspection / spreadsheet)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(["time"] + dataset.channel_names + ["label", "type_label"])
    t = np.arange(dataset.data.shape[0]) / dataset.sampling_rate
    stack = np.column_stack([t, dataset.data, dataset.label, dataset.type_label])
    np.savetxt(path, stack, delimiter=",", header=header, comments="")
    return path
