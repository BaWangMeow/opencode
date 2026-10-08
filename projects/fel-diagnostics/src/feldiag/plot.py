"""Quick visualization of a generated dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .anomalies import AnomalyType
from .generator import DiagnosticsDataset

_COLORS = {
    1: "tab:red",      # point
    2: "tab:orange",   # drift
    3: "tab:green",    # oscillation
    4: "tab:purple",   # level shift
    5: "tab:brown",    # noise
}


def plot_dataset(
    dataset: DiagnosticsDataset,
    start: int = 0,
    end: int = 2000,
    n_channels: int | None = None,
    title: str | None = None,
):
    """Plot a slice of the dataset, shading injected anomaly segments."""
    data = dataset.data
    names = dataset.channel_names
    units = dataset.channel_units
    n = data.shape[1] if n_channels is None else min(n_channels, data.shape[1])

    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(12, 2.0 * n))
    if n == 1:
        axes = [axes]

    tt = np.arange(data.shape[0]) / dataset.sampling_rate
    t = tt[start:end]
    for i in range(n):
        ax = axes[i]
        ax.plot(t, data[start:end, i], lw=0.6, color="k", alpha=0.85)
        ax.set_ylabel(f"{names[i]}\n({units[i]})", fontsize=7)
        ax.grid(alpha=0.2)

    for ev in dataset.events:
        if ev.end <= start or ev.start >= end:
            continue
        c0, c1 = max(ev.start, start), min(ev.end, end)
        color = _COLORS.get(int(ev.type), "gray")
        for ch in ev.channels:
            j = names.index(ch)
            if j < n:
                axes[j].axvspan(tt[c0], tt[c1], color=color, alpha=0.35)

    axes[-1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
