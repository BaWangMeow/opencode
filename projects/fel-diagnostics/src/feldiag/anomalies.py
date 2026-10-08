"""Anomaly injection for synthetic diagnostics time series.

Each anomaly modifies a contiguous segment of one or more channels and records
ground-truth labels:

* ``label``         -- (n_samples,)         global binary label (1 = anomalous)
* ``type_label``    -- (n_samples,)         anomaly type code (0 = normal)
* ``channel_label`` -- (n_samples, n_chan)  per-channel binary label
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import AnomalyConfig


class AnomalyType(IntEnum):
    NORMAL = 0
    POINT = 1        # beam-loss spike
    DRIFT = 2        # slow BPM offset drift
    OSCILLATION = 3  # transverse instability
    LEVEL_SHIFT = 4  # RF phase / amplitude step
    NOISE = 5        # variance increase (e.g. timing jitter)


# Channel groups that each anomaly type may target (one group per event).
ANOMALY_CHANNELS: dict[AnomalyType, list[list[str]]] = {
    AnomalyType.POINT: [["BLM.signal", "BCM.charge"]],
    AnomalyType.DRIFT: [
        ["BPM1.x"], ["BPM1.y"], ["BPM2.x"], ["BPM2.y"],
    ],
    AnomalyType.OSCILLATION: [["BPM1.x", "BPM1.y"], ["BPM2.x", "BPM2.y"]],
    AnomalyType.LEVEL_SHIFT: [["RF1.phase"], ["RF2.phase"], ["RF1.amp"]],
    AnomalyType.NOISE: [["Timing.jitter"], ["BPM1.x"]],
}


@dataclass
class InjectedAnomaly:
    """Metadata for one injected anomaly (handy for evaluation)."""

    type: AnomalyType
    start: int
    end: int
    channels: list[str]
    amplitude: float


class AnomalyInjector:
    """Samples and injects anomalies into a clean multi-channel timeline."""

    def __init__(self, config: "AnomalyConfig", channel_names: list[str], seed=None):
        self.config = config
        self.channel_names = channel_names
        self.name2idx = {n: i for i, n in enumerate(channel_names)}
        self.rng = np.random.default_rng(seed)

    # -- sampling helpers --------------------------------------------------

    def _type_weights(self) -> np.ndarray:
        c = self.config
        w = np.asarray(
            [c.point_prob, c.drift_prob, c.oscillation_prob,
             c.level_shift_prob, c.noise_prob], dtype=float
        )
        return w / w.sum()

    def _duration_range(self, type_: AnomalyType) -> tuple[int, int]:
        c = self.config
        return {
            AnomalyType.POINT: c.point_duration,
            AnomalyType.DRIFT: c.drift_duration,
            AnomalyType.OSCILLATION: c.oscillation_duration,
            AnomalyType.LEVEL_SHIFT: c.level_shift_duration,
            AnomalyType.NOISE: c.noise_duration,
        }[type_]

    def _amplitude_range(self, type_: AnomalyType) -> tuple[float, float]:
        c = self.config
        return {
            AnomalyType.POINT: c.point_amplitude,
            AnomalyType.DRIFT: c.drift_amplitude,
            AnomalyType.OSCILLATION: c.oscillation_amplitude,
            AnomalyType.LEVEL_SHIFT: c.level_shift_amplitude,
            AnomalyType.NOISE: c.noise_factor,
        }[type_]

    def _sample_type(self) -> AnomalyType:
        types = [AnomalyType.POINT, AnomalyType.DRIFT, AnomalyType.OSCILLATION,
                 AnomalyType.LEVEL_SHIFT, AnomalyType.NOISE]
        return AnomalyType(int(self.rng.choice(types, p=self._type_weights())))

    def _n_events(self, n_samples: int) -> int:
        if self.config.n_anomalies > 0:
            return self.config.n_anomalies
        avg_duration = 350.0
        return max(1, int(self.config.anomaly_density * n_samples / avg_duration))

    # -- injection ---------------------------------------------------------

    def inject(self, data: np.ndarray):
        """Inject anomalies into ``data`` and return labels + event metadata.

        Returns
        -------
        out : np.ndarray  (n_samples, n_channels)  data with anomalies
        label : np.ndarray (n_samples,)            global binary label
        type_label : np.ndarray (n_samples,)       anomaly type code
        channel_label : np.ndarray (n_samples, n_channels)
        events : list[InjectedAnomaly]
        """
        n_samples = data.shape[0]
        out = data.copy()
        label = np.zeros(n_samples, dtype=np.int8)
        type_label = np.zeros(n_samples, dtype=np.int16)
        channel_label = np.zeros_like(out, dtype=np.int8)

        # Plan non-overlapping events.
        events: list[InjectedAnomaly] = []
        target = self._n_events(n_samples)
        attempts = 0
        while len(events) < target and attempts < 5000:
            attempts += 1
            type_ = self._sample_type()
            lo, hi = self._duration_range(type_)
            dur = int(self.rng.integers(lo, hi + 1))
            max_start = n_samples - dur
            if max_start < 1:
                continue
            start = int(self.rng.integers(0, max_start))
            end = start + dur
            if not self._overlaps(start, end, events):
                events.append(InjectedAnomaly(type_, start, end, [], 0.0))

        events.sort(key=lambda e: e.start)

        for ev in events:
            idx, ch_names = self._pick_channels(ev.type)
            lo, hi = self._amplitude_range(ev.type)
            ev.amplitude = float(self.rng.uniform(lo, hi))
            ev.channels = ch_names
            self._apply(out, ev.type, ev.start, ev.end, idx, ev.amplitude)
            label[ev.start:ev.end] = 1
            type_label[ev.start:ev.end] = int(ev.type)
            channel_label[ev.start:ev.end, idx] = 1

        return out, label, type_label, channel_label, events

    def _overlaps(self, start: int, end: int, events: list[InjectedAnomaly]) -> bool:
        gap = self.config.min_separation
        for ev in events:
            if start < ev.end + gap and end > ev.start - gap:
                return True
        return False

    def _pick_channels(self, type_: AnomalyType) -> tuple[list[int], list[str]]:
        groups = ANOMALY_CHANNELS[type_]
        group = groups[int(self.rng.integers(len(groups)))]
        return [self.name2idx[n] for n in group], list(group)

    def _apply(self, data, type_, start, end, idx, amp):
        rng = self.rng
        seg = slice(start, end)
        t = np.arange(end - start, dtype=float)
        n = len(t)
        if type_ == AnomalyType.POINT:
            sigma = max(n / 6.0, 1.0)
            spike = amp * np.exp(-0.5 * ((t - n / 2.0) / sigma) ** 2)
            data[start:end, idx[0]] += spike
            if len(idx) > 1:
                data[start:end, idx[1]] -= 0.6 * amp * spike
        elif type_ == AnomalyType.DRIFT:
            ramp = np.minimum(1.0, t / max(n * 0.3, 1.0))
            direction = rng.choice([-1.0, 1.0])
            data[seg, idx] += direction * amp * ramp[:, None]
        elif type_ == AnomalyType.OSCILLATION:
            freq = rng.uniform(0.02, 0.10)
            env = np.minimum(1.0, t / max(n * 0.2, 1.0))
            osc = amp * env * np.sin(2.0 * np.pi * freq * t + rng.uniform(0.0, 2.0 * np.pi))
            data[seg, idx] += osc[:, None]
        elif type_ == AnomalyType.LEVEL_SHIFT:
            direction = rng.choice([-1.0, 1.0])
            data[seg, idx] += direction * amp
        elif type_ == AnomalyType.NOISE:
            data[seg, idx] += amp * rng.standard_normal((n, len(idx)))
