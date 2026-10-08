"""Tests for the feldiag generator."""

import numpy as np

from feldiag import GeneratorConfig, generate


def _cfg(n_samples, seed, n_anomalies):
    cfg = GeneratorConfig(n_samples=n_samples, seed=seed)
    cfg.anomalies.n_anomalies = n_anomalies
    return cfg


def test_shapes_and_labels():
    ds = generate(_cfg(2000, 0, 5))
    assert ds.data.shape == (2000, len(ds.channel_names))
    assert ds.label.shape == (2000,)
    assert ds.type_label.shape == (2000,)
    assert ds.channel_label.shape == ds.data.shape
    assert len(ds.channel_names) == len(ds.channel_units)
    # global label == any channel anomalous
    assert np.array_equal(ds.label, ds.channel_label.any(axis=1).astype(np.int8))
    assert np.isfinite(ds.data).all()


def test_reproducible():
    a = generate(_cfg(1000, 123, 4))
    b = generate(_cfg(1000, 123, 4))
    assert np.array_equal(a.data, b.data)
    assert np.array_equal(a.label, b.label)


def test_different_seed_differs():
    a = generate(_cfg(500, 1, 3))
    b = generate(_cfg(500, 2, 3))
    assert not np.array_equal(a.data, b.data)


def test_event_labels_consistent():
    ds = generate(_cfg(2000, 7, 6))
    for ev in ds.events:
        assert ds.label[ev.start:ev.end].all()
        assert (ds.type_label[ev.start:ev.end] == int(ev.type)).all()
        # each event targets at least one channel
        assert len(ev.channels) >= 1
        # per-channel labels set for the targeted channels
        for ch in ev.channels:
            j = ds.channel_names.index(ch)
            assert ds.channel_label[ev.start:ev.end, j].all()
