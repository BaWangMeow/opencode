"""CLI: generate a synthetic FEL diagnostics dataset and save it.

Examples
--------
    python scripts/generate.py --n-samples 50000 --n-anomalies 30 --seed 42 \
        --out data/train.h5

    python scripts/generate.py --format csv --out data/quick.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feldiag import (  # noqa: E402
    AnomalyConfig,
    GeneratorConfig,
    generate,
    to_csv,
    to_hdf5,
    to_npz,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--sampling-rate", type=float, default=10.0)
    p.add_argument("--n-anomalies", type=int, default=30)
    p.add_argument("--noise-level", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/diagnostics.h5")
    p.add_argument("--format", choices=["h5", "npz", "csv"], default="h5")
    args = p.parse_args()

    cfg = GeneratorConfig(
        n_samples=args.n_samples,
        sampling_rate=args.sampling_rate,
        noise_level=args.noise_level,
        seed=args.seed,
        anomalies=AnomalyConfig(n_anomalies=args.n_anomalies),
    )

    dataset = generate(cfg)

    if args.format == "h5":
        out = to_hdf5(dataset, args.out)
    elif args.format == "npz":
        out = to_npz(dataset, args.out)
    else:
        out = to_csv(dataset, args.out)

    n_anom = int(dataset.label.sum())
    print(f"Generated {dataset.data.shape[0]} samples x "
          f"{dataset.data.shape[1]} channels")
    print(f"Anomalies: {len(dataset.events)} events, {n_anom} anomalous samples "
          f"({100.0 * n_anom / dataset.data.shape[0]:.2f}%)")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
