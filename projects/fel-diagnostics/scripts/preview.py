"""One-off check: generate a small dataset and render a plot to PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feldiag import GeneratorConfig, generate, plot_dataset

ds = generate(GeneratorConfig(n_samples=8000, seed=7))

fig = plot_dataset(ds, start=0, end=4000, title="FEL diagnostics (seed=7)")
out = Path(__file__).resolve().parents[1] / "data" / "preview.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)
print("saved", out)

for ev in ds.events[:10]:
    print(f"  type={ev.type.name:<12} start={ev.start:<6} end={ev.end:<6} "
          f"channels={ev.channels} amp={ev.amplitude:.2f}")
