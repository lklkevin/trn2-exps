"""Plot bucket sweep latency results.

Generates one plot per swept dimension showing latency vs actual size,
with one line per bucketing strategy. X-axis includes both N*128 and N*128+1
points to show the cliff behavior at tile boundaries.

Usage:
  python analysis/plot_bucket_sweep.py results/bucket_sweep_m.json
  python analysis/plot_bucket_sweep.py results/bucket_sweep_m.json --output-dir results/plots
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)


STRATEGY_STYLE = {
    "mult256": {"color": "tab:blue",   "marker": "o", "label": "mult256"},
    "mult512": {"color": "tab:orange", "marker": "s", "label": "mult512"},
    "pow2":    {"color": "tab:green",  "marker": "^", "label": "pow2"},
}


def plot_dim(entries, swept_dim, output_dir):
    by_strategy = defaultdict(list)
    for e in entries:
        if e.get("latency_us") is not None:
            by_strategy[e["strategy"]].append(e)

    if not by_strategy:
        print(f"  No latency data for dim={swept_dim}, skipping.")
        return

    all_sizes = sorted({e[swept_dim.upper()] for e in entries})
    boundaries = [s for s in all_sizes if s % 128 == 0]

    # Compute shared y-axis range across all strategies
    all_lats = [e["latency_us"] for e in entries if e.get("latency_us") is not None]
    ymin = max(0, min(all_lats) - 2)
    ymax = max(all_lats) + 4

    fig, axes = plt.subplots(1, len(by_strategy), figsize=(6 * len(by_strategy), 5), sharey=True)
    if len(by_strategy) == 1:
        axes = [axes]

    for ax, (strategy, pts) in zip(axes, sorted(by_strategy.items())):
        pts.sort(key=lambda e: e[swept_dim.upper()])
        sizes = [e[swept_dim.upper()] for e in pts]
        lats  = [e["latency_us"] for e in pts]
        style = STRATEGY_STYLE.get(strategy, {"color": "gray", "marker": "x", "label": strategy})

        ax.plot(sizes, lats, marker=style["marker"], color=style["color"],
                linewidth=1.5, markersize=5)

        for b in boundaries:
            ax.axvline(b, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_xlabel(f"{swept_dim.upper()} (actual)")
        ax.set_title(style["label"])
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Latency (us)")
    fig.suptitle(f"Bucket strategy latency: {swept_dim.upper()} sweep (others fixed at 512)", fontsize=12)
    fig.tight_layout()

    fname = f"{swept_dim}_dim_bucket_latency.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot bucket sweep results")
    parser.add_argument("input", help="Path to bucket_sweep JSON file")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    entries = load_results(args.input)
    print(f"Loaded {len(entries)} entries")

    dims = sorted({e["swept_dim"] for e in entries})
    for dim in dims:
        dim_entries = [e for e in entries if e["swept_dim"] == dim]
        print(f"Plotting dim={dim} ({len(dim_entries)} entries)")
        plot_dim(dim_entries, dim, args.output_dir)


if __name__ == "__main__":
    main()
