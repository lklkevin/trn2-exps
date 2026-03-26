"""Plot zero-tile skip benchmark results.

Generates only the M-dim sparsity figures:
    - 512^3 varying sparsity: grouped bar chart of latency by kernel variant
    - 1024^3 varying sparsity: same

Usage:
    python analysis/plot_zero_skip.py results/zero_skip_results.json
    python analysis/plot_zero_skip.py results/zero_skip_results.json --output-dir results/plots
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)


def group_by_prefix(entries):
    """Group entries by matrix size prefix (512^3, 1024^3, K-dim)."""
    groups = OrderedDict()
    for e in entries:
        groups.setdefault(e["label"], []).append(e)

    # Merge labels into broader config groups
    config_groups = OrderedDict()
    for label, items in groups.items():
        M = items[0]["M"]
        if "K-tiles" in label:
            key = f"{M}x{M}x{M} K-dim skip"
        else:
            key = f"{M}x{M}x{M} M-dim sparsity"
        config_groups.setdefault(key, []).extend(items)

    return config_groups


KERNEL_DISPLAY = {
    "full_baseline": "Baseline",
    "load_skip_matmul": "Load + skip matmul",
    "skip_store": "Skip load + store zeros",
    "skip_nostore": "Skip everything",
    "k_skip": "K-dim skip",
}

KERNEL_COLORS = {
    "full_baseline": "#2c3e50",
    "load_skip_matmul": "#e67e22",
    "skip_store": "#27ae60",
    "skip_nostore": "#8e44ad",
    "k_skip": "#e67e22",
}

TARGET_M_SIZES = {512, 1024}


def kernel_sort_key(name):
    order = ["full_baseline", "load_skip_matmul", "skip_store", "skip_nostore", "k_skip"]
    prefix = name.split("_")[0] if "_" not in name else ""
    for i, o in enumerate(order):
        if name.startswith(o):
            return i
    return len(order)


def get_kernel_prefix(kernel_name):
    """Extract the kernel type prefix from names like 'skip_store_2of4'."""
    for prefix in ["load_skip_matmul", "skip_nostore", "skip_store", "full_baseline", "k_skip"]:
        if kernel_name.startswith(prefix):
            return prefix
    return kernel_name


def get_display_name(kernel_name):
    prefix = get_kernel_prefix(kernel_name)
    return KERNEL_DISPLAY.get(prefix, kernel_name)


def get_color(kernel_name):
    prefix = get_kernel_prefix(kernel_name)
    return KERNEL_COLORS.get(prefix, "#95a5a6")


def plot_m_sparsity_group(entries, group_name, output_dir):
    """Grouped bar chart: latency by kernel variant across sparsity levels."""

    # Group by label (each label = one sparsity level)
    by_label = OrderedDict()
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)

    # Sort entries within each label by kernel type
    for label in by_label:
        by_label[label].sort(key=lambda e: kernel_sort_key(e["kernel"]))

    sparsity_labels = list(by_label.keys())
    # Extract short sparsity label (e.g. "25%", "50%")
    short_labels = []
    for lab in sparsity_labels:
        for token in lab.split():
            if "%" in token:
                short_labels.append(token)
                break
        else:
            short_labels.append(lab)

    # Collect unique kernel types in order
    kernel_types = []
    seen = set()
    for label, items in by_label.items():
        for e in items:
            prefix = get_kernel_prefix(e["kernel"])
            if prefix not in seen:
                seen.add(prefix)
                kernel_types.append(prefix)

    n_groups = len(sparsity_labels)
    n_bars = len(kernel_types)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 2.5), 5))

    for i, ktype in enumerate(kernel_types):
        lats = []
        for label in sparsity_labels:
            match = [e for e in by_label[label] if get_kernel_prefix(e["kernel"]) == ktype]
            lat = match[0]["latency_us"] if match and match[0].get("latency_us") else 0
            lats.append(lat)

        offset = (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, lats, bar_width,
                       label=KERNEL_DISPLAY.get(ktype, ktype),
                       color=KERNEL_COLORS.get(ktype, "#95a5a6"),
                       edgecolor="white", linewidth=0.5)

        # Label each bar with its latency value
        for bar, lat in zip(bars, lats):
            if lat > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{lat:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Zero fraction")
    ax.set_ylabel("Latency (us)")
    ax.set_title(f"{group_name}: latency by kernel variant")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    fname = group_name.replace(" ", "_").replace("x", "x").lower() + "_latency.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_k_dim_group(entries, group_name, output_dir):
    """Simple bar chart for K-dim skip (single sparsity level)."""
    entries_sorted = sorted(entries, key=lambda e: kernel_sort_key(e["kernel"]))

    names = [get_display_name(e["kernel"]) for e in entries_sorted]
    lats = [e.get("latency_us", 0) or 0 for e in entries_sorted]
    colors = [get_color(e["kernel"]) for e in entries_sorted]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, lats, color=colors, edgecolor="white", linewidth=0.5)

    for bar, lat in zip(bars, lats):
        if lat > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{lat:.1f}us", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Latency (us)")
    ax.set_title(f"{group_name}: latency comparison")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    fname = group_name.replace(" ", "_").replace("x", "x").lower() + "_latency.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_speedup_summary(entries, output_dir):
    """Line plot: speedup of realistic smart DMA vs zero fraction, one line per matrix size."""

    # Group by label
    by_label = OrderedDict()
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)

    # Collect (size_group, zero_frac, speedup) for M-dim configs only
    series = {}  # size_group -> [(zero_frac, speedup)]
    for label, items in by_label.items():
        if "K-tiles" in label:
            continue

        baseline = next((e for e in items if e["kernel"] == "full_baseline"), None)
        load_skip = next((e for e in items if e["kernel"].startswith("load_skip_matmul")), None)
        if not baseline or not load_skip:
            continue
        if not baseline.get("latency_us") or not load_skip.get("latency_us"):
            continue

        M = baseline["M"]
        size_key = f"{M}x{M}x{M}"
        speedup = baseline["latency_us"] / load_skip["latency_us"]

        # Extract zero fraction from label
        zero_frac = None
        for token in label.split():
            if "%" in token:
                zero_frac = int(token.replace("%", "")) / 100
                break
        if zero_frac is None:
            continue

        series.setdefault(size_key, []).append((zero_frac, speedup))

    if not series:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i, (size_key, points) in enumerate(sorted(series.items())):
        points.sort()
        fracs = [p[0] for p in points]
        speedups = [p[1] for p in points]
        color = colors[i % len(colors)]
        ax.plot(fracs, speedups, "o-", color=color, linewidth=2, markersize=7, label=size_key)

    # Ideal line (linear scaling)
    fracs_range = np.linspace(0, 0.95, 50)
    ideal = 1 / (1 - fracs_range)
    ax.plot(fracs_range, ideal, "--", color="gray", alpha=0.5, label="Ideal (1/(1-f))")

    ax.set_xlabel("Fraction of zero M-tiles")
    ax.set_ylabel("Speedup vs baseline")
    ax.set_title("Smart DMA speedup (load + skip matmul) vs zero fraction")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    path = os.path.join(output_dir, "zero_skip_speedup_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot zero-skip benchmark results")
    parser.add_argument("input", help="Path to zero_skip_results.json")
    parser.add_argument("--output-dir", default="results/plots",
                        help="Directory for output plots (default: results/plots)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    entries = load_results(args.input)
    print(f"Loaded {len(entries)} entries\n")

    config_groups = group_by_prefix(entries)

    for group_name, group_entries in config_groups.items():
        if "K-dim" not in group_name and group_entries[0]["M"] in TARGET_M_SIZES:
            plot_m_sparsity_group(group_entries, group_name, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
