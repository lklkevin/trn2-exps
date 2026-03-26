"""Benchmark zero-tile skipping on Trainium.

Simulates a smart DMA engine that detects all-zero tiles and skips matmul.
Tests multiple shapes and sparsity levels to build a complete picture.

Configurations:
  1. 512x512x512   — vary zero fraction: 25%, 50%, 75%
  2. 1024x1024x1024 — vary zero fraction: 25%, 50%, 75%, 87.5%
  3. 2048x2048x2048 — 50%, 75%
  4. Non-aligned zeros: 512x512x512 with 192 zero rows
     -> only 1 of 4 M-tiles is fully zero (rows 0-127),
        tile for rows 128-255 still has 64 zero + 64 live rows, must compute fully
     -> shows tile-granularity limitation
  5. K-dim zeros: 512x512x512, zero out first 256 cols of LHS (= first 2 K-tiles)
     -> tests skipping along the contraction dimension instead of M

Usage:
  python experiments/zero_skip_benchmark.py --mode simulate
  python experiments/zero_skip_benchmark.py --mode benchmark
  python experiments/zero_skip_benchmark.py --mode benchmark --config all
  python experiments/zero_skip_benchmark.py --mode benchmark --config quick
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from kernels.matmul_skip_zeros import (
    TILE_M, TILE_K, TILE_N,
    matmul_full,
    make_skip_kernel,
    make_skip_kernel_load_only,
    make_skip_kernel_no_store,
)

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

def build_configs(config_set="all"):
    """Build list of experiment configs.

    Each config is a dict:
      M, K, N:          matrix dimensions (tile-aligned)
      zero_rows:        number of leading LHS rows zeroed
      zero_m_tiles:     number of fully-zero M-tiles (what we can skip)
      label:            human-readable description
      skip_type:        "m" (zero rows in LHS) or "k" (zero cols in LHS)
      zero_k_tiles:     (only for skip_type="k") number of zero K-tiles
    """
    configs = []

    # --- Group 1: 512x512x512, varying M-sparsity ---
    M, K, N = 512, 512, 512
    total_m = M // TILE_M  # 4
    for zero_frac, zero_tiles in [(0.25, 1), (0.50, 2), (0.75, 3)]:
        configs.append({
            "M": M, "K": K, "N": N,
            "zero_rows": zero_tiles * TILE_M,
            "zero_m_tiles": zero_tiles,
            "total_m_tiles": total_m,
            "label": f"512^3 {zero_frac:.0%} zero M-tiles",
            "skip_type": "m",
        })

    # --- Group 2: 1024x1024x1024, varying M-sparsity ---
    M, K, N = 1024, 1024, 1024
    total_m = M // TILE_M  # 8
    for zero_frac, zero_tiles in [(0.25, 2), (0.50, 4), (0.75, 6), (0.875, 7)]:
        configs.append({
            "M": M, "K": K, "N": N,
            "zero_rows": zero_tiles * TILE_M,
            "zero_m_tiles": zero_tiles,
            "total_m_tiles": total_m,
            "label": f"1024^3 {zero_frac:.0%} zero M-tiles",
            "skip_type": "m",
        })

    # --- Group 3: K-dimension zeros (contraction dim sparsity) ---
    # Zero out first 256 columns of LHS = first 2 K-tiles are zero.
    # Every M-tile still needs to run, but the inner K-loop can skip 2 of 4 iters.
    M, K, N = 512, 512, 512
    configs.append({
        "M": M, "K": K, "N": N,
        "zero_rows": 0,
        "zero_m_tiles": 0,
        "total_m_tiles": M // TILE_M,
        "label": "512^3 50% zero K-tiles (contraction dim)",
        "skip_type": "k",
        "zero_k_tiles": 2,
        "total_k_tiles": K // TILE_K,
    })

    if config_set == "quick":
        # Just the 512^3 configs for a fast run
        configs = [c for c in configs if c["M"] == 512]

    return configs


# ---------------------------------------------------------------------------
# K-dimension skip kernels
# ---------------------------------------------------------------------------

def make_k_skip_kernel(num_zero_k_tiles, num_total_k_tiles):
    """Generate a kernel that skips the first `num_zero_k_tiles` K-tiles.

    The inner K accumulation loop only runs over live K-tiles.
    """
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa

    num_live_k = num_total_k_tiles - num_zero_k_tiles

    @nki.jit
    def matmul_k_skip(lhsT, rhs):
        K, M = lhsT.shape
        K2, N = rhs.shape

        result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

        for m in nl.affine_range(M // TILE_M):
            for n in nl.affine_range(N // TILE_N):
                accum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                # Only iterate over live K-tiles (skip first num_zero_k_tiles)
                for k_offset in nl.affine_range(num_live_k):
                    k = k_offset + num_zero_k_tiles
                    lhsT_tile = nl.load(
                        lhsT[k * TILE_K : (k + 1) * TILE_K,
                              m * TILE_M : (m + 1) * TILE_M]
                    )
                    rhs_tile = nl.load(
                        rhs[k * TILE_K : (k + 1) * TILE_K,
                            n * TILE_N : (n + 1) * TILE_N]
                    )
                    accum += nisa.nc_matmul(lhsT_tile, rhs_tile)

                result_sbuf = nl.copy(accum, dtype=nl.float32)
                nl.store(
                    result[m * TILE_M : (m + 1) * TILE_M,
                           n * TILE_N : (n + 1) * TILE_N],
                    value=result_sbuf
                )

        return result

    return matmul_k_skip


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def make_inputs(cfg):
    """Create inputs according to config."""
    M, K, N = cfg["M"], cfg["K"], cfg["N"]

    lhs = np.random.randn(M, K).astype(np.float32)

    if cfg["skip_type"] == "m":
        lhs[:cfg["zero_rows"], :] = 0.0
    elif cfg["skip_type"] == "k":
        zero_cols = cfg["zero_k_tiles"] * TILE_K
        lhs[:, :zero_cols] = 0.0

    rhs = np.random.randn(K, N).astype(np.float32)
    ref = lhs @ rhs
    lhsT = lhs.T.copy()  # [K, M], contiguous

    return lhsT, rhs, ref


def run_kernel(kernel_fn, lhsT, rhs, mode, multi_output=False):
    """Run a kernel and return (result, latency_us).

    multi_output=True: kernel returns (result, ...) tuple; only first is kept.
    """
    import neuronxcc.nki as nki

    def extract(out):
        return out[0] if multi_output else out

    if mode == "simulate":
        result = extract(nki.simulate_kernel(kernel_fn, lhsT, rhs))
        return result, None
    elif mode == "baremetal":
        result = extract(nki.baremetal(kernel_fn)(lhsT, rhs))
        return result, None
    elif mode == "benchmark":
        bench_fn = nki.benchmark(kernel_fn)
        out = bench_fn(lhsT, rhs)
        latency_us = bench_fn.benchmark_result.nc_latency.get_latency_percentile(50)
        return extract(out), latency_us
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_config(cfg, mode, variants=None):
    """Run baseline + skip variants for one config, return list of result dicts."""
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    label = cfg["label"]
    lhsT, rhs, ref = make_inputs(cfg)

    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"  M={M}, K={K}, N={N}")
    print(f"{'='*60}")

    # variants=None means run all; otherwise only run named variants plus baseline
    def want(name):
        return variants is None or name in variants

    entries = []

    # --- Baseline (full matmul) ---
    print(f"  [baseline] full matmul ...", end=" ", flush=True)
    result_full, lat_full = run_kernel(matmul_full, lhsT, rhs, mode)
    if mode != "benchmark":
        ok = np.allclose(result_full[:M, :N], ref, atol=1e-2, rtol=1e-2)
        print(f"correct={ok}")
    else:
        ok = None
        print(f"latency={lat_full:.1f} us" if lat_full else "no latency")
    entries.append({
        "label": label, "kernel": "full_baseline",
        "M": M, "K": K, "N": N, "latency_us": lat_full, "correct": ok,
        **_derived_metrics(cfg, lat_full, is_skip=False),
    })

    # --- Skip variant(s) ---
    if cfg["skip_type"] == "m":
        zero_t = cfg["zero_m_tiles"]
        total_t = cfg["total_m_tiles"]

        # Skip + store zeros
        if want("skip_store"):
            skip_kernel = make_skip_kernel(zero_t, total_t)
            print(f"  [skip+store] skip {zero_t}/{total_t} M-tiles ...", end=" ", flush=True)
            result_skip, lat_skip = run_kernel(skip_kernel, lhsT, rhs, mode)
            if mode != "benchmark":
                ok = np.allclose(result_skip[:M, :N], ref, atol=1e-2, rtol=1e-2)
                print(f"correct={ok}")
            else:
                ok = None
                print(f"latency={lat_skip:.1f} us" if lat_skip else "no latency")
            entries.append({
                "label": label, "kernel": f"skip_store_{zero_t}of{total_t}",
                "M": M, "K": K, "N": N, "latency_us": lat_skip, "correct": ok,
                **_derived_metrics(cfg, lat_skip, is_skip=True),
            })

        # Load + skip matmul (realistic smart DMA)
        if want("load_skip_matmul"):
            load_only_kernel = make_skip_kernel_load_only(zero_t, total_t)
            print(f"  [load+skip]  skip {zero_t}/{total_t} M-tiles (load, no matmul) ...", end=" ", flush=True)
            result_lo, lat_lo = run_kernel(load_only_kernel, lhsT, rhs, mode, multi_output=True)
            if mode != "benchmark":
                ok = np.allclose(result_lo[:M, :N], ref, atol=1e-2, rtol=1e-2)
                print(f"correct={ok}")
            else:
                ok = None
                print(f"latency={lat_lo:.1f} us" if lat_lo else "no latency")
            entries.append({
                "label": label, "kernel": f"load_skip_matmul_{zero_t}of{total_t}",
                "M": M, "K": K, "N": N, "latency_us": lat_lo, "correct": ok,
                **_derived_metrics(cfg, lat_lo, is_skip=True),
            })

        # Skip, no store (aggressive)
        if want("skip_nostore"):
            skip_ns_kernel = make_skip_kernel_no_store(zero_t, total_t)
            print(f"  [skip only]  skip {zero_t}/{total_t} M-tiles (no zero store) ...", end=" ", flush=True)
            result_ns, lat_ns = run_kernel(skip_ns_kernel, lhsT, rhs, mode)
            if mode != "benchmark":
                live_start = zero_t * TILE_M
                ok = np.allclose(result_ns[live_start:M, :N], ref[live_start:, :], atol=1e-2, rtol=1e-2)
                print(f"correct(live region)={ok}")
            else:
                ok = None
                print(f"latency={lat_ns:.1f} us" if lat_ns else "no latency")
            entries.append({
                "label": label, "kernel": f"skip_nostore_{zero_t}of{total_t}",
                "M": M, "K": K, "N": N, "latency_us": lat_ns, "correct": ok,
                **_derived_metrics(cfg, lat_ns, is_skip=True),
            })

    elif cfg["skip_type"] == "k":
        zero_k = cfg["zero_k_tiles"]
        total_k = cfg["total_k_tiles"]

        k_skip_kernel = make_k_skip_kernel(zero_k, total_k)
        print(f"  [k-skip] skip {zero_k}/{total_k} K-tiles ...", end=" ", flush=True)
        result_k, lat_k = run_kernel(k_skip_kernel, lhsT, rhs, mode)
        if mode != "benchmark":
            ok = np.allclose(result_k[:M, :N], ref, atol=1e-2, rtol=1e-2)
            print(f"correct={ok}")
        else:
            ok = None
            print(f"latency={lat_k:.1f} us" if lat_k else "no latency")
        entries.append({
            "label": label, "kernel": f"k_skip_{zero_k}of{total_k}",
            "M": M, "K": K, "N": N, "latency_us": lat_k, "correct": ok,
            **_derived_metrics(cfg, lat_k, is_skip=True),
        })

    return entries


def _derived_metrics(cfg, latency_us, is_skip):
    """Compute TFLOPS and other derived metrics."""
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    flops_full = 2 * M * K * N

    if cfg["skip_type"] == "m":
        live_rows = M - cfg["zero_rows"]
        flops_useful = 2 * live_rows * K * N
    elif cfg["skip_type"] == "k":
        live_cols = K - cfg["zero_k_tiles"] * TILE_K
        flops_useful = 2 * M * live_cols * N
    else:
        flops_useful = flops_full

    metrics = {
        "flops_full": flops_full,
        "flops_useful": flops_useful,
        "zero_fraction": 1 - flops_useful / flops_full if flops_full > 0 else 0,
    }

    if latency_us and latency_us > 0:
        metrics["achieved_tflops"] = flops_full / (latency_us * 1e-6) / 1e12
        metrics["useful_tflops"] = flops_useful / (latency_us * 1e-6) / 1e12

    return metrics


def print_summary(all_entries):
    """Print comparison table grouped by config."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Group by label
    from collections import OrderedDict
    groups = OrderedDict()
    for e in all_entries:
        groups.setdefault(e["label"], []).append(e)

    for label, entries in groups.items():
        print(f"\n  {label}")
        baseline_lat = None
        for e in entries:
            if e["kernel"] == "full_baseline":
                baseline_lat = e.get("latency_us")

        for e in entries:
            lat = e.get("latency_us")
            if lat is None:
                print(f"    {e['kernel']:35s}  no latency data")
                continue

            speedup = baseline_lat / lat if baseline_lat and lat else 0
            savings_pct = (1 - lat / baseline_lat) * 100 if baseline_lat else 0
            tflops = e.get("achieved_tflops", 0) or 0
            print(f"    {e['kernel']:35s}  {lat:8.1f} us  "
                  f"{speedup:.2f}x  ({savings_pct:+.1f}%)  "
                  f"{tflops:.3f} TFLOPS")


def main():
    parser = argparse.ArgumentParser(description="Zero-tile skip benchmark")
    parser.add_argument(
        "--mode", choices=["simulate", "baremetal", "benchmark"],
        default="simulate",
    )
    parser.add_argument(
        "--config", choices=["all", "quick"], default="all",
        help="'quick' runs only 512^3 configs",
    )
    parser.add_argument("--output", default="results/zero_skip_results.json")
    parser.add_argument(
        "--variants", default=None,
        help="Comma-separated subset of variants to run: "
             "skip_store, load_skip_matmul, skip_nostore, k_skip. "
             "Baseline always runs. Default: all variants.",
    )
    args = parser.parse_args()

    variants = set(args.variants.split(",")) if args.variants else None

    configs = build_configs(args.config)
    print(f"Running {len(configs)} configurations in {args.mode} mode")
    if variants:
        print(f"Variants: baseline + {sorted(variants)}")

    all_entries = []
    for cfg in configs:
        entries = run_config(cfg, args.mode, variants=variants)
        all_entries.extend(entries)

    if args.mode == "benchmark":
        print_summary(all_entries)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_entries, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
