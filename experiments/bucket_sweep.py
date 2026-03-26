"""Latency sweep comparing bucket strategies for M, K, N dimensions.

For each sweep size, the actual input is either a multiple of 128 (tile-aligned)
or a multiple of 128 + 1 (just over a tile boundary). The bucketing strategy
determines what padded size the kernel actually runs at.

Strategies:
  mult256      - round up to nearest multiple of 256
  mult512      - round up to nearest multiple of 512
  pow2         - round up to nearest power of 2

Usage:
  python experiments/bucket_sweep.py --mode benchmark --dim m
  python experiments/bucket_sweep.py --mode benchmark --dim k
  python experiments/bucket_sweep.py --mode benchmark --dim n
  python experiments/bucket_sweep.py --mode simulate --dim m
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from experiments.utils import TILE_M, TILE_K, TILE_N, pad_to_multiple
from kernels.matmul_tiled import matmul_tiled


FIXED_DIM = 512

# Each strategy maps a dimension value to a padded bucket size.
# Buckets must be multiples of 128 (tile size).
def _pow2_buckets(max_val=8192):
    b, buckets = 128, []
    while b <= max_val:
        buckets.append(b)
        b *= 2
    return buckets

BUCKET_STRATEGIES = {
    "mult256": list(range(256, 8192 + 256, 256)),
    "mult512": list(range(512, 8192 + 512, 512)),
    "pow2":    _pow2_buckets(),
}

# Sweep: for each tile boundary N*128, test both N*128 and N*128+1
_boundaries = [128 * i for i in range(1, 9)]  # 128 .. 2048
SWEEP_SIZES = sorted(set(
    [b     for b in _boundaries] +
    [b + 1 for b in _boundaries]
))


def bucket_dim(x, buckets):
    """Round x up to the smallest bucket >= x that is also tile-aligned."""
    tile = TILE_M  # all dims use 128 tile
    for b in sorted(buckets):
        if b >= x and b % tile == 0:
            return b
    # Fallback: tile boundary
    return pad_to_multiple(x, tile)


def pad_inputs(lhs, rhs, M_pad, K_pad, N_pad):
    lhs_padded = np.zeros((M_pad, K_pad), dtype=lhs.dtype)
    lhs_padded[:lhs.shape[0], :lhs.shape[1]] = lhs
    rhs_padded = np.zeros((K_pad, N_pad), dtype=rhs.dtype)
    rhs_padded[:rhs.shape[0], :rhs.shape[1]] = rhs
    return lhs_padded, rhs_padded


def run_one(M, K, N, M_pad, K_pad, N_pad, mode):
    import neuronxcc.nki as nki

    lhs = np.random.randn(M, K).astype(np.float32)
    rhs = np.random.randn(K, N).astype(np.float32)
    lhs_padded, rhs_padded = pad_inputs(lhs, rhs, M_pad, K_pad, N_pad)
    lhsT_padded = lhs_padded.T.copy()

    metrics = {
        "M": M, "K": K, "N": N,
        "M_pad": M_pad, "K_pad": K_pad, "N_pad": N_pad,
        "flop_waste_ratio": (2*M_pad*K_pad*N_pad - 2*M*K*N) / (2*M*K*N),
        "latency_us": None,
        "correct": None,
    }

    if mode == "simulate":
        result = nki.simulate_kernel(matmul_tiled, lhsT_padded, rhs_padded)
        metrics["correct"] = bool(np.allclose(result[:M, :N], lhs @ rhs, atol=1e-2, rtol=1e-2))
    elif mode == "baremetal":
        result = nki.baremetal(matmul_tiled)(lhsT_padded, rhs_padded)
        metrics["correct"] = bool(np.allclose(result[:M, :N], lhs @ rhs, atol=1e-2, rtol=1e-2))
    elif mode == "benchmark":
        bench = nki.benchmark(matmul_tiled)
        bench(lhsT_padded, rhs_padded)
        metrics["latency_us"] = bench.benchmark_result.nc_latency.get_latency_percentile(50)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return metrics


def sweep_dim(swept_dim, strategies, mode):
    results = []

    for size in SWEEP_SIZES:
        if swept_dim == "m":
            M, K, N = size, FIXED_DIM, FIXED_DIM
        elif swept_dim == "k":
            M, K, N = FIXED_DIM, size, FIXED_DIM
        elif swept_dim == "n":
            M, K, N = FIXED_DIM, FIXED_DIM, size
        else:
            raise ValueError(swept_dim)

        print(f"  size={size}  (M={M}, K={K}, N={N})")

        # Fixed dims always use tile boundary
        K_pad_fixed = pad_to_multiple(K, TILE_K)
        N_pad_fixed = pad_to_multiple(N, TILE_N)
        M_pad_fixed = pad_to_multiple(M, TILE_M)

        for strategy_name, buckets in strategies.items():
            if swept_dim == "m":
                M_pad = bucket_dim(M, buckets)
                K_pad, N_pad = K_pad_fixed, N_pad_fixed
            elif swept_dim == "k":
                K_pad = bucket_dim(K, buckets)
                M_pad, N_pad = M_pad_fixed, N_pad_fixed
            elif swept_dim == "n":
                N_pad = bucket_dim(N, buckets)
                M_pad, K_pad = M_pad_fixed, K_pad_fixed

            try:
                entry = run_one(M, K, N, M_pad, K_pad, N_pad, mode)
            except Exception as e:
                print(f"    [{strategy_name}] ERROR: {e}")
                entry = {
                    "M": M, "K": K, "N": N,
                    "M_pad": M_pad, "K_pad": K_pad, "N_pad": N_pad,
                    "latency_us": None, "correct": None, "error": str(e),
                }

            entry["swept_dim"] = swept_dim
            entry["strategy"] = strategy_name
            results.append(entry)

            lat_str = f"{entry['latency_us']:.1f}us" if entry.get("latency_us") else "n/a"
            print(f"    [{strategy_name:8s}] {M_pad}x{K_pad}x{N_pad}  "
                  f"latency={lat_str}  waste={entry.get('flop_waste_ratio', 0):.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bucket strategy latency sweep")
    parser.add_argument(
        "--mode", choices=["simulate", "baremetal", "benchmark"],
        default="simulate",
    )
    parser.add_argument(
        "--dim", choices=["m", "k", "n", "all"], default="all",
    )
    parser.add_argument("--output", default="results/bucket_sweep.json")
    args = parser.parse_args()

    dims = ["m", "k", "n"] if args.dim == "all" else [args.dim]

    all_results = []
    for dim in dims:
        print(f"\n{'='*60}")
        print(f"Sweeping {dim.upper()} (others fixed at {FIXED_DIM})")
        print(f"{'='*60}")
        all_results.extend(sweep_dim(dim, BUCKET_STRATEGIES, args.mode))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
