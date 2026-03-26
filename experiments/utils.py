import math
import numpy as np

TILE_M = 128
TILE_K = 128
TILE_N = 128


def pad_to_multiple(x, multiple):
    return math.ceil(x / multiple) * multiple


def pad_for_matmul(lhs, rhs):
    """Pad lhs [M,K] and rhs [K,N] to tile-aligned shapes.

    Returns: (lhs_padded, rhs_padded, original_dims)
    """
    M, K = lhs.shape
    K2, N = rhs.shape
    assert K == K2

    M_pad = pad_to_multiple(M, TILE_M)
    K_pad = pad_to_multiple(K, TILE_K)
    N_pad = pad_to_multiple(N, TILE_N)

    lhs_padded = np.zeros((M_pad, K_pad), dtype=lhs.dtype)
    lhs_padded[:M, :K] = lhs

    rhs_padded = np.zeros((K_pad, N_pad), dtype=rhs.dtype)
    rhs_padded[:K, :N] = rhs

    return lhs_padded, rhs_padded, (M, K, N)


def compute_padding_metrics(M, K, N):
    """Compute waste metrics for given dimensions (no kernel execution)."""
    M_pad = pad_to_multiple(M, TILE_M)
    K_pad = pad_to_multiple(K, TILE_K)
    N_pad = pad_to_multiple(N, TILE_N)

    flops_orig = 2 * M * K * N
    flops_pad = 2 * M_pad * K_pad * N_pad

    mem_orig = (M * K + K * N + M * N) * 4
    mem_pad = (M_pad * K_pad + K_pad * N_pad + M_pad * N_pad) * 4

    return {
        "M": M, "K": K, "N": N,
        "M_padded": M_pad, "K_padded": K_pad, "N_padded": N_pad,
        "num_tiles": (M_pad // TILE_M) * (K_pad // TILE_K) * (N_pad // TILE_N),
        "flops_original": flops_orig,
        "flops_padded": flops_pad,
        "flop_waste_ratio": (flops_pad - flops_orig) / flops_orig if flops_orig > 0 else 0,
        "memory_bytes_original": mem_orig,
        "memory_bytes_padded": mem_pad,
        "memory_waste_ratio": (mem_pad - mem_orig) / mem_orig if mem_orig > 0 else 0,
    }


def save_results(results, filepath):
    """Save list of result dicts as JSON lines."""
    import json
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')


def load_results(filepath):
    """Load JSON lines file into list of dicts."""
    import json
    with open(filepath) as f:
        return [json.loads(line) for line in f]
