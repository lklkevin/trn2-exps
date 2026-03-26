"""Matmul kernel variants for zero-tile skipping experiment.

Generates kernels that hardcode which M-tiles to skip, simulating a smart DMA
engine that detects all-zero tiles at load time and avoids the matmul.
"""

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


TILE_M = 128  # nl.tile_size.gemm_stationary_fmax
TILE_K = 128  # nl.tile_size.pmax
TILE_N = 128  # nl.tile_size.gemm_stationary_fmax


@nki.jit
def matmul_full(lhsT, rhs):
    """Baseline: full matmul over all tiles, no skipping."""
    K, M = lhsT.shape
    K2, N = rhs.shape

    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            accum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
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


def make_skip_kernel(num_zero_m_tiles, num_total_m_tiles):
    """Generate a kernel that skips the first `num_zero_m_tiles` M-tiles.

    Zero tiles: store zeros directly (no loads, no matmul).
    Live tiles: normal load/matmul/accumulate/store.

    The closure captures the tile counts as compile-time constants.
    """
    num_live = num_total_m_tiles - num_zero_m_tiles

    @nki.jit
    def matmul_skip(lhsT, rhs):
        K, M = lhsT.shape
        K2, N = rhs.shape

        result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

        # --- Zero M-tiles: just store zeros ---
        if num_zero_m_tiles > 0:
            for m in nl.affine_range(num_zero_m_tiles):
                for n in nl.affine_range(N // TILE_N):
                    zero_accum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                    zero_sbuf = nl.copy(zero_accum, dtype=nl.float32)
                    nl.store(
                        result[m * TILE_M : (m + 1) * TILE_M,
                               n * TILE_N : (n + 1) * TILE_N],
                        value=zero_sbuf
                    )

        # --- Live M-tiles: full compute ---
        if num_live > 0:
            for m_offset in nl.affine_range(num_live):
                m = m_offset + num_zero_m_tiles
                for n in nl.affine_range(N // TILE_N):
                    accum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                    for k in nl.affine_range(K // TILE_K):
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

    return matmul_skip


def make_skip_kernel_load_only(num_zero_m_tiles, num_total_m_tiles):
    """Realistic smart DMA: load lhsT tiles (to detect zeros), skip matmul entirely.

    For zero M-tiles: loads each K-tile of lhsT to simulate DMA inspection,
    stores the loaded tiles into a scratch HBM buffer to keep the loads live,
    and writes zeros to the result. rhs is never loaded and nc_matmul is never
    called.

    For live M-tiles: normal load/matmul/accumulate/store.
    """
    num_live = num_total_m_tiles - num_zero_m_tiles

    @nki.jit
    def matmul_skip_load_only(lhsT, rhs):
        K, M = lhsT.shape
        K2, N = rhs.shape

        result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

        # --- Zero M-tiles: load lhsT, skip rhs + nc_matmul ---
        # Scratch HBM buffer sized for all (m, k) tile combinations so both
        # m and k appear in the store destination address, satisfying NKI's
        # loop variable dependency rule. Zero compute.
        scratch = nl.ndarray((num_zero_m_tiles * K, TILE_M), dtype=nl.float32, buffer=nl.shared_hbm)
        if num_zero_m_tiles > 0:
            for m in nl.affine_range(num_zero_m_tiles):
                for n in nl.affine_range(N // TILE_N):
                    for k in nl.affine_range(K // TILE_K):
                        lhsT_tile = nl.load(
                            lhsT[k * TILE_K : (k + 1) * TILE_K,
                                  m * TILE_M : (m + 1) * TILE_M]
                        )
                        nl.store(
                            scratch[m * K + k * TILE_K : m * K + (k + 1) * TILE_K, :],
                            value=lhsT_tile
                        )
                    zero = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                    nl.store(
                        result[m * TILE_M : (m + 1) * TILE_M,
                               n * TILE_N : (n + 1) * TILE_N],
                        value=nl.copy(zero, dtype=nl.float32)
                    )

        # --- Live M-tiles: full compute ---
        if num_live > 0:
            for m_offset in nl.affine_range(num_live):
                m = m_offset + num_zero_m_tiles
                for n in nl.affine_range(N // TILE_N):
                    accum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                    for k in nl.affine_range(K // TILE_K):
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

        return result, scratch

    return matmul_skip_load_only


def make_skip_kernel_no_store(num_zero_m_tiles, num_total_m_tiles):
    """Like make_skip_kernel but doesn't even store zeros for skipped tiles.

    Measures pure compute savings — output garbage in zero region.
    """
    num_live = num_total_m_tiles - num_zero_m_tiles

    @nki.jit
    def matmul_skip_no_store(lhsT, rhs):
        K, M = lhsT.shape
        K2, N = rhs.shape

        result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

        # Only compute live M-tiles
        for m_offset in nl.affine_range(num_live):
            m = m_offset + num_zero_m_tiles
            for n in nl.affine_range(N // TILE_N):
                accum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

                for k in nl.affine_range(K // TILE_K):
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

    return matmul_skip_no_store
