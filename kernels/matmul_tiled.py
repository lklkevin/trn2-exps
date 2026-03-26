import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


@nki.jit
def matmul_tiled(lhsT, rhs):
    """Tiled matmul: result = lhsT.T @ rhs

    Args:
        lhsT: float32 tensor of shape [K, M], K and M must be multiples of 128
        rhs:  float32 tensor of shape [K, N], K multiple of 128, N multiple of 512

    Returns:
        result: float32 tensor of shape [M, N]
    """
    K, M = lhsT.shape
    K2, N = rhs.shape

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax                   # 128
    TILE_N = nl.tile_size.gemm_stationary_fmax    # 128

    result = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)

    # Tile loops over M and N dimensions
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Accumulator in PSUM
            accum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            # Inner loop over K dimension
            for k in nl.affine_range(K // TILE_K):
                # Load tiles from HBM to SBUF
                lhsT_tile = nl.load(
                    lhsT[k * TILE_K : (k + 1) * TILE_K,
                          m * TILE_M : (m + 1) * TILE_M]
                )
                rhs_tile = nl.load(
                    rhs[k * TILE_K : (k + 1) * TILE_K,
                        n * TILE_N : (n + 1) * TILE_N]
                )
                accum += nisa.nc_matmul(lhsT_tile, rhs_tile)

            # Copy PSUM -> SBUF, then store to HBM
            result_sbuf = nl.copy(accum, dtype=nl.float32)
            nl.store(
                result[m * TILE_M : (m + 1) * TILE_M,
                       n * TILE_N : (n + 1) * TILE_N],
                value=result_sbuf
            )

    return result
