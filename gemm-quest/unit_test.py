import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
import quik

def find_scale_int4_baseline(
        x: torch.Tensor,
        scale: torch.Tensor,
) -> torch.Tensor:
    # OPTIMAL_GAUSSIAN_SCALES = {
    #     1: 0.7978845587140913,
    #     2: 1.4935346200015913,
    #     3: 2.051068354131873,
    #     4: 2.513930578568423,
    #     5: 2.9160938834961225,
    #     6: 3.276597282593217,
    #     7: 3.6010497188221655,
    #     8: 3.884938678807525,
    # }
    x: torch.Tensor = x.to(dtype=torch.float32)  # fp32, (..., K)
    scale_: torch.Tensor = (x * x).mean(dim=-1, keepdim=True).sqrt() * (2.513930578568423 * 2. / ((1 << 4) - 1)) + 1e-8  # fp32, (..., 1)
    return scale.copy_(scale_)  # fp, (..., 1)


def quantize_int4_baseline(
        x: torch.Tensor,
        scale: torch.Tensor,
        x_int: torch.Tensor = None,
        x_int_packed: torch.Tensor = None,
        x_int_row_sum: torch.Tensor = None,
        do_dequantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    x_q: torch.Tensor = (x.to(dtype=torch.float32) / scale.to(dtype=torch.float32)).floor().clamp(16 // -2, 16 // 2 - 1)  # fp32, (..., K)

    if do_dequantize:
        x_dq: torch.Tensor = x_q * scale + (scale * .5)  # fp32, (..., K)
        x.copy_(x_dq)  # fp, (..., K)

    write_dense: bool = x_int is not None
    write_packed: bool = x_int_packed is not None
    write_row_sum: bool = x_int_row_sum is not None

    if write_dense or write_packed or write_row_sum:
        x_int_: torch.Tensor = x_q.to(dtype=torch.int8)  # int8, (..., K)

        if write_dense:
            x_int.copy_(x_int_)  # int8, (..., K)

        if write_packed:
            x_int_packed_: torch.Tensor = (x_int_[..., 1::2] << 4) | (x_int_[..., ::2] & 0xF)  # int8, (..., K // 2)
            x_int_packed.copy_(x_int_packed_)  # uint8, (..., K)

        if write_row_sum:
            x_int_row_sum.copy_(x_int_.to(dtype=torch.int32).sum(dim=-1, keepdim=True))  # int32, (..., 1)

    return x, x_int, x_int_packed, x_int_row_sum


def quantize_int4_sparse(
        x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize using 4:8 sparsity
    Args:
        x: fp, (..., K), inplace

    Returns:
        x: fp, (..., K), inplace
        ...

    """
    p: float = 2.  # norm to find sparse pairs
    _, nonzero_idx = torch.linalg.vector_norm(
        x.unflatten(dim=-1, sizes=(-1, 4, 2)), ord=p, dim=-1, keepdim=True,
    ).topk(
        k=2, dim=-2, largest=True, sorted=False,
    )  # int64, (..., K // 8, 2, 1)
    nonzero_idx, _ = nonzero_idx.sort(dim=-2, descending=False)  # int64, (..., K // 8, 2, 1)
    nonzero_mask: torch.Tensor = torch.zeros_like(x, dtype=torch.bool).unflatten(dim=-1, sizes=(-1, 4, 2)).scatter_(
        dim=-2,
        index=nonzero_idx.expand(*nonzero_idx.shape[:-1], 2),  # int64, (..., K //8 , 2, 2)
        src=torch.ones((), dtype=torch.bool, device=x.device).expand(*nonzero_idx.shape[:-1], 2),
        # int64, (..., K // 8, 2, 2)
    ).flatten(start_dim=-3)  # bool, (..., K)

    scale: torch.Tensor = torch.empty(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)  # fp, (..., 1)
    x_int: torch.Tensor = torch.empty(x.shape, dtype=torch.int8, device=x.device)  # int8, (..., K)
    scale = find_scale_int4_baseline(x=x, scale=scale)  # fp, (..., 1)
    _, x_int, _, _ = quantize_int4_baseline(x=x, scale=scale, x_int=x_int, x_int_packed=None, x_int_row_sum=None, do_dequantize=False)  # int8, (..., K)
    x_int *= nonzero_mask  # int8, (..., K)

    x_dq: torch.Tensor = (x_int.to(dtype=scale.dtype) + .5) * scale * nonzero_mask  # fp, (..., K)
    x.copy_(x_dq)  # fp, (..., K)

    x_int_packed: torch.Tensor = ((x_int[..., 1::2] << 4) | (x_int[..., ::2] & 0xF)).to(dtype=torch.uint8)  # uint8 = 2 * int4, (..., K // 2)
    x_int_compressed: torch.Tensor = x_int.unflatten(dim=-1, sizes=(-1, 4, 2)).take_along_dim(nonzero_idx, dim=-2).flatten(start_dim=-3)  # int8, (..., K // 2)
    x_int_compressed_packed: torch.Tensor = ((x_int_compressed[..., 1::2] << 4) | (x_int_compressed[..., ::2] & 0xF)).to(dtype=torch.uint8)  # uint8 = 2 * int4, (..., K // 4)

    meta_e: torch.Tensor = nonzero_idx[..., 0].to(dtype=torch.int32)  # int32, (..., K // 8, 2)
    meta_e = meta_e[..., 0] | (meta_e[..., 1] << 2)  # int32 = 2 * int2, (..., K // 8)
    meta_e = meta_e[..., ::2] | (meta_e[..., 1::2] << 4)  # int32 = 4 * int2, (..., K // 16)
    meta_e = meta_e[..., ::2] | (meta_e[..., 1::2] << 8)  # int32 = 8 * int2, (..., K // 32)
    meta_e = meta_e[..., ::2] | (meta_e[..., 1::2] << 16)  # int32 = 16 * int2, (..., K // 64)
    meta_e = meta_e.to(dtype=torch.uint32)  # uint32, (..., K // 64)
    meta_e_no_reorder: torch.Tensor = meta_e.clone()  # uint32, (..., K // 64)
    # meta_e: torch.Tensor = quest.reorder_meta48_int4(meta_e_in=meta_e)  # uint32, (..., K // 64)

    return x, scale, x_int, x_int_packed, x_int_compressed, x_int_compressed_packed, nonzero_mask, nonzero_idx, meta_e_no_reorder, meta_e


def add_mul_vv_baseline(
        vec_a_add: torch.Tensor,
        vec_a_mul: torch.Tensor,
        vec_b_add: torch.Tensor,
        vec_b_mul: torch.Tensor,
        mat_c: torch.Tensor,
        mat_d: torch.Tensor,
) -> torch.Tensor:
    accum_dtype: torch.dtype = torch.float32
    mat_d_: torch.Tensor = (mat_c * 2 + vec_a_add + vec_b_add.t()).to(dtype=accum_dtype) * (.5 * vec_a_mul.to(dtype=accum_dtype) * vec_b_mul.t().to(dtype=accum_dtype))  # fp32, (M, N)
    return mat_d.copy_(mat_d_)  # fp, (M, N)


def foo(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        vec_a_add: torch.Tensor,
        vec_b_add: torch.Tensor,
        vec_a_mul: torch.Tensor,
        vec_b_mul: torch.Tensor,
        mat_d: torch.Tensor,
) -> torch.Tensor:
    """
    mat_c = A @ B.t() = matmul(mat_a, mat_b)
    mat_d = (mat_c * 2 + vec_a_add + vec_b_add.t()) * (.5 * vec_a_mul * vec_b_mul.t())
    Args:
        mat_a: uint8 = 2 * int4, (M, K // 2)
        mat_b: uint8 = 2 * int4, (N, K // 2)
        vec_a_add: int32, (M, 1)
        vec_a_mul: fp, (M, 1)
        vec_b_add: int32, (N, 1)
        vec_b_mul: fp, (N, 1)
        mat_d: fp, (M, N), inplace

    Returns:
        mat_d: fp, (M, N), inplace

    """
    import quest
    c = quest.matmul_int4_int4t_int32(mat_a=mat_a, mat_b=mat_b, out=None)  # int32, (M, N)
    add_mul_vv_baseline(vec_a_add=vec_a_add, vec_a_mul=vec_a_mul, vec_b_add=vec_b_add, vec_b_mul=vec_b_mul, mat_c=c, mat_d=mat_d)  # fp, (M, N)
    return mat_d


def _unit_test(M: int = 32 * 5, K: int = 256 * 3, N: int = 32 * 7) -> None:
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device('cuda')

    weight: torch.Tensor = torch.randn(N, K, dtype=dtype, device=device)  # fp, (N, K)
    activation: torch.Tensor = torch.randn(M, K, dtype=dtype, device=device)  # fp, (M, K)

    w_sp_dq, w_scale, w_int, w_int_packed, w_int_compressed, w_int_compressed_packed, w_mask, w_mask_idx, w_meta_e_no_reorder, w_meta_e = quantize_int4_sparse(x=weight.clone())
    w_add = w_int.to(dtype=torch.int32).sum(dim=-1, keepdim=True).to(dtype=torch.int32) + K // 2  # int32, (N, 1)

    a_scale = torch.empty(M, 1, dtype=dtype, device=device)  # fp, (M, 1)
    a_int = torch.empty(M, K, dtype=torch.int8, device=device)  # int8, (M, K)
    a_int_packed = torch.empty(M, K // 2, dtype=torch.uint8, device=device)  # uint8, (M, K // 2)
    a_int_row_sum = torch.empty(M, 1, dtype=torch.int32, device=device)  # int32, (M, 1)
    find_scale_int4_baseline(x=activation, scale=a_scale)
    quantize_int4_baseline(x=activation.clone(), scale=a_scale, x_int=a_int, x_int_packed=a_int_packed, x_int_row_sum=a_int_row_sum, do_dequantize=False)

    # UNIT TEST BEGIN

    a_dq = (a_int.to(dtype=torch.float64) + .5) * a_scale.to(dtype=torch.float64)  # fp64, (M, K)
    w_dq = (w_int.to(dtype=torch.float64) + .5) * w_scale.to(dtype=torch.float64)  # fp64, (M, K)
    m_ref = a_dq @ w_dq.t()  # fp64, (M, N)

    m = a_scale.to(dtype=torch.float64) * (
            2. * a_int.to(dtype=torch.float64) @ w_int.to(dtype=torch.float64).t()
            + a_int.to(dtype=torch.float64).sum(dim=-1, keepdim=True)
            + w_int.to(dtype=torch.float64).sum(dim=-1, keepdim=True).t()
            + .5 * K
    ) * (.5 * w_scale.to(dtype=torch.float64)).t()  # fp64, (M, N)
    assert m.equal(m_ref)

    print(a_int_row_sum.shape, w_add.shape)

    # TODO: write new kernel to replace the function foo()
    m = torch.empty(M, N, dtype=torch.float16, device=device)  # fp, (M, N)
    _ = quik.symmetric.int4FusedDequantize(a_int_packed,
                                        w_int_packed,
                                        a_scale.to(dtype=torch.float16),
                                        w_scale.to(dtype=torch.float16),
                                        w_add.to(dtype=torch.float32),
                                        a_int_row_sum.to(dtype=torch.float32),
                                        m)
    print(m[:6,:6])
    print(m_ref.half()[:6, :6])

    assert m.equal(m_ref.half())

    # UNIT TEST END

    print('Unit Test OK!')


if __name__ == '__main__':
    _unit_test()