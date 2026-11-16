# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX entry points for the cudnn-frontend fused MXFP8 SwiGLU grouped GEMMs (SM100 only).

Two calls carry the SwiGLU of a gated expert MLP, ``y = (silu(gate) * up) @ w2`` with
``[gate, up] = x @ w13``:

- :func:`mxfp8_fused_swiglu`: ``x_q @ w13`` followed in the epilogue by SwiGLU and the MXFP8
  quantization of the hidden activation ``h`` in both orientations, storing the bf16
  pre-activation for the backward.
- :func:`mxfp8_fused_dswiglu`: ``g_q @ w2^T`` followed by the SwiGLU backward against the saved
  pre-activation, the MXFP8 quantization of its result in both orientations, and each row's
  ``<h, dh>``.

Gate/up contract: the fused SwiGLU pairs gate and up columns in alternating 32-column blocks of the
``2 * I`` axis (even blocks gate, odd blocks up). ``w13`` must be laid out that way before it is
quantized (:func:`levanter.grug._moe.mxfp8.layout.interleave_gate_up_blocks`), and the
pre-activation ``c13`` and its gradient come back in the same order.

Row layout, scale layouts and launcher memoization follow :mod:`.adapter`. The routing-weight
knobs of the kernels (alpha, beta, prob, norm_const) are pinned to 1. The backward runs the
unified dGLU kernel in its deterministic mode, which writes the routing-weight gradient into one
slot per N tile instead of adding across tiles with float32 atomics.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp
from cudnn.gemm.cutedsl.grouped.dglu.moe_blockscaled_grouped_gemm_dglu_dbias import (
    BlockScaledMoEGroupedGemmDgluDbiasKernel,
)
from cudnn.gemm.cutedsl.grouped.moe_utils import MoEWeightMode
from cudnn.gemm.cutedsl.grouped.swiglu.grouped_gemm_swiglu_quant import BlockScaledContiguousGroupedGemmKernel

from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug._moe.mxfp8.adapter import (
    _BATCH_LAST,
    _SPEC,
    _TWO_CTA_TILE_M,
    _UNUSED_WORKSPACE_BYTES,
    _as_batch_last,
    _check_padded_rows,
    _check_quantized,
    _check_scales,
    _dprob_slot_count,
    _max_active_clusters,
    _require_sm100,
    _sf_col_bytes,
    _sf_row_bytes,
    _ones,
)
from levanter.grug._moe.mxfp8.layout import GATE_UP_BLOCK
from levanter.grug._moe.mxfp8.quantize import MXFP8_BLOCK_SIZE

# (mma_tiler_mn, cluster_shape_mn) per kernel. A sweep of tile and cluster shapes on one B200 at
# H = I = 3072, 3 experts and 302,336 rows found no other configuration more than 3% faster for
# either kernel.
_SWIGLU_TILING = ((256, 256), (2, 1))
_DSWIGLU_TILING = ((256, 256), (2, 1))
# The scalar dSwiGLU epilogue takes 3.98 ms at that shape against 4.52 ms for the packed-f32 one.
# Its rounding differs, so dc13 bytes differ from the packed epilogue's.
_DSWIGLU_VECTORIZED_F32 = False
# The dGLU kernel adds each routing-gradient partial into its slot, so the slots go in zeroed as the
# last input and come back as the last output in the same buffer.
_DPROB_SLOTS_INPUT = 10
_DPROB_SLOTS_OUTPUT = 5


class SwigluOutputs(NamedTuple):
    """:func:`mxfp8_fused_swiglu`'s outputs for ``M`` rows and intermediate size ``I``."""

    c13: jax.Array
    """(M, 2I) bf16 pre-activation, gate and up interleaved in 32-column blocks."""
    h: jax.Array
    """(M, I) e4m3 SwiGLU output quantized along features, the down GEMM's operand."""
    h_col: jax.Array
    """(M, I) e4m3 SwiGLU output quantized along tokens, the down weight gradient's operand."""
    h_scales: jax.Array
    """Flat row-orientation scales of ``h``."""
    h_col_scales: jax.Array
    """Flat column-orientation scales of ``h_col``."""


class DSwigluOutputs(NamedTuple):
    """:func:`mxfp8_fused_dswiglu`'s outputs for ``M`` rows and intermediate size ``I``."""

    dc13: jax.Array
    """(M, 2I) e4m3 gradient of the interleaved pre-activation, quantized along features."""
    dc13_col: jax.Array
    """(M, 2I) e4m3 gradient of the interleaved pre-activation, quantized along tokens."""
    dc13_scales: jax.Array
    """Flat row-orientation scales of ``dc13``."""
    dc13_col_scales: jax.Array
    """Flat column-orientation scales of ``dc13_col``."""
    row_dot: jax.Array
    """(M,) float32 routing-weight gradient ``<swiglu(c13), dh>`` with ``dh = g @ w2^T``."""


# --------------------------------------------------------------------------- #
# fwd w13: grouped GEMM + SwiGLU + dual-orientation MXFP8 quantize
# --------------------------------------------------------------------------- #


@cute_launcher_factory
def _swiglu_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, max_active_clusters):
    """Signature: (stream, x, w13, sfx, sfw13, offs, alpha, prob, norm, c13, h, h_col, sfh_row, sfh_col)."""

    @cute.jit
    def launcher(stream, mA, mB, mSFA, mSFB, mOffs, mAlpha, mProb, mNorm, mC, mD, mDCol, mSFDRow, mSFDCol):
        kernel = BlockScaledContiguousGroupedGemmKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vector_f32=False,
            generate_sfd=True,
            discrete_col_sfd=True,
            expert_cnt=expert_count,
        )
        kernel(
            mA,
            mB,
            mC,
            mD,
            mDCol,
            mSFA,
            mSFB,
            mSFDRow,
            mSFDCol,
            None,  # amax: only produced on the bf16-output path
            mNorm,
            mOffs,
            mAlpha,
            mProb,
            max_active_clusters,
            stream,
        )

    return launcher


def mxfp8_fused_swiglu(x_q, x_sf, w13i_q, w13i_sf, offs) -> SwigluOutputs:
    """Fused forward w13: ``swiglu(x @ w13)`` with a dual-orientation MXFP8 epilogue.

    Args:
        x_q: (M, H) float8_e4m3fn tokens, experts contiguous with ``GROUP_PAD``-padded rows.
        x_sf: row-orientation swizzled scales of ``x_q``, any shape with ``_sf_row_bytes(M, H)`` bytes.
        w13i_q: (E, 2I, H) float8_e4m3fn, gate/up interleaved in 32-column blocks along ``2I``,
            quantized along ``H``.
        w13i_sf: per-expert swizzled scales of ``w13i_q``.
        offs: (E,) int32 padded END offsets, each a multiple of ``GROUP_PAD``.

    Returns:
        The interleaved pre-activation, both quantized copies of the SwiGLU output, and their
        scales; see :class:`SwigluOutputs`.
    """
    _require_sm100()
    m, hidden = x_q.shape
    experts, n2, k = w13i_q.shape
    if k != hidden:
        raise ValueError(f"contraction mismatch: x_q {x_q.shape}, w13i_q {w13i_q.shape}")
    if n2 % (2 * GATE_UP_BLOCK):
        raise ValueError(f"2I={n2} must be a multiple of {2 * GATE_UP_BLOCK} (32-column gate/up block pairs)")
    _check_padded_rows(m, offs, experts)
    _check_quantized(x_q, w13i_q)
    _check_scales(x_sf, w13i_sf)
    intermediate = n2 // 2
    mma_tiler_mn, cluster_shape_mn = _SWIGLU_TILING
    launcher = _swiglu_launcher(
        expert_count=experts,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        max_active_clusters=_max_active_clusters(cluster_shape_mn),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, m, n2), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, m, intermediate), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((_sf_row_bytes(m, intermediate),), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((_sf_col_bytes(m, intermediate),), jnp.float8_e8m0fnu),
        ),
        input_spec=(_BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC),
        output_spec=(_BATCH_LAST, _BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC),
        use_static_tensors=True,
    )
    c13, h, h_col, sfh_row, sfh_col = call(
        _as_batch_last(x_q),
        w13i_q,
        x_sf.reshape(-1),
        w13i_sf.reshape(-1),
        offs.astype(jnp.int32),
        _ones(experts),
        jnp.ones((m, 1, 1), jnp.float32),
        _ones(1),
    )
    return SwigluOutputs(c13[0], h[0], h_col[0], sfh_row, sfh_col)


# --------------------------------------------------------------------------- #
# bwd: dgrad-w2 GEMM + dSwiGLU + dual-orientation MXFP8 quantize + routing gradient
# --------------------------------------------------------------------------- #


@cute_launcher_factory
def _dswiglu_launcher(*, expert_count, mma_tiler_mn, cluster_shape_mn, vectorized_f32, max_active_clusters):
    """Signature: (stream, g, w2, c13, sfg, sfw2, offs, alpha, beta, prob, norm, dprob_slots, dc, dc_col,
    sfdc_row, sfdc_col, workspace); ``dprob_slots`` is the zeroed input the last output aliases."""

    @cute.jit
    def launcher(
        stream,
        mA,
        mB,
        mC,
        mSFA,
        mSFB,
        mOffs,
        mAlpha,
        mBeta,
        mProb,
        mNorm,
        mDProbSlots,
        mD,
        mDCol,
        mSFDRow,
        mSFDCol,
        mWorkspace,
    ):
        kernel = BlockScaledMoEGroupedGemmDgluDbiasKernel(
            sf_vec_size=MXFP8_BLOCK_SIZE,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=mma_tiler_mn[0] == _TWO_CTA_TILE_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vectorized_f32,
            discrete_col_sfd=True,
            expert_cnt=expert_count,
            weight_mode=MoEWeightMode.DENSE,
            use_dynamic_sched=False,
            act_func="dswiglu",
            deterministic=True,
        )
        kernel(
            mA,
            mB,
            mSFB,
            cutlass.Int32(0),  # n: discrete weight mode only
            cutlass.Int32(0),  # k: discrete weight mode only
            cutlass.Int64(0),  # b_stride_size: discrete weight mode only
            cute.nvgpu.OperandMajorMode.K,  # b_major_mode: discrete weight mode only
            mWorkspace.iterator,
            mC,
            mD,
            mDCol,
            mSFA,
            mSFDRow,
            mSFDCol,
            None,  # amax: only produced on the bf16-output path
            mNorm,
            mOffs,
            mAlpha,
            mBeta,
            mProb,
            mDProbSlots,
            None,  # dbias: the expert MLP has no bias
            max_active_clusters,
            stream,
        )

    return launcher


def mxfp8_fused_dswiglu(g_q, g_sf, w2_q, w2_sf, c13, offs) -> DSwigluOutputs:
    """Fused backward: ``dc13 = dSwiGLU(g @ w2^T, c13)`` with a dual MXFP8 epilogue.

    Args:
        g_q: (M, H) float8_e4m3fn output cotangent, quantized along ``H``.
        g_sf: row-orientation swizzled scales of ``g_q``.
        w2_q: (E, I, H) float8_e4m3fn, ``w2`` in its stored layout quantized along ``H``.
        w2_sf: per-expert swizzled scales of ``w2_q``.
        c13: (M, 2I) bf16 interleaved pre-activation from :func:`mxfp8_fused_swiglu`.
        offs: (E,) int32 padded END offsets, each a multiple of ``GROUP_PAD``.

    Returns:
        Both quantized copies of the interleaved pre-activation's gradient, their scales, and each
        row's routing-weight gradient; see :class:`DSwigluOutputs`. The kernel adds each N tile's
        routing-gradient partial into that tile's zeroed slot, the only write to the slot, and the
        slots are summed here, so the row dot is deterministic.
    """
    _require_sm100()
    m, hidden = g_q.shape
    experts, intermediate, k = w2_q.shape
    if k != hidden:
        raise ValueError(f"contraction mismatch: g_q {g_q.shape}, w2_q {w2_q.shape}")
    n2 = 2 * intermediate
    if c13.shape != (m, n2) or c13.dtype != jnp.bfloat16:
        raise ValueError(f"c13 must be bf16 of shape ({m}, {n2}), got {c13.dtype} {c13.shape}")
    _check_padded_rows(m, offs, experts)
    _check_quantized(g_q, w2_q)
    _check_scales(g_sf, w2_sf)
    mma_tiler_mn, cluster_shape_mn = _DSWIGLU_TILING
    slots = _dprob_slot_count(intermediate, _DSWIGLU_TILING)
    launcher = _dswiglu_launcher(
        expert_count=experts,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vectorized_f32=_DSWIGLU_VECTORIZED_F32,
        max_active_clusters=_max_active_clusters(cluster_shape_mn),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, m, n2), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, m, n2), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((_sf_row_bytes(m, n2),), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((_sf_col_bytes(m, n2),), jnp.float8_e8m0fnu),
            jax.ShapeDtypeStruct((_UNUSED_WORKSPACE_BYTES,), jnp.uint8),
            jax.ShapeDtypeStruct((m, slots, 1), jnp.float32),
        ),
        input_spec=(_BATCH_LAST, _BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC, _SPEC),
        output_spec=(_BATCH_LAST, _BATCH_LAST, _SPEC, _SPEC, _SPEC, _SPEC),
        input_output_aliases=((_DPROB_SLOTS_INPUT, _DPROB_SLOTS_OUTPUT),),
        use_static_tensors=True,
    )
    dc, dc_col, sfdc_row, sfdc_col, _workspace, dprob_slots = call(
        _as_batch_last(g_q),
        w2_q,
        _as_batch_last(c13),
        g_sf.reshape(-1),
        w2_sf.reshape(-1),
        offs.astype(jnp.int32),
        _ones(experts),
        _ones(experts),
        jnp.ones((m, 1, 1), jnp.float32),
        _ones(1),
        jnp.zeros((m, slots, 1), jnp.float32),
    )
    return DSwigluOutputs(dc[0], dc_col[0], sfdc_row, sfdc_col, jnp.sum(dprob_slots[:, :, 0], axis=-1))
