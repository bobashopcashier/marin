# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerics and determinism of the cudnn-frontend MXFP8 grouped kernels (SM100).

Each kernel is checked against per-expert f32 matmuls on the dequantized operands it consumes, so
the comparison isolates the kernel from quantization noise. The operands are quantized here by an
independent implementation of the MXFP8 rule and the SM100 scale layouts. Quantized outputs are
checked after dequantizing both the kernel's bytes and scales and an independent reference
quantization of the f32 reference result.
"""

import functools
import importlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.mxfp8.layout import GROUP_PAD
from levanter.grug._moe.mxfp8.quantize import WeightMajor
from levanter.testing.devices import nvidia_compute_capability

_SF_VEC = 32
_E4M3_MAX = 448.0
_HIGHEST = jax.lax.Precision.HIGHEST
_SM100_COMPUTE_CAPABILITY_MAJOR = 10
# The bytes of an E4M3 NaN and an E8M0 NaN, for operand rows the kernels must not read.
_E4M3_NAN_BYTE = 0x7F
_E8M0_NAN_BYTE = 0xFF

# Bounds on the relative Frobenius error against the f32 reference rounded to the output's
# precision. The kernels accumulate in f32, so the measured errors are 1e-4 or less.
_GEMM_TOLERANCE = 1e-3
_WGRAD_TOLERANCE = 2e-3
_QUANTIZED_TOLERANCE = 2e-3
_DPROB_TOLERANCE = 2e-2
# Width of the gate and up column blocks the fused SwiGLU epilogue pairs.
_GATE_UP_BLOCK = 32


class _Problem(NamedTuple):
    hidden: int
    intermediate: int
    # Token count per expert before padding; zero marks an expert that receives no tokens.
    group_sizes: tuple[int, ...]


_SMALL_PROBLEMS = [
    # One expert gets no tokens.
    _Problem(hidden=256, intermediate=512, group_sizes=(300, 0, 1000, 213)),
    # The last expert gets no tokens; H and I end inside a 256-wide N tile of both kernels.
    _Problem(hidden=384, intermediate=640, group_sizes=(700, 512, 0)),
]
# One expert chunk of the EP hero run on one GPU: H = I = 3072, 3 experts, 302,336 rows.
_HERO_PROBLEM = _Problem(hidden=3072, intermediate=3072, group_sizes=(140_000, 100_000, 62_000))
_SMALL = [pytest.param(problem, id=f"h{problem.hidden}-i{problem.intermediate}") for problem in _SMALL_PROBLEMS]
_PROBLEMS = [*_SMALL, pytest.param(_HERO_PROBLEM, id="hero-chunk", marks=pytest.mark.slow)]


def _require_sm100():
    capability = nvidia_compute_capability()
    if capability is None or int(capability) != _SM100_COMPUTE_CAPABILITY_MAJOR:
        pytest.skip("fused MXFP8 kernels require an SM100 GPU")
    pytest.importorskip("cutlass")
    pytest.importorskip("cudnn")
    pytest.importorskip("quack")
    return importlib.import_module("levanter.grug._moe.mxfp8.adapter")


def _padded_sizes(problem: _Problem) -> tuple[int, ...]:
    return tuple(-(-size // GROUP_PAD) * GROUP_PAD for size in problem.group_sizes)


# --------------------------------------------------------------------------- #
# reference MXFP8 quantization and SM100 scale-factor layouts
# --------------------------------------------------------------------------- #


def _e8m0_round_up(x):
    """Exponent byte of the smallest power of two >= x, for finite x >= 0 (byte 0 is 2^-127)."""
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    exponent = bits >> 23
    mantissa = bits & 0x7FFFFF
    round_up = (mantissa > 0) & (exponent != 0xFE) & ~((exponent == 0) & (mantissa <= 0x400000))
    return jnp.where(round_up, exponent + 1, exponent).astype(jnp.uint8)


def _e8m0_decode(scale):
    exponent = scale.astype(jnp.uint32)
    bits = jnp.where(exponent == 0, jnp.uint32(0x00400000), exponent << 23)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _quantize(x, axis):
    """MXFP8 along ``axis`` in blocks of 32: (float8_e4m3fn values, uint8 e8m0 scales)."""
    moved = jnp.moveaxis(x.astype(jnp.float32), axis, -1)
    blocks = moved.reshape(*moved.shape[:-1], moved.shape[-1] // _SF_VEC, _SF_VEC)
    scale = _e8m0_round_up(jnp.max(jnp.abs(blocks), axis=-1) / _E4M3_MAX)
    q = jnp.clip(blocks / _e8m0_decode(scale)[..., None], -_E4M3_MAX, _E4M3_MAX).astype(jnp.float8_e4m3fn)
    return jnp.moveaxis(q.reshape(moved.shape), -1, axis), jnp.moveaxis(scale, -1, axis)


def _dequantize(q, scale, axis):
    moved = jnp.moveaxis(q.astype(jnp.float32), axis, -1)
    blocks = moved.reshape(*moved.shape[:-1], moved.shape[-1] // _SF_VEC, _SF_VEC)
    values = blocks * _e8m0_decode(jnp.moveaxis(scale, axis, -1))[..., None]
    return jnp.moveaxis(values.reshape(moved.shape), -1, axis)


def _swizzle(scale):
    """(rows, cols) uint8 scales -> flat bytes in the SM100 32x4x4 atom order (128x4 tiles, row-major)."""
    rows, cols = scale.shape
    assert rows % 128 == 0 and cols % 4 == 0, scale.shape
    tiles = scale.reshape(rows // 128, 128, cols // 4, 4).transpose(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).swapaxes(1, 2).reshape(-1)


def _unswizzle(flat, rows, cols):
    tiles = jax.lax.bitcast_convert_type(flat.reshape(-1), jnp.uint8)
    tiles = tiles.reshape(-1, 32, 4, 4).swapaxes(1, 2).reshape(rows // 128, cols // 4, 128, 4)
    return tiles.transpose(0, 2, 1, 3).reshape(rows, cols)


def _as_e8m0(raw):
    return jax.lax.bitcast_convert_type(raw, jnp.float8_e8m0fnu)


def _column_scales(scale, padded_sizes):
    """(T/32, F) token-axis scales -> per-expert swizzled (F, T/32 slice) chunks, as an (F, T/32) array."""
    by_feature = scale.T
    chunks, start = [], 0
    for size in padded_sizes:
        chunks.append(_swizzle(by_feature[:, start : start + size // _SF_VEC]))
        start += size // _SF_VEC
    return _as_e8m0(jnp.concatenate(chunks).reshape(by_feature.shape))


def _unswizzle_columns(flat, features, padded_sizes):
    """Inverse of :func:`_column_scales`: kernel column-orientation scales -> (T/32, F) uint8."""
    flat = flat.reshape(-1)
    columns, start = [], 0
    for size in padded_sizes:
        count = features * size // _SF_VEC
        columns.append(_unswizzle(flat[start : start + count], features, size // _SF_VEC))
        start += count
    return jnp.concatenate(columns, axis=1).T


def _weight_scales(scale):
    """(E, N, K/32) weight scales -> one swizzled block per expert."""
    return _as_e8m0(jnp.stack([_swizzle(scale[e]) for e in range(scale.shape[0])]))


# --------------------------------------------------------------------------- #
# operands, kernel calls, and the f32 reference
# --------------------------------------------------------------------------- #


class _Inputs(NamedTuple):
    x: jax.Array  # (M, H) bf16, padding rows zero
    g: jax.Array  # (M, I) bf16, padding rows zero
    w: jax.Array  # (E, H, I) bf16


class _KernelOperands(NamedTuple):
    x: jax.Array  # quantized along H, the GEMM's A operand
    x_sf: jax.Array
    x_col: jax.Array  # quantized along tokens, a weight-gradient operand
    x_col_sf: jax.Array
    g_col: jax.Array
    g_col_sf: jax.Array
    w_n: jax.Array  # (E, H, I) quantized along H in place, read N-major
    w_k: jax.Array  # the same values as their (E, I, H) transpose, read K-major
    w_sf: jax.Array  # scales over (I, H / 32), shared by both layouts


def _make_inputs(problem: _Problem, seed: int) -> _Inputs:
    padded = _padded_sizes(problem)
    valid = np.concatenate([np.arange(p) < s for s, p in zip(problem.group_sizes, padded, strict=True)])
    valid = jnp.asarray(valid)[:, None]
    experts, hidden, intermediate = len(padded), problem.hidden, problem.intermediate
    kx, kg, kw = jax.random.split(jax.random.key(seed), 3)
    rows = sum(padded)
    x = jnp.where(valid, jax.random.normal(kx, (rows, hidden), jnp.bfloat16), 0)
    g = jnp.where(valid, jax.random.normal(kg, (rows, intermediate), jnp.bfloat16), 0)
    w = (jax.random.normal(kw, (experts, hidden, intermediate), jnp.float32) / hidden**0.5).astype(jnp.bfloat16)
    return _Inputs(x=x, g=g, w=w)


def _quantize_operands(inputs: _Inputs, padded_sizes: tuple[int, ...]) -> _KernelOperands:
    x, x_scale = _quantize(inputs.x, axis=1)
    x_col, x_col_scale = _quantize(inputs.x, axis=0)
    g_col, g_col_scale = _quantize(inputs.g, axis=0)
    w_n, w_scale = _quantize(inputs.w, axis=1)  # scales [E, H/32, I]
    return _KernelOperands(
        x=x,
        x_sf=_as_e8m0(_swizzle(x_scale)),
        x_col=x_col,
        x_col_sf=_column_scales(x_col_scale, padded_sizes),
        g_col=g_col,
        g_col_sf=_column_scales(g_col_scale, padded_sizes),
        w_n=w_n,
        w_k=jnp.swapaxes(w_n, 1, 2),
        w_sf=_weight_scales(jnp.swapaxes(w_scale, 1, 2)),
    )


class _Outputs(NamedTuple):
    y: jax.Array  # (M, I) bf16, from the N-major weights
    y_k_major: jax.Array  # (M, I) bf16, from the K-major transpose
    dw: jax.Array  # (E, H, I) bf16


def _run_kernels(adapter, padded_sizes: tuple[int, ...], ops: _KernelOperands) -> _Outputs:
    offs = jnp.asarray(np.cumsum(padded_sizes), jnp.int32)
    y = adapter.mxfp8_fused_gemm(ops.x, ops.x_sf, ops.w_n, ops.w_sf, offs, b_major=WeightMajor.N)
    y_k_major = adapter.mxfp8_fused_gemm(ops.x, ops.x_sf, ops.w_k, ops.w_sf, offs, b_major=WeightMajor.K)
    dw = adapter.mxfp8_fused_wgrad(ops.x_col, ops.x_col_sf, ops.g_col, ops.g_col_sf, offs)
    return _Outputs(y, y_k_major, dw)


def _grouped_matmul(a, b, padded_sizes):
    """Per expert ``a[rows_e] @ b[e]`` in f32: a (M, K), b (E, K, N) -> (M, N)."""
    outputs, start = [], 0
    for expert, size in enumerate(padded_sizes):
        outputs.append(jnp.matmul(a[start : start + size], b[expert], precision=_HIGHEST))
        start += size
    return jnp.concatenate(outputs)


def _grouped_wgrad(a, b, padded_sizes):
    """Per expert ``a[rows_e].T @ b[rows_e]`` in f32: a (M, K), b (M, N) -> (E, K, N)."""
    outputs, start = [], 0
    for size in padded_sizes:
        outputs.append(jnp.matmul(a[start : start + size].T, b[start : start + size], precision=_HIGHEST))
        start += size
    return jnp.stack(outputs)


def _relative_error(actual, expected):
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    return jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)


def _dequantize_rows(q, sf):
    rows, features = q.shape
    return _dequantize(q, _unswizzle(sf, rows, features // _SF_VEC), axis=1)


def _dequantize_columns(q, sf, padded_sizes):
    return _dequantize(q, _unswizzle_columns(sf, q.shape[1], padded_sizes), axis=0)


def _dequantize_n_major_weight(q, sf):
    experts, k, n = q.shape
    scale = jnp.stack([_unswizzle(sf[e], n, k // _SF_VEC) for e in range(experts)])
    return _dequantize(q, jnp.swapaxes(scale, 1, 2), axis=1)


# The reference is built in separately jitted stages. Within one XLA computation the
# excess-precision rewrite may drop an f32 -> fp8/bf16 -> f32 round trip, which would compare the
# kernels against an unquantized or unrounded reference. Values crossing a jit boundary are
# materialized in their dtype, so every dequantization reads materialized bytes and the bf16
# rounding of the reference happens in its own stage.


class _References(NamedTuple):
    """f32 reference of each kernel output, computed from the operands that kernel consumed."""

    y: jax.Array
    dw: jax.Array


def _references(padded_sizes: tuple[int, ...], ops: _KernelOperands) -> _References:
    x = _dequantize_rows(ops.x, ops.x_sf)
    x_col = _dequantize_columns(ops.x_col, ops.x_col_sf, padded_sizes)
    g_col = _dequantize_columns(ops.g_col, ops.g_col_sf, padded_sizes)
    w = _dequantize_n_major_weight(ops.w_n, ops.w_sf)  # (E, H, I)
    return _References(y=_grouped_matmul(x, w, padded_sizes), dw=_grouped_wgrad(x_col, g_col, padded_sizes))


def _round_references(ref: _References) -> _References:
    return _References(y=ref.y.astype(jnp.bfloat16), dw=ref.dw.astype(jnp.bfloat16))


def _errors(out: _Outputs, ref: _References) -> dict[str, jax.Array]:
    return {"y": _relative_error(out.y, ref.y), "dw": _relative_error(out.dw, ref.dw)}


_TOLERANCES = {"y": _GEMM_TOLERANCE, "dw": _WGRAD_TOLERANCE}


def _bits(a):
    unsigned = {1: jnp.uint8, 2: jnp.uint16, 4: jnp.uint32}[a.dtype.itemsize]
    return jax.lax.bitcast_convert_type(a, unsigned)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _PROBLEMS)
def test_mxfp8_grouped_gemm_and_wgrad_match_dequantized_f32_reference(problem):
    adapter = _require_sm100()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_operands, padded_sizes=padded))(_make_inputs(problem, seed=0))
    out = jax.jit(functools.partial(_run_kernels, adapter, padded))(operands)
    references = jax.jit(functools.partial(_references, padded))(operands)
    rounded = jax.jit(_round_references)(references)
    del references
    errors = {name: float(value) for name, value in jax.jit(_errors)(out, rounded).items()}
    print(f"{problem}: " + ", ".join(f"{name}={value:.3e}" for name, value in errors.items()))

    failures = {name: value for name, value in errors.items() if not value < _TOLERANCES[name]}
    assert not failures, f"kernels over tolerance: {failures} (tolerances {_TOLERANCES})"

    for expert, size in enumerate(problem.group_sizes):
        if size == 0:
            assert not np.any(np.asarray(out.dw[expert])), f"expert {expert} has no tokens but a nonzero dw"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _PROBLEMS)
def test_mxfp8_grouped_gemm_and_wgrad_are_bitwise_identical_across_reruns(problem):
    adapter = _require_sm100()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_operands, padded_sizes=padded))(_make_inputs(problem, seed=1))
    run = jax.jit(functools.partial(_run_kernels, adapter, padded))
    first = run(operands)
    for rerun in range(2):
        again = run(operands)
        identical = jax.tree.map(lambda a, b: bool(jnp.array_equal(_bits(a), _bits(b))), first, again)
        differing = [name for name, same in identical._asdict().items() if not same]
        assert not differing, f"rerun {rerun + 1} changed {differing}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _PROBLEMS)
def test_mxfp8_grouped_gemm_reads_n_major_weights_as_their_k_major_transpose_bitwise(problem):
    # An expert MLP's forward GEMMs read their weights N-major so that no weight is transposed;
    # they must compute exactly what the K-major weight-transpose layout computes.
    adapter = _require_sm100()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_operands, padded_sizes=padded))(_make_inputs(problem, seed=2))
    out = jax.jit(functools.partial(_run_kernels, adapter, padded))(operands)

    assert bool(jnp.array_equal(_bits(out.y), _bits(out.y_k_major)))


def _with_unread_tail(ops: _KernelOperands, tail_rows: int) -> _KernelOperands:
    """``ops`` with ``tail_rows`` more rows after the last expert's end, holding NaN values and NaN scales."""

    def nan_rows(q):
        tail = jnp.full((tail_rows, q.shape[1]), _E4M3_NAN_BYTE, jnp.uint8)
        return jnp.concatenate([q, jax.lax.bitcast_convert_type(tail, jnp.float8_e4m3fn)])

    def nan_row_scales(sf, features):
        tail = jnp.full((tail_rows * features // _SF_VEC,), _E8M0_NAN_BYTE, jnp.uint8)
        return jnp.concatenate([sf.reshape(-1), _as_e8m0(tail)])

    def nan_column_scales(sf):
        # The experts' chunks are concatenated flat, so the tail's scales follow the last chunk.
        features = sf.shape[0]
        tail = jnp.full((features * tail_rows // _SF_VEC,), _E8M0_NAN_BYTE, jnp.uint8)
        return jnp.concatenate([sf.reshape(-1), _as_e8m0(tail)]).reshape(features, -1)

    return ops._replace(
        x=nan_rows(ops.x),
        x_sf=nan_row_scales(ops.x_sf, ops.x.shape[1]),
        x_col=nan_rows(ops.x_col),
        x_col_sf=nan_column_scales(ops.x_col_sf),
        g_col=nan_rows(ops.g_col),
        g_col_sf=nan_column_scales(ops.g_col_sf),
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SMALL)
def test_mxfp8_grouped_gemm_and_wgrad_do_not_read_rows_past_the_last_experts_end(problem):
    # A receiver buffer keeps unused capacity after the last expert. The kernels schedule their
    # tiles from the experts' end offsets, so NaN operands and scales in those rows must change no
    # output row before the end and no weight gradient.
    adapter = _require_sm100()
    padded = _padded_sizes(problem)
    rows = sum(padded)
    operands = jax.jit(functools.partial(_quantize_operands, padded_sizes=padded))(_make_inputs(problem, seed=3))
    run = jax.jit(functools.partial(_run_kernels, adapter, padded))
    exact = run(operands)
    with_tail = run(jax.jit(functools.partial(_with_unread_tail, tail_rows=2 * GROUP_PAD))(operands))

    assert with_tail.y.shape[0] == rows + 2 * GROUP_PAD
    assert bool(jnp.array_equal(_bits(with_tail.y[:rows]), _bits(exact.y)))
    assert bool(jnp.array_equal(_bits(with_tail.dw), _bits(exact.dw)))


# --------------------------------------------------------------------------- #
# the six legs of a SwiGLU expert MLP
# --------------------------------------------------------------------------- #

# One expert gets no tokens; the intermediate sizes give 2 and 3 dSwiGLU column tiles.
_SWIGLU_SMALL_PROBLEMS = [
    _Problem(hidden=256, intermediate=512, group_sizes=(300, 0, 1000, 213)),
    _Problem(hidden=512, intermediate=768, group_sizes=(700, 512, 0)),
]
_SWIGLU_PROBLEMS = [
    *(pytest.param(problem, id=f"h{problem.hidden}-i{problem.intermediate}") for problem in _SWIGLU_SMALL_PROBLEMS),
    pytest.param(_HERO_PROBLEM, id="hero-chunk", marks=pytest.mark.slow),
]


def _require_swiglu_adapters():
    return _require_sm100(), importlib.import_module("levanter.grug._moe.mxfp8.adapter_swiglu")


class _SwigluInputs(NamedTuple):
    x: jax.Array  # (M, H) bf16, padding rows zero
    g: jax.Array  # (M, H) bf16 output cotangent, padding rows zero
    w13i: jax.Array  # (E, 2I, H) bf16, gate/up interleaved in 32-column blocks
    w2: jax.Array  # (E, I, H) bf16


class _SwigluOperands(NamedTuple):
    x: jax.Array
    x_sf: jax.Array
    x_col: jax.Array
    x_col_sf: jax.Array
    g: jax.Array
    g_sf: jax.Array
    g_col: jax.Array
    g_col_sf: jax.Array
    # Every weight K-major, (E, N, K) quantized along K.
    w13i: jax.Array  # (E, 2I, H), the gate/up GEMM's
    w13i_sf: jax.Array
    w2f: jax.Array  # (E, H, I), the down GEMM's
    w2f_sf: jax.Array
    w2: jax.Array  # (E, I, H), the down data gradient's
    w2_sf: jax.Array
    w13dg: jax.Array  # (E, H, 2I), the gate/up data gradient's
    w13dg_sf: jax.Array


def _make_swiglu_inputs(problem: _Problem, seed: int) -> _SwigluInputs:
    padded = _padded_sizes(problem)
    valid = np.concatenate([np.arange(p) < s for s, p in zip(problem.group_sizes, padded, strict=True)])
    valid = jnp.asarray(valid)[:, None]
    experts, hidden, intermediate = len(padded), problem.hidden, problem.intermediate
    kx, kg, k13, k2 = jax.random.split(jax.random.key(seed), 4)
    rows = sum(padded)
    x = jnp.where(valid, jax.random.normal(kx, (rows, hidden), jnp.bfloat16), 0)
    g = jnp.where(valid, jax.random.normal(kg, (rows, hidden), jnp.bfloat16), 0)
    w13i = (jax.random.normal(k13, (experts, 2 * intermediate, hidden), jnp.float32) / hidden**0.5).astype(
        jnp.bfloat16
    )
    w2 = (jax.random.normal(k2, (experts, intermediate, hidden), jnp.float32) / intermediate**0.5).astype(jnp.bfloat16)
    return _SwigluInputs(x=x, g=g, w13i=w13i, w2=w2)


def _quantize_swiglu_operands(inputs: _SwigluInputs, padded_sizes: tuple[int, ...]) -> _SwigluOperands:
    def rows(t):
        q, scale = _quantize(t, axis=1)
        return q, _as_e8m0(_swizzle(scale))

    def columns(t):
        q, scale = _quantize(t, axis=0)
        return q, _column_scales(scale, padded_sizes)

    def weight(w):
        q, scale = _quantize(w, axis=2)
        return q, _weight_scales(scale)

    return _SwigluOperands(
        *rows(inputs.x),
        *columns(inputs.x),
        *rows(inputs.g),
        *columns(inputs.g),
        *weight(inputs.w13i),
        *weight(jnp.swapaxes(inputs.w2, 1, 2)),
        *weight(inputs.w2),
        *weight(jnp.swapaxes(inputs.w13i, 1, 2)),
    )


class _SwigluOutputs(NamedTuple):
    c13: jax.Array
    h: jax.Array
    h_col: jax.Array
    sfh_row: jax.Array
    sfh_col: jax.Array
    y: jax.Array
    dc: jax.Array
    dc_col: jax.Array
    sfdc_row: jax.Array
    sfdc_col: jax.Array
    dprob: jax.Array
    dx: jax.Array
    dw13i: jax.Array
    dw2: jax.Array


def _run_swiglu_legs(adapters, padded_sizes: tuple[int, ...], ops: _SwigluOperands) -> _SwigluOutputs:
    adapter, adapter_swiglu = adapters
    offs = jnp.asarray(np.cumsum(padded_sizes), jnp.int32)
    n2 = ops.w13i.shape[1]
    k_major = WeightMajor.K
    c13, h, h_col, sfh_row, sfh_col = adapter_swiglu.mxfp8_fused_swiglu(ops.x, ops.x_sf, ops.w13i, ops.w13i_sf, offs)
    y = adapter.mxfp8_fused_gemm(h, sfh_row, ops.w2f, ops.w2f_sf, offs, b_major=k_major)
    dc, dc_col, sfdc_row, sfdc_col, dprob = adapter_swiglu.mxfp8_fused_dswiglu(
        ops.g, ops.g_sf, ops.w2, ops.w2_sf, c13, offs
    )
    dx = adapter.mxfp8_fused_gemm(dc, sfdc_row, ops.w13dg, ops.w13dg_sf, offs, b_major=k_major)
    dw13i = adapter.mxfp8_fused_wgrad(ops.x_col, ops.x_col_sf, dc_col, sfdc_col.reshape(n2, -1), offs)
    dw2 = adapter.mxfp8_fused_wgrad(h_col, sfh_col.reshape(n2 // 2, -1), ops.g_col, ops.g_col_sf, offs)
    return _SwigluOutputs(c13, h, h_col, sfh_row, sfh_col, y, dc, dc_col, sfdc_row, sfdc_col, dprob, dx, dw13i, dw2)


def _swiglu(c):
    blocks = c.reshape(*c.shape[:-1], -1, 2, _GATE_UP_BLOCK)
    gate = blocks[..., 0, :].reshape(*c.shape[:-1], -1)
    up = blocks[..., 1, :].reshape(*c.shape[:-1], -1)
    return jax.nn.silu(gate) * up


def _dequantize_k_major_weight(q, sf):
    experts, n, k = q.shape
    scale = jnp.stack([_unswizzle(sf[e], n, k // _SF_VEC) for e in range(experts)])
    return _dequantize(q, scale, axis=2)


class _SwigluReferences(NamedTuple):
    """f32 reference of each kernel output, computed from the operands that kernel consumed."""

    c13: jax.Array
    h: jax.Array
    y: jax.Array
    dc: jax.Array
    dprob: jax.Array
    dx: jax.Array
    dw13i: jax.Array
    dw2: jax.Array


def _swiglu_references(padded_sizes: tuple[int, ...], ops: _SwigluOperands, out: _SwigluOutputs) -> _SwigluReferences:
    def matmul(a, w):
        # a (M, K) against K-major weights (E, N, K).
        return _grouped_matmul(a, jnp.swapaxes(w, 1, 2), padded_sizes)

    x = _dequantize_rows(ops.x, ops.x_sf)
    g = _dequantize_rows(ops.g, ops.g_sf)
    x_col = _dequantize_columns(ops.x_col, ops.x_col_sf, padded_sizes)
    g_col = _dequantize_columns(ops.g_col, ops.g_col_sf, padded_sizes)
    w13i = _dequantize_k_major_weight(ops.w13i, ops.w13i_sf)
    w2f = _dequantize_k_major_weight(ops.w2f, ops.w2f_sf)
    w2 = _dequantize_k_major_weight(ops.w2, ops.w2_sf)
    w13dg = _dequantize_k_major_weight(ops.w13dg, ops.w13dg_sf)
    # Downstream legs consume the kernels' own quantized intermediates.
    h = _dequantize_rows(out.h, out.sfh_row)
    h_col = _dequantize_columns(out.h_col, out.sfh_col, padded_sizes)
    dc = _dequantize_rows(out.dc, out.sfdc_row)
    dc_col = _dequantize_columns(out.dc_col, out.sfdc_col, padded_sizes)

    c13_ref = matmul(x, w13i)
    dh = matmul(g, w2)
    # The dSwiGLU epilogue reads the bf16 pre-activation the forward kernel stored.
    c13 = out.c13.astype(jnp.float32)
    _, swiglu_pullback = jax.vjp(_swiglu, c13)
    (dc_ref,) = swiglu_pullback(dh)
    return _SwigluReferences(
        c13=c13_ref,
        h=_swiglu(c13_ref),
        y=matmul(h, w2f),
        dc=dc_ref,
        dprob=jnp.sum(_swiglu(c13) * dh, axis=-1),
        dx=matmul(dc, w13dg),
        dw13i=_grouped_wgrad(x_col, dc_col, padded_sizes),
        dw2=_grouped_wgrad(h_col, g_col, padded_sizes),
    )


class _RoundedSwigluReferences(NamedTuple):
    """The references in the output's precision: bf16 for GEMM outputs, MXFP8 (values, scales) otherwise."""

    c13: jax.Array
    y: jax.Array
    dx: jax.Array
    dw13i: jax.Array
    dw2: jax.Array
    h_rows: tuple[jax.Array, jax.Array]
    h_cols: tuple[jax.Array, jax.Array]
    dc_rows: tuple[jax.Array, jax.Array]
    dc_cols: tuple[jax.Array, jax.Array]
    dprob: jax.Array


def _round_swiglu_references(ref: _SwigluReferences) -> _RoundedSwigluReferences:
    return _RoundedSwigluReferences(
        c13=ref.c13.astype(jnp.bfloat16),
        y=ref.y.astype(jnp.bfloat16),
        dx=ref.dx.astype(jnp.bfloat16),
        dw13i=ref.dw13i.astype(jnp.bfloat16),
        dw2=ref.dw2.astype(jnp.bfloat16),
        h_rows=_quantize(ref.h, axis=1),
        h_cols=_quantize(ref.h, axis=0),
        dc_rows=_quantize(ref.dc, axis=1),
        dc_cols=_quantize(ref.dc, axis=0),
        dprob=ref.dprob,
    )


def _swiglu_leg_errors(
    padded_sizes: tuple[int, ...], out: _SwigluOutputs, ref: _RoundedSwigluReferences
) -> dict[str, jax.Array]:
    """Relative Frobenius error of every kernel output against its rounded reference."""
    return {
        "c13": _relative_error(out.c13, ref.c13),
        "y": _relative_error(out.y, ref.y),
        "dx": _relative_error(out.dx, ref.dx),
        "dw13i": _relative_error(out.dw13i, ref.dw13i),
        "dw2": _relative_error(out.dw2, ref.dw2),
        "h": _relative_error(_dequantize_rows(out.h, out.sfh_row), _dequantize(*ref.h_rows, axis=1)),
        "h_col": _relative_error(
            _dequantize_columns(out.h_col, out.sfh_col, padded_sizes), _dequantize(*ref.h_cols, axis=0)
        ),
        "dc": _relative_error(_dequantize_rows(out.dc, out.sfdc_row), _dequantize(*ref.dc_rows, axis=1)),
        "dc_col": _relative_error(
            _dequantize_columns(out.dc_col, out.sfdc_col, padded_sizes), _dequantize(*ref.dc_cols, axis=0)
        ),
        "dprob": _relative_error(out.dprob, ref.dprob),
    }


_SWIGLU_TOLERANCES = {
    "c13": _GEMM_TOLERANCE,
    "y": _GEMM_TOLERANCE,
    "dx": _GEMM_TOLERANCE,
    "h": _QUANTIZED_TOLERANCE,
    "h_col": _QUANTIZED_TOLERANCE,
    "dc": _QUANTIZED_TOLERANCE,
    "dc_col": _QUANTIZED_TOLERANCE,
    "dw13i": _QUANTIZED_TOLERANCE,
    "dw2": _QUANTIZED_TOLERANCE,
    "dprob": _DPROB_TOLERANCE,
}


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SWIGLU_PROBLEMS)
def test_fused_mxfp8_swiglu_legs_match_dequantized_f32_reference(problem):
    adapters = _require_swiglu_adapters()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_swiglu_operands, padded_sizes=padded))(
        _make_swiglu_inputs(problem, seed=0)
    )
    out = jax.jit(functools.partial(_run_swiglu_legs, adapters, padded))(operands)
    references = jax.jit(functools.partial(_swiglu_references, padded))(operands, out)
    rounded = jax.jit(_round_swiglu_references)(references)
    del references
    errors = jax.jit(functools.partial(_swiglu_leg_errors, padded))(out, rounded)
    errors = {name: float(value) for name, value in errors.items()}
    print(f"{problem}: " + ", ".join(f"{name}={value:.3e}" for name, value in errors.items()))

    failures = {name: value for name, value in errors.items() if not value < _SWIGLU_TOLERANCES[name]}
    assert not failures, f"legs over tolerance: {failures} (tolerances {_SWIGLU_TOLERANCES})"

    for expert, size in enumerate(problem.group_sizes):
        if size == 0:
            assert not np.any(np.asarray(out.dw13i[expert])), f"expert {expert} has no tokens but a nonzero dw13"
            assert not np.any(np.asarray(out.dw2[expert])), f"expert {expert} has no tokens but a nonzero dw2"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _SWIGLU_PROBLEMS)
def test_fused_mxfp8_swiglu_legs_are_bitwise_identical_across_reruns(problem):
    adapters = _require_swiglu_adapters()
    padded = _padded_sizes(problem)
    operands = jax.jit(functools.partial(_quantize_swiglu_operands, padded_sizes=padded))(
        _make_swiglu_inputs(problem, seed=1)
    )
    run = jax.jit(functools.partial(_run_swiglu_legs, adapters, padded))
    first = run(operands)
    for rerun in range(2):
        again = run(operands)
        identical = jax.tree.map(lambda a, b: bool(jnp.array_equal(_bits(a), _bits(b))), first, again)
        differing = [name for name, same in identical._asdict().items() if not same]
        assert not differing, f"rerun {rerun + 1} changed {differing}"


# The hero chunk's 303,104 padded rows with skewed routing and an expert that gets no tokens.
_SKEWED_HERO_PROBLEM = _Problem(hidden=3072, intermediate=3072, group_sizes=(182_000, 0, 121_000))
_FORWARD_ONLY_PROBLEMS = [
    *(pytest.param(problem, id=f"h{problem.hidden}-i{problem.intermediate}") for problem in _SWIGLU_SMALL_PROBLEMS),
    pytest.param(_SKEWED_HERO_PROBLEM, id="hero-chunk-skewed", marks=pytest.mark.slow),
]


def _gate_up_operands(inputs: _SwigluInputs) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The gate/up GEMM's operands: row-quantized tokens and weights, with their swizzled scales."""
    x, x_scale = _quantize(inputs.x, axis=1)
    w13i, w13i_scale = _quantize(inputs.w13i, axis=2)
    return x, _as_e8m0(_swizzle(x_scale)), w13i, _weight_scales(w13i_scale)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("problem", _FORWARD_ONLY_PROBLEMS)
def test_forward_only_swiglu_writes_the_upstream_kernels_h_and_row_scales_bitwise(problem):
    # The full call runs upstream's kernel class and the forward-only call the Marin subclass, so
    # this also holds the copied methods to upstream's output.
    _adapter, adapter_swiglu = _require_swiglu_adapters()
    operands = jax.jit(_gate_up_operands)(_make_swiglu_inputs(problem, seed=2))
    offs = jnp.asarray(np.cumsum(_padded_sizes(problem)), jnp.int32)

    _c13, h, _h_col, sfh_row, _sfh_col = jax.jit(adapter_swiglu.mxfp8_fused_swiglu)(*operands, offs)
    h_forward, sfh_row_forward = jax.jit(adapter_swiglu.mxfp8_fused_swiglu_forward)(*operands, offs)

    assert bool(jnp.array_equal(_bits(h_forward), _bits(h))), "h differs from the upstream kernel's"
    assert bool(jnp.array_equal(_bits(sfh_row_forward), _bits(sfh_row))), "sfh_row differs from the upstream kernel's"
