# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 options of the Grug MoE routed experts."""

from enum import StrEnum


class ExpertMxfp8(StrEnum):
    """Which routed-expert GEMMs run with MXFP8 operands on block-scaled SM100 kernels.

    MXFP8 stores E4M3 values with one E8M0 scale per 32 elements. Only the `ragged_all_to_all`
    backend runs it, on SM100 GPUs, for SiLU experts and bfloat16 tokens.
    """

    NONE = "none"
    """No GEMM: every operand stays in the token dtype, bfloat16 for the hero."""
    ALL = "all"
    """The forward and data-gradient GEMMs and the weight gradients, quantized along the token axis per expert."""
