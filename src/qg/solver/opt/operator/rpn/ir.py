"""
rpn_ir.py
---------
Internal representation (IR) types and token-level utilities for the RPN PDE compiler.

Everything here is pure data / pure functions — no torch, no derivative context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Public output type
# ---------------------------------------------------------------------------

@dataclass
class CompiledPDE:
    linear_operator: Optional["torch.Tensor"]   # noqa: F821  (torch imported by compiler)
    nonlinear_source: Optional[Callable]
    tokens: Sequence[str]


# ---------------------------------------------------------------------------
# Internal expression IR
# ---------------------------------------------------------------------------

@dataclass
class _TermMeta:
    """Tracks linearity bookkeeping for one multiplicative term."""
    state_factor_count: int
    linear_multiplier: Optional["torch.Tensor"]   # noqa: F821
    scalar_value: Optional[float]


@dataclass
class _Expr:
    """A scalar expression node in the compiled expression tree."""
    eval_fn: Callable
    depends_on_state: bool
    const_value: Optional[Union[float, "torch.Tensor"]]  # noqa: F821
    linear_multiplier: Optional["torch.Tensor"]           # noqa: F821
    terms: Sequence[_TermMeta]
    in_physical_domain: bool = False


@dataclass
class _VecExpr:
    """A 2-D vector expression node (x and y scalar components)."""
    x: _Expr
    y: _Expr


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------

def _is_vec(expr) -> bool:
    return isinstance(expr, _VecExpr)


def _normalize_token(token: str) -> str:
    stripped = token.strip()
    if stripped in {"∇", "nabla", "del"}:
        return "nabla"
    if stripped in {"Δ", "laplacian", "delta", "del2"}:
        return "lap"
    if stripped in {"Δinv", "invlaplacian", "lapinv"}:
        return "invlap"
    return stripped.lower()


def _parse_tokens(rpn: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(rpn, str):
        return [t for t in rpn.strip().split() if t]
    return list(rpn)
