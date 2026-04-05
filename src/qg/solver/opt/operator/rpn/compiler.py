"""
rpn_compiler.py
---------------
Compiles an RPN token sequence into a CompiledPDE that carries both a
spectral linear operator and a nonlinear source callable.

Imports all IR types from rpn_ir; no embedding logic lives here.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch

from qg.solver.opt.basis import to_physical, to_spectral
from .ir import (
    CompiledPDE,
    _Expr,
    _TermMeta,
    _VecExpr,
    _is_vec,
    _normalize_token,
    _parse_tokens,
)


# ---------------------------------------------------------------------------
# ExprBuilder — construction and transformation of _Expr / _VecExpr nodes
# ---------------------------------------------------------------------------

class ExprBuilder:
    """Static helpers that build or transform _Expr / _VecExpr nodes."""

    # ------------------------------------------------------------------
    # Leaf constructors
    # ------------------------------------------------------------------

    @staticmethod
    def const(value) -> _Expr:
        scalar_value = float(value) if isinstance(value, (int, float)) else None
        return _Expr(
            eval_fn=lambda state, _v=value: float(_v) if isinstance(_v, (int, float)) else _v,
            depends_on_state=False,
            const_value=float(value) if isinstance(value, (int, float)) else value,
            linear_multiplier=None,
            terms=[_TermMeta(state_factor_count=0, linear_multiplier=None, scalar_value=scalar_value)],
            in_physical_domain=False,
        )

    @staticmethod
    def const_field(field: torch.Tensor, in_physical_domain: bool = False) -> _Expr:
        """Wraps a precomputed constant tensor field (not a scalar).

        Unlike ``const``, ``const_value`` is left as ``None`` so that
        downstream ops (add, mul, apply_linear …) never try to treat the
        tensor as a plain Python scalar.
        """
        return _Expr(
            eval_fn=lambda state, _f=field: _f,
            depends_on_state=False,
            const_value=None,
            linear_multiplier=None,
            terms=[_TermMeta(state_factor_count=0, linear_multiplier=None, scalar_value=None)],
            in_physical_domain=in_physical_domain,
        )

    # ------------------------------------------------------------------
    # Term-list helpers
    # ------------------------------------------------------------------

    @staticmethod
    def negate_terms(terms: Sequence[_TermMeta]) -> list:
        return [
            _TermMeta(
                state_factor_count=t.state_factor_count,
                linear_multiplier=-t.linear_multiplier if t.linear_multiplier is not None else None,
                scalar_value=-t.scalar_value if t.scalar_value is not None else None,
            )
            for t in terms
        ]

    @staticmethod
    def scale_terms(terms: Sequence[_TermMeta], scalar) -> list:
        if not isinstance(scalar, (int, float)):
            return [_TermMeta(t.state_factor_count, None, None) for t in terms]
        s = float(scalar)
        return [
            _TermMeta(
                state_factor_count=t.state_factor_count,
                linear_multiplier=s * t.linear_multiplier if t.linear_multiplier is not None else None,
                scalar_value=s * t.scalar_value if t.scalar_value is not None else None,
            )
            for t in terms
        ]

    # ------------------------------------------------------------------
    # Internal utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_physical_safe(value, in_physical: bool):
        if isinstance(value, torch.Tensor):
            return value if in_physical else to_physical(value)
        return value

    # ------------------------------------------------------------------
    # Binary scalar operations
    # ------------------------------------------------------------------

    @staticmethod
    def add(a: _Expr, b: _Expr) -> _Expr:
        in_phys = a.in_physical_domain or b.in_physical_domain

        if in_phys:
            def eval_fn(state):
                av = ExprBuilder._to_physical_safe(a.eval_fn(state), a.in_physical_domain)
                bv = ExprBuilder._to_physical_safe(b.eval_fn(state), b.in_physical_domain)
                if isinstance(av, torch.Tensor) and av.dim() == 2:
                    av = av.unsqueeze(0)
                if isinstance(bv, torch.Tensor) and bv.dim() == 2:
                    bv = bv.unsqueeze(0)
                return av + bv
        else:
            eval_fn = lambda state: a.eval_fn(state) + b.eval_fn(state)

        depends = a.depends_on_state or b.depends_on_state
        const_value = (
            a.const_value + b.const_value
            if (not depends) and a.const_value is not None and b.const_value is not None
            else None
        )

        lin = None
        if a.linear_multiplier is not None and b.linear_multiplier is not None:
            lin = a.linear_multiplier + b.linear_multiplier
        elif a.linear_multiplier is not None and not b.depends_on_state:
            lin = a.linear_multiplier
        elif b.linear_multiplier is not None and not a.depends_on_state:
            lin = b.linear_multiplier

        terms = ExprBuilder._cross_terms_add(a.terms, b.terms, sign=+1)

        return _Expr(eval_fn=eval_fn, depends_on_state=depends, const_value=const_value,
                     linear_multiplier=lin, terms=terms, in_physical_domain=in_phys)

    @staticmethod
    def sub(a: _Expr, b: _Expr) -> _Expr:
        in_phys = a.in_physical_domain or b.in_physical_domain

        if in_phys:
            def eval_fn(state):
                av = ExprBuilder._to_physical_safe(a.eval_fn(state), a.in_physical_domain)
                bv = ExprBuilder._to_physical_safe(b.eval_fn(state), b.in_physical_domain)
                if isinstance(av, torch.Tensor) and av.dim() == 2:
                    av = av.unsqueeze(0)
                if isinstance(bv, torch.Tensor) and bv.dim() == 2:
                    bv = bv.unsqueeze(0)
                return torch.add(av, bv, alpha=-1) # avoids add_kernel on complex<float>
        else:
            eval_fn = lambda state: a.eval_fn(state) - b.eval_fn(state)

        depends = a.depends_on_state or b.depends_on_state
        const_value = (
            a.const_value - b.const_value
            if (not depends) and a.const_value is not None and b.const_value is not None
            else None
        )

        lin = None
        if a.linear_multiplier is not None and b.linear_multiplier is not None:
            lin = a.linear_multiplier - b.linear_multiplier
        elif a.linear_multiplier is not None and not b.depends_on_state:
            lin = a.linear_multiplier
        elif b.linear_multiplier is not None and not a.depends_on_state:
            lin = -b.linear_multiplier

        terms = ExprBuilder._cross_terms_add(a.terms, b.terms, sign=-1)

        return _Expr(eval_fn=eval_fn, depends_on_state=depends, const_value=const_value,
                     linear_multiplier=lin, terms=terms, in_physical_domain=in_phys)

    @staticmethod
    def mul(a: _Expr, b: _Expr) -> _Expr:
        in_phys = a.in_physical_domain or b.in_physical_domain

        if in_phys:
            def eval_fn(state):
                av = ExprBuilder._to_physical_safe(a.eval_fn(state), a.in_physical_domain)
                bv = ExprBuilder._to_physical_safe(b.eval_fn(state), b.in_physical_domain)
                return av * bv
        else:
            eval_fn = lambda state: a.eval_fn(state) * b.eval_fn(state)

        depends = a.depends_on_state or b.depends_on_state
        const_value = (
            a.const_value * b.const_value
            if (not depends) and a.const_value is not None and b.const_value is not None
            else None
        )

        lin = None
        if a.linear_multiplier is not None and not b.depends_on_state and b.const_value is not None:
            lin = a.linear_multiplier * b.const_value
        elif b.linear_multiplier is not None and not a.depends_on_state and a.const_value is not None:
            lin = b.linear_multiplier * a.const_value

        terms = ExprBuilder._cross_terms_mul(a.terms, b.terms)

        return _Expr(eval_fn=eval_fn, depends_on_state=depends, const_value=const_value,
                     linear_multiplier=lin, terms=terms, in_physical_domain=in_phys)

    # ------------------------------------------------------------------
    # Unary operations
    # ------------------------------------------------------------------

    @staticmethod
    def apply_linear(expr: _Expr, op_multiplier, op_name: str) -> _Expr:

        if not expr.depends_on_state:
            raw = expr.eval_fn(None)
            if not isinstance(raw, torch.Tensor):
                raw = torch.tensor(float(raw))
            field_h = to_spectral(raw) if expr.in_physical_domain else raw
            result_h = op_multiplier * field_h
            return ExprBuilder.const_field(result_h, in_physical_domain=False)

        lin = op_multiplier * expr.linear_multiplier if expr.linear_multiplier is not None else None
        terms = [
            _TermMeta(
                state_factor_count=t.state_factor_count,
                linear_multiplier=op_multiplier * t.linear_multiplier if t.linear_multiplier is not None else None,
                scalar_value=None,
            )
            for t in expr.terms
        ]

        return _Expr(
            eval_fn=lambda state: op_multiplier * (
                to_spectral(expr.eval_fn(state)) if expr.in_physical_domain else expr.eval_fn(state)
            ),
            depends_on_state=expr.depends_on_state,
            const_value=None,
            linear_multiplier=lin,
            terms=terms,
            in_physical_domain=False,
        )

    @staticmethod
    def apply_nonlinear(expr: _Expr, func, func_name: str) -> _Expr:
        def _eval(state):
            val = expr.eval_fn(state) if expr.in_physical_domain else (
                to_physical(expr.eval_fn(state))
                if isinstance(expr.eval_fn(state), torch.Tensor)
                else expr.eval_fn(state)
            )
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(float(val))
            return func(val)

        return _Expr(
            eval_fn=_eval,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=None,
            terms=[_TermMeta(t.state_factor_count, None, None) for t in expr.terms],
            in_physical_domain=True,
        )

    # ------------------------------------------------------------------
    # Vector-aware dispatch wrappers
    # ------------------------------------------------------------------

    @staticmethod
    def vec_add(a, b):
        if _is_vec(a) and _is_vec(b):
            return _VecExpr(x=ExprBuilder.add(a.x, b.x), y=ExprBuilder.add(a.y, b.y))
        if not _is_vec(a) and not _is_vec(b):
            return ExprBuilder.add(a, b)
        raise ValueError("RPN type error: '+' requires scalar+scalar or vector+vector")

    @staticmethod
    def vec_sub(a, b):
        if _is_vec(a) and _is_vec(b):
            return _VecExpr(x=ExprBuilder.sub(a.x, b.x), y=ExprBuilder.sub(a.y, b.y))
        if not _is_vec(a) and not _is_vec(b):
            return ExprBuilder.sub(a, b)
        raise ValueError("RPN type error: '-' requires scalar-scalar or vector-vector")

    @staticmethod
    def vec_mul(a, b):
        if _is_vec(a) and _is_vec(b):
            raise ValueError("RPN type error: vector*vector is undefined; use 'dot' or 'inner'")
        if _is_vec(a):
            return _VecExpr(x=ExprBuilder.mul(a.x, b), y=ExprBuilder.mul(a.y, b))
        if _is_vec(b):
            return _VecExpr(x=ExprBuilder.mul(a, b.x), y=ExprBuilder.mul(a, b.y))
        return ExprBuilder.mul(a, b)

    @staticmethod
    def vec_apply_linear(expr, op_multiplier, op_name: str):
        if _is_vec(expr):
            return _VecExpr(
                x=ExprBuilder.apply_linear(expr.x, op_multiplier, op_name),
                y=ExprBuilder.apply_linear(expr.y, op_multiplier, op_name),
            )
        return ExprBuilder.apply_linear(expr, op_multiplier, op_name)

    @staticmethod
    def vec_apply_nonlinear(expr, func, func_name: str):
        if _is_vec(expr):
            return _VecExpr(
                x=ExprBuilder.apply_nonlinear(expr.x, func, func_name),
                y=ExprBuilder.apply_nonlinear(expr.y, func, func_name),
            )
        return ExprBuilder.apply_nonlinear(expr, func, func_name)

    # ------------------------------------------------------------------
    # Private cross-product term helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_terms_add(a_terms, b_terms, sign: int) -> list:
        terms = []
        for ta in a_terms:
            for tb in b_terms:
                lin = None
                if ta.linear_multiplier is not None and tb.linear_multiplier is not None:
                    lin = ta.linear_multiplier * tb.linear_multiplier
                elif ta.linear_multiplier is not None and tb.scalar_value is not None:
                    lin = ta.linear_multiplier * tb.scalar_value
                elif tb.linear_multiplier is not None and ta.scalar_value is not None:
                    lin = sign * tb.linear_multiplier * ta.scalar_value
                terms.append(_TermMeta(
                    state_factor_count=ta.state_factor_count + tb.state_factor_count,
                    linear_multiplier=lin,
                    scalar_value=None,
                ))
        return terms

    @staticmethod
    def _cross_terms_mul(a_terms, b_terms) -> list:
        terms = []
        for ta in a_terms:
            for tb in b_terms:
                lin = None
                if ta.linear_multiplier is not None and tb.scalar_value is not None:
                    lin = ta.linear_multiplier * tb.scalar_value
                elif tb.linear_multiplier is not None and ta.scalar_value is not None:
                    lin = tb.linear_multiplier * ta.scalar_value
                terms.append(_TermMeta(
                    state_factor_count=ta.state_factor_count + tb.state_factor_count,
                    linear_multiplier=lin,
                    scalar_value=None,
                ))
        return terms


# ---------------------------------------------------------------------------
# OperatorRegistry — operator tables, resolved against a derivative context
# ---------------------------------------------------------------------------

class OperatorRegistry:
    """Holds operator look-up tables, fully resolved for a given derivative context.

    Unstable operators (tan, acos, asin, atan, log, log10, sign, ceil,
    floor, round) are intentionally excluded from NONLINEAR_UNARY so that
    the compiler and the embedding vocabulary stay in sync.
    """

    # Stable nonlinear unary operators only.
    NONLINEAR_UNARY: dict = {
        "sqrt":   torch.sqrt,
        "cos":    torch.cos,
        "sin":    torch.sin,
        "cosh":   torch.cosh,
        "sinh":   torch.sinh,
        "tanh":   torch.tanh,
        "exp":    torch.exp,
        "square": lambda x: x ** 2,
        "cube":   lambda x: x ** 3,
        # "abs":    torch.abs,
    }

    def __init__(self, derivative):
        self.derivative = d = derivative

        self.linear_unary: dict = {
            "dx":     d.dx,
            "dy":     d.dy,
            "lap":    d.laplacian,
            "invlap": d.inv_laplacian,
        }

        self.binary: dict = {
            "+":   ExprBuilder.vec_add,
            "-":   ExprBuilder.vec_sub,
            "*":   ExprBuilder.vec_mul,
            "mul": ExprBuilder.vec_mul,
        }

    def is_linear_unary(self, token: str) -> bool:
        return token in self.linear_unary

    def is_nonlinear_unary(self, token: str) -> bool:
        return token in self.NONLINEAR_UNARY

    def is_binary(self, token: str) -> bool:
        return token in self.binary

    def get_linear_unary(self, token: str):
        return self.linear_unary[token]

    def get_nonlinear_unary(self, token: str):
        return self.NONLINEAR_UNARY[token]

    def apply_binary(self, token: str, a, b):
        return self.binary[token](a, b)


# ---------------------------------------------------------------------------
# Variable table factory
# ---------------------------------------------------------------------------

def _build_variable_table(derivative) -> dict:
    d = derivative
    one = torch.ones_like(d.laplacian)

    def _state_expr(eval_fn, lin):
        return _Expr(
            eval_fn=eval_fn,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=lin,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=lin, scalar_value=None)],
            in_physical_domain=False,
        )

    return {
        "q":     _state_expr(lambda s: s.qh, one),
        "omega": _state_expr(lambda s: s.qh, one),
        "psi":   _state_expr(lambda s: s.ph,  d.inv_laplacian),
        "ph":    _state_expr(lambda s: s.ph,  d.inv_laplacian),
        "u":     _state_expr(lambda s: s.uh, -1 * d.dy * d.inv_laplacian),
        "uh":    _state_expr(lambda s: s.uh, -1 * d.dy * d.inv_laplacian),
        "v":     _state_expr(lambda s: s.vh,  d.dx * d.inv_laplacian),
        "vh":    _state_expr(lambda s: s.vh,  d.dx * d.inv_laplacian),
        "x": _Expr(
            eval_fn=lambda s: torch.meshgrid(d.grid.y, d.grid.x, indexing="ij")[1].unsqueeze(0),
            depends_on_state=False, const_value=None, linear_multiplier=None,
            terms=[_TermMeta(0, None, None)], in_physical_domain=True,
        ),
        "y": _Expr(
            eval_fn=lambda s: torch.meshgrid(d.grid.y, d.grid.x, indexing="ij")[0].unsqueeze(0),
            depends_on_state=False, const_value=None, linear_multiplier=None,
            terms=[_TermMeta(0, None, None)], in_physical_domain=True,
        ),
    }


# ---------------------------------------------------------------------------
# RPNCompiler
# ---------------------------------------------------------------------------

class RPNCompiler:
    """Compiles an RPN token sequence into a CompiledPDE."""

    def __init__(self, derivative, pde_params):
        self.derivative = derivative
        self.ops = OperatorRegistry(derivative)
        self.variables = _build_variable_table(derivative)
        self.scalar_constants = self._extract_scalar_constants(pde_params)

    def compile(self, rpn: Union[str, Sequence[str]]) -> CompiledPDE:
        tokens = _parse_tokens(rpn)
        if not tokens:
            raise ValueError("pde.rpn is empty")

        stack = []
        for token in tokens:
            self._process_token(token, stack)

        if len(stack) != 1:
            raise ValueError("RPN parse error: expression did not reduce to a single result")
        expr = stack[0]
        if _is_vec(expr):
            raise ValueError("RPN type error: final expression must be scalar")

        return self._assemble(expr, tokens)

    def _process_token(self, token: str, stack: list) -> None:
        lower = _normalize_token(token)

        if lower in self.variables:
            stack.append(self.variables[lower])
            return

        if lower in self.scalar_constants:
            stack.append(ExprBuilder.const(self.scalar_constants[lower]))
            return

        try:
            stack.append(ExprBuilder.const(float(token)))
            return
        except ValueError:
            pass

        special = {
            "neg":      self._handle_neg,
            "dealias":  self._handle_dealias,
            "grad":     self._handle_grad,
            "nabla":    self._handle_grad,
            "div":      self._handle_div,
            "curl":     self._handle_curl,
            "dot":      self._handle_dot,
            "inner":    self._handle_dot,
            "jacobian": self._handle_jacobian,
            "j":        self._handle_jacobian,
        }
        if lower in special:
            special[lower](lower, stack)
            return

        if self.ops.is_linear_unary(lower):
            a = self._pop(stack, lower, n=1)
            stack.append(ExprBuilder.vec_apply_linear(a, self.ops.get_linear_unary(lower), lower))
            return

        if self.ops.is_nonlinear_unary(lower):
            a = self._pop(stack, lower, n=1)
            func = self.ops.get_nonlinear_unary(lower)
            if _is_vec(a):
                if not a.x.depends_on_state and not a.y.depends_on_state:
                    # Constant vector field — precompute each component.
                    rx = func(ExprBuilder._to_physical_safe(a.x.eval_fn(None), a.x.in_physical_domain))
                    ry = func(ExprBuilder._to_physical_safe(a.y.eval_fn(None), a.y.in_physical_domain))
                    stack.append(_VecExpr(
                        x=ExprBuilder.const_field(rx, in_physical_domain=True),
                        y=ExprBuilder.const_field(ry, in_physical_domain=True),
                    ))
                    return
            else:
                if not a.depends_on_state:
                    # Constant scalar or field — precompute eagerly.
                    raw = ExprBuilder._to_physical_safe(a.eval_fn(None), a.in_physical_domain)
                    if not isinstance(raw, torch.Tensor):
                        raw = torch.tensor(float(raw))
                    result = func(raw)
                    if isinstance(result, torch.Tensor) and result.dim() == 0:
                        # True scalar result (e.g. sin(3)) — wrap as plain const.
                        stack.append(ExprBuilder.const(result.item()))
                    else:
                        # Tensor field result (e.g. sin(x)) — wrap as const_field.
                        stack.append(ExprBuilder.const_field(result, in_physical_domain=True))
                    return
            stack.append(ExprBuilder.vec_apply_nonlinear(a, func, lower))
            return

        if self.ops.is_binary(lower):
            b, a = self._pop(stack, lower, n=2)
            stack.append(self.ops.apply_binary(lower, a, b))
            return

        raise ValueError(f"Unknown RPN token: '{token}'")

    def _handle_neg(self, token: str, stack: list) -> None:
        a = self._pop(stack, token, n=1)
        if _is_vec(a):
            neg_one = ExprBuilder.const(-1.0)
            stack.append(_VecExpr(x=ExprBuilder.mul(neg_one, a.x), y=ExprBuilder.mul(neg_one, a.y)))
            return

        def _neg_eval(state, _a=a):
            val = to_spectral(_a.eval_fn(state)) if _a.in_physical_domain else _a.eval_fn(state)
            if isinstance(val, torch.Tensor):
                return torch.mul(val, -1)  # avoids neg_kernel on complex<float>
            return -val

        stack.append(_Expr(
            eval_fn=_neg_eval,
            depends_on_state=a.depends_on_state,
            const_value=-a.const_value if a.const_value is not None else None,
            linear_multiplier=-a.linear_multiplier if a.linear_multiplier is not None else None,
            terms=ExprBuilder.negate_terms(a.terms),
        ))

    def _handle_dealias(self, token: str, stack: list) -> None:
        d = self.derivative
        a = self._pop(stack, token, n=1)

        def _dealias_scalar(s: _Expr) -> _Expr:
            return _Expr(
                eval_fn=lambda state, _s=s: d.dealias(
                    (to_spectral(_s.eval_fn(state)) if _s.in_physical_domain else _s.eval_fn(state)).clone()
                ),
                depends_on_state=s.depends_on_state,
                const_value=None,
                linear_multiplier=(d.dealias(s.linear_multiplier.clone()) if s.linear_multiplier is not None else None),
                terms=[
                    _TermMeta(
                        state_factor_count=t.state_factor_count,
                        linear_multiplier=(d.dealias(t.linear_multiplier.clone()) if t.linear_multiplier is not None else None),
                        scalar_value=None,
                    )
                    for t in s.terms
                ],
                in_physical_domain=False,
            )

        if _is_vec(a):
            stack.append(_VecExpr(x=_dealias_scalar(a.x), y=_dealias_scalar(a.y)))
        else:
            stack.append(_dealias_scalar(a))

    def _handle_grad(self, token: str, stack: list) -> None:
        a = self._pop(stack, token, n=1)
        if _is_vec(a):
            raise ValueError(f"RPN type error: '{token}' expects a scalar")
        d = self.derivative
        stack.append(_VecExpr(
            x=ExprBuilder.apply_linear(a, d.dx, token),
            y=ExprBuilder.apply_linear(a, d.dy, token),
        ))

    def _handle_div(self, token: str, stack: list) -> None:
        a = self._pop(stack, token, n=1)
        if not _is_vec(a):
            raise ValueError("RPN type error: 'div' expects a vector expression")
        d = self.derivative
        stack.append(ExprBuilder.add(
            ExprBuilder.apply_linear(a.x, d.dx, "div"),
            ExprBuilder.apply_linear(a.y, d.dy, "div"),
        ))

    def _handle_curl(self, token: str, stack: list) -> None:
        a = self._pop(stack, token, n=1)
        if not _is_vec(a):
            raise ValueError("RPN type error: 'curl' expects a vector expression")
        d = self.derivative
        stack.append(ExprBuilder.sub(
            ExprBuilder.apply_linear(a.y, d.dx, "curl"),
            ExprBuilder.apply_linear(a.x, d.dy, "curl"),
        ))

    def _handle_dot(self, token: str, stack: list) -> None:
        b, a = self._pop(stack, token, n=2)
        if not _is_vec(a) or not _is_vec(b):
            raise ValueError(f"RPN type error: '{token}' expects two vectors")
        stack.append(_Expr(
            eval_fn=lambda state: (
                to_physical(a.x.eval_fn(state)) * to_physical(b.x.eval_fn(state))
                + to_physical(a.y.eval_fn(state)) * to_physical(b.y.eval_fn(state))
            ),
            depends_on_state=True,
            const_value=None,
            linear_multiplier=None,
            terms=[_TermMeta(0, None, None)],
            in_physical_domain=True,
        ))

    def _handle_jacobian(self, token: str, stack: list) -> None:
        b, a = self._pop(stack, token, n=2)
        if _is_vec(a) or _is_vec(b):
            raise ValueError(f"RPN type error: '{token}' expects scalar operands")
        d = self.derivative
        stack.append(_Expr(
            eval_fn=lambda state, _a=a, _b=b: self._jacobian(
                _a.eval_fn(state), _b.eval_fn(state),
                a_physical=_a.in_physical_domain,
                b_physical=_b.in_physical_domain,
            ),
            depends_on_state=a.depends_on_state or b.depends_on_state,
            const_value=None,
            linear_multiplier=None,
            terms=[
                _TermMeta(ta.state_factor_count + tb.state_factor_count, None, None)
                for ta in a.terms for tb in b.terms
            ],
            in_physical_domain=True,
        ))

    def _assemble(self, expr: _Expr, tokens: Sequence[str]) -> CompiledPDE:
        linear_operator = None
        if not expr.in_physical_domain:
            for term in expr.terms:
                if term.state_factor_count <= 1 and term.linear_multiplier is not None:
                    linear_operator = (
                        term.linear_multiplier if linear_operator is None
                        else linear_operator + term.linear_multiplier
                    )

        nonlinear_source = None
        if expr.depends_on_state or expr.const_value is not None or expr.in_physical_domain:
            d = self.derivative

            def _rhs(state, _expr=expr, _lin=linear_operator):
                rhs = _expr.eval_fn(state)
                if _expr.in_physical_domain:
                    rhs = to_spectral(rhs)
                if not torch.is_tensor(rhs):
                    rhs = float(rhs) * torch.ones_like(state.qh)
                if _lin is not None:
                    rhs = rhs - _lin * state.qh
                return d.dealias(rhs)

            nonlinear_source = _rhs

        return CompiledPDE(
            linear_operator=linear_operator,
            nonlinear_source=nonlinear_source,
            tokens=tokens,
        )

    def _jacobian(self, a_h, b_h, a_physical: bool = False, b_physical: bool = False) -> torch.Tensor:
        d = self.derivative

        # Promote plain scalars to a spectral field of the right shape
        if not isinstance(a_h, torch.Tensor):
            a_h = float(a_h) * torch.ones_like(d.laplacian).unsqueeze(0)
            a_physical = False
        if not isinstance(b_h, torch.Tensor):
            b_h = float(b_h) * torch.ones_like(d.laplacian).unsqueeze(0)
            b_physical = False

        if a_physical:
            a_h = to_spectral(a_h)
        if b_physical:
            b_h = to_spectral(b_h)

        # Ensure batch dim
        if a_h.dim() == 2:
            a_h = a_h.unsqueeze(0)
        if b_h.dim() == 2:
            b_h = b_h.unsqueeze(0)

        a_x = to_physical(d.dx * a_h)
        a_y = to_physical(d.dy * a_h)
        b_x = to_physical(d.dx * b_h)
        b_y = to_physical(d.dy * b_h)
        return a_x * b_y - a_y * b_x

    @staticmethod
    def _pop(stack: list, token: str, n: int):
        if len(stack) < n:
            raise ValueError(f"RPN parse error: '{token}' needs {n} operand{'s' if n > 1 else ''}")
        if n == 1:
            return stack.pop()
        return stack.pop(), stack.pop()

    @staticmethod
    def _extract_scalar_constants(pde_params) -> dict:
        iterable = pde_params.items() if hasattr(pde_params, "items") else vars(pde_params).items()
        return {str(k): float(v) for k, v in iterable if isinstance(v, (int, float))}