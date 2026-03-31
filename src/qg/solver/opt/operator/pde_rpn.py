from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import torch

from qg.solver.opt.basis import to_physical, to_spectral


@dataclass
class CompiledPDE:
    linear_operator: Optional[torch.Tensor]
    nonlinear_source: Optional[Callable]
    tokens: Sequence[str]


@dataclass
class _TermMeta:
    state_factor_count: int
    linear_multiplier: Optional[torch.Tensor]
    scalar_value: Optional[float]


@dataclass
class _Expr:
    eval_fn: Callable
    depends_on_state: bool
    const_value: Optional[Union[float, torch.Tensor]]
    linear_multiplier: Optional[torch.Tensor]
    terms: Sequence[_TermMeta]


@dataclass
class _VecExpr:
    x: _Expr
    y: _Expr


def _parse_tokens(rpn: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(rpn, str):
        return [token for token in rpn.strip().split() if token]
    return list(rpn)


def _make_const_expr(value):
    scalar_value = float(value) if isinstance(value, (int, float)) else None
    return _Expr(
        eval_fn=lambda state, _value=value: _value,
        depends_on_state=False,
        const_value=value,
        linear_multiplier=None,
        terms=[_TermMeta(state_factor_count=0, linear_multiplier=None, scalar_value=scalar_value)],
    )


def _negate_terms(terms):
    negated = []
    for term in terms:
        linear_multiplier = None
        if term.linear_multiplier is not None:
            linear_multiplier = -term.linear_multiplier
        scalar_value = None
        if term.scalar_value is not None:
            scalar_value = -term.scalar_value
        negated.append(_TermMeta(
            state_factor_count=term.state_factor_count,
            linear_multiplier=linear_multiplier,
            scalar_value=scalar_value,
        ))
    return negated


def _scale_terms(terms, scalar):
    if not isinstance(scalar, (int, float)):
        return [_TermMeta(t.state_factor_count, None, None) for t in terms]

    scaled = []
    for term in terms:
        linear_multiplier = None
        if term.linear_multiplier is not None:
            linear_multiplier = float(scalar) * term.linear_multiplier
        scalar_value = None
        if term.scalar_value is not None:
            scalar_value = float(scalar) * term.scalar_value
        scaled.append(_TermMeta(
            state_factor_count=term.state_factor_count,
            linear_multiplier=linear_multiplier,
            scalar_value=scalar_value,
        ))
    return scaled


def _is_vec(expr):
    return isinstance(expr, _VecExpr)


def _binary_add_scalar(a: _Expr, b: _Expr):
    const_value = None
    if not a.depends_on_state and not b.depends_on_state:
        const_value = a.const_value + b.const_value

    linear_multiplier = None
    if a.linear_multiplier is not None and b.linear_multiplier is not None:
        linear_multiplier = a.linear_multiplier + b.linear_multiplier
    elif a.linear_multiplier is not None:
        linear_multiplier = a.linear_multiplier
    elif b.linear_multiplier is not None:
        linear_multiplier = b.linear_multiplier

    return _Expr(
        eval_fn=lambda state: a.eval_fn(state) + b.eval_fn(state),
        depends_on_state=a.depends_on_state or b.depends_on_state,
        const_value=const_value,
        linear_multiplier=linear_multiplier,
        terms=[*a.terms, *b.terms],
    )


def _binary_sub_scalar(a: _Expr, b: _Expr):
    const_value = None
    if not a.depends_on_state and not b.depends_on_state:
        const_value = a.const_value - b.const_value

    linear_multiplier = None
    if a.linear_multiplier is not None and b.linear_multiplier is not None:
        linear_multiplier = a.linear_multiplier - b.linear_multiplier
    elif a.linear_multiplier is not None:
        linear_multiplier = a.linear_multiplier
    elif b.linear_multiplier is not None:
        linear_multiplier = -b.linear_multiplier

    return _Expr(
        eval_fn=lambda state: a.eval_fn(state) - b.eval_fn(state),
        depends_on_state=a.depends_on_state or b.depends_on_state,
        const_value=const_value,
        linear_multiplier=linear_multiplier,
        terms=[*a.terms, *_negate_terms(b.terms)],
    )


def _binary_mul_scalar(a: _Expr, b: _Expr):
    const_value = None
    if not a.depends_on_state and not b.depends_on_state:
        const_value = a.const_value * b.const_value

    linear_multiplier = None
    if (not a.depends_on_state) and (b.linear_multiplier is not None):
        linear_multiplier = a.const_value * b.linear_multiplier
    elif (not b.depends_on_state) and (a.linear_multiplier is not None):
        linear_multiplier = b.const_value * a.linear_multiplier

    if (not a.depends_on_state) and (not b.depends_on_state):
        eval_fn = lambda state: a.eval_fn(state) * b.eval_fn(state)
        terms = [_TermMeta(
            state_factor_count=0,
            linear_multiplier=None,
            scalar_value=(float(a.const_value) * float(b.const_value)) if isinstance(a.const_value, (int, float)) and isinstance(b.const_value, (int, float)) else None,
        )]
    elif (not a.depends_on_state):
        eval_fn = lambda state: a.eval_fn(state) * b.eval_fn(state)
        terms = _scale_terms(b.terms, a.const_value)
    elif (not b.depends_on_state):
        eval_fn = lambda state: a.eval_fn(state) * b.eval_fn(state)
        terms = _scale_terms(a.terms, b.const_value)
    else:
        eval_fn = lambda state: to_spectral(to_physical(a.eval_fn(state)) * to_physical(b.eval_fn(state)))
        terms = []
        for ta in a.terms:
            for tb in b.terms:
                term_linear_multiplier = None
                if ta.scalar_value is not None and tb.linear_multiplier is not None and tb.state_factor_count <= 1:
                    term_linear_multiplier = ta.scalar_value * tb.linear_multiplier
                elif tb.scalar_value is not None and ta.linear_multiplier is not None and ta.state_factor_count <= 1:
                    term_linear_multiplier = tb.scalar_value * ta.linear_multiplier

                term_scalar = None
                if ta.scalar_value is not None and tb.scalar_value is not None:
                    term_scalar = ta.scalar_value * tb.scalar_value

                terms.append(_TermMeta(
                    state_factor_count=ta.state_factor_count + tb.state_factor_count,
                    linear_multiplier=term_linear_multiplier,
                    scalar_value=term_scalar,
                ))

    return _Expr(
        eval_fn=eval_fn,
        depends_on_state=a.depends_on_state or b.depends_on_state,
        const_value=const_value,
        linear_multiplier=linear_multiplier,
        terms=terms,
    )


def _binary_add(a, b):
    if _is_vec(a) and _is_vec(b):
        return _VecExpr(
            x=_binary_add_scalar(a.x, b.x),
            y=_binary_add_scalar(a.y, b.y),
        )
    if (not _is_vec(a)) and (not _is_vec(b)):
        return _binary_add_scalar(a, b)
    raise ValueError("RPN type error: '+' requires scalar+scalar or vector+vector")


def _binary_sub(a, b):
    if _is_vec(a) and _is_vec(b):
        return _VecExpr(
            x=_binary_sub_scalar(a.x, b.x),
            y=_binary_sub_scalar(a.y, b.y),
        )
    if (not _is_vec(a)) and (not _is_vec(b)):
        return _binary_sub_scalar(a, b)
    raise ValueError("RPN type error: '-' requires scalar-scalar or vector-vector")


def _binary_mul(a, b):
    if _is_vec(a) and _is_vec(b):
        raise ValueError("RPN type error: vector*vector is undefined; use 'dot' or 'inner'")
    if _is_vec(a):
        return _VecExpr(
            x=_binary_mul_scalar(a.x, b),
            y=_binary_mul_scalar(a.y, b),
        )
    if _is_vec(b):
        return _VecExpr(
            x=_binary_mul_scalar(a, b.x),
            y=_binary_mul_scalar(a, b.y),
        )
    return _binary_mul_scalar(a, b)


def _apply_linear_unary_scalar(expr: _Expr, op_multiplier, op_name: str):
    if (not expr.depends_on_state) and (expr.const_value is not None):
        raise ValueError(f"Cannot apply '{op_name}' to a state-independent scalar expression")

    expr_linear_multiplier = None
    if expr.linear_multiplier is not None:
        expr_linear_multiplier = op_multiplier * expr.linear_multiplier

    terms = []
    for term in expr.terms:
        linear_multiplier = None
        if term.linear_multiplier is not None:
            linear_multiplier = op_multiplier * term.linear_multiplier
        terms.append(_TermMeta(
            state_factor_count=term.state_factor_count,
            linear_multiplier=linear_multiplier,
            scalar_value=None,
        ))

    return _Expr(
        eval_fn=lambda state: op_multiplier * expr.eval_fn(state),
        depends_on_state=expr.depends_on_state,
        const_value=None,
        linear_multiplier=expr_linear_multiplier,
        terms=terms,
    )


def _apply_linear_unary(expr, op_multiplier, op_name: str):
    if _is_vec(expr):
        return _VecExpr(
            x=_apply_linear_unary_scalar(expr.x, op_multiplier, op_name),
            y=_apply_linear_unary_scalar(expr.y, op_multiplier, op_name),
        )
    return _apply_linear_unary_scalar(expr, op_multiplier, op_name)


def _jacobian(a_h, b_h, derivative):
    a_x = to_physical(derivative.dx * a_h)
    a_y = to_physical(derivative.dy * a_h)
    b_x = to_physical(derivative.dx * b_h)
    b_y = to_physical(derivative.dy * b_h)
    return to_spectral(a_x * b_y - a_y * b_x)


def _normalize_token(token: str) -> str:
    stripped = token.strip()
    if stripped in {"∇", "nabla", "del"}:
        return "nabla"
    if stripped in {"Δ", "laplacian", "delta", "del2"}:
        return "lap"
    if stripped in {"Δinv", "invlaplacian", "lapinv"}:
        return "invlap"
    return stripped.lower()


def compile_pde_rpn(rpn, derivative, pde_params) -> CompiledPDE:
    tokens = _parse_tokens(rpn)
    if not tokens:
        raise ValueError("pde.rpn is empty")

    scalar_constants = {}
    if hasattr(pde_params, "items"):
        iterable = pde_params.items()
    else:
        iterable = vars(pde_params).items()

    for key, value in iterable:
        if isinstance(value, (int, float)):
            scalar_constants[str(key)] = float(value)

    one = torch.ones_like(derivative.laplacian)

    variable_expr = {
        "q": _Expr(
            eval_fn=lambda state: state.qh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=one,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=one, scalar_value=None)],
        ),
        "omega": _Expr(
            eval_fn=lambda state: state.qh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=one,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=one, scalar_value=None)],
        ),
        "psi": _Expr(
            eval_fn=lambda state: state.ph,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=derivative.inv_laplacian, scalar_value=None)],
        ),
        "ph": _Expr(
            eval_fn=lambda state: state.ph,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=derivative.inv_laplacian, scalar_value=None)],
        ),
        "u": _Expr(
            eval_fn=lambda state: state.uh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=-derivative.dy * derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=-derivative.dy * derivative.inv_laplacian, scalar_value=None)],
        ),
        "uh": _Expr(
            eval_fn=lambda state: state.uh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=-derivative.dy * derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=-derivative.dy * derivative.inv_laplacian, scalar_value=None)],
        ),
        "v": _Expr(
            eval_fn=lambda state: state.vh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=derivative.dx * derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=derivative.dx * derivative.inv_laplacian, scalar_value=None)],
        ),
        "vh": _Expr(
            eval_fn=lambda state: state.vh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=derivative.dx * derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=derivative.dx * derivative.inv_laplacian, scalar_value=None)],
        ),
    }

    unary_linear_ops = {
        "dx": derivative.dx,
        "dy": derivative.dy,
        "lap": derivative.laplacian,
        "invlap": derivative.inv_laplacian,
        "hodge": one,
        "star": one,
    }

    stack = []

    for token in tokens:
        lower = _normalize_token(token)

        if lower in variable_expr:
            stack.append(variable_expr[lower])
            continue

        if lower in scalar_constants:
            stack.append(_make_const_expr(scalar_constants[lower]))
            continue

        try:
            numeric = float(token)
            stack.append(_make_const_expr(numeric))
            continue
        except ValueError:
            pass

        if lower == "neg":
            if len(stack) < 1:
                raise ValueError("RPN parse error: 'neg' needs one operand")
            a = stack.pop()
            if _is_vec(a):
                stack.append(_VecExpr(
                    x=_binary_mul_scalar(_make_const_expr(-1.0), a.x),
                    y=_binary_mul_scalar(_make_const_expr(-1.0), a.y),
                ))
                continue
            stack.append(_Expr(
                eval_fn=lambda state: -a.eval_fn(state),
                depends_on_state=a.depends_on_state,
                const_value=(-a.const_value if a.const_value is not None else None),
                linear_multiplier=(-a.linear_multiplier if a.linear_multiplier is not None else None),
                terms=_negate_terms(a.terms),
            ))
            continue

        if lower == "dealias":
            if len(stack) < 1:
                raise ValueError("RPN parse error: 'dealias' needs one operand")
            a = stack.pop()
            if _is_vec(a):
                stack.append(_VecExpr(
                    x=_Expr(
                        eval_fn=lambda state: derivative.dealias(a.x.eval_fn(state).clone()),
                        depends_on_state=a.x.depends_on_state,
                        const_value=None,
                        linear_multiplier=(derivative.dealias(a.x.linear_multiplier.clone()) if a.x.linear_multiplier is not None else None),
                        terms=[
                            _TermMeta(
                                state_factor_count=t.state_factor_count,
                                linear_multiplier=(derivative.dealias(t.linear_multiplier.clone()) if t.linear_multiplier is not None else None),
                                scalar_value=None,
                            )
                            for t in a.x.terms
                        ],
                    ),
                    y=_Expr(
                        eval_fn=lambda state: derivative.dealias(a.y.eval_fn(state).clone()),
                        depends_on_state=a.y.depends_on_state,
                        const_value=None,
                        linear_multiplier=(derivative.dealias(a.y.linear_multiplier.clone()) if a.y.linear_multiplier is not None else None),
                        terms=[
                            _TermMeta(
                                state_factor_count=t.state_factor_count,
                                linear_multiplier=(derivative.dealias(t.linear_multiplier.clone()) if t.linear_multiplier is not None else None),
                                scalar_value=None,
                            )
                            for t in a.y.terms
                        ],
                    ),
                ))
                continue
            stack.append(_Expr(
                eval_fn=lambda state: derivative.dealias(a.eval_fn(state).clone()),
                depends_on_state=a.depends_on_state,
                const_value=None,
                linear_multiplier=(derivative.dealias(a.linear_multiplier.clone()) if a.linear_multiplier is not None else None),
                terms=[
                    _TermMeta(
                        state_factor_count=t.state_factor_count,
                        linear_multiplier=(derivative.dealias(t.linear_multiplier.clone()) if t.linear_multiplier is not None else None),
                        scalar_value=None,
                    )
                    for t in a.terms
                ],
            ))
            continue

        if lower in {"grad", "nabla", "del"}:
            if len(stack) < 1:
                raise ValueError(f"RPN parse error: '{token}' needs one operand")
            a = stack.pop()
            if _is_vec(a):
                raise ValueError(f"RPN type error: '{token}' expects a scalar")
            stack.append(_VecExpr(
                x=_apply_linear_unary_scalar(a, derivative.dx, token),
                y=_apply_linear_unary_scalar(a, derivative.dy, token),
            ))
            continue

        if lower == "div":
            if len(stack) < 1:
                raise ValueError("RPN parse error: 'div' needs one operand")
            a = stack.pop()
            if not _is_vec(a):
                raise ValueError("RPN type error: 'div' expects a vector expression")
            div_expr = _binary_add_scalar(
                _apply_linear_unary_scalar(a.x, derivative.dx, "div"),
                _apply_linear_unary_scalar(a.y, derivative.dy, "div"),
            )
            stack.append(div_expr)
            continue

        if lower == "curl":
            if len(stack) < 1:
                raise ValueError("RPN parse error: 'curl' needs one operand")
            a = stack.pop()
            if not _is_vec(a):
                raise ValueError("RPN type error: 'curl' expects a vector expression")
            curl_expr = _binary_sub_scalar(
                _apply_linear_unary_scalar(a.y, derivative.dx, "curl"),
                _apply_linear_unary_scalar(a.x, derivative.dy, "curl"),
            )
            stack.append(curl_expr)
            continue

        if lower in unary_linear_ops:
            if len(stack) < 1:
                raise ValueError(f"RPN parse error: '{token}' needs one operand")
            a = stack.pop()
            stack.append(_apply_linear_unary(a, unary_linear_ops[lower], token))
            continue

        if lower in {"+", "-", "*", "mul"}:
            if len(stack) < 2:
                raise ValueError(f"RPN parse error: '{token}' needs two operands")
            b = stack.pop()
            a = stack.pop()
            if lower == "+":
                stack.append(_binary_add(a, b))
            elif lower == "-":
                stack.append(_binary_sub(a, b))
            else:
                stack.append(_binary_mul(a, b))
            continue

        if lower in {"dot", "inner"}:
            if len(stack) < 2:
                raise ValueError(f"RPN parse error: '{token}' needs two operands")
            b = stack.pop()
            a = stack.pop()
            if (not _is_vec(a)) or (not _is_vec(b)):
                raise ValueError(f"RPN type error: '{token}' expects two vectors")
            stack.append(
                _binary_add_scalar(
                    _binary_mul_scalar(a.x, b.x),
                    _binary_mul_scalar(a.y, b.y),
                )
            )
            continue

        if lower in {"jacobian", "j"}:
            if len(stack) < 2:
                raise ValueError(f"RPN parse error: '{token}' needs two operands")
            b = stack.pop()
            a = stack.pop()
            if _is_vec(a) or _is_vec(b):
                raise ValueError(f"RPN type error: '{token}' expects scalar operands")
            stack.append(_Expr(
                eval_fn=lambda state: _jacobian(a.eval_fn(state), b.eval_fn(state), derivative),
                depends_on_state=True,
                const_value=None,
                linear_multiplier=None,
                terms=[
                    _TermMeta(
                        state_factor_count=ta.state_factor_count + tb.state_factor_count,
                        linear_multiplier=None,
                        scalar_value=None,
                    )
                    for ta in a.terms for tb in b.terms
                ],
            ))
            continue

        raise ValueError(f"Unknown RPN token: '{token}'")

    if len(stack) != 1:
        raise ValueError("RPN parse error: expression did not reduce to a single result")

    expr = stack[0]
    if _is_vec(expr):
        raise ValueError("RPN type error: final expression must be scalar")

    linear_operator = None
    for term in expr.terms:
        if term.state_factor_count <= 1 and term.linear_multiplier is not None:
            if linear_operator is None:
                linear_operator = term.linear_multiplier
            else:
                linear_operator = linear_operator + term.linear_multiplier

    nonlinear_source = None
    if expr.depends_on_state or expr.const_value is not None:
        def _rhs(state):
            rhs = expr.eval_fn(state)
            if not torch.is_tensor(rhs):
                rhs = float(rhs) * torch.ones_like(state.qh)
            if linear_operator is not None:
                rhs = rhs - linear_operator * state.qh
            return derivative.dealias(rhs)

        nonlinear_source = _rhs

    return CompiledPDE(
        linear_operator=linear_operator,
        nonlinear_source=nonlinear_source,
        tokens=tokens,
    )
