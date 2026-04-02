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
    in_physical_domain: bool = False  # True if this expression must be evaluated in physical space


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
        eval_fn=lambda state, _value=value: float(_value) if isinstance(_value, (int, float)) else _value,
        depends_on_state=False,
        const_value=float(value) if isinstance(value, (int, float)) else value,
        linear_multiplier=None,
        terms=[_TermMeta(state_factor_count=0, linear_multiplier=None, scalar_value=scalar_value)],
        in_physical_domain=False,
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


def _to_physical_safe(value, in_physical):
    if isinstance(value, torch.Tensor):
        return value if in_physical else to_physical(value)
    return value


# Operator registry tables, centralized for easy modularization and random PDE graph generation.
unary_linear_ops = {
    "dx": None,  # filled in compile_pde_rpn from derivative context
    "dy": None,
    "lap": None,
    "invlap": None,
    "hodge": None,
    "star": None,
}

nonlinear_unary_ops = {
    "sqrt": torch.sqrt,
    "cos": torch.cos,
    "sin": torch.sin,
    "tan": torch.tan,
    "acos": torch.acos,
    "asin": torch.asin,
    "atan": torch.atan,
    "cosh": torch.cosh,
    "sinh": torch.sinh,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "log": torch.log,
    "log10": lambda x: torch.log10(x),
    "square": lambda x: x**2,
    "cube": lambda x: x**3,
    "abs": torch.abs,
    "sign": torch.sign,
    "ceil": torch.ceil,
    "floor": torch.floor,
    "round": torch.round,
}







def _binary_add(a, b):
    if _is_vec(a) and _is_vec(b):
        return _VecExpr(
            x=_binary_add_scalar(a.x, b.x, "+"),
            y=_binary_add_scalar(a.y, b.y, "+"),
        )
    if (not _is_vec(a)) and (not _is_vec(b)):
        return _binary_add_scalar(a, b, "+")
    raise ValueError("RPN type error: '+' requires scalar+scalar or vector+vector")


def _binary_sub(a, b):
    if _is_vec(a) and _is_vec(b):
        return _VecExpr(
            x=_binary_sub_scalar(a.x, b.x, "-"),
            y=_binary_sub_scalar(a.y, b.y, "-"),
        )
    if (not _is_vec(a)) and (not _is_vec(b)):
        return _binary_sub_scalar(a, b, "-")
    raise ValueError("RPN type error: '-' requires scalar-scalar or vector-vector")


def _binary_mul(a, b):
    if _is_vec(a) and _is_vec(b):
        raise ValueError("RPN type error: vector*vector is undefined; use 'dot' or 'inner'")
    if _is_vec(a):
        return _VecExpr(
            x=_binary_mul_scalar(a.x, b, "*"),
            y=_binary_mul_scalar(a.y, b, "*"),
        )
    if _is_vec(b):
        return _VecExpr(
            x=_binary_mul_scalar(a, b.x, "*"),
            y=_binary_mul_scalar(a, b.y, "*"),
        )
    return _binary_mul_scalar(a, b, "*")


def _binary_add_scalar(a: _Expr, b: _Expr, op: str) -> _Expr:
    in_physical_domain = a.in_physical_domain or b.in_physical_domain
    if in_physical_domain:
        def eval_fn(state):
            a_val = _to_physical_safe(a.eval_fn(state), a.in_physical_domain)
            b_val = _to_physical_safe(b.eval_fn(state), b.in_physical_domain)
            if isinstance(a_val, torch.Tensor) and a_val.dim() == 2:
                a_val = a_val.unsqueeze(0)
            if isinstance(b_val, torch.Tensor) and b_val.dim() == 2:
                b_val = b_val.unsqueeze(0)
            return a_val + b_val
    else:
        eval_fn = lambda state: a.eval_fn(state) + b.eval_fn(state)
    
    depends_on_state = a.depends_on_state or b.depends_on_state
    const_value = None
    if (not depends_on_state) and (a.const_value is not None) and (b.const_value is not None):
        const_value = a.const_value + b.const_value
    
    linear_multiplier = None
    if a.linear_multiplier is not None and b.linear_multiplier is not None:
        linear_multiplier = a.linear_multiplier + b.linear_multiplier
    elif a.linear_multiplier is not None and (not b.depends_on_state):
        linear_multiplier = a.linear_multiplier
    elif b.linear_multiplier is not None and (not a.depends_on_state):
        linear_multiplier = b.linear_multiplier
    
    terms = []
    for ta in a.terms:
        for tb in b.terms:
            term_linear_multiplier = None
            if ta.linear_multiplier is not None and tb.linear_multiplier is not None:
                term_linear_multiplier = ta.linear_multiplier * tb.linear_multiplier
            elif ta.linear_multiplier is not None and tb.scalar_value is not None:
                term_linear_multiplier = ta.linear_multiplier * tb.scalar_value
            elif tb.linear_multiplier is not None and ta.scalar_value is not None:
                term_linear_multiplier = tb.linear_multiplier * ta.scalar_value
            terms.append(_TermMeta(
                state_factor_count=ta.state_factor_count + tb.state_factor_count,
                linear_multiplier=term_linear_multiplier,
                scalar_value=None,
            ))
    
    return _Expr(
        eval_fn=eval_fn,
        depends_on_state=depends_on_state,
        const_value=const_value,
        linear_multiplier=linear_multiplier,
        terms=terms,
        in_physical_domain=in_physical_domain,
    )


def _binary_sub_scalar(a: _Expr, b: _Expr, op: str) -> _Expr:
    in_physical_domain = a.in_physical_domain or b.in_physical_domain
    if in_physical_domain:
        def eval_fn(state):
            a_val = _to_physical_safe(a.eval_fn(state), a.in_physical_domain)
            b_val = _to_physical_safe(b.eval_fn(state), b.in_physical_domain)
            if isinstance(a_val, torch.Tensor) and a_val.dim() == 2:
                a_val = a_val.unsqueeze(0)
            if isinstance(b_val, torch.Tensor) and b_val.dim() == 2:
                b_val = b_val.unsqueeze(0)
            if op == "-":
                return a_val - b_val
            elif op == "+":
                return a_val + b_val
            elif op == "*":
                return a_val * b_val
            elif op == "jacobian":
                return _jacobian(a_val, b_val, derivative)
            else:
                raise ValueError(f"Unsupported op in physical domain: {op}")
    else:
        eval_fn = lambda state: _binary_op_funcs[op](a.eval_fn(state), b.eval_fn(state))
    
    depends_on_state = a.depends_on_state or b.depends_on_state
    const_value = None
    if (not depends_on_state) and (a.const_value is not None) and (b.const_value is not None):
        const_value = a.const_value - b.const_value
    
    linear_multiplier = None
    if a.linear_multiplier is not None and b.linear_multiplier is not None:
        linear_multiplier = a.linear_multiplier - b.linear_multiplier
    elif a.linear_multiplier is not None and (not b.depends_on_state):
        linear_multiplier = a.linear_multiplier
    elif b.linear_multiplier is not None and (not a.depends_on_state):
        linear_multiplier = -b.linear_multiplier
    
    terms = []
    for ta in a.terms:
        for tb in b.terms:
            term_linear_multiplier = None
            if ta.linear_multiplier is not None and tb.linear_multiplier is not None:
                term_linear_multiplier = ta.linear_multiplier * tb.linear_multiplier
            elif ta.linear_multiplier is not None and tb.scalar_value is not None:
                term_linear_multiplier = ta.linear_multiplier * tb.scalar_value
            elif tb.linear_multiplier is not None and ta.scalar_value is not None:
                term_linear_multiplier = -tb.linear_multiplier * ta.scalar_value
            terms.append(_TermMeta(
                state_factor_count=ta.state_factor_count + tb.state_factor_count,
                linear_multiplier=term_linear_multiplier,
                scalar_value=None,
            ))
    
    return _Expr(
        eval_fn=eval_fn,
        depends_on_state=depends_on_state,
        const_value=const_value,
        linear_multiplier=linear_multiplier,
        terms=terms,
        in_physical_domain=in_physical_domain,
    )


def _binary_mul_scalar(a: _Expr, b: _Expr, op: str) -> _Expr:
    in_physical_domain = a.in_physical_domain or b.in_physical_domain
    if in_physical_domain:
        def eval_fn(state):
            a_val = _to_physical_safe(a.eval_fn(state), a.in_physical_domain)
            b_val = _to_physical_safe(b.eval_fn(state), b.in_physical_domain)
            return a_val * b_val
    else:
        eval_fn = lambda state: a.eval_fn(state) * b.eval_fn(state)
    
    depends_on_state = a.depends_on_state or b.depends_on_state
    const_value = None
    if (not depends_on_state) and (a.const_value is not None) and (b.const_value is not None):
        const_value = a.const_value * b.const_value
    
    linear_multiplier = None
    if a.linear_multiplier is not None and (not b.depends_on_state) and (b.const_value is not None):
        linear_multiplier = a.linear_multiplier * b.const_value
    elif b.linear_multiplier is not None and (not a.depends_on_state) and (a.const_value is not None):
        linear_multiplier = b.linear_multiplier * a.const_value
    
    terms = []
    for ta in a.terms:
        for tb in b.terms:
            term_linear_multiplier = None
            if ta.linear_multiplier is not None and tb.scalar_value is not None:
                term_linear_multiplier = ta.linear_multiplier * tb.scalar_value
            elif tb.linear_multiplier is not None and ta.scalar_value is not None:
                term_linear_multiplier = tb.linear_multiplier * ta.scalar_value
            terms.append(_TermMeta(
                state_factor_count=ta.state_factor_count + tb.state_factor_count,
                linear_multiplier=term_linear_multiplier,
                scalar_value=None,
            ))
    
    return _Expr(
        eval_fn=eval_fn,
        depends_on_state=depends_on_state,
        const_value=const_value,
        linear_multiplier=linear_multiplier,
        terms=terms,
        in_physical_domain=in_physical_domain,
    )

binary_ops = {
    "+": _binary_add,
    "-": _binary_sub,
    "*": _binary_mul,
    "mul": _binary_mul,
    "dot": None,  # handled explicitly in compile_pde_rpn
    "inner": None,
    "jacobian": None,
}


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
        eval_fn=lambda state: op_multiplier * (to_spectral(expr.eval_fn(state)) if expr.in_physical_domain else expr.eval_fn(state)),
        depends_on_state=expr.depends_on_state,
        const_value=None,
        linear_multiplier=expr_linear_multiplier,
        terms=terms,
        in_physical_domain=False,  # Linear ops result in spectral
    )


def _apply_linear_unary(expr, op_multiplier, op_name):
    if _is_vec(expr):
        return _VecExpr(
            x=_apply_linear_unary_scalar(expr.x, op_multiplier, op_name),
            y=_apply_linear_unary_scalar(expr.y, op_multiplier, op_name),
        )
    return _apply_linear_unary_scalar(expr, op_multiplier, op_name)


def _apply_nonlinear_unary(expr, func, func_name):
    if _is_vec(expr):
        return _VecExpr(
            x=_apply_nonlinear_unary_scalar(expr.x, func, func_name),
            y=_apply_nonlinear_unary_scalar(expr.y, func, func_name),
        )
    return _apply_nonlinear_unary_scalar(expr, func, func_name)


def _apply_nonlinear_unary_scalar(expr: _Expr, func, func_name):
    if not expr.depends_on_state:
        # For constant expressions, compute once in physical space
        raw_value = expr.eval_fn(None)
        physical_value = raw_value if expr.in_physical_domain else (to_physical(raw_value) if isinstance(raw_value, torch.Tensor) else raw_value)
        result = func(physical_value)
        return _Expr(
            eval_fn=lambda state: result,
            depends_on_state=False,
            const_value=None,
            linear_multiplier=None,
            terms=[_TermMeta(state_factor_count=0, linear_multiplier=None, scalar_value=None)],
            in_physical_domain=True,  # Nonlinear, physical
        )
    else:
        # For state-dependent, apply in physical space per evaluation
        return _Expr(
            eval_fn=lambda state: func(
                expr.eval_fn(state) if expr.in_physical_domain else (
                    to_physical(expr.eval_fn(state)) if isinstance(expr.eval_fn(state), torch.Tensor) else expr.eval_fn(state)
                )
            ),
            depends_on_state=True,
            const_value=None,
            linear_multiplier=None,
            terms=[_TermMeta(state_factor_count=t.state_factor_count, linear_multiplier=None, scalar_value=None) for t in expr.terms],
            in_physical_domain=True,  # Nonlinear, physical
        )


def _jacobian(a_h, b_h, derivative):
    a_x = to_physical(derivative.dx * a_h)
    a_y = to_physical(derivative.dy * a_h)
    b_x = to_physical(derivative.dx * b_h)
    b_y = to_physical(derivative.dy * b_h)
    return a_x * b_y - a_y * b_x


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
            in_physical_domain=False,
        ),
        "omega": _Expr(
            eval_fn=lambda state: state.qh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=one,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=one, scalar_value=None)],
            in_physical_domain=False,
        ),
        "psi": _Expr(
            eval_fn=lambda state: state.ph,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=derivative.inv_laplacian, scalar_value=None)],
            in_physical_domain=False,
        ),
        "ph": _Expr(
            eval_fn=lambda state: state.ph,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=derivative.inv_laplacian, scalar_value=None)],
            in_physical_domain=False,
        ),
        "u": _Expr(
            eval_fn=lambda state: state.uh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=-derivative.dy * derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=-derivative.dy * derivative.inv_laplacian, scalar_value=None)],
            in_physical_domain=False,
        ),
        "uh": _Expr(
            eval_fn=lambda state: state.uh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=-derivative.dy * derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=-derivative.dy * derivative.inv_laplacian, scalar_value=None)],
            in_physical_domain=False,
        ),
        "v": _Expr(
            eval_fn=lambda state: state.vh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=derivative.dx * derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=derivative.dx * derivative.inv_laplacian, scalar_value=None)],
            in_physical_domain=False,
        ),
        "vh": _Expr(
            eval_fn=lambda state: state.vh,
            depends_on_state=True,
            const_value=None,
            linear_multiplier=derivative.dx * derivative.inv_laplacian,
            terms=[_TermMeta(state_factor_count=1, linear_multiplier=derivative.dx * derivative.inv_laplacian, scalar_value=None)],
            in_physical_domain=False,
        ),
        "x": _Expr(
            eval_fn=lambda state: torch.meshgrid(derivative.grid.y, derivative.grid.x, indexing='ij')[1].unsqueeze(0),
            depends_on_state=False,
            const_value=None,
            linear_multiplier=None,
            terms=[_TermMeta(state_factor_count=0, linear_multiplier=None, scalar_value=None)],
            in_physical_domain=True,
        ),
        "y": _Expr(
            eval_fn=lambda state: torch.meshgrid(derivative.grid.y, derivative.grid.x, indexing='ij')[0].unsqueeze(0),
            depends_on_state=False,
            const_value=None,
            linear_multiplier=None,
            terms=[_TermMeta(state_factor_count=0, linear_multiplier=None, scalar_value=None)],
            in_physical_domain=True,
        ),
    }

    unary_linear_ops = {
        "dx": derivative.dx,
        "dy": derivative.dy,
        "lap": derivative.laplacian,
        "invlap": derivative.inv_laplacian,
        # "hodge": one,
        # "star": one,
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
                        eval_fn=lambda state, a=a: -(to_spectral(a.eval_fn(state)) if a.in_physical_domain else a.eval_fn(state)),
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
                        eval_fn=lambda state, a=a: derivative.dealias((to_spectral(a.x.eval_fn(state)) if a.x.in_physical_domain else a.x.eval_fn(state)).clone()),
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
                        in_physical_domain=False,
                    ),
                    y=_Expr(
                        eval_fn=lambda state, a=a: derivative.dealias((to_spectral(a.y.eval_fn(state)) if a.y.in_physical_domain else a.y.eval_fn(state)).clone()),
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
                        in_physical_domain=False,
                    ),
                ))
                continue
            stack.append(_Expr(
                eval_fn=lambda state, a=a: derivative.dealias((to_spectral(a.eval_fn(state)) if a.in_physical_domain else a.eval_fn(state)).clone()),
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
                in_physical_domain=False,
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

        if lower in nonlinear_unary_ops:
            if len(stack) < 1:
                raise ValueError(f"RPN parse error: '{token}' needs one operand")
            a = stack.pop()
            stack.append(_apply_nonlinear_unary(a, nonlinear_unary_ops[lower], token))
            continue

        if lower in {"+", "-", "*", "mul"}:
            if len(stack) < 2:
                raise ValueError(f"RPN parse error: '{token}' needs two operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(binary_ops[lower](a, b))
            continue

        if lower in {"dot", "inner"}:
            if len(stack) < 2:
                raise ValueError(f"RPN parse error: '{token}' needs two operands")
            b = stack.pop()
            a = stack.pop()
            if (not _is_vec(a)) or (not _is_vec(b)):
                raise ValueError(f"RPN type error: '{token}' expects two vectors")
            dot_expr = _Expr(
                eval_fn=lambda state: to_physical(a.x.eval_fn(state)) * to_physical(b.x.eval_fn(state)) + to_physical(a.y.eval_fn(state)) * to_physical(b.y.eval_fn(state)),
                depends_on_state=True,
                const_value=None,
                linear_multiplier=None,
                terms=[_TermMeta(state_factor_count=0, linear_multiplier=None, scalar_value=None)],  # Simplified
                in_physical_domain=True,  # Dot product is nonlinear, physical
            )
            stack.append(dot_expr)
            continue

        if lower in {"jacobian", "j"}:
            if len(stack) < 2:
                raise ValueError(f"RPN parse error: '{token}' needs two operands")
            b = stack.pop()
            a = stack.pop()
            if _is_vec(a) or _is_vec(b):
                raise ValueError(f"RPN type error: '{token}' expects scalar operands")
            stack.append(_Expr(
                eval_fn=lambda state, a=a, b=b: _jacobian(a.eval_fn(state), b.eval_fn(state), derivative),
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
                in_physical_domain=True,  # Nonlinear
            ))
            continue

        raise ValueError(f"Unknown RPN token: '{token}'")

    if len(stack) != 1:
        raise ValueError("RPN parse error: expression did not reduce to a single result")

    expr = stack[0]
    if _is_vec(expr):
        raise ValueError("RPN type error: final expression must be scalar")

    linear_operator = None
    if not expr.in_physical_domain:
        for term in expr.terms:
            if term.state_factor_count <= 1 and term.linear_multiplier is not None:
                if linear_operator is None:
                    linear_operator = term.linear_multiplier
                else:
                    linear_operator = linear_operator + term.linear_multiplier

    nonlinear_source = None
    if expr.depends_on_state or expr.const_value is not None or expr.in_physical_domain:
        def _rhs(state):
            rhs = expr.eval_fn(state)
            if expr.in_physical_domain:
                rhs = to_spectral(rhs)
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
