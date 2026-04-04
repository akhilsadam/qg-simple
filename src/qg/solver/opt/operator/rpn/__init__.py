from .compiler import RPNCompiler

# Optional: contrastive / algebra training utilities (lazy names for importers)
from . import algebra  # noqa: F401
from . import contrastive  # noqa: F401


def compile_pde_rpn(rpn, derivative, pde_params):
    return RPNCompiler(derivative, pde_params).compile(rpn)
