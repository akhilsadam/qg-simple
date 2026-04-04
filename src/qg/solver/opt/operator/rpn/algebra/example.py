"""Minimal smoke test: composite rules + Jacobian antisymmetry on token IDs.

Use the ``autoencoders`` virtualenv, e.g. ``Desktop/ml/autoencoders/.venv``::

    cd /path/to/autoencoders && PYTHONPATH=packages/qg/src .venv/bin/python \\
        packages/qg/src/qg/solver/opt/operator/rpn/algebra/example.py
"""

import torch

from qg.solver.opt.operator.rpn.embeddings import TOKEN_TO_ID
from qg.solver.opt.operator.rpn.algebra import create_composite_ruleset


def main() -> None:
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__scalar__"])
    # J(psi, q) in RPN: psi q jacobian
    t = torch.tensor(
        [[TOKEN_TO_ID["psi"], TOKEN_TO_ID["q"], TOKEN_TO_ID["jacobian"]]],
        dtype=torch.long,
    )
    res = rules.apply_random_rule(t)
    assert res is not None
    print("original:", t)
    print("rewritten:", res.tokens)
    print("matched:", res.matched)


if __name__ == "__main__":
    main()
