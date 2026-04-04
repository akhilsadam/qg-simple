"""Quick check for composite rule application.

Run with the ``autoencoders`` venv (``Desktop/ml/autoencoders/.venv``), e.g.::

    cd /path/to/autoencoders && PYTHONPATH=packages/qg/src .venv/bin/python \\
        packages/qg/src/qg/solver/opt/operator/rpn/algebra/test_modular.py
"""

import torch

from qg.solver.opt.operator.rpn.algebra import create_composite_ruleset
from qg.solver.opt.operator.rpn.embeddings import TOKEN_TO_ID


def test_modular_system() -> None:
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__scalar__"])
    print(f"{len(rules.rules)} rules loaded")
    a_id = TOKEN_TO_ID["q"]
    b_id = TOKEN_TO_ID["psi"]
    plus_id = TOKEN_TO_ID["+"]
    token_ids = torch.tensor([[a_id, b_id, plus_id]], dtype=torch.long)
    res = rules.apply_random_rule(token_ids)
    if res is not None:
        print("commutative rewrite:", res.tokens, "matched:", res.matched)


if __name__ == "__main__":
    test_modular_system()
