"""
RPN smoke tests (algebra rules + contrastive forward).

Always run with the venv at Desktop/ml/autoencoders/.venv (see repo root
``smoke_rpn.sh``), for example::

    cd /path/to/autoencoders
    export PYTHONPATH=packages/qg/src
    .venv/bin/python packages/qg/src/qg/solver/opt/operator/rpn/_smoke_test.py
"""

from qg.solver.opt.operator.rpn.algebra import create_composite_ruleset
from qg.solver.opt.operator.rpn.contrastive import ContrastiveRPNTrainer
from qg.solver.opt.operator.rpn.embeddings import TOKEN_TO_ID

import torch


def main() -> None:
    rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__scalar__"])
    t = torch.tensor(
        [[TOKEN_TO_ID["psi"], TOKEN_TO_ID["q"], TOKEN_TO_ID["jacobian"]]],
        dtype=torch.long,
    )
    r = rules.apply_random_rule(t)
    assert r is not None
    print("algebra OK", tuple(r.tokens.shape))

    m = ContrastiveRPNTrainer()
    z1, z2, loss = m.forward_from_rpn_strings(["psi q jacobian", "q psi +"])
    print("contrastive OK", tuple(z1.shape), float(loss))
    print("smoke tests passed")


if __name__ == "__main__":
    main()
