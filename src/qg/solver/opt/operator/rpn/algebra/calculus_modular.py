"""
Calculus / spectral-operator equivalences on RPN (placeholders).

Future: linearity of ``dx``/``dy``/``lap`` on sums; composition identities
that are valid in this compiler's semantics.
"""

from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet


def create_calculus_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    return AlgebraicRuleSet("calculus", vocab, pad_token_id=pad_token_id)
