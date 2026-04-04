"""
Vector calculus identities (``grad``/``div``/``curl``/``dot``) — placeholders.

Future: e.g. ``div grad`` vs ``lap`` where equivalent under compiler semantics.
"""

from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet


def create_vector_calculus_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    return AlgebraicRuleSet("vector_calculus", vocab, pad_token_id=pad_token_id)
