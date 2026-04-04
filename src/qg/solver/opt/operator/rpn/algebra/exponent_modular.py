"""Exponent / ``exp`` / power stack identities (placeholders)."""

from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet


def create_exponent_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    return AlgebraicRuleSet("exponent", vocab, pad_token_id=pad_token_id)
