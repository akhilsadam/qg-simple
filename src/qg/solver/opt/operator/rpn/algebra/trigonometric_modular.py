"""Trig identities on ``sin``/``cos``/… RPN suffixes (placeholders)."""

from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet


def create_trigonometric_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    return AlgebraicRuleSet("trigonometric", vocab, pad_token_id=pad_token_id)
