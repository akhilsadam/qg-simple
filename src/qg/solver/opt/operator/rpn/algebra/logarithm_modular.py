"""Log-type identities — vocabulary has no ``log`` in stable mode; reserved for future."""

from typing import Dict, Optional

from .modular_rules import AlgebraicRuleSet


def create_logarithm_rules(vocab: Dict[str, int], pad_token_id: Optional[int] = None) -> AlgebraicRuleSet:
    return AlgebraicRuleSet("logarithm", vocab, pad_token_id=pad_token_id)
