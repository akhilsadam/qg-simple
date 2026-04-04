"""
Tensor-based algebraic equivalence rules for RPN token ID sequences.

Used to generate *positive views* for contrastive learning: applying a sound
rewrite yields a different token sequence with the same mathematical meaning.

Design
------
- **SimpleAlgebraicRule** — a match function (which sequences end with a
  rewriteable pattern) and a transform (rewrite the suffix, preserving prefix).
- **AlgebraicRuleSet** — named collection with utilities to sample and apply rules.
- Batched sequences may have different effective lengths; use ``pad_token_id``
  (typically ``__scalar__`` from :mod:`qg.solver.opt.operator.rpn.embeddings`)
  only for right-padding when aligning variable-length rewrite outputs.

Scalar literals: commutative swaps must eventually be paired with swapped
``scalar_vals`` in the training loop — see ``qg.solver.opt.operator.rpn.contrastive``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

# Match: (token_ids (B,L), vocab) -> (B,) bool
MatchFn = Callable[[torch.Tensor, Dict[str, int]], torch.Tensor]
# Transform: (token_ids (B',L), vocab) -> (B', L') long — only called on matched rows
TransformFn = Callable[[torch.Tensor, Dict[str, int]], torch.Tensor]


class RuleCategory(str, Enum):
    ARITHMETIC = "arithmetic"
    EXPONENT = "exponent"
    LOGARITHM = "logarithm"
    TRIGONOMETRIC = "trigonometric"
    CALCULUS = "calculus"
    VECTOR_CALCULUS = "vector_calculus"
    JACOBIAN = "jacobian"


@dataclass
class TransformResult:
    """Result of applying a rule to a batch of token sequences."""

    tokens: torch.Tensor
    """(B, L') token IDs, right-padded with ``pad_token_id`` if needed."""

    matched: torch.Tensor
    """(B,) bool — which batch rows were rewritten."""

    pad_token_id: int
    """Padding ID used in ``tokens`` for alignment (not a semantic token)."""


def _pad_rows_to_length(
    rows: Sequence[torch.Tensor],
    pad_id: int,
) -> torch.Tensor:
    """Stack 1D token rows into (B, max_len) with right padding."""
    if not rows:
        raise ValueError("rows must be non-empty")
    max_len = max(int(r.numel()) for r in rows)
    device = rows[0].device
    dtype = rows[0].dtype
    B = len(rows)
    out = torch.full((B, max_len), pad_id, dtype=dtype, device=device)
    for i, r in enumerate(rows):
        L = r.numel()
        out[i, :L] = r
    return out


class SimpleAlgebraicRule:
    """
    Concrete rule: ``match_fn`` selects batch rows; ``transform_fn`` rewrites
    those rows (suffix patterns only, in all current implementations).
    """

    def __init__(
        self,
        name: str,
        category: RuleCategory,
        description: str,
        pattern_length: int,
        output_length: int,
        match_fn: MatchFn,
        transform_fn: TransformFn,
    ) -> None:
        self.name = name
        self.category = category
        self.description = description
        self.pattern_length = pattern_length
        self.output_length = output_length
        self._match_fn = match_fn
        self._transform_fn = transform_fn

    def matches(self, token_ids: torch.Tensor, vocab: Dict[str, int]) -> torch.Tensor:
        return self._match_fn(token_ids, vocab)

    def apply(
        self,
        token_ids: torch.Tensor,
        vocab: Dict[str, int],
        pad_token_id: int,
    ) -> TransformResult:
        """
        Rewrite every batch row where ``matches`` is True; others are unchanged
        (then padded/truncated so all rows share the same width).
        """
        matched = self.matches(token_ids, vocab)
        if not matched.any():
            return TransformResult(token_ids.clone(), matched, pad_token_id)

        B, _ = token_ids.shape
        device = token_ids.device
        rows_out: List[torch.Tensor] = []
        for b in range(B):
            row = token_ids[b].contiguous()
            if matched[b]:
                sub = row.unsqueeze(0)
                new_sub = self._transform_fn(sub, vocab).squeeze(0)
                rows_out.append(new_sub)
            else:
                rows_out.append(row)
        stacked = _pad_rows_to_length(rows_out, pad_id=pad_token_id)
        return TransformResult(stacked, matched, pad_token_id)


class AlgebraicRuleSet:
    """Named list of rules sharing one vocabulary mapping (token string -> id)."""

    def __init__(
        self,
        name: str,
        vocab: Dict[str, int],
        pad_token_id: Optional[int] = None,
    ) -> None:
        self.name = name
        self.vocab = vocab
        self.pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else int(vocab.get("__scalar__", 0))
        )
        self.rules: List[SimpleAlgebraicRule] = []

    def add_rule(self, rule: SimpleAlgebraicRule) -> None:
        self.rules.append(rule)

    def applicable_rule_indices(self, token_ids: torch.Tensor) -> List[int]:
        """Indices of rules that match at least one row in the batch."""
        out: List[int] = []
        for i, rule in enumerate(self.rules):
            if rule.matches(token_ids, self.vocab).any():
                out.append(i)
        return out

    def apply_rule(self, rule_index: int, token_ids: torch.Tensor) -> TransformResult:
        """Apply one rule by index (raises if index invalid)."""
        rule = self.rules[rule_index]
        return rule.apply(token_ids, self.vocab, self.pad_token_id)

    def apply_random_rule(
        self,
        token_ids: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[TransformResult]:
        """
        Pick uniformly among rules that match at least one sequence; return
        ``None`` if no rule applies.
        """
        idxs = self.applicable_rule_indices(token_ids)
        if not idxs:
            return None
        if generator is None:
            j = int(torch.randint(len(idxs), (1,)).item())
        else:
            j = int(torch.randint(len(idxs), (1,), generator=generator).item())
        return self.apply_rule(idxs[j], token_ids)

    def random_positive_view(
        self,
        token_ids: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience for contrastive learning: ``(anchor, positive)`` token IDs
        of equal shape. If no rule applies, returns ``(anchor, anchor)``.
        """
        anchor = token_ids.clone()
        res = self.apply_random_rule(anchor, generator=generator)
        if res is None:
            return anchor, anchor
        return anchor, res.tokens


class CompositeAlgebraicRuleSet(AlgebraicRuleSet):
    """Flatten several :class:`AlgebraicRuleSet` instances into one."""

    def __init__(self, *rule_sets: AlgebraicRuleSet) -> None:
        if not rule_sets:
            raise ValueError("CompositeAlgebraicRuleSet needs at least one AlgebraicRuleSet")
        first = rule_sets[0]
        super().__init__("composite", first.vocab, first.pad_token_id)
        for rs in rule_sets:
            if rs.pad_token_id != first.pad_token_id:
                raise ValueError("All rule sets must use the same pad_token_id")
            self.rules.extend(rs.rules)
