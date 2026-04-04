"""
Contrastive learning for RPN token embeddings.

**Anchor** = tokenized expression. **Positive** = one sound algebraic rewrite
(from :mod:`algebra`). **Negatives** = other sequences in the batch (InfoNCE).

This module keeps a single encoder path: :class:`RPNTokenEmbedder` → pool →
projection. Pooling uses a mask so padded ``__scalar__`` slots do not contribute.

TODO (scalar-aware rewrites)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Commutative rules swap operand tokens; if either operand was a numeric literal,
the parallel ``scalar_vals`` / ``scalar_mask`` tensors from
:func:`embeddings.batch_tokenize_rpn` must be swapped as well. The hook is
``ContrastiveRPNTrainer._align_scalars_after_rewrite`` (placeholder).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebra import AlgebraicRuleSet, create_composite_ruleset
from .embeddings import (
    TOKEN_TO_ID,
    RPNTokenEmbedder,
    batch_tokenize_rpn,
)


def masked_mean_pool(
    seq: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Parameters
    ----------
    seq : (B, L, E)
    mask : (B, L) bool — True = real token (not padding)
    """
    w = mask.unsqueeze(-1).float()
    num = (seq * w).sum(dim=1)
    den = w.sum(dim=1).clamp_min(eps)
    return num / den


def infonce_logits(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Symmetric InfoNCE logits (B, B): diagonal = positive pairs.
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    return (z_a @ z_b.T) / temperature


def infonce_symmetric_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy on both directions; positives are batch-aligned."""
    logits_ab = infonce_logits(z_anchor, z_positive, temperature)
    logits_ba = infonce_logits(z_positive, z_anchor, temperature)
    B = z_anchor.size(0)
    targets = torch.arange(B, device=z_anchor.device)
    loss_a = F.cross_entropy(logits_ab, targets)
    loss_b = F.cross_entropy(logits_ba, targets)
    return 0.5 * (loss_a + loss_b)


class RPNEncoderHead(nn.Module):
    """Pool sequence embeddings and project to contrastive space."""

    def __init__(self, embed_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(pooled)


def _pad_2d_right(x: torch.Tensor, new_len: int, pad_value) -> torch.Tensor:
    """Pad or truncate (B, L) along L to ``new_len`` (``pad_value`` may be float or int)."""
    B, L = x.shape
    if L == new_len:
        return x
    if L > new_len:
        return x[:, :new_len]
    pad = torch.full((B, new_len - L), pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _pad_2d_bool(x: torch.Tensor, new_len: int) -> torch.Tensor:
    B, L = x.shape
    if L == new_len:
        return x
    if L > new_len:
        return x[:, :new_len]
    pad = torch.zeros((B, new_len - L), dtype=torch.bool, device=x.device)
    return torch.cat([x, pad], dim=-1)


def _pad_1d_right(row: torch.Tensor, new_len: int, pad_id: int) -> torch.Tensor:
    L = row.numel()
    if L >= new_len:
        return row[:new_len]
    out = row.new_full((new_len,), pad_id)
    out[:L] = row
    return out


class ContrastiveRPNTrainer(nn.Module):
    """
    End-to-end: RPN strings → pooled embedding → projection → InfoNCE vs
    algebra-augmented positives.

    Uses ``rules.random_positive_view`` on *token IDs*; for expressions that
    admit no rewrite, anchor == positive (loss still valid but weaker).
    """

    def __init__(
        self,
        embed_dim: int = 32,
        proj_dim: int = 64,
        scalar_fourier: int = 16,
        rules: Optional[AlgebraicRuleSet] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.pad_token_id = (
            int(pad_token_id) if pad_token_id is not None else TOKEN_TO_ID["__scalar__"]
        )
        self.temperature = temperature
        self.embedder = RPNTokenEmbedder(embed_dim=embed_dim, scalar_fourier=scalar_fourier)
        self.head = RPNEncoderHead(embed_dim, proj_dim)
        self.rules = rules or create_composite_ruleset(TOKEN_TO_ID, pad_token_id=self.pad_token_id)

    def encode_token_batch(
        self,
        token_ids: torch.Tensor,
        scalar_vals: torch.Tensor,
        scalar_mask: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pad_mask : (B, L) bool — True for non-padding positions (inverse of padding column).
        """
        seq = self.embedder(token_ids, scalar_vals, scalar_mask)
        pooled = masked_mean_pool(seq, pad_mask)
        return self.head(pooled)

    def _align_scalars_after_rewrite(
        self,
        anchor_ids: torch.Tensor,
        positive_ids: torch.Tensor,
        anchor_vals: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Placeholder: when rules swap operands that include ``__scalar__`` slots,
        mirror the swap in ``scalar_vals`` / ``scalar_mask``. Currently returns
        anchor tensors unchanged (safe for symbol-only rewrites).
        """
        return anchor_vals, anchor_mask

    def forward_from_token_tensors(
        self,
        anchor_ids: torch.Tensor,
        anchor_vals: torch.Tensor,
        anchor_scalars: torch.Tensor,
        anchor_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build positives by applying a random rewrite row-wise.

        Sequences are right-padded to a common width (rewrites may lengthen the
        suffix, e.g. Jacobian antisymmetry).
        """
        B, L0 = anchor_ids.shape
        pos_rows: List[torch.Tensor] = []

        for b in range(B):
            row = anchor_ids[b : b + 1, :]
            _, pos = self.rules.random_positive_view(row)
            pos_rows.append(pos.squeeze(0))

        max_len = max(L0, max(int(r.numel()) for r in pos_rows))

        anchor_ids_e = _pad_2d_right(anchor_ids, max_len, self.pad_token_id)
        anchor_vals_e = _pad_2d_right(anchor_vals, max_len, 0.0)
        anchor_scalars_e = _pad_2d_bool(anchor_scalars, max_len)
        anchor_pad_e = _pad_2d_bool(anchor_pad_mask, max_len)

        positive_ids = torch.stack(
            [_pad_1d_right(r, max_len, self.pad_token_id) for r in pos_rows],
            dim=0,
        )

        pos_vals, pos_mask = self._align_scalars_after_rewrite(
            anchor_ids_e, positive_ids, anchor_vals_e, anchor_scalars_e
        )

        pos_pad = self._pad_mask_from_ids(positive_ids)

        z_a = self.encode_token_batch(anchor_ids_e, anchor_vals_e, anchor_scalars_e, anchor_pad_e)
        z_p = self.encode_token_batch(positive_ids, pos_vals, pos_mask, pos_pad)

        loss = infonce_symmetric_loss(z_a, z_p, self.temperature)
        return z_a, z_p, loss

    def _pad_mask_from_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """True where not right-padding with pad_token_id (heuristic)."""
        B, L = token_ids.shape
        mask = torch.zeros(B, L, dtype=torch.bool, device=token_ids.device)
        for b in range(B):
            row = token_ids[b]
            non_pad = (row != self.pad_token_id).nonzero(as_tuple=False)
            if non_pad.numel() == 0:
                continue
            last = int(non_pad[-1].item()) + 1
            mask[b, :last] = True
        return mask

    def forward_from_rpn_strings(
        self,
        rpns: Sequence[str],
        scalar_params_list: Optional[Sequence[Optional[Dict[str, float]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize with :func:`batch_tokenize_rpn` then :meth:`forward_from_token_tensors`."""
        if scalar_params_list is None:
            scalar_params_list = [None] * len(rpns)
        token_ids, scalar_vals, scalar_mask = batch_tokenize_rpn(
            rpns, list(scalar_params_list), max_len=None
        )
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)
        scalar_vals = scalar_vals.to(device)
        scalar_mask = scalar_mask.to(device)
        pad_mask = self._pad_mask_from_ids(token_ids)
        return self.forward_from_token_tensors(token_ids, scalar_vals, scalar_mask, pad_mask)
