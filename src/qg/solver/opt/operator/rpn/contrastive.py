"""
Contrastive learning for RPN token embeddings.

**Anchor** = tokenized expression. **Positive** = one sound algebraic rewrite
(from :mod:`algebra`). **Negatives** = other sequences in the batch (InfoNCE).

This module keeps a single encoder path: :class:`RPNTokenEmbedder` → pool →
projection. Pooling uses a mask so padded ``__scalar__`` slots do not contribute.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .algebra import AlgebraicRuleSet, create_composite_ruleset
from .embeddings import (
    TOKEN_TO_ID,
    ID_TO_TOKEN,
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

def infonce_single_loss(
    z_anchor: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy on both directions; positives are batch-aligned."""
    logits_ab = infonce_logits(z_anchor, z_anchor, temperature)
    B = z_anchor.size(0)
    targets = torch.arange(B, device=z_anchor.device)
    return F.cross_entropy(logits_ab, targets)

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

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # MHA expects (Batch, Seq, Feature) if batch_first=True
        attn_output, _ = self.mha(x, x, x)
        return self.linear(attn_output) + x  # Residual connection


class RPN_AE(nn.Module):
    """Pool sequence embeddings and project to contrastive space."""

    def __init__(self, seq_len=100, embed_dim: int=32, proj_dim: int=64, num_heads=4):
        super().__init__()
        self.proj = nn.Sequential(
            SelfAttention(embed_dim, num_heads=num_heads),
            nn.SiLU(),
            SelfAttention(embed_dim, num_heads=num_heads),
            nn.SiLU(),
            nn.Linear(embed_dim, proj_dim),
        )
        
        self.unproj = nn.Sequential(
            SelfAttention(proj_dim + embed_dim, num_heads=num_heads),
            nn.SiLU(),
            nn.Linear(proj_dim + embed_dim, embed_dim),
            nn.SiLU(),
            SelfAttention(embed_dim, num_heads=num_heads),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pe_fwd = nn.Parameter(0.01 * torch.randn(seq_len, embed_dim))
        self.pe_rev = nn.Parameter(0.01 * torch.randn(seq_len, embed_dim))  # Learnable reverse positional encoding

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.proj(
            rep + self.pe_fwd[None,...]
            ).mean(dim=-1) # B, proj_dim
    
    def reverse(self, rep, pooled):
        return self.unproj(
            torch.cat([
                    rep + self.pe_rev[None,...], 
                    pooled.unsqueeze(1).expand(-1, self.seq_len, -1)
                ], dim=-1)
            ) # B, seq_len, embed_dim
    
class ContrastiveRPN(nn.Module):
    """
    End-to-end: RPN strings → pooled embedding → projection → InfoNCE vs
    algebra-augmented positives.
    """

    def __init__(
        self,
        seq_len: int = 100,
        embed_dim: int = 32,
        proj_dim: int = 64,
        rules: Optional[AlgebraicRuleSet] = None,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.temperature = temperature
        self.embedder = RPNTokenEmbedder(embed_dim=embed_dim)
        self.head = RPN_AE(embed_dim, proj_dim)
        if rules is None:
            self.rules = None
        else:
            self.rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__scalar__"])
        self.seq_len = seq_len
        
        self.criterion = nn.MSELoss()
        
    def encode_token_batch(
        self,
        token_ids: torch.Tensor,
        amp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pad_mask : (B, L) bool — True for non-padding positions (inverse of padding column).
        """
        pooled = self.embedder(token_ids, amp)
        return self.head(pooled)

    def loss(
        self,
        rpns: Sequence[str],
    ) -> torch.Tensor:
        
        ### basic tokenization
        token_ids, amp = self.tokenize(rpns)
        device = self.embedder.token_embed.weight.device
        
        ### encode original batch
        pooled = self.embedder(token_ids.to(device), amp.to(device))
        z_a = self.head(pooled)
        
        ### contrastive loss (simple)
        loss = infonce_single_loss(z_a, self.temperature)
        
        ### denoiser (reconstruction via conditional flow-matching)
        noise = torch.randn_like(pooled)
        t = torch.rand(pooled.shape[0], device=device)[:, None] * 0.5 # less info needed
        pooled_noised = pooled * t + noise * (1 - t)
        denoise_loss = self.criterion(self.head.reverse(pooled_noised, z_a), pooled)
        loss = loss + denoise_loss
        
        ### symmetry-based contrastive loss (algebra)
        if self.rules is not None:
            r_token_ids, r_amp = self.rules.random_positive_view(token_ids, amp)
            z_p = self.head(self.embedder(r_token_ids.to(device), r_amp.to(device)))
            rule_loss = infonce_symmetric_loss(z_a, z_p, self.temperature)
            loss = loss + rule_loss
            # apply random rewrite to each expression in the batch, encode with same head, compute contrastive loss
        else:
            rule_loss = 0.0
        
        return loss, denoise_loss, rule_loss

    def tokenize(self, rpns: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize with :func:`batch_tokenize_rpn`."""
        token_ids, amp = batch_tokenize_rpn(
            rpns, max_len = self.seq_len
        )
        return token_ids, amp
    
    def detokenize(self, token_ids: torch.Tensor, amp: torch.Tensor) -> List[str]:
        """Convert token IDs back to RPN strings."""
        __scalar__
        npy_ids = token_ids.cpu().numpy()
        amps = amp.cpu().numpy()
        
        rpns = []
        for seq_ids, seq_amp in zip(npy_ids, amps):
            tokens = [ID_TO_TOKEN[token_id] for token_id in seq_ids]
            rpn = []
            for token, a in zip(tokens, seq_amp):
                if token == "__scalar__":
                    rpn.append(f"{a:.6f}")
                else:
                    rpn.append(token)
            rpns.append(" ".join(rpn))
    
        return rpns
    
    def forward(
        self,
        rpns: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids, amp = self.tokenize(rpns)
        device = self.embedder.token_embed.weight.device
        token_ids = token_ids.to(device)
        amp = amp.to(device)
        return self.encode_token_batch(token_ids, amp)

    def decode(self, encoded):
        noisy_pooled = torch.randn((encoded.shape[0], self.seq_len, self.embed_dim), device=encoded.device)
        decoded = self.head.reverse(noisy_pooled, encoded)
        decoded_norm = decoded.norm(dim=-1, keepdim=True)
        decoded_normalized = decoded / (decoded_norm + 1e-8)
        amp = decoded_norm.squeeze(-1)
        
        # find nearest token in embedding space
        token_ids = self.embedder.token_embed.decode(decoded_normalized) 
        return token_ids, amp