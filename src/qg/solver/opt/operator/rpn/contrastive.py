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
    TOKEN_TO_CAT,
    ID_TO_TOKEN,
    RPNTokenEmbedder,
    batch_tokenize_rpn,
    TokenCategory,
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

def validate_rpn_syntax(token_ids: torch.Tensor) -> torch.Tensor:
    """
    Validate RPN syntax per sequence via stack-based evaluation.
    Uses token categories from embeddings.py to determine operator arity.
    Valid RPN ends with stack_depth = 1 (single value).
    
    Stack semantics by category:
      - SCALAR_CONST, VARIABLE: push value (stack += 1)
      - NONLINEAR_UNARY, LINEAR_DIFF, VECTOR_OP, JACOBIAN, MISC_OP: unary (pop 1, push 1)
      - BINARY_OP: binary (pop 2, push 1)
    
    Returns
    -------
    validity : (B,) binary tensor — 1.0 if valid, 0.0 if invalid
    """
    batch_size = token_ids.shape[0]
    validity = torch.zeros(batch_size, device=token_ids.device, dtype=torch.float32)
    
    for b in range(batch_size):
        stack_depth = 0
        valid = True
        
        for token_id in token_ids[b]:
            token_id_val = token_id.item()
            token = ID_TO_TOKEN.get(token_id_val, "__pad__")
            
            # Get token category
            category = TOKEN_TO_CAT.get(token, None)
            
            # Scalar constants and variables: push to stack
            if category in (
                TokenCategory.SCALAR_CONST,
                TokenCategory.VARIABLE
            ):
                stack_depth += 1
            # Unary operators: pop 1, push 1 (no net change, but need >= 1)
            elif category in (
                TokenCategory.NONLINEAR_UNARY,
                TokenCategory.LINEAR_DIFF,
                TokenCategory.VECTOR_OP,
                TokenCategory.MISC_OP,
            ):
                if stack_depth < 1:
                    valid = False
                    break
                # Stack depth unchanged (pop 1, push 1)
            # Binary operators: pop 2, push 1
            elif category in (
                TokenCategory.BINARY_OP,
                TokenCategory.JACOBIAN,
            ):
                if stack_depth < 2:
                    valid = False
                    break
                stack_depth -= 1  # pop 2, push 1
            # Padding: SKIP
            elif token == "__pad__":
                continue
            else:
                # Unknown token
                valid = False
                break
        
        # Valid if syntax check passed and final stack has exactly 1 element
        if valid and stack_depth == 1:
            validity[b] = 1.0
    
    return validity

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
        self.proj_dim = proj_dim
        self.pe_fwd = nn.Parameter(0.01 * torch.randn(seq_len, embed_dim))
        self.pe_rev = nn.Parameter(0.01 * torch.randn(seq_len, embed_dim))  # Learnable reverse positional encoding

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.proj(
            rep + self.pe_fwd[None,...]
            ).sum(dim=1) # B, proj_dim
    
    def reverse(self, rep, pooled):
        return self.unproj(
            torch.cat([
                    rep + self.pe_rev[None,...], 
                    pooled[:,None,:].expand(-1, self.seq_len, self.proj_dim)
                ], dim=-1)
            ) # B, seq_len, embed_dim
    
class ContrastiveRPN(nn.Module):
    """
    End-to-end: RPN strings → pooled embedding → projection → InfoNCE vs
    algebra-augmented positives.
    """
    # TODO fix naming so pooled actually represents the pooled embedding not
    # the sequence embedding

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
        self.head = RPN_AE(seq_len, embed_dim, proj_dim)
        
        self.use_rules = rules
        self.rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=TOKEN_TO_ID["__pad__"])
            
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.criterion = nn.MSELoss()
        
    def masked_criterion(self, pred: torch.Tensor, target: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # Apply the padding mask to the loss
        w = 0.99 # mostly not padding!
        mask = (ids == TOKEN_TO_ID["__pad__"]) * w + (1-w)
        mask = mask[:, :, None]
        return self.criterion(pred*mask, target*mask)

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
        device = self.head.pe_fwd.device
        
        ### encode original batch
        pooled = self.embedder(token_ids.to(device), amp.to(device))
        z_a = self.head(pooled)
        
        ### contrastive loss (simple)
        loss = infonce_single_loss(z_a, self.temperature)
        
        ### denoiser (reconstruction via one-step conditional flow-matching)
        noise = torch.randn_like(pooled)
        t = torch.rand(pooled.shape[0], device=device)[:, None, None] * 0.2 # less info needed
        pooled_noised = pooled * t + noise * (1 - t)
        decoded = self.head.reverse(pooled_noised, z_a)
        denoise_distortion_loss = self.masked_criterion(decoded, pooled, token_ids)
        loss = loss + denoise_distortion_loss
        
        recoded = self.head(decoded)
        denoise_perception_loss = self.criterion(recoded, z_a)
        loss = loss + denoise_perception_loss
        
        ### symmetry-based contrastive loss (algebra)
        if self.use_rules or not self.training:
            r_token_ids, r_amp = self.rules.random_positive_view(token_ids, amp)
            
            # truncate to what's available TODO check that this is properly padded and only padding is truncated
            # r_token_ids = r_token_ids[:,:self.seq_len,:]
            if r_token_ids.shape[1] == self.seq_len:
                
                z_p = self.head(self.embedder(r_token_ids.to(device), r_amp.to(device)))
                rule_loss = infonce_symmetric_loss(z_a, z_p, self.temperature)
                if self.use_rules:
                    loss = loss + rule_loss
            else:
                rule_loss = 0.0
            # apply random rewrite to each expression in the batch, encode with same head, compute contrastive loss
        else:
            rule_loss = 0.0
        
        ### GRPO-style syntax reward: sample multiple rollouts and encourage valid ones
        syntax_loss = self._grpo_syntax_loss(z_a, pooled, device)
        loss = loss + syntax_loss
        
        return loss, denoise_distortion_loss, denoise_perception_loss, syntax_loss, rule_loss
    
    def _grpo_syntax_loss(
        self,
        z_a: torch.Tensor,
        pooled: torch.Tensor,
        device: torch.device,
        num_samples: int = 3,
    ) -> torch.Tensor:
        """
        GRPO-style loss: sample multiple decoded rollouts, compute syntax validity,
        then use relative advantages to encourage valid RPN generation.
        
        Parameters
        ----------
        z_a : (B, proj_dim) — encoded embeddings
        pooled : (B, seq_len, embed_dim) — pooled sequence representation
        num_samples : int — number of rollout samples per batch element
        
        Returns
        -------
        syntax_loss : scalar tensor
        """
        batch_size = pooled.shape[0]
        syntax_loss = torch.tensor(0.0, device=device)
        
        # Sample multiple decoded sequences
        validity_scores = []
        reconstruction_errors = []
        
        for _ in range(num_samples):
            # Sample noisy input and decode
            noise = torch.randn_like(pooled)
            decoded = self.head.reverse(noise, z_a)
            
            # Get token predictions
            token_ids_sample, _ = self._decode_tokens(decoded)
            
            # Validate syntax
            validity = validate_rpn_syntax(token_ids_sample)
            validity_scores.append(validity)
            
            # Reconstruction error (lower is better)
            recon_error = self.criterion(decoded, pooled)
            reconstruction_errors.append(recon_error)
        
        # Stack validity scores: (num_samples, B)
        validity_scores = torch.stack(validity_scores, dim=0)  # (num_samples, B)
        reconstruction_errors = torch.stack(reconstruction_errors, dim=0)  # (num_samples,)
        
        # Compute relative advantages (GRPO style)
        # Higher validity → lower loss
        mean_validity = validity_scores.mean(dim=0, keepdim=True)  # (1, B)
        validity_advantage = validity_scores - mean_validity  # (num_samples, B)
        
        # Weight samples by validity advantage: encourage high-validity samples
        # and penalize low-validity ones (relative to batch mean)
        weighted_errors = reconstruction_errors.unsqueeze(-1) * (1.0 - validity_advantage)
        
        # Average over samples and batch
        syntax_loss = weighted_errors.mean()
        
        return syntax_loss
    
    def _decode_tokens(self, decoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to decode embeddings to token IDs. Returns token_ids, amplitudes."""
        amp = decoded.norm(dim=-1, keepdim=True)
        decoded_normalized = decoded / (amp + 1e-8)
        token_ids = self.embedder.token_embed.decode(decoded_normalized)

        return token_ids, amp.squeeze(-1)

    def tokenize(self, rpns: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize with :func:`batch_tokenize_rpn`."""
        token_ids, amp = batch_tokenize_rpn(
            rpns, max_len = self.seq_len
        )
        return token_ids, amp
    
    def detokenize(self, token_ids: torch.Tensor, amp: torch.Tensor) -> List[str]:
        """Convert token IDs back to RPN strings."""
        npy_ids = token_ids.detach().cpu().numpy()
        amps = amp.detach().cpu().numpy()
        
        rpns = []
        for seq_ids, seq_amp in zip(npy_ids, amps):
            rpn = []
            for token_id, a in zip(seq_ids, seq_amp):
                # Convert numpy int to Python int for safe dictionary lookup
                token_id_int = int(token_id)
                
                # Bounds check: skip if token_id is out of range
                if token_id_int < 0 or token_id_int >= len(ID_TO_TOKEN):
                    print(f"Warning: token_id {token_id_int} out of range [0, {len(ID_TO_TOKEN)})")
                    continue
                
                token = ID_TO_TOKEN.get(token_id_int, "__pad__")
                
                # Stop at padding
                if token == "__pad__":
                    continue
                
                # If scalar, use the amplitude value
                if token == "__scalar__" and abs(float(a)) > 1e-8:
                    rpn.append(f"{float(a):.6f}")
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
        return self._decode_tokens(decoded)