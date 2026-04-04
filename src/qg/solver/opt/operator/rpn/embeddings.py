"""
rpn_embeddings.py
-----------------
Token vocabulary and per-token learned embeddings for RPN-encoded PDEs.

Design goals
~~~~~~~~~~~~
1. Every stable token in the compiler has exactly one embedding vector.
2. Numeric scalar constants are mapped to a continuous embedding via a small
   MLP acting on the scalar value — they are NOT discretised.
3. Named pde_params constants (e.g. "r", "beta") share a single UNKNOWN_PARAM
   slot; their numeric value is handled by the scalar MLP at encode-time.
4. The vocabulary is the single source of truth: OperatorRegistry and the
   compiler's variable/special tables must not diverge from it.
5. The module is self-contained — it does NOT import from rpn_compiler or
   rpn_ir, so it can be loaded and trained without a derivative context.

Token categories (mirrored from the compiler)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  VARIABLE       — state fields: q, psi, u, v, x, y
  LINEAR_DIFF    — spectral derivative operators: dx, dy, lap, invlap
  NONLINEAR_UNARY — stable pointwise functions: sqrt, cos, sin, …
  BINARY_OP      — arithmetic: +, -, *
  VECTOR_OP      — vector calculus: grad, div, curl, dot
  JACOBIAN       — jacobian / j
  MISC_OP        — neg, dealias
  SCALAR_CONST   — a numeric literal or named param (handled by scalar MLP)

Embedding structure per token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  final_embed(token) = LayerNorm(
      token_embed(token_id)          # E-dim lookup
    + category_embed(category_id)    # E-dim categorical bias
    + scalar_proj(value)             # E-dim, nonzero only for SCALAR_CONST
  )

  token_embed  : nn.Embedding(vocab_size, embed_dim)
  category_embed: nn.Embedding(n_categories, embed_dim)
  scalar_proj  : ScalarMLP(1 → embed_dim), applied only to scalar tokens
"""

from __future__ import annotations

import math
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Token categories
# ---------------------------------------------------------------------------

class TokenCategory(IntEnum):
    VARIABLE        = 0
    LINEAR_DIFF     = auto()
    NONLINEAR_UNARY = auto()
    BINARY_OP       = auto()
    VECTOR_OP       = auto()
    JACOBIAN        = auto()
    MISC_OP         = auto()
    SCALAR_CONST    = auto()   # numeric literals + named params


# ---------------------------------------------------------------------------
# Vocabulary definition
# ---------------------------------------------------------------------------
# Each entry: canonical_name → (token_id, category)
# Aliases are resolved to canonical names by the tokenizer (see normalize_token).

_VOCAB_DEF: List[Tuple[str, TokenCategory]] = [
    # --- variables ---
    ("q",        TokenCategory.VARIABLE),
    ("psi",      TokenCategory.VARIABLE),
    ("u",        TokenCategory.VARIABLE),
    ("v",        TokenCategory.VARIABLE),
    ("x",        TokenCategory.VARIABLE),
    ("y",        TokenCategory.VARIABLE),

    # --- linear differential operators ---
    ("dx",       TokenCategory.LINEAR_DIFF),
    ("dy",       TokenCategory.LINEAR_DIFF),
    ("lap",      TokenCategory.LINEAR_DIFF),
    ("invlap",   TokenCategory.LINEAR_DIFF),

    # --- stable nonlinear unary functions ---
    ("sqrt",     TokenCategory.NONLINEAR_UNARY),
    ("cos",      TokenCategory.NONLINEAR_UNARY),
    ("sin",      TokenCategory.NONLINEAR_UNARY),
    ("cosh",     TokenCategory.NONLINEAR_UNARY),
    ("sinh",     TokenCategory.NONLINEAR_UNARY),
    ("tanh",     TokenCategory.NONLINEAR_UNARY),
    ("exp",      TokenCategory.NONLINEAR_UNARY),
    ("square",   TokenCategory.NONLINEAR_UNARY),
    ("cube",     TokenCategory.NONLINEAR_UNARY),
    ("abs",      TokenCategory.NONLINEAR_UNARY),

    # --- binary arithmetic ---
    ("+",        TokenCategory.BINARY_OP),
    ("-",        TokenCategory.BINARY_OP),
    ("*",        TokenCategory.BINARY_OP),

    # --- vector calculus ---
    ("grad",     TokenCategory.VECTOR_OP),
    ("div",      TokenCategory.VECTOR_OP),
    ("curl",     TokenCategory.VECTOR_OP),
    ("dot",      TokenCategory.VECTOR_OP),

    # --- Jacobian ---
    ("jacobian", TokenCategory.JACOBIAN),

    # --- misc ---
    ("neg",      TokenCategory.MISC_OP),
    ("dealias",  TokenCategory.MISC_OP),

    # --- scalar placeholder (numeric literals and named params) ---
    ("__scalar__", TokenCategory.SCALAR_CONST),
]

# Build forward and reverse maps once at import time.
TOKEN_TO_ID:  Dict[str, int]           = {name: i for i, (name, _) in enumerate(_VOCAB_DEF)}
TOKEN_TO_CAT: Dict[str, TokenCategory] = {name: cat for name, cat in _VOCAB_DEF}
ID_TO_TOKEN:  Dict[int, str]           = {i: name for i, (name, _) in enumerate(_VOCAB_DEF)}

VOCAB_SIZE     = len(_VOCAB_DEF)
N_CATEGORIES   = len(TokenCategory)
SCALAR_TOKEN_ID = TOKEN_TO_ID["__scalar__"]

# Aliases: map compiler synonyms → canonical vocab name.
_ALIASES: Dict[str, str] = {
    # variables
    "omega": "q",
    "ph":    "psi",
    "uh":    "u",
    "vh":    "v",
    # operators
    "nabla": "grad",
    "del":   "grad",
    "inner": "dot",
    "j":     "jacobian",
    "mul":   "*",
}

# Tokens that are removed from the compiler but may appear in old data;
# raise a clear error if encountered.
_REMOVED_TOKENS = frozenset({
    "tan", "acos", "asin", "atan", "log", "log10",
    "sign", "ceil", "floor", "round",
})


def normalize_token(token: str) -> str:
    """
    Map a raw RPN token string to its canonical vocabulary name.

    Returns "__scalar__" for numeric literals and unknown named params.
    Raises ValueError for removed (unstable) tokens.
    """
    s = token.strip()

    # Numeric literal?
    try:
        float(s)
        return "__scalar__"
    except ValueError:
        pass

    lower = s.lower()

    if lower in _REMOVED_TOKENS:
        raise ValueError(
            f"Token '{token}' has been removed from the stable vocabulary "
            f"(numerically unstable). Remove it from the RPN expression."
        )

    # Resolve alias first.
    canonical = _ALIASES.get(lower, lower)

    if canonical in TOKEN_TO_ID:
        return canonical

    # Named scalar parameter (e.g. "r", "beta", "f0").
    return "__scalar__"


def tokenize_rpn(
    rpn: Union[str, Sequence[str]],
    scalar_params: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], List[Optional[float]]]:
    """
    Convert a raw RPN expression to (canonical_tokens, scalar_values).

    canonical_tokens : list of canonical vocab names
    scalar_values    : parallel list; float value for __scalar__ tokens, else None

    Parameters
    ----------
    rpn : str or list of str
        Raw RPN expression.
    scalar_params : dict, optional
        Mapping from named param token (e.g. "r") to its float value.
        Named params not found here get value 0.0 with a warning.
    """
    if isinstance(rpn, str):
        raw_tokens = [t for t in rpn.strip().split() if t]
    else:
        raw_tokens = list(rpn)

    if scalar_params is None:
        scalar_params = {}

    canonical: List[str]             = []
    values:    List[Optional[float]] = []

    for tok in raw_tokens:
        canon = normalize_token(tok)
        canonical.append(canon)

        if canon == "__scalar__":
            # Try numeric literal first.
            try:
                values.append(float(tok))
            except ValueError:
                # Named param.
                val = scalar_params.get(tok.lower(), None)
                if val is None:
                    import warnings
                    warnings.warn(
                        f"Named param '{tok}' not found in scalar_params; defaulting to 0.0",
                        stacklevel=2,
                    )
                    val = 0.0
                values.append(val)
        else:
            values.append(None)

    return canonical, values


# ---------------------------------------------------------------------------
# Scalar MLP — maps a single float to an embed_dim vector
# ---------------------------------------------------------------------------

class ScalarEmbedding(nn.Module):
    """
    Embeds a single scalar value (numeric constant or param value) into R^embed_dim.

    Architecture: Fourier feature encoding → 2-layer MLP
    The Fourier features help the network distinguish small vs large scalars
    and negative vs positive values without requiring scale normalisation.
    """

    def __init__(self, embed_dim: int, n_fourier: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_fourier = n_fourier

        # Fixed Fourier feature frequencies (not learned).
        # Shape: (n_fourier,)
        freqs = torch.exp(
            torch.linspace(math.log(0.01), math.log(100.0), n_fourier)
        )
        self.register_buffer("freqs", freqs)

        # MLP: (2*n_fourier + 1) → embed_dim
        in_dim = 2 * n_fourier + 1  # sin, cos features + raw value
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        values : (B,) or (B, L) float tensor of scalar values

        Returns
        -------
        (B, embed_dim) or (B, L, embed_dim) tensor
        """
        orig_shape = values.shape
        v = values.reshape(-1, 1)                    # (N, 1)
        phases = v * self.freqs.unsqueeze(0)         # (N, n_fourier)
        feats = torch.cat([
            torch.sin(phases),
            torch.cos(phases),
            v,
        ], dim=-1)                                   # (N, 2*n_fourier + 1)
        out = self.mlp(feats)                        # (N, embed_dim)
        return out.reshape(*orig_shape, self.embed_dim)


# ---------------------------------------------------------------------------
# Token embedding table
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """
    Per-token embedding table for the RPN vocabulary.

    For non-scalar tokens: looks up token_id in the embedding table and
    adds a category-level bias.

    For __scalar__ tokens: returns zero (the caller adds ScalarEmbedding output).

    All embeddings are L2-normalised before return so that contrastive
    distances are on a consistent scale.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed    = nn.Embedding(VOCAB_SIZE,   embed_dim)
        self.category_embed = nn.Embedding(N_CATEGORIES, embed_dim)
        self.layer_norm     = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # Use category-aware initialisation: tokens in the same category
        # start near each other so the contrastive loss can separate them
        # based on algebraic role rather than random noise.
        nn.init.normal_(self.token_embed.weight,    std=0.02)
        nn.init.normal_(self.category_embed.weight, std=0.1)

        # Explicitly set category embeddings to human-interpretable directions.
        # This gives the model a warm-start that respects the operator taxonomy.
        with torch.no_grad():
            for name, cat in TOKEN_TO_CAT.items():
                if name == "__scalar__":
                    continue
                tid = TOKEN_TO_ID[name]
                cid = int(cat)
                # Category embed already initialised; no override needed.
                # Just zero the token embed so pure category signal dominates
                # at init — the token embed learns fine-grained distinctions.
                self.token_embed.weight[tid].zero_()

    def _build_id_to_cat_buffer(self, device: torch.device) -> torch.Tensor:
        """Precomputed tensor: vocab_id → category_id, shape (VOCAB_SIZE,)."""
        mapping = [int(TOKEN_TO_CAT[ID_TO_TOKEN[i]]) for i in range(VOCAB_SIZE)]
        return torch.tensor(mapping, dtype=torch.long, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : (B, L) long tensor of token IDs from TOKEN_TO_ID

        Returns
        -------
        (B, L, embed_dim) float tensor, layer-normed
        """
        # Build or reuse the vocab→category mapping buffer (avoids Python loop).
        if not hasattr(self, "_id_to_cat") or self._id_to_cat.device != token_ids.device:
            self._id_to_cat = self._build_id_to_cat_buffer(token_ids.device)

        cat_ids = self._id_to_cat[token_ids]                   # (B, L)

        tok_emb = self.token_embed(token_ids)          # (B, L, E)
        cat_emb = self.category_embed(cat_ids)         # (B, L, E)
        return self.layer_norm(tok_emb + cat_emb)      # (B, L, E)


# ---------------------------------------------------------------------------
# Full RPN token embedder (combines TokenEmbedding + ScalarEmbedding)
# ---------------------------------------------------------------------------

class RPNTokenEmbedder(nn.Module):
    """
    Maps a batch of tokenized RPN expressions to a sequence of embeddings.

    Input
    -----
    token_ids   : (B, L) long     — token IDs (use TOKEN_TO_ID)
    scalar_vals : (B, L) float    — scalar values; 0.0 for non-scalar tokens
    scalar_mask : (B, L) bool     — True where token is __scalar__

    Output
    ------
    embeddings  : (B, L, embed_dim) float, layer-normed per position

    The embedder can be used standalone (for inspection / debugging) or as
    the first stage of the RPNEncoder defined below.
    """

    def __init__(self, embed_dim: int = 32, scalar_fourier: int = 16):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_emb  = TokenEmbedding(embed_dim)
        self.scalar_emb = ScalarEmbedding(embed_dim, n_fourier=scalar_fourier)
        self.out_norm   = nn.LayerNorm(embed_dim)

    def forward(
        self,
        token_ids:   torch.Tensor,   # (B, L)
        scalar_vals: torch.Tensor,   # (B, L)  float
        scalar_mask: torch.Tensor,   # (B, L)  bool
    ) -> torch.Tensor:               # (B, L, E)

        tok_emb = self.token_emb(token_ids)          # (B, L, E)

        # Scalar embedding — compute for all positions, zero-out non-scalars.
        sc_emb = self.scalar_emb(scalar_vals)        # (B, L, E)
        sc_emb = sc_emb * scalar_mask.unsqueeze(-1).float()

        # For scalar positions, replace token embed with scalar embed.
        # For non-scalar positions, keep token embed (scalar contrib is zero).
        combined = torch.where(
            scalar_mask.unsqueeze(-1).expand_as(tok_emb),
            sc_emb,
            tok_emb,
        )
        return self.out_norm(combined)               # (B, L, E)

    # ------------------------------------------------------------------
    # Convenience: build inputs from tokenize_rpn output
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        canonical_tokens: List[str],
        scalar_values:    List[Optional[float]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert tokenize_rpn output to tensors for a single expression (B=1).

        Returns (token_ids, scalar_vals, scalar_mask), all on `device`.
        """
        ids    = [TOKEN_TO_ID[t] for t in canonical_tokens]
        vals   = [v if v is not None else 0.0 for v in scalar_values]
        is_sc  = [v is not None for v in scalar_values]

        t_ids  = torch.tensor(ids,  dtype=torch.long).unsqueeze(0)
        t_vals = torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
        t_mask = torch.tensor(is_sc, dtype=torch.bool).unsqueeze(0)

        if device is not None:
            t_ids  = t_ids.to(device)
            t_vals = t_vals.to(device)
            t_mask = t_mask.to(device)

        return t_ids, t_vals, t_mask

    def embed_rpn(
        self,
        rpn: Union[str, Sequence[str]],
        scalar_params: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        End-to-end convenience: raw RPN string → (1, L, embed_dim) tensor.

        Primarily for debugging and inspection; training should use batched inputs.
        """
        canonical, values = tokenize_rpn(rpn, scalar_params)
        ids, vals, mask   = self.prepare_inputs(canonical, values, device)
        with torch.no_grad():
            return self.forward(ids, vals, mask)


# ---------------------------------------------------------------------------
# Utility: pretty-print vocabulary
# ---------------------------------------------------------------------------

def print_vocab() -> None:
    """Print the full token vocabulary with IDs and categories."""
    print(f"{'ID':>4}  {'Token':<14}  {'Category'}")
    print("-" * 42)
    for i, (name, cat) in enumerate(_VOCAB_DEF):
        print(f"{i:>4}  {name:<14}  {cat.name}")


def get_tokens_by_category(category: TokenCategory) -> List[str]:
    """Returns canonical token names for the requested category."""
    return [t for t, c in _VOCAB_DEF if c == category]


def get_variable_tokens() -> List[str]:
    return get_tokens_by_category(TokenCategory.VARIABLE)


def get_binary_tokens() -> List[str]:
    return get_tokens_by_category(TokenCategory.BINARY_OP)


def get_unary_tokens() -> List[str]:
    return get_tokens_by_category(TokenCategory.NONLINEAR_UNARY) + get_tokens_by_category(TokenCategory.MISC_OP)


def get_vector_tokens() -> List[str]:
    return get_tokens_by_category(TokenCategory.VECTOR_OP) + get_tokens_by_category(TokenCategory.JACOBIAN)


def batch_tokenize_rpn(
    rpns: Sequence[Union[str, Sequence[str]]],
    scalar_params_list: Optional[Sequence[Optional[Dict[str, float]]]] = None,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize a batch of RPN expressions, padding to max_len.

    Returns
    -------
    token_ids : (B, L) long
    scalar_vals: (B, L) float
    scalar_mask: (B, L) bool
    """
    if scalar_params_list is None:
        scalar_params_list = [None] * len(rpns)
    if len(scalar_params_list) != len(rpns):
        raise ValueError("scalar_params_list must match rpns length")

    canonicals = []
    values = []
    lengths = []

    for rpn, params in zip(rpns, scalar_params_list):
        c, v = tokenize_rpn(rpn, params)
        canonicals.append(c)
        values.append(v)
        lengths.append(len(c))

    if max_len is None:
        max_len = max(lengths)
    if max_len <= 0:
        raise ValueError("max_len must be positive")

    B = len(rpns)
    token_ids = torch.zeros((B, max_len), dtype=torch.long)
    scalar_vals = torch.zeros((B, max_len), dtype=torch.float)
    scalar_mask = torch.zeros((B, max_len), dtype=torch.bool)

    # Use padding token __scalar__ (zero value) for simplicity
    pad_id = TOKEN_TO_ID["__scalar__"]

    token_ids.fill_(pad_id)

    for i in range(B):
        L = min(lengths[i], max_len)
        for j in range(L):
            token_ids[i, j] = TOKEN_TO_ID[canonicals[i][j]]
            if values[i][j] is not None:
                scalar_vals[i, j] = float(values[i][j])
                scalar_mask[i, j] = True

    return token_ids, scalar_vals, scalar_mask


def _rand_scalar_literal() -> str:
    """Generate a random numeric constant token as string."""
    value = float(torch.empty(1).uniform_(-10.0, 10.0).item())
    # Keep readable and stable for tokenization
    return f"{value:.4f}"


def generate_random_rpn(
    max_length: int = 32,
    min_leaves: int = 2,
    max_leaves: Optional[int] = None,
    include_unary: bool = True,
    include_vector_ops: bool = False,
) -> str:
    """
    Generate a syntactically valid random RPN expression.

    This generator is meant for data augmentation / contrastive self-supervision.
    It guarantees that the expression is valid in terms of stack balance (for
    binary/unary operators) but does not guarantee physical meaning.
    """
    import random

    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    if max_leaves is None:
        max_leaves = max(2, max_length // 2)

    variables = get_variable_tokens()
    binary_ops = get_binary_tokens()
    unary_ops = (["neg", "dealias"] if include_unary else []) + (get_unary_tokens() if include_unary else [])
    vector_ops = get_vector_tokens() if include_vector_ops else []

    # Build a random binary tree with random leaf count
    n_leaves = random.randint(min_leaves, min(max_leaves, max(2, max_length // 2)))

    # Recursive tree node as nested tuple or leaf token
    def _build_tree(leaves: int):
        if leaves == 1:
            return random.choice(variables + [_rand_scalar_literal()])
        left_leaves = random.randint(1, leaves - 1)
        right_leaves = leaves - left_leaves
        return (
            _build_tree(left_leaves),
            _build_tree(right_leaves),
            random.choice(binary_ops + vector_ops),
        )

    def _to_postfix(node):
        if isinstance(node, tuple):
            left, right, op = node
            return _to_postfix(left) + _to_postfix(right) + [op]
        return [node]

    tree = _build_tree(n_leaves)
    tokens = _to_postfix(tree)

    # Add unary operations to the end to keep sequence valid.
    if include_unary:
        n_unary = max(0, max_length - len(tokens))
        n_unary = min(n_unary, len(unary_ops))
        # Randomly apply 0..n_unary unary ops onto final stack result.
        for _ in range(n_unary):
            tokens.append(random.choice(unary_ops))

    # Truncate to max length, if necessary
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    return " ".join(map(str, tokens))


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_vocab()
    print()

    embedder = RPNTokenEmbedder(embed_dim=32)
    print(f"Parameters: {sum(p.numel() for p in embedder.parameters()):,}")
    print()

    # QG vorticity advection: J(psi, q)
    rpn1 = "psi q jacobian"
    # Linear Ekman drag: -r * q
    rpn2 = "q r *"
    # Beta plane: beta * v  where v = dpsi/dx
    rpn3 = "psi dx beta *"
    # Hyperdiffusion: -nu * lap(lap(q))
    rpn4 = "q lap lap nu *"

    params = {"r": 0.01, "beta": 1.5, "nu": 1e-6}

    for label, rpn in [("J(ψ,q)", rpn1), ("-r·q", rpn2), ("β·v", rpn3), ("ν·Δ²q", rpn4)]:
        canon, vals = tokenize_rpn(rpn, params)
        emb = embedder.embed_rpn(rpn, params)
        print(f"{label:<10}  tokens={canon}  shape={tuple(emb.shape)}")
