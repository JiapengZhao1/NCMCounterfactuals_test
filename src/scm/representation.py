"""Utilities for categorical representations.

Conventions used for the categorical upgrade:
- External representation: index in {0..K-1}, tensor shape [n, 1] (Long).
- Internal representation (model inputs): one-hot float tensor shape [n, K].

This module is intentionally small and dependency-free so it can be imported
from runner/pipeline/scm without circular imports.
"""

from __future__ import annotations

from typing import Optional

import torch as T


def ensure_index(x: T.Tensor) -> T.Tensor:
    """Ensure x is a Long index tensor of shape [n, 1]."""
    if not T.is_tensor(x):
        x = T.as_tensor(x)
    if x.dim() == 0:
        x = x.view(1, 1)
    elif x.dim() == 1:
        x = x.view(-1, 1)
    elif x.dim() == 2 and x.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Expected index tensor of shape [n,1] or [n], got {tuple(x.shape)}")
    return x.long()


def index_to_onehot(x: T.Tensor, k: int, *, dtype: T.dtype = T.float32) -> T.Tensor:
    """Convert an index tensor [n,1]/[n] to one-hot [n,k]."""
    x = ensure_index(x).squeeze(1)
    if (x < 0).any() or (x >= k).any():
        raise ValueError(f"Index out of range for K={k}: min={int(x.min())}, max={int(x.max())}")
    return T.nn.functional.one_hot(x, num_classes=int(k)).to(dtype)


def onehot_to_index(x: T.Tensor) -> T.Tensor:
    """Convert one-hot/soft tensor [n,k] to index [n,1] via argmax."""
    if x.dim() != 2:
        raise ValueError(f"Expected [n,k] tensor, got {tuple(x.shape)}")
    return T.argmax(x, dim=1).view(-1, 1).long()


def maybe_onehot_to_index(x: T.Tensor) -> T.Tensor:
    """If x looks like a categorical one-hot/soft vector ([n,k], k>1), argmax it.

    Otherwise return x unchanged.
    """
    if T.is_tensor(x) and x.dim() == 2 and x.shape[1] > 1:
        return onehot_to_index(x)
    return x
