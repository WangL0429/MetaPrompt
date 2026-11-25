"""Class-balanced pairwise ranking for one missing state of a query episode."""
import torch
from torch.nn import functional as F


def query_ranking_loss(logits, target, *, task):
    """Average logistic pair loss over classes with positive and negative examples.

    Supply predictions for a single missing state. Multiclass targets are class
    indices, multilabel targets are indicator matrices, and binary targets can
    accompany either one logit or two logits ordered [negative, positive].
    """
    if logits.ndim == 1:
        logits = logits[:, None]
    if logits.ndim != 2 or len(target) != len(logits):
        raise ValueError("Expected one label and one vector of logits per sample")
    scores = logits if logits.dtype == torch.float64 else logits.float()
    if task == 'multilabel':
        if target.shape != scores.shape:
            raise ValueError("Multilabel targets must match the logits")
        positive = target.bool()
    elif task == 'multiclass':
        if scores.shape[1] < 2 or target.numel() != len(scores):
            raise ValueError("Multiclass classification requires class indices")
        positive = F.one_hot(target.reshape(-1).long(), scores.shape[1]).bool()
        scores = F.log_softmax(scores, dim=1)
    elif task == 'binary':
        if scores.shape[1] not in (1, 2) or target.numel() != len(scores):
            raise ValueError("Binary classification requires one target and one or two logits")
        if scores.shape[1] == 2:
            scores = scores[:, 1:] - scores[:, :1]
        positive = target.reshape(-1, 1).bool()
    else:
        raise ValueError(f"Unknown classification task {task!r}")
    pairs = positive[:, None, :] & ~positive[None, :, :]
    counts = pairs.sum(dim=(0, 1))
    losses = F.softplus(scores[None, :, :] - scores[:, None, :])
    per_class = (losses * pairs).sum(dim=(0, 1)) / counts.clamp_min(1)
    return per_class.sum() / (counts > 0).sum().clamp_min(1)
