"""Dataset-specific classification and reporting settings for the shared training pipeline."""
import os

import torch
from sklearn.metrics import roc_auc_score

TASK = os.environ.get("METAPROMPT_SINGLE_LABEL_TASK", "food101")
if TASK not in ("food101", "hatememes"):
    raise ValueError(f"Unsupported dataset {TASK}")
BINARY = TASK == "hatememes"
N_CLASSES = 2 if BINARY else 101
PRIMARY = "auroc" if BINARY else "acc"
TAG = "hatememes" if BINARY else "food"


def head(net):
    return getattr(net, f"{TASK}_classifier")


def decision(logits):
    if BINARY:
        # The log odds of the positive class keep scores comparable across input states.
        return logits[...,1].double()-logits[...,0].double()
    return logits.argmax(-1)


def measure(scores, labels, metric=PRIMARY):
    if metric == "auroc":
        return 100 * roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    classes = scores.gt(0).long() if BINARY else scores
    return 100 * (classes==labels).double().mean().item()
