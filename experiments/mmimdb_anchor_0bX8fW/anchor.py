import math


def soft_grid(logits, labels):
    """Nine expected-confusion Macro-F1 scores, in [0, 1]."""
    probs = logits.sigmoid()
    weights = []
    for mode in ("both", "image", "text"):
        for rate in (.5, .7, .9):
            text_missing = {"text":rate, "image":0., "both":rate/2}[mode]
            weights.append([1-rate, text_missing, rate-text_missing])
    w = probs.new_tensor(weights)
    tp = w @ (probs*labels).sum(dim=1)
    predicted = w @ probs.sum(dim=1)
    return (2*tp / (predicted + labels.sum(dim=0)).clamp_min(1e-8)).mean(dim=1)


def worst_drop(reference, current):
    return (reference.detach() - current).max().relu()


def eligible(grid, baseline):
    return (len(grid) == len(baseline) == 9 and
            all(math.isfinite(v) and v >= b-.25 for v,b in zip(grid, baseline)))
