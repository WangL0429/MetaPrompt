"""First-order coefficients of the mean scenario soft Macro-F1 with respect to label probabilities."""
import torch


def scenario_weights(reference):
    return reference.new_tensor([
        (1-rate, rate*fraction, rate*(1-fraction))
        for fraction in (0., .5, 1.) for rate in (.5, .7, .9)
    ])


def grid_coefficients(probabilities, labels):
    """Inputs have shapes (items, states, classes) and (items, classes)."""
    weights = scenario_weights(probabilities)
    population = labels.mean(dim=0)
    true_positive = weights @ (probabilities*labels[:,None]).mean(dim=0)
    denominator = (weights @ probabilities.mean(dim=0) + population).clamp_min(1e-8)
    # Normalize per observed state so the curriculum still controls missing-view exposure.
    conditional = weights / weights.sum(dim=0, keepdim=True)
    slope = 2 * conditional.T @ denominator.reciprocal()
    intercept = 2 * conditional.T @ (true_positive / denominator.square())
    return slope, intercept
