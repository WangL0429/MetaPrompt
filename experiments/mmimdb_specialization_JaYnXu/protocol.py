"""Mask construction and CPU audit for the matched high-missing training design."""

import argparse
import hashlib
import json
import random
import runpy
from pathlib import Path


ARMS = ("course_shared", "high_shared", "high_text", "high_image")
TYPES = ("text", "image", "both")


def training_rate(arm, course_rate):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}")
    return course_rate if arm == "course_shared" else 0.9


def state_fractions(arm, rate):
    """Complete, text missing, image missing. Use for population risk statistics."""
    if arm not in ARMS or not 0 <= rate <= 1:
        raise ValueError("Invalid training arm or missing rate")
    if arm == "high_text":
        return 1 - rate, rate, 0.0
    if arm == "high_image":
        return 1 - rate, 0.0, rate
    return 1 - rate, rate / 2, rate / 2


def episode_masks(arm, course_rate, seed, rank, part, size):
    """Return state codes in the caller's sample order; consumes no data-loader RNG."""
    if size < 0 or part not in ("support", "router", "live_query", "fast_query"):
        raise ValueError("Invalid episode part or size")
    rate = training_rate(arm, course_rate)
    if not 0 <= rate <= 1:
        raise ValueError("Missing rate must be between zero and one")
    rng = random.Random(f"{seed}/{rank}/{part}")
    order = list(range(size))
    rng.shuffle(order)
    count = round(size * rate)
    modes = TYPES if part in ("support", "router") else ("both",)
    masks = []
    for index, kind in enumerate(modes):
        offset = index * size // len(modes)
        missing = (order[offset:] + order[:offset])[:count]
        split = (count + rng.randrange(2)) // 2
        effective_kind = {"high_text": "text", "high_image": "image"}.get(arm, kind)
        if effective_kind == "text":
            split = count
        elif effective_kind == "image":
            split = 0
        mask = [0] * size
        for j, row in enumerate(missing):
            mask[row] = 1 if j < split else 2
        masks.append(mask)
    return masks


def audit(schedule, seed):
    fingerprint = hashlib.sha256()
    for rate, kind, mask_seed in schedule:
        fingerprint.update(f"{rate.hex()}/{kind}/{mask_seed}\n".encode())
    rows, steps = 32278, 64
    counts = {arm: [0, 0, 0] for arm in ARMS}
    for step, (rate, _, mask_seed) in enumerate(schedule):
        batch = rows // steps + int(step % steps < rows % steps)
        for rank in (0, 1):
            local = (batch + (1 - rank)) // 2
            for part, size in (("support", 8), ("router", 12),
                               ("live_query", (local + 1) // 2), ("fast_query", local // 2)):
                by_arm = {a: episode_masks(a, rate, mask_seed, rank, part, size) for a in ARMS}
                for a, masks in by_arm.items():
                    k = round(training_rate(a, rate) * size)
                    for m in masks:
                        assert len(m) - m.count(0) == k
                        if a == "high_text":
                            assert 2 not in m
                        if a == "high_image":
                            assert 1 not in m
                    if len(masks) == 3 and 0 < k < size:
                        assert len({tuple(v == 0 for v in m) for m in masks}) == 3
                    if part.endswith("query"):
                        for state in range(3):
                            counts[a][state] += masks[0].count(state)
                for triple in zip(by_arm["high_shared"], by_arm["high_text"], by_arm["high_image"]):
                    assert len({tuple(v == 0 for v in m) for m in triple}) == 1
    for c in counts.values():
        assert sum(c) == rows * 20
    return dict(seed=seed, steps=len(schedule), course_sha256=fingerprint.hexdigest(),
                query_counts=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    make_schedule = runpy.run_path(str(args.repo / "clip/datamodules/missing_schedule.py"))["make_schedule"]
    result = [audit(make_schedule(1280, seed, "curriculum"), seed) for seed in (0, 1)]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
