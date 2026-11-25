"""Join the matched MMIMDb grid and compute per-condition differences."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evaluate_mmimdb import GRID, MODES, RATES, SEEDS


def main():
    directory = Path(__file__).resolve().parent
    scores = defaultdict(list)
    for path in [directory / f"{arm}.csv" for arm in ("curriculum", "random")]:
        for row in csv.DictReader(path.open()):
            key = row["missing_type"], float(row["missing_rate"]), row["method"]
            scores[key].append(float(row["macro_f1_percent"]))
    with (directory / "dpl_specialists" / "summary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["metric"] == "f1_macro":
                key = row["test_missing_type"], float(row["test_missing_rate"]), "DPL specialist"
                scores[key] = [100 * float(row[f"seed{seed}"]) for seed in SEEDS]

    with (GRID / "combined_primary.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["method"] == "DPL" and row["dataset"] == "mmimdb" and row["metric"] == "f1_macro":
                key = row["missing_type"], float(row["missing_rate"]), "DPL fixed Both70"
                scores[key] = [float(row[f"seed{seed}_percent"]) for seed in SEEDS]

    rows = []
    for mode in MODES:
        for rate in RATES:
            ratio = rate / 100
            by_method = {}
            for method in ("curriculum", "random", "DPL fixed Both70", "DPL specialist"):
                values = scores[mode, ratio, method]
                if len(values) != len(SEEDS):
                    raise ValueError(f"Incomplete mask results for {mode=} {ratio=} {method=}")
                by_method[method] = mean(values)
            course = by_method["curriculum"]
            specialist = by_method["DPL specialist"]
            rows.append(
                {
                    "missing_type": mode,
                    "missing_rate": ratio,
                    "curriculum_macro_f1_percent": course,
                    "random_macro_f1_percent": by_method["random"],
                    "curriculum_minus_random_pp": course - by_method["random"],
                    "DPL_fixed_Both70_macro_f1_percent": by_method["DPL fixed Both70"],
                    "DPL_specialist_macro_f1_percent": specialist,
                    "curriculum_minus_DPL_specialist_pp": course - specialist,
                    "random_minus_DPL_specialist_pp": by_method["random"] - specialist,
                }
            )
    with (directory / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    gaps = [row["curriculum_minus_DPL_specialist_pp"] for row in rows]
    worst_course = min(rows, key=lambda row: row["curriculum_minus_DPL_specialist_pp"])
    worst_random = min(rows, key=lambda row: row["random_minus_DPL_specialist_pp"])
    summary = {
        "conditions": len(rows),
        "mask_seeds": SEEDS,
        "curriculum_wins_vs_random": sum(row["curriculum_minus_random_pp"] > 0 for row in rows),
        "mean_curriculum_minus_random_pp": mean(row["curriculum_minus_random_pp"] for row in rows),
        "matched_specialist_conditions": len(gaps),
        "curriculum_wins_vs_specialists": sum(gap > 0 for gap in gaps),
        "worst_curriculum_minus_DPL_specialist_pp": worst_course["curriculum_minus_DPL_specialist_pp"],
        "worst_curriculum_condition": [worst_course["missing_type"], worst_course["missing_rate"]],
        "worst_random_minus_DPL_specialist_pp": worst_random["random_minus_DPL_specialist_pp"],
        "worst_random_condition": [worst_random["missing_type"], worst_random["missing_rate"]],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
