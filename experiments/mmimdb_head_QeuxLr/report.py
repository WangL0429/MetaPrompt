import csv
import json
from statistics import mean
from pathlib import Path

job = Path(__file__).resolve().parent
previous = job.parent / "mmimdb_curriculum_20261119"
with (previous / "comparison.csv").open() as stream:
    old = list(csv.DictReader(stream))
with (job / "selected_test.csv").open() as stream:
    measured = list(csv.DictReader(stream))
assert len(measured) == 27
rows = []
for base in old:
    mode, rate = base["missing_type"], float(base["missing_rate"])
    tests = [r for r in measured if r["missing_type"] == mode and float(r["missing_rate"]) == rate]
    assert len(tests) == 3 and {int(r["mask_seed"]) for r in tests} == {42,2026,3407}
    score = mean(float(r["macro_f1_percent"]) for r in tests)
    original = float(base["curriculum_macro_f1_percent"])
    expert = float(base["DPL_specialist_macro_f1_percent"])
    rows.append(dict(missing_type=mode, missing_rate=rate, original=original, optimized=score,
                     change_pp=score-original, DPL_specialist=expert, gap_pp=score-expert))
assert len(rows) == 9
with (job/"comparison.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows))
