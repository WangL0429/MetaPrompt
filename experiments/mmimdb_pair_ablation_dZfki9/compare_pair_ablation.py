"""Measure the incremental effect of the paired loss with matched selection rules and at a fixed epoch."""
import json
import numpy as np
import torch

from compare_semantics import compare, read_csv
from fit_semantics import get_weights, predictions, read
from predictor import SemanticPredictor
from semantic_io import JOB, csv_out, setup


def selection(path):
    return json.loads((path/"selection.json").read_text())


def check_training(paired, disabled):
    assert paired.get("semantic_weight", .35) == .35 and disabled["semantic_weight"] == 0.
    for key in ("seed", "steps", "epochs", "k", "trainable", "native_cal"):
        assert paired[key] == disabled[key], key
    assert len(paired["history"]) == len(disabled["history"]) == 20
    for a,b in zip(paired["history"],disabled["history"]):
        for key in ("epoch","train_plots","query_state_counts"):
            assert a[key] == b[key], key
        assert a["mean_training"][2] == b["mean_training"][2]


def main():
    setup()
    on, off = JOB/"k1_paired", JOB/"k1"
    paired, disabled = selection(on), selection(off)
    check_training(paired, disabled)
    # The reference run's calibration-selected epoch is fixed before examining ablation test scores.
    epoch = paired["picks"]["mean9"]
    label = f"off_epoch{epoch}"
    cached = off/f"test_epoch_{epoch}.pt"
    if cached.is_file():
        fixed = torch.load(cached,map_location="cpu")
    else:
        model = SemanticPredictor(1).cuda().eval()
        model.load_state_dict(torch.load(off/f"epoch_{epoch:02}.pt",map_location="cpu"))
        fixed,_ = predictions(model,read("test"),get_weights())
    pairs = [(f"on_{rule}",f"off_{rule}") for rule in ("mean9","text90")]
    pairs += [(f"{arm}_mean9","base") for arm in ("off","on")]
    pairs.append(("on_mean9",label))
    compare({"on":on,"off":off},pairs,{label:fixed.numpy()},
            interval_scenarios=(("text",50),("text",70),("text",90),("image",90)))
    original = {(r["mode"],r["rate"],r["seed"]):r for r in read_csv(on.resolve().parent/"per_seed.csv")}
    for row in read_csv(JOB/"per_seed.csv"):
        other = original[row["mode"],row["rate"],row["seed"]]
        for col, prior_col in (("base","base"),("on_mean9","k1_mean9"),("on_text90","k1_text90"),("DPL","DPL")):
            assert abs(float(row[col])-float(other[prior_col])) < 1e-7
    effects = []
    for row in read_csv(JOB/"test_grid.csv"):
        for rule in ("mean9","text90"):
            a,b,base = (float(row[name]) for name in (f"on_{rule}",f"off_{rule}","base"))
            effects.append(dict(mode=row["mode"],rate=row["rate"],selector=rule,on_epoch=paired["picks"][rule],
                                off_epoch=disabled["picks"][rule],base=base,without_pair=b,with_pair=a,
                                off_minus_base=b-base,on_minus_off=a-b,on_minus_base=a-base))
    csv_out(JOB/"pair_effect.csv",effects)


if __name__=="__main__":
    main()
