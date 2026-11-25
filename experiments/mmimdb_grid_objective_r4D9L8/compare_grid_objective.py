"""Evaluate the two continuation objectives against each other and their actual starting weights."""
import json
import torch

from compare_semantics import compare, read_csv
from fit_semantics import predictions, read, get_weights
from predictor import SemanticPredictor
from semantic_io import JOB, csv_out, setup


def evaluate_pair():
    setup()
    arms = {name: JOB/name for name in ("control","grid")}
    selections = [json.loads((path/"selection.json").read_text()) for path in arms.values()]
    control,grid = selections
    assert control["recipe"]["initialization"] == grid["recipe"]["initialization"]
    assert control["recipe"]["lr"] == grid["recipe"]["lr"] == .0001
    assert not control["recipe"]["grid_objective"] and grid["recipe"]["grid_objective"]
    for field in ("epochs","steps","seed","trainable","native_cal","semantic_weight"):
        assert control[field] == grid[field],field
    assert len(control["history"]) == len(grid["history"]) == 20
    for a,b in zip(control["history"],grid["history"]):
        assert a["query_state_counts"] == b["query_state_counts"]
        assert a["train_plots"] == b["train_plots"] == 32278
        assert a["mean_training"][2] == b["mean_training"][2]
    anchor = torch.load(JOB/"previous/test_epoch_11.pt",map_location="cpu").numpy()
    fixed = {"start":anchor}
    test,w = read("test"),get_weights()
    for arm,path in arms.items():
        net = SemanticPredictor(1).eval().cuda()
        net.load_state_dict(torch.load(path/"epoch_20.pt",map_location="cpu"),strict=True)
        scores,_ = predictions(net,test,w)
        fixed[f"{arm}_epoch20"] = scores.numpy()
    pairs = [(f"grid_{rule}",f"control_{rule}") for rule in ("mean9","text90","epoch20")]
    pairs.extend((f"{arm}_{rule}","start") for arm in arms for rule in ("mean9","text90"))
    compare(arms,pairs,fixed,interval_scenarios=(("text",90),("image",90)))
    old_grid = read_csv((JOB/"previous").resolve().parent/"test_grid.csv")
    old = {(r["mode"],r["rate"]):float(r["off_text90"]) for r in old_grid}
    results = []
    for row in read_csv(JOB/"test_grid.csv"):
        start = float(row["start"])
        assert abs(start-old[row["mode"],row["rate"]])<1e-7
        for rule in ("mean9","text90","epoch20"):
            a,b = (float(row[f"{arm}_{rule}"]) for arm in arms)
            results.append(dict(mode=row["mode"],rate=row["rate"],selector=rule,start=start,
                                control=a,grid=b,DPL=float(row["DPL"]),grid_minus_control=b-a,grid_minus_start=b-start))
    csv_out(JOB/"objective_effect.csv",results)


if __name__=="__main__":
    evaluate_pair()
