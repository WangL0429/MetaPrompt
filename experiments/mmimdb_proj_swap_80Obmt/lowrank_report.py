"""Compare the single selected checkpoint to the shared baseline and the matched DPL experts."""
import csv
import json
from pathlib import Path
from statistics import mean

import torch

from extract import JOB, SHARED


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def keyed(path):
    rows=read_rows(path)
    output={(r["missing_type"], round(100*float(r["missing_rate"])), int(r["mask_seed"])):
            float(r["macro_f1_percent"]) for r in rows}
    if len(rows)!=27 or len(output)!=27:
        raise ValueError(f"Expected nine scenarios, three masks each in {path}")
    return output


def changed_tensors(original, path):
    revised=torch.load(path,map_location="cpu")["state_dict"]
    assert set(original)==set(revised)
    changes={}
    for k in original:
        a,b=original[k],revised[k]
        assert a.shape==b.shape and a.dtype==b.dtype
        if not torch.equal(a,b):
            changes[k]=int((a!=b).sum().item())
    return changes


def main():
    selected=JOB/"lowrank_selected.ckpt"
    trained=JOB/"lowrank_trained.ckpt"
    original=torch.load(SHARED,map_location="cpu")["state_dict"]
    changes=changed_tensors(original, selected)
    trained_changes=changed_tensors(original, trained)
    key={"model.image_encoder.proj","model.text_encoder.text_projection"}
    selection=json.loads((JOB/"lowrank_selection.json").read_text())
    assert set(changes)==(key if selection["selected"] else set())
    assert set(trained_changes)==key
    before=keyed(SHARED.parent/"test.csv")
    after=keyed(JOB/"lowrank_test.csv")
    actual=keyed(JOB/"trained_test.csv")
    assert set(before)==set(after)==set(actual)
    baselines=read_rows(SHARED.parent/"test_comparison.csv")
    scenarios=[]
    for ref in baselines:
        mode=ref["missing_type"]
        rate=round(100*float(ref["missing_rate"]))
        seeds=(42,2026,3407)
        av=mean(after[mode,rate,s] for s in seeds)
        bv=mean(before[mode,rate,s] for s in seeds)
        candidate=mean(actual[mode,rate,s] for s in seeds)
        assert abs(float(ref["selected"])-bv)<1e-8
        specialist=float(ref["DPL"])
        scenarios.append(dict(missing_type=mode,missing_rate=rate,baseline_F1=bv,selected_F1=av,
                              trained_F1=candidate,trained_gain_pp=candidate-bv,matched_DPL_F1=specialist,
                              trained_gap_to_DPL_pp=candidate-specialist,
                              **{"trained_gain_seed_"+str(s):actual[mode,rate,s]-before[mode,rate,s] for s in seeds}))
    with (JOB/"lowrank_comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=scenarios[0])
        writer.writeheader()
        writer.writerows(scenarios)
    def average(key):
        return mean(r[key] for r in scenarios)
    results=dict(epoch=selection["selected"],train_epochs=len(selection["history"])-1,
                 eligible_epochs=sum(r["eligible"] for r in selection["history"][1:]),
                 shared_mean=average("baseline_F1"),selected_mean=average("selected_F1"),
                 trained_epoch=selection["best_nonzero_epoch"],trained_mean=average("trained_F1"),
                 DPL_mean=average("matched_DPL_F1"),
                 trained_gains_by_mask={s:average("trained_gain_seed_"+str(s)) for s in (42,2026,3407)},
                 baseline_wins=sum(r["baseline_F1"]>r["matched_DPL_F1"] for r in scenarios),
                 trained_wins=sum(r["trained_gap_to_DPL_pp"]>0 for r in scenarios),
                 improved_scenarios=sum(r["trained_gain_pp"]>0 for r in scenarios),
                 changed_weights=changes,changed_weights_trained=trained_changes,
                 extra_trainable_parameters=selection["params"],
                 inference_parameter_count_change=0,checkpoint=str(selected.resolve()))
    (JOB/"lowrank_summary.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
    print((JOB/"lowrank_comparison.csv").read_text())


if __name__=="__main__":
    torch.set_num_threads(2)
    main()

