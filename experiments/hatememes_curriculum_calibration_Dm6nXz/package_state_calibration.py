"""Package the validation-locked arm and verify its standalone positive-class log odds."""
import json
from pathlib import Path

import torch

from evaluate import infer, load_model
from food_io import SingleLabelDataset, condition_masks, grid, loader, save, setup
from single_label import decision

OUT = Path(__file__).resolve().parent


def load_calibrated(path):
    model = load_model(path)
    params = torch.load(path, map_location="cpu")["state_calibration"]["state_affine"]
    return model, params.cuda()


@torch.inference_mode()
def positive_margin(bundle, images, tokens, modes):
    model, affine = bundle
    margin = decision(infer(model,images,tokens,modes))
    return margin * affine[modes,0] + affine[modes,1]


def main():
    setup()
    lock = json.loads((OUT / "selection.json").read_text())
    selected = lock["selected"]
    use_raw = selected in ("ranked_raw","ranked_raw_cal")
    calibrated = selected.endswith("_cal")
    arm = "ranked_raw" if use_raw else "ranked"
    params = lock["scale_and_bias"][arm] if calibrated else [[1.,0.]]*3
    ck = torch.load(lock["rank_source"],map_location="cpu")
    if use_raw:
        ck["hatememes_image_readout"] = torch.load(lock["raw_head_source"],map_location="cpu")["hatememes_image_readout"]
    ck["state_calibration"] = dict(selected=selected, state_affine=torch.tensor(params,dtype=torch.float64),
                                   states=["complete","image_only","text_only"])
    path = OUT/"validation_selected.ckpt"
    save(ck,path)
    del ck
    bundle = load_calibrated(path)
    dataset = SingleLabelDataset("test")
    scores = []
    for images,tokens,_,_ in loader(dataset):
        images,tokens = images.cuda(),tokens.cuda()
        pure = torch.stack([positive_margin(bundle,images,tokens,torch.full((len(tokens),),s,device="cuda"))
                            for s in range(3)],1)
        if not scores:
            mode = torch.arange(len(tokens),device="cuda")%3
            mixed = positive_margin(bundle,images.masked_fill((mode==2)[:,None,None,None],0),
                                    tokens.masked_fill((mode==1)[:,None],0),mode)
            torch.testing.assert_close(mixed,pure[torch.arange(len(tokens),device="cuda"),mode],atol=.025,rtol=.01)
        scores.append(pure.cpu())
    ref = json.loads((OUT/"test.json").read_text())["metrics"][selected]
    masks = condition_masks(dataset.ids,"test")
    result = {}
    for metric in ("auroc","acc"):
        values,_ = grid(torch.cat(scores),dataset.labels,masks,metric)
        diff = max(abs(ref[metric+"_"+k]-v) for k,v in values.items())
        assert diff < .001, (metric,diff)
        result[metric+"_max_abs_delta_points"] = diff
    result.update(selected=selected,test_ids=len(dataset),checkpoint=str(path))
    (OUT / "replay.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
