"""Matched continuation from the previous text90-selected single-candidate checkpoint."""
import argparse
import json
import torch

from evaluate_semantics import evaluate
from fit_semantics import fit
from grid_objective import grid_coefficients, scenario_weights
from semantic_io import JOB, SOURCE


def check_coefficients():
    probabilities = torch.linspace(.03,.98,72,dtype=torch.float64).reshape(8,3,3).requires_grad_()
    labels = (torch.arange(24).reshape(8,3)%3 == torch.arange(8)[:,None]%3).double()
    weights = scenario_weights(probabilities)
    tp = weights @ (probabilities*labels[:,None]).mean(0)
    den = weights @ probabilities.mean(0) + labels.mean(0)
    gradient, = torch.autograd.grad((2*tp/den).mean(), probabilities)
    slope,intercept = grid_coefficients(probabilities.detach(),labels)
    expected = (labels[:,None]*slope-intercept)*weights.mean(0)[None,:,None]/labels.numel()
    torch.testing.assert_close(gradient,expected)


def run(arm):
    start = JOB/"previous/text90.ckpt"
    previous = json.loads((JOB/"previous/selection.json").read_text())
    assert previous["semantic_weight"] == 0 and previous["picks"]["text90"] == 11
    starting = torch.load(start,map_location="cpu")
    native = torch.load(SOURCE,map_location="cpu")["state_dict"]
    assert starting["semantic_predictor"]["epoch"] == 11
    assert native.keys() == starting["state_dict"].keys()
    assert all(torch.equal(value,starting["state_dict"][key]) for key,value in native.items())
    del native,starting
    check_coefficients()
    fit(1,out=JOB/arm,warm_start=start,grid_objective=arm=="grid",lr=.0001,paired_supervision=False)
    evaluate(1,out=JOB/arm)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("arm",choices=("grid","control"))
    run(parser.parse_args().arm)
