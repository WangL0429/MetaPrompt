import sys
from pathlib import Path
import torch
from torch.func import functional_call

JOB = Path(__file__).resolve().parent
VIS = JOB.parent / 'mmimdb_visual_prompt_NOSgpO'
CROSS = JOB.parent / 'mmimdb_cross_meta_k9dMAw'
sys.path[:0] = [str(CROSS),str(VIS)]
from common import INITIAL, load, plots
from train_meta import validate, validation_data, views
from meta_weights import choose_weights
from clip.modules import clip


def outputs(net, images, tokens, state, params=None):
    # The unused modality has identically zero features in native inference.
    learner = net.model.prompt_learner
    pi, pt = learner([state]) if params is None else functional_call(learner, params, ([state],))
    n = len(tokens)
    modes = [state]*n
    im = tx = None
    if state != 2:
        im = net.model.image_encoder(images.type(net.model.dtype),
                                     [p.expand(n,-1,-1) for p in pi],modes)
    if state != 1:
        tx = net.model.text_encoder(tokens,[p.expand(n,-1,-1) for p in pt],modes)
    if im is None:
        im = torch.zeros_like(tx)
    if tx is None:
        tx = torch.zeros_like(im)
    return net.mmimdb_classifier(torch.cat((im,tx),1)).float(),im.float(),tx.float()
