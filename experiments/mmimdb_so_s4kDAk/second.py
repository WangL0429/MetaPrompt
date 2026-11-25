import json
import sys
from pathlib import Path

import torch

JOB = Path(__file__).resolve().parent
PAIRED = JOB.parent / 'mmimdb_anchor_0bX8fW'
sys.path.insert(0,str(PAIRED))
from train_anchor import OLD, INITIAL, episode, gradients, load, plots


def query_gradient(net, params, support, query, measure=False):
    fast = net._inner_adapt_prompts(support,net.combined_loss_fn,inner_lr=.01,inner_steps=1)
    if measure:
        # Evaluate the first-order approximation at identical adapted weights.
        leaves = {n:p.detach().requires_grad_() for n,p in fast.items()}
        plain = gradients(net._mmimdb_bce_loss(query,params=leaves),tuple(leaves.values()))
    loss = net._mmimdb_bce_loss(query,params=fast)
    full = gradients(loss,params)
    if measure:
        return full, plain
    return full


def meta_gradient(net, params, support, query):
    terms = [query_gradient(net,params,s,q) for s,q in zip(support,query)]
    return torch.stack(terms).mean(0)


def check():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    net, ck = load(INITIAL)
    cfg = ck['hyper_parameters']['config']
    del ck
    net.cuda().eval().requires_grad_(False)
    net.model.prompt_learner.requires_grad_(True)
    net.hparams.config['meta_second_order'] = True
    net.current_tasks = ['mmimdb']
    params = [net.model.get_parameter(n) for n in net._prompt_param_names()]
    schedule = json.loads((OLD/'episodes.json').read_text())
    s,q = episode(plots(cfg,'train'),schedule[0])
    data = {}
    for mode, support,query in zip(('text','image','both'),s,q):
        full, plain = query_gradient(net,params,support,query,measure=True)
        change = ((full-plain).norm()/plain.norm()).item()
        assert full.isfinite().all() and plain.isfinite().all() and change>0
        data[mode] = dict(relative_correction=change,cosine=torch.nn.functional.cosine_similarity(
                          full,plain,dim=0).item())
    data['peak_gib'] = torch.cuda.max_memory_allocated()/2**30
    (JOB/'gradient_check.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)

if __name__ == '__main__':
    check()
