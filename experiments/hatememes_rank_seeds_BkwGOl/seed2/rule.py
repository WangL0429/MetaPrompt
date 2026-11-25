import torch


def flat(loss, params, graph=False):
    parts = torch.autograd.grad(loss,params,create_graph=graph,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p,g in zip(params,parts)])


def unit(vector):
    return vector / vector.norm().clamp_min(1e-12)


def curvature(support_loss, params, query_grad):
    direction = unit(flat(support_loss,params,graph=True))
    return flat(direction @ query_grad.detach(),params)
