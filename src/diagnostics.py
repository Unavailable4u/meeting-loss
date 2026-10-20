"""Diagnostics for H3 (gradient alignment) and for collapse. UNTESTED starter code."""
import torch
import torch.nn.functional as F


def grad_alignment(model, x, y):
    """Cosine similarity between the task gradient and the meeting gradient
    on the lower layers (arm D only)."""
    params = list(model.low.parameters())
    h = model.low(x)
    z = model.up(h)
    L_task = F.cross_entropy(z, y)
    p = z.detach().softmax(-1)
    with torch.no_grad():
        e_y = F.one_hot(y, model.n_cls).float()
        b = h.detach() + model.G(e_y) - model.G(p)
    L_meet = F.mse_loss(h, b)
    g_task = torch.autograd.grad(L_task, params, retain_graph=True)
    g_meet = torch.autograd.grad(L_meet, params)
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
    return F.cosine_similarity(flat(g_task), flat(g_meet), dim=0).item()


@torch.no_grad()
def mid_state_variance(model, x):
    """Mean per-feature variance of h across the batch (collapse indicator)."""
    return model.low(x).var(dim=0).mean().item()
