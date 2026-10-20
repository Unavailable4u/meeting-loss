"""Model and losses for the meeting-loss project.

UNTESTED starter code: run `pytest` (Stage 2 of docs/PLAN.md) before trusting any result.
The method is defined in docs/METHOD.md. If this file and METHOD.md disagree, that is a
drift event (docs/DRIFT_LOG.md).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeetNet(nn.Module):
    """MLP split at a middle layer, with extra modules per experimental arm.

    arm: "base" (A), "deepsup" (B), "ae" (C), "meet" (D)
    """

    def __init__(self, d_in=784, d_hid=256, d_mid=128, n_cls=10, arm="meet"):
        super().__init__()
        self.arm, self.n_cls = arm, n_cls
        self.low = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_mid))
        self.up = nn.Sequential(nn.ReLU(), nn.Linear(d_mid, d_hid), nn.ReLU(), nn.Linear(d_hid, n_cls))
        if arm == "meet":
            self.G = nn.Sequential(nn.Linear(n_cls, d_hid), nn.ReLU(), nn.Linear(d_hid, d_mid))
        elif arm == "deepsup":
            self.aux = nn.Linear(d_mid, n_cls)
        elif arm == "ae":
            self.dec = nn.Sequential(nn.Linear(d_mid, d_hid), nn.ReLU(), nn.Linear(d_hid, d_in))
        elif arm != "base":
            raise ValueError(f"unknown arm: {arm}")

    def forward(self, x):
        return self.up(self.low(x))


def compute_losses(model, x, y, cfg, step):
    """Returns (total_loss, dict_of_parts, h, z).

    cfg keys: lam_inv, lam_meet, warmup_steps, lam_aux.
    """
    h = model.low(x)
    z = model.up(h)
    parts = {"task": F.cross_entropy(z, y)}
    total = parts["task"]

    if model.arm == "meet":
        h_d = h.detach()
        p = z.detach().softmax(-1)
        e_y = F.one_hot(y, model.n_cls).float()
        gp = model.G(p)                              # trained via L_inv
        L_inv = F.mse_loss(gp, h_d)
        with torch.no_grad():                        # the target is a constant
            b = h_d + model.G(e_y) - gp.detach()
        L_meet = F.mse_loss(h, b)                    # gradient reaches F_low only
        ramp = min(1.0, step / max(1, cfg["warmup_steps"]))
        total = total + cfg["lam_inv"] * L_inv + cfg["lam_meet"] * ramp * L_meet
        parts.update(inv=L_inv, meet=L_meet)

    elif model.arm == "deepsup":
        L_aux = F.cross_entropy(model.aux(h), y)
        total = total + cfg["lam_aux"] * L_aux
        parts["aux"] = L_aux

    elif model.arm == "ae":
        L_aux = F.mse_loss(model.dec(h), x)
        total = total + cfg["lam_aux"] * L_aux
        parts["aux"] = L_aux

    return total, parts, h, z
