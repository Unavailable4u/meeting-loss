"""Model and losses for the meeting-loss project (method v1.1, see docs/METHOD.md).

Arms (docs/DECISIONS.md D-013):
  base         A  plain network
  deepsup      B  DSN-style deep supervision: companion CE head on the middle state
  ae           C  autoencoder-reconstruction auxiliary on the middle state
  meet         D  meeting loss (SDTP-style target at the middle layer), default configuration
  meet_ao      V2 meeting loss, augmented output: G also sees fixed random features of h
  meet_frozen  V3 meeting loss with a frozen random G and no inverse loss
  meet_grad    F  control: DTP-style exact-gradient target  h - eta * dL/dh
  meet_rho        check G0 (D-020, docs/notes/mechanism-dose-response.md): learned G as in arm D, but the meeting
                  signal's strength is set directly as rho times the task gradient at h (not a method: a dose)
Variant V1 (noise in the inverse loss) is the config key `sigma_inv > 0` on meet / meet_ao.
Control E (lower-layer learning-rate multiplier) is the config key `low_lr_mult` (see src/train.py).

If this file and docs/METHOD.md disagree, that is a drift event (docs/DRIFT_LOG.md).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MEET_ARMS = ("meet", "meet_ao", "meet_frozen")
ARMS = ("base", "deepsup", "ae") + MEET_ARMS + ("meet_grad", "selective", "meet_rho")


def _act(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


class MeetNet(nn.Module):
    """MLP split at a middle layer, with extra modules per experimental arm."""

    def __init__(self, d_in=784, d_hid=256, d_mid=128, n_cls=10, arm="meet", act="relu", d_aux=128):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.arm, self.n_cls, self.d_aux = arm, n_cls, d_aux
        self.low = nn.Sequential(nn.Linear(d_in, d_hid), _act(act), nn.Linear(d_hid, d_mid))
        self.up = nn.Sequential(_act(act), nn.Linear(d_mid, d_hid), _act(act), nn.Linear(d_hid, n_cls))
        if arm in MEET_ARMS:
            g_in = n_cls + (d_aux if arm == "meet_ao" else 0)
            self.G = nn.Sequential(nn.Linear(g_in, d_hid), _act(act), nn.Linear(d_hid, d_mid))
            if arm == "meet_ao":
                # fixed random features of h (not trained, not a parameter): Bartunov et al. 2018, AO-SDTP
                self.register_buffer("R", torch.randn(d_mid, d_aux) / d_mid ** 0.5)
            if arm == "meet_frozen":
                self.G.requires_grad_(False)  # frozen random feedback network (no inverse loss)
        elif arm == "meet_rho":
            # same G as arm D (created after low and up, so the initial weights of low and up equal arm A's for a given seed)
            self.G = nn.Sequential(nn.Linear(n_cls, d_hid), _act(act), nn.Linear(d_hid, d_mid))
        elif arm == "deepsup":
            self.aux = nn.Linear(d_mid, n_cls)
        elif arm == "ae":
            self.dec = nn.Sequential(nn.Linear(d_mid, d_hid), _act(act), nn.Linear(d_hid, d_in))

    def forward(self, x):
        return self.up(self.low(x))


def _g_in(model, p, h_ref):
    """Input to the backward network G: class probabilities (or one-hot labels), plus
    fixed random features of h for the augmented-output variant."""
    if model.arm == "meet_ao":
        return torch.cat([p, torch.tanh(h_ref @ model.R)], dim=-1)
    return p


def _aux_weight(cfg, step):
    """Weight of the auxiliary loss at this step (arms B and C).
    aux_sched: "const" (default) or "linear" (decays to zero over max_steps, as in the DSN paper)."""
    base = cfg.get("lam_aux", 0.0)
    if cfg.get("aux_sched", "const") == "linear":
        return base * max(0.0, 1.0 - step / max(1, cfg.get("max_steps", 1)))
    return base


def compute_losses(model, x, y, cfg, step):
    """Returns (total_loss, dict_of_parts, h, z).

    cfg keys used: lam_inv, lam_meet, warmup_steps (meet arms); lam_aux, aux_sched, aux_gamma,
    max_steps (arms B, C); sigma_inv (V1); eta_hat (meet_grad); rho (required) and rho_warmup_steps
    (default 200), lam_inv (meet_rho). Missing optional keys default
    to the values in docs/METHOD.md.
    """
    h = model.low(x)
    z = model.up(h)
    parts = {"task": F.cross_entropy(z, y)}
    total = parts["task"]
    ramp = min(1.0, step / max(1, cfg.get("warmup_steps", 1)))

    if model.arm in MEET_ARMS:
        h_d = h.detach()
        p = z.detach().softmax(-1)
        e_y = F.one_hot(y, model.n_cls).float()
        sigma = cfg.get("sigma_inv", 0.0)

        if model.arm == "meet_frozen":
            with torch.no_grad():
                b = h_d + model.G(_g_in(model, e_y, h_d)) - model.G(_g_in(model, p, h_d))
            L_inv = torch.zeros((), device=h.device)
            lam_inv = 0.0
        elif sigma == 0.0:
            gp = model.G(_g_in(model, p, h_d))           # trained via L_inv
            L_inv = F.mse_loss(gp, h_d)
            with torch.no_grad():                        # the target is a constant
                b = h_d + model.G(_g_in(model, e_y, h_d)) - gp.detach()
            lam_inv = cfg["lam_inv"]
        else:                                            # V1: denoising inverse (DTP/SDTP)
            with torch.no_grad():
                b = h_d + model.G(_g_in(model, e_y, h_d)) - model.G(_g_in(model, p, h_d))
                h_t = h_d + sigma * torch.randn_like(h_d)
                p_t = model.up(h_t).softmax(-1)
            L_inv = F.mse_loss(model.G(_g_in(model, p_t, h_t)), h_t)
            lam_inv = cfg["lam_inv"]

        L_meet = F.mse_loss(h, b)                        # gradient reaches F_low only
        total = total + lam_inv * L_inv + cfg["lam_meet"] * ramp * L_meet
        parts.update(inv=L_inv, meet=L_meet)

    elif model.arm == "meet_rho":
        # Check G0: the learned direction u_i = G(e_y) - G(p) (as in arm D), with the strength set explicitly.
        #   L_meet = -rho * ramp * sum_i n_i * <u_i / ||u_i||, h_i>,   n_i = ||dL_task/dh_i|| (units of the mean loss)
        # so the gradient at h_i has norm rho * ramp * n_i and points against u_i (descent moves h along u_i).
        # u_i and n_i are constants; rows with p_y > 0.9999 get u_i = 0. G's only loss is L_inv (as in arm D).
        ramp_rho = min(1.0, step / max(1, cfg.get("rho_warmup_steps", 200)))
        h_d = h.detach()
        p = z.detach().softmax(-1)
        e_y = F.one_hot(y, model.n_cls).float()
        gp = model.G(p)                                  # trained via L_inv
        L_inv = F.mse_loss(gp, h_d)
        with torch.no_grad():
            u = model.G(e_y) - gp.detach()
            u = torch.where(p.gather(1, y[:, None]) > 0.9999, torch.zeros_like(u), u)
            u_hat = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        g_h = torch.autograd.grad(parts["task"], h, retain_graph=True)[0]
        n_i = g_h.norm(dim=1, keepdim=True).detach()
        L_meet = -(cfg["rho"] * ramp_rho) * (n_i * u_hat * h).sum()
        total = total + cfg["lam_inv"] * L_inv + L_meet
        parts.update(inv=L_inv, meet=L_meet)

    elif model.arm == "meet_grad":
        # Control F: DTP-style target built from the exact gradient (per-sample units).
        g_h = torch.autograd.grad(parts["task"], h, retain_graph=True)[0]
        b = (h - cfg.get("eta_hat", 0.1) * h.shape[0] * g_h).detach()
        L_meet = F.mse_loss(h, b)
        total = total + cfg["lam_meet"] * ramp * L_meet
        parts["meet"] = L_meet

    elif model.arm == "deepsup":
        L_aux = F.cross_entropy(model.aux(h), y)
        gamma = cfg.get("aux_gamma", 0.0)
        L_eff = F.relu(L_aux - gamma) if gamma > 0 else L_aux   # companion loss vanishes below gamma
        total = total + _aux_weight(cfg, step) * L_eff
        parts["aux"] = L_aux

    elif model.arm == "ae":
        L_aux = F.mse_loss(model.dec(h), x)
        total = total + _aux_weight(cfg, step) * L_aux
        parts["aux"] = L_aux

    return total, parts, h, z
