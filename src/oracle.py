"""Oracle directions for the root-cause checks B and D (docs/notes/mechanism-root-cause.md, D-028, DR-019).

Not a method: these quantities are computed exactly from the network's own Jacobian, so their cost is meaningless and
is ignored. They answer one question: if the meeting signal had a PERFECT inverse, would it help (check B), or carry the
lower half alone (check D)?

Directions in the middle space h, per example, descent directions for h (as in exploratory/mechanism.py, check A):
  grad   J^T (e_y - p)                        what backprop sends to h (the pure rescale control F')
  pinv   (S J)^+ (e_y - p)                    a perfect local inverse; also the Gauss-Newton step (full-row-rank J)
  logit  J^T (J J^T + alpha I)^-1 (e_y - p)   damped least-squares inverse in logit space
with J = dz/dh (n_cls x d_mid) of model.up, S = diag(p) - p p^T, alpha_i = alpha_mult * 0.01 * mean eigenvalue of J_i J_i^T.

The injected signal is a linear surrogate loss  L = sum_i <c_i, h_i>  with c_i detached, so that its gradient at h_i is c_i:
  c_i = -rho * ramp * n_i * u_i / ||u_i||,   n_i = ||dL_task/dh_i||  (units of the mean loss)
i.e. the gradient at h_i has norm rho * ramp * n_i and points against u_i (a descent step moves h_i along u_i), exactly as
the dose arm `meet_rho` (check G0). Rows with p_y > SKIP_PY get c_i = 0 (their loss is numerically zero and the
pseudo-inverse is unstable), except for type `grad` without a bound, where c_i = rho * ramp * g_i is used as it is so that
the exact-gradient replacement is arithmetically identical to backpropagation (tests/test_oracle.py).

Bounded scale (`oracle_bound: radial`): the radial component of u_i along h_i is removed before normalising, so the
injected step cannot stretch or shrink h_i to first order. This is a first-order, not a strict, bound; h_var is logged to
check it (G0 showed the unbounded linear term blowing up h in ReLU, F-014).
"""
import torch
import torch.nn.functional as F

SKIP_PY = 0.9999          # as exploratory/mechanism.py and meet_rho
ALPHA_REL = 0.01          # damping of the logit-space inverse (check B note)
KINDS = ("grad", "pinv", "logit")
BOUNDS = ("none", "radial")


def _act_derivative(module, a):
    if isinstance(module, torch.nn.ReLU):
        return (a > 0).to(a.dtype)
    if isinstance(module, torch.nn.Tanh):
        return 1.0 - torch.tanh(a) ** 2
    if isinstance(module, torch.nn.Identity):
        return torch.ones_like(a)
    raise ValueError(f"unsupported activation {type(module).__name__}")


def up_jacobian(model, h):
    """Closed-form dz/dh of model.up = [act, Linear(d_mid, d_hid), act, Linear(d_hid, n_cls)], shape [B, n_cls, d_mid],
    in the dtype of h."""
    act1, lin1, act2, lin2 = model.up
    w1, b1, w2 = lin1.weight.to(h.dtype), lin1.bias.to(h.dtype), lin2.weight.to(h.dtype)
    s1 = _act_derivative(act1, h)
    z1 = F.linear(act1(h), w1, b1)
    s2 = _act_derivative(act2, z1)
    return torch.einsum("ke,be,ed,bd->bkd", w2, s2, w1, s1)


def up_logits(model, h):
    """Logits of model.up at h, in the dtype of h (the weights are cast)."""
    act1, lin1, act2, lin2 = model.up
    z1 = F.linear(act1(h), lin1.weight.to(h.dtype), lin1.bias.to(h.dtype))
    return F.linear(act2(z1), lin2.weight.to(h.dtype), lin2.bias.to(h.dtype))


def direction(model, h, p, e_y, kind, alpha_mult=1.0):
    """Unnormalised per-row direction u [B, d_mid] (descent direction for h) of the given kind, computed in float64 and
    returned in the dtype of h. Rows with p_y > SKIP_PY (judged on the given p) are returned as zeros. No gradient.
    The probabilities are RECOMPUTED in float64 from h: float32 probabilities do not sum to 1 closely enough, which
    perturbs the null direction of S = diag(p) - p p^T and makes the pseudo-inverse of S J blow up (found by comparing
    with exploratory/mechanism.py, which works in float64)."""
    if kind not in KINDS:
        raise ValueError(f"unknown oracle_type {kind!r}; choose from {KINDS}")
    with torch.no_grad():
        out = torch.zeros_like(h)
        keep = (p * e_y).sum(1) <= SKIP_PY
        if not bool(keep.any()):
            return out
        h64, e64 = h[keep].double(), e_y[keep].double()
        p64 = up_logits(model, h64).softmax(-1)
        delta = e64 - p64
        J = up_jacobian(model, h64)
        if kind == "grad":
            u = torch.einsum("bkd,bk->bd", J, delta)
        elif kind == "pinv":
            S = torch.diag_embed(p64) - p64.unsqueeze(2) * p64.unsqueeze(1)
            u = torch.einsum("bdk,bk->bd", torch.linalg.pinv(S @ J), delta)
        else:
            JJt = J @ J.transpose(1, 2)
            alpha = alpha_mult * ALPHA_REL * JJt.diagonal(dim1=1, dim2=2).mean(-1)
            eye = torch.eye(p64.shape[1], dtype=p64.dtype, device=p64.device)
            w = torch.linalg.solve(JJt + alpha[:, None, None] * eye, delta.unsqueeze(-1)).squeeze(-1)
            u = torch.einsum("bkd,bk->bd", J, w)
        out[keep] = u.to(h.dtype)
        return out


def _unit(u):
    return u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)


def remove_radial(v, h):
    """v with its component along h removed, per row (first-order bound of the scale of h). Zero rows of v stay zero."""
    h_hat = _unit(h)
    return v - (v * h_hat).sum(1, keepdim=True) * h_hat


def ramp(cfg, step):
    """Linear ramp of the injected signal over `oracle_warmup_steps` (default 0: full strength from the first step; an
    exact oracle has no inverse to learn)."""
    w = cfg.get("oracle_warmup_steps", 0)
    return 1.0 if w <= 0 else min(1.0, step / w)


def coefficients(model, h, p, e_y, g_h, cfg, step):
    """c [B, d_mid] (detached): the gradient the surrogate loss sends to h. `g_h` is the task gradient at h (units of the
    mean loss). Required config keys: oracle_type, oracle_rho; optional: oracle_warmup_steps, oracle_alpha_mult,
    oracle_bound."""
    kind, rho = cfg["oracle_type"], float(cfg["oracle_rho"])
    bound = cfg.get("oracle_bound", "none")
    if bound not in BOUNDS:
        raise ValueError(f"unknown oracle_bound {bound!r}; choose from {BOUNDS}")
    scale = rho * ramp(cfg, step)
    g_h = g_h.detach()
    if kind == "grad" and bound == "none":
        return scale * g_h                                    # exact: total gradient at h is (1 + rho) times backprop's
    u = direction(model, h.detach(), p, e_y, kind, cfg.get("oracle_alpha_mult", 1.0))
    u_hat = _unit(u)
    if bound == "radial":
        h_hat = _unit(h.detach())
        u_hat = _unit(u_hat - (u_hat * h_hat).sum(1, keepdim=True) * h_hat)
        u_hat = u_hat * (u.norm(dim=1, keepdim=True) > 0)    # rows that were skipped stay zero
    n = g_h.norm(dim=1, keepdim=True)
    return -scale * n * u_hat


def surrogate(h, c):
    """Linear surrogate loss whose gradient at h is c (c detached)."""
    return (c.detach() * h).sum()
