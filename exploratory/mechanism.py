"""Direction geometry (check A) and the linking measurement (check L) of the meeting signal.

Plans, written BEFORE the first run: docs/notes/mechanism-a-direction-geometry.md and
docs/notes/linking-measurement-concentration.md. Exploratory (D-015). Nothing here touches the global
random state or leaves gradients behind, so measuring never changes the training trajectory.

Directions in the middle space h (128 dimensions), per example, descent directions for h:
  bp    J^T (e_y - p)                      what backprop sends to h (equals -J^T r with r = p - e_y)
  meet  G(e_y) - G(p)                      what the meeting loss pushes h by (arm D's learned G)
  pinv  (S J)^+ (e_y - p)                  a perfect local inverse (min-norm change of h that moves p to e_y)
  lsq   J^T (J J^T + alpha I)^-1 (e_y - p) damped least-squares inverse in logit space
with J = dz/dh (n_cls x d_mid) and S = diag(p) - p p^T.
Fact used (checked numerically and by a test): the Gauss-Newton step (J^T S J)^+ J^T (e_y - p) equals `pinv`
whenever J has full row rank (generic here: 128 -> 10), because both are the minimum-norm solution in the
row space of J of S J J^T w = e_y - p. So there is ONE curvature direction, `pinv`; `lsq` is the distinct one.
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jacrev, vmap

from exploratory.gradconc import per_example_grads, subset_metrics
from src.model import MeetNet, compute_losses
from src.train import build_optimizer, evaluate
from src.utils import set_seed

# ---- decision rules (docs/notes/mechanism-a-direction-geometry.md and linking-measurement-concentration.md) ----
RULE_STEPS = (500, 1000, 2000)
Q1_COS, Q1_RATIO = 0.6, 0.1            # REDUNDANT: cos(g_task, g_meet) >= 0.6 and |dL(perp)| <= 0.1 |dL(task)|
Q2_MARGIN = 0.10                       # YES: cos(meet, curvature) >= cos(meet, bp) + 0.10 at steps 500 and 1000
Q3_SAME, Q3_DIFF = 0.9, 0.8            # BACKPROP ~ GN: cos(bp, pinv) >= 0.9; GN DIFFERS: <= 0.8 (all three steps)
L_SHARE_HI, L_CORR, L_SHARE_LO = 0.90, 0.5, 0.60

PRIMARY_FRACTION = 0.3                 # S30 selection rule
SKIP_PY = 0.9999                       # rows with p_y above this have numerically zero loss
ALPHA_REL = 0.01                       # lsq damping: alpha_i = 0.01 * mean eigenvalue of J_i J_i^T
STEP_REL = 0.01                        # one-step probe length: 0.01 * ||theta_low||
PAIRS = (("meet", "bp"), ("meet", "pinv"), ("meet", "lsq"), ("bp", "pinv"), ("bp", "lsq"), ("pinv", "lsq"))


# ------------------------------------------------------------------ Jacobian of the top half
def _act_derivative(module, a):
    if isinstance(module, torch.nn.ReLU):
        return (a > 0).to(a.dtype)
    if isinstance(module, torch.nn.Tanh):
        return 1.0 - torch.tanh(a) ** 2
    if isinstance(module, torch.nn.Identity):      # linear top half (used in tests)
        return torch.ones_like(a)
    raise ValueError(f"unsupported activation {type(module).__name__}")


def up_jacobian(model, h):
    """Closed-form dz/dh of model.up = [act, Linear(d_mid, d_hid), act, Linear(d_hid, n_cls)], shape [B, n_cls, d_mid]."""
    act1, lin1, act2, lin2 = model.up
    s1 = _act_derivative(act1, h)
    z1 = F.linear(act1(h), lin1.weight, lin1.bias)
    s2 = _act_derivative(act2, z1)
    return torch.einsum("ke,be,ed,bd->bkd", lin2.weight, s2, lin1.weight, s1)


def up_jacobian_autograd(model, h):
    """Same Jacobian by automatic differentiation (used to test the closed form)."""
    return vmap(jacrev(model.up))(h)


def softmax_hessian(p):
    """S = diag(p) - p p^T per row, shape [B, K, K]."""
    return torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)


# ------------------------------------------------------------------ Part 1: directions in h-space
def directions(model64, x, y, with_meet):
    """Per-row directions (float64) for the rows with p_y <= SKIP_PY. `model64` is a float64 copy of the model.
    Returns (dict of [n, d_mid] tensors, n_kept, n_total); the dict is empty if no row is kept."""
    with torch.no_grad():
        h = model64.low(x)
        p = model64.up(h).softmax(-1)
        n_total = x.shape[0]
        keep = p.gather(1, y.unsqueeze(1)).squeeze(1) <= SKIP_PY
        n_kept = int(keep.sum())
        if n_kept == 0:
            return {}, 0, n_total
        h, p, y = h[keep], p[keep], y[keep]
        ey = F.one_hot(y, p.shape[1]).to(p.dtype)
        delta = ey - p
        J = up_jacobian(model64, h)
        S = softmax_hessian(p)
        d = {"bp": torch.einsum("bkd,bk->bd", J, delta),
             "pinv": torch.einsum("bdk,bk->bd", torch.linalg.pinv(S @ J), delta)}
        JJt = J @ J.transpose(1, 2)
        alpha = ALPHA_REL * JJt.diagonal(dim1=1, dim2=2).mean(-1)
        eye = torch.eye(p.shape[1], dtype=p.dtype)
        w = torch.linalg.solve(JJt + alpha[:, None, None] * eye, delta.unsqueeze(-1)).squeeze(-1)
        d["lsq"] = torch.einsum("bkd,bk->bd", J, w)
        if with_meet:
            d["meet"] = model64.G(ey) - model64.G(p)
    return d, n_kept, n_total


def row_cos(a, b):
    return F.cosine_similarity(a, b, dim=1, eps=1e-12)


def weighted_mean(v, w):
    return float((v * w).sum() / (w.sum() + 1e-300))


def geometry_h_space(model64, x, y, with_meet):
    """Cosines between the directions (weighted by ||bp|| = the rows that matter for the update, and unweighted),
    the norm ratio ||meet|| / ||bp|| (ratio of sums) and the share of rows skipped."""
    d, n_kept, n_total = directions(model64, x, y, with_meet)
    out = {"skipped_share": 1.0 - n_kept / n_total}
    if n_kept == 0:
        return out
    w = d["bp"].norm(dim=1)
    for a, b in PAIRS:
        if a not in d or b not in d:
            continue
        c = row_cos(d[a], d[b])
        out[f"cos_{a}_{b}_w"] = weighted_mean(c, w)
        out[f"cos_{a}_{b}_u"] = float(c.mean())
    if with_meet:
        out["norm_ratio"] = float(d["meet"].norm(dim=1).sum() / (d["bp"].norm(dim=1).sum() + 1e-300))
    return out


# ------------------------------------------------------------------ Part 2: parameter space (F_low)
def split_parallel(g_task, g_meet):
    """(parallel part, orthogonal part) of g_meet with respect to g_task."""
    t = g_task / (g_task.norm() + 1e-30)
    par = (g_meet @ t) * t
    return par, g_meet - par


def _flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def _set_flat(params, vec):
    o = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(vec[o:o + n].view_as(p))
            o += n


def geometry_param_space(model, cfg, x, y, probe):
    """Gradients of the task loss and of the (unscaled) meeting loss on the lower layers, their parallel and
    orthogonal parts, and a one-step probe on a separate batch `probe` = (xp, yp): the change of the task loss
    after moving F_low by STEP_REL * ||theta_low|| along -g_task, -g_meet and -g_perp. Parameters are restored
    exactly."""
    params = list(model.low.parameters())
    _, parts, _, _ = compute_losses(model, x, y, cfg, step=10 ** 9)
    g_task = _flat(torch.autograd.grad(parts["task"], params, retain_graph=True))
    g_meet = _flat(torch.autograd.grad(parts["meet"], params, retain_graph=True))
    par, perp = split_parallel(g_task, g_meet)
    out = {"cos_task_meet": float(F.cosine_similarity(g_task, g_meet, dim=0, eps=1e-12)),
           "perp_ratio": float(perp.norm() / (g_task.norm() + 1e-30)),
           "meet_ratio": float(g_meet.norm() / (g_task.norm() + 1e-30))}
    out["par_share"] = out["cos_task_meet"] ** 2
    xp, yp = probe
    saved = [p.detach().clone() for p in params]
    theta = torch.cat([p.detach().reshape(-1) for p in params])
    step_len = STEP_REL * theta.norm()
    with torch.no_grad():
        base = F.cross_entropy(model(xp), yp).item()
        for name, g in (("task", g_task), ("meet", g_meet), ("perp", perp)):
            u = -g / (g.norm() + 1e-30)
            _set_flat(params, theta + step_len * u)
            out[f"dl_{name}"] = F.cross_entropy(model(xp), yp).item() - base
        for p, s in zip(params, saved):
            p.copy_(s)
    return out


# ------------------------------------------------------------------ Check L: where does the meeting gradient live?
def per_example_low_meet_grads(model, x, y):
    """Per-example gradients of l_i = mean((h_i - b_i)^2) with respect to the lower-layer parameters, shape [B, P_low].
    b is the constant meeting target h + G(e_y) - G(p). The mean over examples of l_i is F.mse_loss(h, b)."""
    with torch.no_grad():
        h = model.low(x)
        p = model.up(h).softmax(-1)
        ey = F.one_hot(y, p.shape[1]).float()
        b = h + model.G(ey) - model.G(p)
    params = {k: v.detach() for k, v in model.low.named_parameters()}

    def loss_fn(pr, xi, bi):
        hi = functional_call(model.low, pr, (xi.unsqueeze(0),)).squeeze(0)
        return ((hi - bi) ** 2).mean()

    g = vmap(grad(loss_fn), in_dims=(None, 0, 0))(params, x, b)
    return torch.cat([g[k].reshape(x.shape[0], -1) for k in params], dim=1)


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm() + 1e-30))


def link_metrics(model, x, y, with_meet, fraction=PRIMARY_FRACTION):
    """Concentration of the task gradient (all parameters) and, if `with_meet`, of the meeting gradient
    (lower-layer parameters) in the top-`fraction` rows by loss (E-009 definitions, exploratory/gradconc.py),
    the row-level coupling with the loss, and the overlap of the two gradients on the unselected rows."""
    flat, losses = per_example_grads(model, x, y)
    B = x.shape[0]
    k = min(B, max(1, math.ceil(fraction * B)))
    order = torch.argsort(losses, descending=True)
    idx, rest = order[:k], order[k:]
    m = subset_metrics(flat, losses, idx)
    out = {"task_norm_share": m["norm_share"], "task_cos": m["cos"], "task_omitted": m["omitted"]}
    with torch.no_grad():
        py = model(x).softmax(-1).gather(1, y.unsqueeze(1)).squeeze(1)
    out["confident_share"] = float((py > SKIP_PY).float().mean())
    if with_meet:
        n_low = sum(p.numel() for p in model.low.parameters())      # F_low parameters come first in `flat`
        mflat = per_example_low_meet_grads(model, x, y)
        mm = subset_metrics(mflat, losses, idx)
        out.update(meet_norm_share=mm["norm_share"], meet_cos=mm["cos"], meet_omitted=mm["omitted"])
        if len(rest) > 0:
            out["overlap_cos_unselected"] = float(F.cosine_similarity(
                mflat[rest].sum(0), flat[rest, :n_low].sum(0), dim=0, eps=1e-12))
        with torch.no_grad():
            h = model.low(x)
            p = model.up(h).softmax(-1)
            ey = F.one_hot(y, p.shape[1]).float()
            u_meet = (model.G(ey) - model.G(p)).norm(dim=1)
            g_h = up_jacobian(model, h)
            u_bp = torch.einsum("bkd,bk->bd", g_h, ey - p).norm(dim=1)
        out["corr_loss_umeet"] = spearman(losses, u_meet)
        out["corr_loss_gbp"] = spearman(losses, u_bp)
    return out


# ------------------------------------------------------------------ one checkpoint, averaged over the pool
def _average(dicts):
    keys = {k for d in dicts for k in d}
    out = {}
    for k in sorted(keys):
        v = [d[k] for d in dicts if k in d and not math.isnan(d[k])]
        out[k] = sum(v) / len(v) if v else float("nan")
    return out


def measure_checkpoint(model, cfg, pool, probe, device="cpu"):
    """Average the quantities of checks A and L over the fixed pool of training batches. Geometry is only
    computed for the meeting arm; the link metrics for the task gradient are computed for every arm."""
    with_meet = cfg["arm"] == "meet"
    model64 = copy.deepcopy(model).double()
    probe = (probe[0].to(device), probe[1].to(device))
    per_batch = []
    for x, y in pool:
        x, y = x.to(device), y.to(device)
        d = {}
        d.update(link_metrics(model, x, y, with_meet))
        if with_meet:
            d.update(geometry_h_space(model64, x.double(), y, True))
            d.update(geometry_param_space(model, cfg, x, y, probe))
        per_batch.append(d)
    return _average(per_batch)


def run_measured(cfg, train_loader, val_loader, pool, probe, checkpoints, device="cpu", measure=True):
    """Train one run exactly like src.train.run and, at the checkpoints, measure. Returns (rows, model); a row is
    (seed, arm, step, quantity, value). Measuring never changes the training."""
    set_seed(cfg["seed"])
    model = MeetNet(arm=cfg["arm"], act=cfg.get("act", "relu")).to(device)
    opt, sched = build_optimizer(model, cfg)
    # src.train.run draws one batch before training (to count FLOPs); that consumes the loader's random
    # generator, so do the same to follow exactly the same data order.
    next(iter(train_loader))
    cps = set(checkpoints)
    rows, step = [], 0
    while step < cfg["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss, _, _, _ = compute_losses(model, x, y, cfg, step)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step in cps:
                rows.append((cfg["seed"], cfg["arm"], step, "val_acc", evaluate(model, val_loader, device)))
                if measure:
                    with torch.random.fork_rng(devices=[]):
                        q = measure_checkpoint(model, cfg, pool, probe, device)
                    rows.extend((cfg["seed"], cfg["arm"], step, k, v) for k, v in q.items())
            if step >= cfg["max_steps"]:
                break
    return rows, model


# ------------------------------------------------------------------ decision rules
def _vals(d, steps):
    return [d.get(s, float("nan")) for s in steps]


def _missing(v):
    return any(math.isnan(x) for x in v)


def decide_q1(cos_by_step, dl_perp_by_step, dl_task_by_step, steps=RULE_STEPS):
    """Q1 (redundancy of the signal in parameter space). REDUNDANT if cos(g_task, g_meet) >= 0.6 at all three steps
    and |dL(-g_perp)| <= 0.1 |dL(-g_task)| at all three; NOT REDUNDANT if the second condition fails at two of the three."""
    c, p, t = _vals(cos_by_step, steps), _vals(dl_perp_by_step, steps), _vals(dl_task_by_step, steps)
    if _missing(c + p + t):
        return "N/A (rule steps missing)"
    small = [abs(pi) <= Q1_RATIO * abs(ti) for pi, ti in zip(p, t)]
    if all(ci >= Q1_COS for ci in c) and all(small):
        return "REDUNDANT"
    if sum(not s for s in small) >= 2:
        return "NOT REDUNDANT"
    return "INCONCLUSIVE"


def decide_q2(cos_meet_curv_by_step, cos_meet_bp_by_step, steps=(500, 1000)):
    """Q2 (does G move toward the curvature direction?). YES if cos(meet, curv) >= cos(meet, bp) + 0.10 at steps 500 and 1000;
    NO if cos(meet, curv) < cos(meet, bp) at both; otherwise UNCLEAR."""
    a, b = _vals(cos_meet_curv_by_step, steps), _vals(cos_meet_bp_by_step, steps)
    if _missing(a + b):
        return "N/A (rule steps missing)"
    if all(x >= y + Q2_MARGIN for x, y in zip(a, b)):
        return "YES"
    if all(x < y for x, y in zip(a, b)):
        return "NO"
    return "UNCLEAR"


def decide_q3(cos_bp_curv_by_step, steps=RULE_STEPS):
    """Q3 (does curvature matter for this network?). BACKPROP ~ GN if cos(bp, pinv) >= 0.9 at all three steps;
    GN DIFFERS if <= 0.8 at all three; otherwise GREY."""
    v = _vals(cos_bp_curv_by_step, steps)
    if _missing(v):
        return "N/A (rule steps missing)"
    if all(x >= Q3_SAME for x in v):
        return "BACKPROP ~ GN"
    if all(x <= Q3_DIFF for x in v):
        return "GN DIFFERS"
    return "GREY"


def decide_link(share_by_step, corr_by_step, steps=RULE_STEPS):
    """Check L. ALREADY SELECTIVE if the meeting gradient's norm share in the selected rows is >= 0.90 and the loss correlation
    of ||G(e_y) - G(p)|| is >= 0.5 at all three steps; NOT SELECTIVE if the share is <= 0.60 at two of the three; else PARTIAL."""
    s, c = _vals(share_by_step, steps), _vals(corr_by_step, steps)
    if _missing(s + c):
        return "N/A (rule steps missing)"
    if all(x >= L_SHARE_HI for x in s) and all(x >= L_CORR for x in c):
        return "ALREADY SELECTIVE"
    if sum(x <= L_SHARE_LO for x in s) >= 2:
        return "NOT SELECTIVE"
    return "PARTIAL"
