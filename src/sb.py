"""Selective-Backprop (arm "sb"), the reference arm of the Improve track (D-025).

Follows Jiang et al. 2019 and the released code (github.com/angelajiang/SelectiveBackprop, commit 7c41937;
docs/notes/improve-reference-arms.md):

* every incoming batch of `batch_size` rows is forwarded; each row's loss goes into a history of the last `sb_history`
  (1024) forward losses; the row is kept independently with probability (share of history entries strictly below its
  loss)^sb_beta (floor `sb_sampling_min`, default 0). The history is updated with the batch before the percentiles are
  taken, as in the released code. With uniform percentiles the expected kept share is 1/(beta + 1).
* kept rows wait in a queue; as soon as the queue holds `batch_size` rows one update is made on the first `batch_size`
  of them (at most one update per forward batch, as in the released code). The update is an ordinary step of arm A on
  that batch (same loss, optimizer and schedule): the released code runs a SECOND forward pass on the kept rows, and so
  do we.
* the first `sb_prime_steps` updates (default: one epoch = len(train_loader), as the released code primes with one epoch
  of images) are ordinary steps of arm A on every incoming batch. The released code also runs a redundant second forward
  pass during this phase; we charge arm A's normal step cost so that the arm is not penalised for that artefact. These
  steps are bit-identical to arm A's (tests/test_sb.py).

FLOPs (src/flops.py counter, matrix multiplications): every forward batch of the selection phase costs one forward pass
of `batch_size` rows; every update costs one ordinary step of arm A (forward + backward). `flops` is that, the cost of
the released algorithm ("uncached"). `flops_cached` charges only the backward part of the update, i.e. what an
implementation that reuses the selection forward pass would cost; the gradient is still computed with a fresh forward
pass (identical to a cache except for rows queued across an update), so both counts come from the same run.

Step counts are UPDATES, as for every other arm: `eval_every`, `max_steps` and the cosine schedule count updates.
Config keys: sb_beta (2.0), sb_history (1024), sb_prime_steps (one epoch), sb_sampling_min (0.0),
sb_max_forward_batches (safety stop, default 200 * max_steps).
"""
import time

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from src.flops import flops_per_step
from src.model import MeetNet, compute_losses
from src.train import build_optimizer, evaluate
from src.utils import set_seed


class LossHistory:
    """The last `size` losses (a ring buffer, the same set as a deque with maxlen) and strict-percentile lookups."""

    def __init__(self, size=1024):
        self.size, self.buf, self.n, self.pos = size, torch.zeros(size), 0, 0

    def extend(self, losses):
        m = losses.shape[0]
        if m > self.size:
            losses, m = losses[-self.size:], self.size
        idx = (self.pos + torch.arange(m)) % self.size
        self.buf[idx] = losses.detach().float().cpu()
        self.pos = (self.pos + m) % self.size
        self.n = min(self.size, self.n + m)

    def percentile(self, losses):
        """Share (0 to 1) of the history strictly below each loss."""
        h = self.buf[: self.n]
        return (h[None, :] < losses.detach().float().cpu()[:, None]).float().mean(dim=1)


def keep_probability(percentile, beta, sampling_min=0.0):
    return (percentile ** beta).clamp(min=sampling_min)


def forward_flops(model, x):
    """FLOPs of one forward pass on x (no gradient)."""
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


def run_sb(cfg, device="cpu", data=None):
    """Train one Selective-Backprop run; returns the log (same row format as src.train.run, plus the extra keys
    flops_cached, n_fwd, rows_seen, sel_rate, queue)."""
    set_seed(cfg["seed"])
    if data is None:
        from src.data import get_data               # imported lazily: needs torchvision
        data = get_data(cfg["batch_size"], cfg["seed"])
    train_loader, val_loader = data
    model = MeetNet(arm="sb", act=cfg.get("act", "relu")).to(device)
    opt, sched = build_optimizer(model, cfg)

    beta, size = cfg.get("sb_beta", 2.0), cfg.get("sb_history", 1024)
    smin = cfg.get("sb_sampling_min", 0.0)
    prime = cfg.get("sb_prime_steps")
    prime = len(train_loader) if prime is None else int(prime)
    max_fwd = cfg.get("sb_max_forward_batches", 200 * cfg["max_steps"])

    bx, by = next(iter(train_loader))
    bx, by = bx.to(device), by.to(device)
    fps = flops_per_step(model, opt, (bx, by), cfg)          # an ordinary step of arm A on one batch
    f_fwd = forward_flops(model, bx)                        # one forward pass on one batch

    hist = LossHistory(size)
    gen = torch.Generator().manual_seed(1_000_003 + cfg["seed"])   # selection draws; independent of the data order
    qx, qy = [], []                                          # queue of kept rows
    qn = 0
    log, step, t0 = [], 0, time.time()
    cum, cum_cached = 0, 0
    n_fwd = rows_seen = sel_rows = sel_seen = 0

    while step < cfg["max_steps"]:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            n_fwd += 1
            rows_seen += x.shape[0]
            if n_fwd > max_fwd:
                raise RuntimeError(f"sb: {n_fwd} forward batches for {step} updates; selection is too strict (sb_beta {beta})")
            updated = False
            if step < prime:                                 # priming: ordinary step of arm A (bit-identical)
                opt.zero_grad(set_to_none=True)
                loss, parts, _, _ = compute_losses(model, x, y, cfg, step)
                loss.backward()
                opt.step()
                sched.step()
                cum += fps
                cum_cached += fps
                step += 1
                updated = True
                last = parts["task"]
            else:
                with torch.no_grad():
                    losses = F.cross_entropy(model(x), y, reduction="none")
                cum += f_fwd
                cum_cached += f_fwd
                hist.extend(losses)
                p = keep_probability(hist.percentile(losses), beta, smin)
                keep = torch.rand(p.shape[0], generator=gen) < p
                sel_seen += keep.shape[0]
                sel_rows += int(keep.sum())
                if keep.any():
                    kd = keep.to(x.device)
                    qx.append(x[kd]); qy.append(y[kd])
                    qn += int(keep.sum())
                if qn >= x.shape[0]:
                    B = x.shape[0]
                    xs, ys = torch.cat(qx), torch.cat(qy)
                    xb, yb, qx, qy, qn = xs[:B], ys[:B], [xs[B:]], [ys[B:]], xs.shape[0] - B
                    opt.zero_grad(set_to_none=True)
                    loss, parts, _, _ = compute_losses(model, xb, yb, cfg, step)
                    loss.backward()
                    opt.step()
                    sched.step()
                    cum += fps
                    cum_cached += fps - f_fwd
                    step += 1
                    updated = True
                    last = parts["task"]
            if updated and step % cfg["eval_every"] == 0:
                log.append(dict(step=step, flops=cum, flops_cached=cum_cached, time=time.time() - t0,
                                val_acc=evaluate(model, val_loader, device), task=last.item(),
                                n_fwd=n_fwd, rows_seen=rows_seen,
                                sel_rate=(sel_rows / sel_seen) if sel_seen else 1.0, queue=qn))
            if step >= cfg["max_steps"]:
                break
    return log
