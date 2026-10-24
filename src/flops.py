"""FLOP counting per training step. UNTESTED starter code (see test in tests/)."""
from torch.utils.flop_counter import FlopCounterMode

from src.model import compute_losses


def flops_per_step(model, opt, batch, cfg, step=10**9):
    """FLOPs of one full training step (forward + backward) for this arm,
    including G or auxiliary heads. Optimizer FLOPs are ignored (small)."""
    x, y = batch
    opt.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        if model.arm == "selective":
            from src.selective import selective_step
            selective_step(model, x, y, cfg, step)
        else:
            loss, _, _, _ = compute_losses(model, x, y, cfg, step=step)
            loss.backward()
    opt.zero_grad(set_to_none=True)
    return fc.get_total_flops()
