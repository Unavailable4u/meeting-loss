"""Study 1b, the meeting NETWORK (docs/notes/meeting-network.md, D-030, DR-023). Exploratory (D-015).

The idea (not the auxiliary loss of arm D): a FORWARD chain goes from the input toward a state, a BACKWARD chain goes from the desired
output (a label code) toward the same state, and the two meet there. Nodes of the 4-layer MNIST MLP: dims (784, 256, 128, 256, 10).
Forward chain: layers 0..k-1 of the MLP (input -> node k). Backward chain: the mirrored layers from the label code to node k
(for k = 1 and a one-hot code: 10 -> 256 -> 128 -> 256). The meeting layer k is fixed BEFORE training by the equal-cost rule
(`equal_cost_k`): the k at which the cumulative multiply-adds of the forward part (layers 0..k-1) and of the mirrored backward part
(layers k..L-1) are closest. Nothing is measured on both chains at full depth.

Meeting loss: cosine logits between the forward state of an example and the backward state of each class, with a fixed scale
(`mn_scale`), cross-entropy over the classes. The other classes' backward states are the negatives that stop a collapse. Both chains
are trained jointly by this loss (backprop inside each half). The backward chain runs on the n_cls codes of a step only, not on every
example, so its cost per step is small and is counted (FlopCounterMode).

State: the post-activation output of the last layer of a chain (`mn_post_act`, default true). With the pre-activation state, the
equal-cost meeting point k = 1 would make the forward chain one linear map (about 92% accuracy on MNIST by construction).

Arms (config `arm`):
  mn_match  both chains, cosine meeting loss; the prediction is the nearest backward state (readout "match")
  mn_fwd    as mn_match plus the upper layers k..L-1 trained by cross-entropy on the DETACHED meeting state (readout "fwd");
            both readouts are logged (val_acc = fwd, val_acc_match = match, val_acc_fwd = fwd)
  mn_table  ablation of mn_match: the backward chain is replaced by a table of n_cls free vectors
  shallow   baseline: forward chain 0..k-1, activation, Linear(d_k, n_cls), ordinary cross-entropy (no backward chain)
Label codes (config `mn_code`): onehot (default); noise = one-hot plus `mn_noise_dim` Gaussian components (std `mn_sigma`), drawn once
per class per step and set to zero at test time; multihot = a fixed random +-1 code of MH_DIM components (same information as one-hot).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import _act

MN_ARMS = ("mn_match", "mn_fwd", "mn_table", "shallow")
DIMS = (784, 256, 128, 256, 10)
CODES = ("onehot", "noise", "multihot")
MH_DIM = 16
CODE_SEED = 20261127            # fixed: the multi-hot code is a constant of the study, not a random seed of a run
READOUTS = {"mn_match": ("match",), "mn_table": ("match",), "mn_fwd": ("fwd", "match"), "shallow": ("fwd",)}


def layer_macs(dims=DIMS):
    return [dims[i] * dims[i + 1] for i in range(len(dims) - 1)]


def equal_cost_k(dims=DIMS):
    """Meeting node k in 1..L-1 (L layers) that balances the cumulative multiply-adds of the forward part (layers 0..k-1) and of the
    mirrored backward part (layers k..L-1); ties go to the smaller k. For the 4-layer MNIST MLP: 200,704 / 68,096 at k=1."""
    m = layer_macs(dims)
    return min(range(1, len(m)), key=lambda k: (abs(sum(m[:k]) - sum(m[k:])), k))


def multihot_codes(n_cls=10, dim=MH_DIM):
    """Fixed random +-1 codes with distinct rows (deterministic)."""
    g = torch.Generator().manual_seed(CODE_SEED)
    while True:
        c = torch.randint(0, 2, (n_cls, dim), generator=g).float() * 2 - 1
        if len({tuple(r.tolist()) for r in c}) == n_cls:
            return c


class MeetingNet(nn.Module):
    def __init__(self, arm="mn_match", act="relu", k=None, code="onehot", post_act=True, noise_dim=8, sigma=0.5, dims=DIMS):
        super().__init__()
        if arm not in MN_ARMS:
            raise ValueError(f"unknown arm: {arm}")
        if code not in CODES:
            raise ValueError(f"unknown mn_code {code!r}; choose from {CODES}")
        L = len(dims) - 1
        k = equal_cost_k(dims) if k is None else int(k)
        if not 1 <= k <= L - 1:
            raise ValueError(f"meeting node k must be in 1..{L - 1}")
        self.arm, self.k, self.code, self.dims, self.L = arm, k, code, dims, L
        self.n_cls, self.noise_dim, self.sigma = dims[-1], noise_dim, sigma
        self.post_act = post_act or arm == "shallow"
        self.act = _act(act)
        self.fwd = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(k)])
        d_k = dims[k]
        if arm in ("mn_match", "mn_fwd"):
            d_code = {"onehot": self.n_cls, "noise": self.n_cls + noise_dim, "multihot": MH_DIM}[code]
            bd = [d_code] + list(dims[::-1][1:L - k + 1])
            self.back = nn.ModuleList([nn.Linear(bd[i], bd[i + 1]) for i in range(len(bd) - 1)])
            if code == "multihot":
                self.register_buffer("codes", multihot_codes(self.n_cls))
        elif arm == "mn_table":
            self.table = nn.Parameter(torch.randn(self.n_cls, d_k))
        if arm == "mn_fwd":
            layers = [] if self.post_act else [_act(act)]
            for i in range(k, L):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < L - 1:
                    layers.append(_act(act))
            self.head = nn.Sequential(*layers)
        elif arm == "shallow":
            self.head = nn.Linear(d_k, self.n_cls)
        self.readout = READOUTS[arm][0]
        self.readouts = READOUTS[arm]

    # ---- chains
    def state(self, x):
        """Forward state at the meeting node (post-activation unless mn_post_act is false)."""
        h = x
        for i, lin in enumerate(self.fwd):
            h = lin(h)
            if i < len(self.fwd) - 1 or self.post_act:
                h = self.act(h)
        return h

    def _codes(self, train):
        n = self.n_cls
        if self.code == "multihot":
            return self.codes
        c = torch.eye(n, device=self.back[0].weight.device)
        if self.code == "noise":
            z = self.sigma * torch.randn(n, self.noise_dim, device=c.device) if train else torch.zeros(n, self.noise_dim, device=c.device)
            c = torch.cat([c, z], dim=1)
        return c

    def targets(self, train=False):
        """Backward states of the n_cls classes at the meeting node, shape (n_cls, d_k)."""
        if self.arm == "mn_table":
            return self.table
        t = self._codes(train)
        for i, lin in enumerate(self.back):
            t = lin(t)
            if i < len(self.back) - 1 or self.post_act:
                t = self.act(t)
        return t

    def cosine(self, s, t):
        return F.normalize(s, dim=1) @ F.normalize(t, dim=1).T

    # ---- prediction
    def predict(self, x, readout):
        s = self.state(x)
        if readout == "match":
            return self.cosine(s, self.targets(train=False))
        if readout == "fwd":
            return self.head(s)
        raise ValueError(f"unknown readout {readout!r}")

    def forward(self, x):
        return self.predict(x, self.readout)

    def mn_losses(self, x, y, cfg, step):
        """(total, parts, state, logits) like src.model.compute_losses. cfg keys: mn_scale (default 10)."""
        s = self.state(x)
        zero = torch.zeros((), device=s.device)
        if self.arm == "shallow":
            z = self.head(s)
            task = F.cross_entropy(z, y)
            return task, {"task": task}, s, z
        logits = cfg.get("mn_scale", 10.0) * self.cosine(s, self.targets(train=True))
        meet = F.cross_entropy(logits, y)
        if self.arm == "mn_fwd":
            z = self.head(s.detach())                          # the upper chain does not shape the meeting state
            task = F.cross_entropy(z, y)
            return meet + task, {"task": task, "meet": meet}, s, z
        return meet, {"task": meet, "meet": meet}, s, logits


def build_model(cfg):
    return MeetingNet(arm=cfg["arm"], act=cfg.get("act", "relu"), k=cfg.get("mn_k"), code=cfg.get("mn_code", "onehot"),
                      post_act=cfg.get("mn_post_act", True), noise_dim=cfg.get("mn_noise_dim", 8), sigma=cfg.get("mn_sigma", 0.5))
