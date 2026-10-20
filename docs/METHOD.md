# Method

**This file is the single source of truth for the method.** It started as a copy of `docs/GUIDE.md`, Section 2, on 2026-10-21. From now on, change the method **here** (and record a decision), not in the guide. The code in `src/` must match this file; if they disagree, that is a drift event (`docs/DRIFT_LOG.md`).

**Method version:** 1.0 (2026-10-21). Add a line to the version history below on every change.

| Version | Date | Change | Decision / drift ID |
|---|---|---|---|
| 1.0 | 2026-10-21 | Initial specification (Design B) | D-002 |

---

## Method specification (Design B)

### Notation

- `x`: input, `y`: label, `e_y`: one-hot label vector.
- `F_low`: lower half of the network (input to the middle). `F_up`: upper half (middle to output logits).
- `G`: backward network, mapping from the output space (class-probability vectors) to the middle space.
- `sg(·)`: stop-gradient (treat as a constant).

### Equations

```
h = F_low(x)                       # forward middle state
z = F_up(h)                        # logits
p = softmax(z)                     # predicted class probabilities

L_task = CrossEntropy(z, y)

L_inv  = || G(sg(p)) − sg(h) ||²             # trains G to invert F_up
b      = sg(h) + G(e_y) − G(sg(p))           # per-example backward target for h (a constant)
L_meet = || h − b ||²                         # pulls h toward its backward target

L = L_task + λ_inv · L_inv + λ_meet(t) · L_meet
λ_meet(t) = λ_meet · min(1, t / T_warmup)     # ramp up while G is still poor
```

### Which loss trains which parameters

| Loss | Trains | Why |
|---|---|---|
| `L_task` | `F_low`, `F_up` | Normal backpropagation |
| `L_inv` | `G` only | `G` learns to approximately invert `F_up` |
| `L_meet` | `F_low` only (through `h`) | The target `b` is a constant, so `G` and `F_up` receive no gradient from it |

### Intuition

If `G` were a perfect inverse of `F_up`, then `G(e_y) − G(p)` is how much the middle state would need to change for the prediction `p` to become the true label `e_y`. The meeting loss moves `h` by exactly that amount. Its gradient on `h` is `−2·(G(e_y) − G(p))`, which you can compare with the ordinary backprop gradient on `h`. That comparison is the H3 diagnostic.

### Design choices to record as decisions

| Choice | Default | Alternatives to test later |
|---|---|---|
| Where to meet | One middle layer | Several depths at once |
| Input to `G` | Softmax probabilities | Logits, one-hot only |
| Distance | Mean squared error | Cosine distance, smooth L1 |
| `G` size | Same width as `F_up` | Smaller (cheaper), larger |
| Warm-up | 200 to 1000 steps | None, longer |
| Backprop kept for `L_task` | Yes (auxiliary use) | Weaker reliance on backprop (future work) |

### Known failure modes and how to detect them

- **Collapse:** the middle state shrinks toward a constant so the meeting loss reaches zero trivially. *Detect:* the variance of `h` across a batch falls toward zero while accuracy stalls.
- **Noisy targets early:** `G` is a poor inverse at the start, so targets mislead. *Detect:* negative or near-zero gradient alignment early. *Mitigate:* warm-up.
- **Extrapolation:** `G` is trained on softmax outputs but queried on one-hot vectors. *Detect:* `||G(e_y) − G(p)||` exploding. *Mitigate:* a smoothed one-hot target, or a smaller `λ_meet`.
- **Hidden cost:** `G` adds compute. *Detect:* always count FLOPs per step per arm.

### The H3 diagnostic (gradient alignment)

At regular points during training, compute on the same batch:

- `g_task`: gradient of `L_task` with respect to the `F_low` parameters
- `g_meet`: gradient of `L_meet` with respect to the same parameters

and log `cos(g_task, g_meet)`. A consistently positive value means the meeting signal agrees with the true gradient and can explain a speedup. Near zero or negative values explain a failure.
