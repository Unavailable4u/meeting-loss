# Method

**This file is the single source of truth for the method.** It started as a copy of `docs/GUIDE.md`, Section 2, on 2026-10-21. From now on, change the method **here** (and record a decision), not in the guide. The code in `src/` must match this file; if they disagree, that is a drift event (`docs/DRIFT_LOG.md`).

**Method version:** 1.1 (2026-10-21). Add a line to the version history below on every change.

| Version | Date | Change | Decision / drift ID |
|---|---|---|---|
| 1.0 | 2026-10-21 | Initial specification (Design B) | D-002 |
| 1.1 | 2026-10-21 | Arms and variants added (E, F, V1 to V3, B-paper/B-code), activation as hyperparameter, schedules; contribution reframed (the target is SDTP/DTP) | D-012, D-013, D-014, DR-002 |

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

---

## v1.1 additions (2026-10-21; D-012 to D-014, DR-002)

The equations above remain the default arm D. Everything below is implemented in `src/model.py` and `src/train.py`, with config keys listed in `configs/template_all_options.yaml`.

### Note on novelty

The target `b = sg(h) + G(e_y) - G(sg(p))` is the simplified difference target propagation (SDTP) target of Bartunov et al. 2018 (and the DTP target of Lee et al. 2015, Eq. 15) applied at a middle layer, with `G` inverting the whole top half. What differs here is the use: an auxiliary loss next to backpropagation, measured for compute efficiency. See `docs/LITERATURE.md`.

### Arms (Stage 3 main comparison, D-013)

| Arm | `arm` key | Definition |
|---|---|---|
| A | `base` | Plain network |
| B | `deepsup` | Companion cross-entropy head on `h`. **B-paper** (primary): weight `lam_aux` decays linearly to zero over `max_steps` (`aux_sched: linear`), optional threshold `aux_gamma` (loss term `relu(L_aux - gamma)`). **B-code** (secondary): constant weight with `aux_lr_decay: power_inv`, the companion head's learning rate decaying as `(1 + 1e-4*step)^-0.75` |
| C | `ae` | Decoder reconstructs the input from `h`, weight `lam_aux` |
| D | `meet` | Meeting loss, default configuration (equations above) |
| E | `base` + `low_lr_mult` | Control: arm A with a learning-rate multiplier on the lower layers `F_low` |
| F | `meet_grad` | Control: DTP-style exact-gradient target `b = sg(h - eta_hat * dL_task/dh)` in per-sample units (the batch size is multiplied back in), `L_meet = ||h - b||^2`, weight `lam_meet`, same warm-up |

### Variants of D (maximum 3 per stage)

- **V1, noise in the inverse loss** (`sigma_inv > 0`): the inverse is trained on corrupted middle states, as in DTP and SDTP. `h_t = sg(h) + sigma * eps`, `p_t = softmax(F_up(h_t))` (no gradient to `F_up`), `L_inv = ||G(p_t) - h_t||^2`. The target `b` is still computed from the clean `h` and `p`.
- **V2, augmented output** (`arm: meet_ao`): `G` receives `[p, z]` where `z = tanh(sg(h) @ R)` and `R` is a fixed random matrix (a buffer, never trained), so that `b = sg(h) + G([e_y, z]) - G([p, z])`. Bartunov et al. introduced this to give the inverse richer, per-example information than the label alone (risk R-09). `d_aux` defaults to 128.
- **V3, frozen random `G`** (`arm: meet_frozen`): `G` is never trained and there is no inverse loss (`lam_inv` ignored). For a near-linear `G`, `G(e_y) - G(p)` is approximately a fixed random linear map of the output error `(e_y - p)`, so this variant behaves like direct feedback alignment. It is also the cheapest meeting-loss variant (F-003). Inspired by fixed-weight DTP.

### Hyperparameters shared by all arms (D-013, D-014)

- `act` in {relu, tanh}. Lee et al. found ReLU hurt TP while backprop prefers it, so it is tuned per arm.
- `lr_schedule` in {constant, cosine}.

### Notes and expectations

- **Adam and control F.** Under AdamW, adding a term whose gradient on the lower layers is parallel to the task gradient mostly rescales that gradient, which Adam normalizes away. So F is nearly a no-op under AdamW and informative under SGD with momentum; control E (learning-rate multiplier) is the one that changes Adam's effective step.
- **Alignment (H3).** Expect positive but imperfect alignment: Lee et al. bound the angle only under perfect inverses and small steps, and Meulemans et al. 2020 argue DTP targets follow Gauss-Newton-like directions rather than the gradient.
- **Parked idea.** Bartunov et al. (appendix 5.6) suggest hybrid targets that mix the Jacobian-transpose target with a learned-inverse target. With backprop kept, an exact-gradient component only rescales the gradient (see arm F), so a hybrid is parked until there is a reason to test it.
