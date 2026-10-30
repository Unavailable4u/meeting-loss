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

## v1.2 additions (2026-10-27; D-021, D-023, DR-007, DR-011): arm S and the groups of `prereg-v2`

### Arm S: selective backward pass (`src/selective.py`)

For each batch of B rows the forward pass runs on all rows and the per-row task loss is computed. The backward pass is written by hand and runs only through the k = ceil(`sel_frac` * B) rows with the highest loss (`sel_mode` top; random rows in the control R30). The gradient is the sum over the kept rows divided by B, so the update equals the full-batch gradient without the part of the omitted rows; it reproduces autograd's gradient on the kept rows (`tests/test_selective.py`). During the first `sel_warmup` steps (100) the full backward pass is used. Only plain MLPs are supported (no dropout, no batch normalisation); the optimizer and its settings are those of arm A. The rationale is the concentration of the gradient on high-loss rows (E-009, F-013).

### FLOP counting

As for every arm, `src/flops.py` counts the matrix multiplications of the forward and backward passes. For arm S a step costs the full-batch forward pass plus the backward pass of the k kept rows (full backward during warm-up); the cumulative count is logged at every evaluation. The top-k selection, the gather of rows and the optimizer are not counted, so wall-clock time is reported separately (F-016).

### Groups of `prereg-v2` and the small-batch arm

The plan is `configs/prereg_v2_plan.yaml` (11 groups, 220 runs, seeds 1300 to 1319); the hypotheses H7a and H7b and the roles of the groups are in `docs/HYPOTHESES.md`. The small-batch arm B48 is arm A at batch size 48, tuned in tag `t9` with A's search space and the same 25 trials, 8,000 steps (equal samples to 3,000 steps at batch 128), evaluated every 125 steps. A target is reached at the first of two consecutive evaluations at or above it for every group (D-018); intervals are 97.5% paired bootstrap intervals (10,000 resamples, seed 0).

## v1.3 additions (2026-10-30; D-025, D-026, DR-014 to DR-017): selection without a warm-up, the random control, Selective-Backprop, and the groups of `prereg-v3`

### Arm S without a warm-up and the random control (`src/selective.py`)

Arm S is as in v1.2. With `sel_warmup` 0 the selection starts at the first update, so the first rows are ranked by the loss of a randomly initialised network. S20w0 is `sel_frac` 0.2, `sel_mode` top, `sel_warmup` 0; S20 and S15 differ from it in the kept fraction or in a warm-up of 100 steps, S20w10 in a warm-up of 10 steps, and the control R20w0 keeps 20% of the rows at random (`sel_mode` random) without warm-up. All use A1's tuned hyperparameters; only the selective keys and the stage name differ from `configs/tuned/t1_base_two.yaml` (`src/prereg3.py`, `tests/test_prereg3.py`). The option `sel_scale` (DR-015) is not used: it had no effect in look 2 (E-025).

### Arm SB: Selective-Backprop as released (`src/sb.py`, DR-014)

Every incoming batch of 128 rows is forwarded and each row's loss enters a history of the last 1,024 losses; a row is kept with probability (share of history entries below its loss) to the power `sb_beta` (2.0). Kept rows queue; whenever the queue holds a batch, one update of arm A's step is made on it, with a second forward pass as in the released code. The first epoch of updates (priming) are ordinary steps of arm A. Steps are updates. **Uncached FLOPs** (`flops`, decisive in H10) charge every selection forward pass and the full update; **cached FLOPs** (`flops_cached`) charge only the backward part of an update and are reported next to it. Its hyperparameters are those of tag `t11` (E-026), tuned with the protocol of arm A (25 trials, two selection seeds, FLOPs to 0.975 under the crossing rule of D-018; `sb_beta`, history and priming not searched).

### Groups, intervals and secondary readings of `prereg-v3`

The plan is `configs/prereg_v3_plan.yaml` (13 groups, 260 runs, seeds 2300 to 2319); the hypotheses H8, H9 and H10 and the roles are in `docs/HYPOTHESES.md`. Each interval is a paired percentile bootstrap over seeds (10,000 resamples, seed 0) with alpha 0.05 / 3 (98.33%). A target is reached at the first of two consecutive evaluations at or above it (D-018); a run that never reaches it is charged its whole-run FLOPs and counted. The **cached score** of a run is `flops_cached` at the same evaluation. The **warm-up credit** of a run with a warm-up of W steps is min(W, step at the crossing) times (FLOPs of a full step minus FLOPs of a selective step); the full step is read from arm A's logs (every step is a full step) and the selective step from the run's own evaluations after the warm-up; subtracting it charges the warm-up steps like selective steps. Neither decides anything. The analysis script also prints the git commits and `git_dirty` flags found in the run headers.
