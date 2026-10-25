# Deferred: fixing G's quality and the meeting loss's cost (Gate 3), only after a step reduction has been shown

**Status:** plan, written BEFORE any run, and **blocked by gates**: do not start any item here until the gate below is met. Type: `exploratory`. Part of the mechanism study (`mechanism-overview.md`).

## Why this is deferred

Every item here makes a meeting-loss variant better or cheaper. That is only worth doing if some version of the signal reduces the *number of steps* at all. E-008 showed the opposite (D needs about 10% more steps than A), so cheaper machinery only makes a loser cost less. The earlier checks decide whether a step reduction exists:

- **Check G0** (dose-response, learned G): DOSE HELPS.
- **Check B** (oracle arms): ORACLE HELPS.
- **Check C** (deep network, SGD): SUPPORTED.
- **Check D**: R2 (oracle replacement) sufficient while R1 (learned G) is not, which says `G`'s quality is the bottleneck.

**Gate (all of the following):** at least one of G0, B or C ends with the helpful outcome above **and** the step reduction against arm A is large enough to pay for the overhead. With the 0.85 FLOPs-ratio line (the GO line of the pilot) and per-step overheads relative to A, the paired steps ratio vs A must be at most:

| Overhead per step (vs A) | Required steps ratio vs A | Meaning |
|---|---|---|
| +23% (arm D as is) | 0.69 | 31% fewer steps |
| +12% (frozen G) | 0.76 | 24% fewer steps |
| +6% (a lazy k = 4 variant, estimate) | 0.80 | 20% fewer steps |

(0.85 divided by 1 + overhead; the warm-up phase is ignored.) If no check reaches these, **stop here**: the mechanism result is the contribution.

## Part 1: fixing G's quality (items G1 and G2)

**G1. Domain mismatch (the one-hot versus softmax query).** `G` is trained on softmax outputs `p` and queried on the one-hot `e_y`. Query instead at a smoothed point `e_y_tilde = (1 - eps) e_y + eps/K` (or at temperature-scaled logits), so the query stays inside the region where `G` has seen data. Notes:
- `eta_hat` in the configs belongs to **control F** (the exact-gradient target step), not to a query shrink, so this is a new option, not an existing one (checked in METHOD).
- Smoothing also shrinks `G(e_y) - G(p)`, so it changes the signal's **magnitude** as well as its domain. Always compare at a normalized strength (rho of check G0), not at a fixed `lam_meet`.

**G2. Stabilizing the inverse.** The V2 warning signs (`h_var` rising from 4 to 58 and the inverse loss from 1.6 to 41 in short runs) come from one variant, in short runs, and have not been shown to cause the large-`lam_meet` failures. Try in this order, one at a time:
1. a smaller learning rate for `G` (`g_lr_mult`), gradient clipping on `G`, weight decay on `G`;
2. only then normalization of `h` (LayerNorm or L2).

Cautions:
- **Normalizing `h` changes the baseline's dynamics too**, so arm A must get the same layer, otherwise the comparison is confounded.
- **Spectral normalization or a Lipschitz bound on `G` conflicts with what an inverse does**: inverting a layer with small singular values needs a large gain (about `1/sigma`), so the constraint can cripple exactly the part that makes `G` useful. Treat it as a last resort and test it against a version without it.

## Part 2: reducing the cost (items C1 to C3)

- **C1. Lazy application.** Apply the meeting gradient and update `G` every `k` steps (for example `k` = 4 or 5). Most direct, and the first thing to try.
- **C2. Smaller `G`.** Reduce the hidden width of `G`. Low rank buys little (its input is already 10-dimensional, so rank 10 at most). A **linear `G` is a weak test**: it cannot depend on the state, so it behaves like a learned feedback matrix, which the frozen-`G` variant already resembles.
- **C3. Fewer rows.** Apply the meeting loss only to the rows selected for backward (the hybrid; see `linking-measurement-concentration.md` for the cost and the rule that decides whether to try it).

## Decision rule for each fix (judged against the gate above, with the helpful check's oracle as the ceiling)

For a fix `X` evaluated in the setting where the gate was met (paired seeds, steps-to-target at both targets):

```
gap_closed(X) = (steps(D) - steps(X)) / (steps(D) - steps(oracle))
```

- **PROMISING:** `gap_closed >= 0.5` at both targets **and** the steps ratio vs A is at or below the required value in the table for that variant's overhead.
- **DROP:** `gap_closed < 0.25` at either target, or the required steps ratio is not reached.
- **MIXED:** anything else (report).

Limits: at most 3 variants per stage (a standing rule of this project) and a decision entry before they run; each variant needs its own seeds and a written rule before its first run.

## What this can NOT show

- It is only meaningful if the oracle helps (`steps(oracle) < steps(D)`); otherwise `gap_closed` is undefined and the whole line is closed.
- It does not tell you whether the combination of fixes works (they are tested one at a time).
- MNIST MLP unless the gate was met by check C.

## If the gate is not met

Close the line for this setup in `FINDINGS` as a mechanism result (A, G0, B, D, C together) and write it up. Do not run this note.
