<!--
LIVING DRAFT RULES (docs/GUIDE.md, Section 7)
1. No numeric or comparative claim without a claim ID like [C-07].
2. Every claim ID has evidence IDs in paper/CLAIMS.md.
3. When a finding changes status, re-check every claim that cites it.
4. Each section keeps a status header (skeleton | draft | stable) and the date it was last checked against docs/FINDINGS.md.
5. Write stable parts early (Method, Setup); Results last.
6. Record limitations as soon as you know them.
Terminology must match docs/METHOD.md. Related Work must match the novelty statement in docs/LITERATURE.md.
One paper, not two: see paper/PAPER_PLAN.md (split triggers). [C-xx] marks with square-bracket text inside a claim are open until the check that supports them has run.
-->

# Which Ideas Actually Save Compute? A Pre-Registered, Compute-Matched Study of a Target-Propagation Auxiliary Loss and a Selective Backward Pass (working title)

Status: draft | Version: v0.2 | Last checked against FINDINGS: 2026-10-25

## Abstract
Status: skeleton | Depends on: all sections | Checked: 2026-10-25
Problem: claims that a training idea saves compute are easy to make and hard to trust (tuning noise, unfair baselines, uncounted costs). Study 1: an auxiliary target loss next to backpropagation did not reduce FLOPs-to-target against a tuned baseline; it needed about 36% more [C-09], although its gradient is aligned with the task gradient [C-03]. Mechanism: [C-15 to C-19, planned]. Study 2: a selective backward pass over the highest-loss rows [C-21, planned; pilot C-13]. Protocol lessons: [C-10, C-11, C-12]. Scope: one MNIST MLP unless stated [C-24]. Write this last.

## 1. Introduction
Status: skeleton | Depends on: H1 to H4, H6, LITERATURE | Checked: 2026-10-25
- The cost of training and the credit-assignment path; why a fair, compute-matched test is needed.
- Study 1 in one paragraph: the loss is the simplified difference target propagation (SDTP) target at a middle layer [C-05]; the contribution is the study, not the loss. It was pre-registered before the final seeds, and it is a null result [C-09].
- Study 2 in one paragraph: backpropagating only the rows that matter. This is known (Jiang et al. 2019, selective backprop); we contribute a compute-matched test with tuning noise controlled [C-21]. The pilot is tentative [C-13].
- Contributions, each a claim ID: [C-09] the null result; [C-15] to [C-19] the mechanism; [C-10] [C-11] [C-12] protocol lessons; [C-21] the selective-backward test (conditional on its outcome). State plainly that the paper does not depend on the last one (`paper/PAPER_PLAN.md`).

## 2. Related work
Status: skeleton | Depends on: LITERATURE.md | Checked: 2026-10-25
- Target propagation, difference target propagation \cite{lee2015dtp}, simplified DTP \cite{bartunov2018scal}: our target is the SDTP target [C-05]; theory \cite{meulemans2020tp}; fixed-weight variants \cite{shibuya2023fwdtp}; feedback alignment \cite{lillicrap2014random} and direct feedback alignment \cite{nokland2016dfa}. Prior work used these as *replacements* for backpropagation and judged final accuracy, not compute.
- Auxiliary and deep-supervision losses \cite{lee2015dsn}, including the paper-versus-code difference in how the companion objective fades; ladder networks \cite{rasmus2015ladder}.
- Selective and importance-based backward passes \cite{jiang2019biggestlosers}; importance sampling and online batch selection (to verify and add).
- Evaluation methodology \cite{schmidt2021crowded}: budgets, schedules, seeds, re-tuning noise.
- How this work differs: must match the novelty statement in `docs/LITERATURE.md` and the check of prior work done before the selective test.

## 3. Method
Status: draft | Depends on: METHOD.md | Checked: 2026-10-25

### 3.1 Notation and the meeting loss
`x`, `y`, one-hot `e_y`; the network splits at a middle layer into `F_low` and `F_up`; `G` maps class probabilities to the middle space; `sg` is stop-gradient. With `h = F_low(x)`, `z = F_up(h)`, `p = softmax(z)`:

```
L_task = CE(z, y)
L_inv  = || G(sg(p)) - sg(h) ||^2          # trains G only
b      = sg(h) + G(e_y) - G(sg(p))          # a constant per-example target for h
L_meet = || h - b ||^2                      # trains F_low only, through h
L      = L_task + lam_inv * L_inv + lam_meet(t) * L_meet,   lam_meet(t) = lam_meet * min(1, t / T_warmup)
```

The auxiliary use (backpropagation kept for `L_task`) is the difference from earlier work [C-05].

### 3.2 Arms and controls
A plain network; B deep supervision (companion head on `h`); C autoencoder auxiliary; D the meeting loss; E a lower-layer learning-rate control; F an exact-gradient-target control (a near-inert negative control whose scale must be set explicitly [C-12]); variants V1 (noisy inverse), V2 (augmented output), V3 (frozen random `G`). Definitions: `docs/METHOD.md`.

### 3.3 The selective backward arm (S)
Forward pass on the whole batch under no gradient, keeping the activations; per-example losses come for free. Select `k = ceil(sel_frac * B)` rows, the highest-loss ones (or random rows as a control); run a hand-written backward pass for the plain MLP on the selected rows only, with the gradient divided by the full batch size; use the full backward pass during a short warm-up. The hand-written backward matches autograd exactly and the FLOP count is exact [C-25].

### 3.4 Cost analysis
Per-step overheads relative to the baseline: meeting loss, deep supervision, autoencoder [C-07]; selective backward [C-25]. All FLOPs are counted with the exact counter (validated against the analytic count, F-001); wall-clock is reported separately [C-23, planned].

## 4. Experimental setup
Status: draft | Depends on: D-003 to D-008, D-014, D-017 to D-019 | Checked: 2026-10-25
- **Data.** MNIST with a fixed train/validation split (D-008); the test set is not used until the end.
- **Model.** A plain MLP split at the middle layer; AdamW; 3,000 steps; batch size 128 for every arm (a small-batch control is planned, [C-22]).
- **Targets and "reaching a target".** Validation accuracy 0.97 and 0.975; a target is reached at the first of two consecutive evaluations at or above it (D-018) [C-11]. A 0.98 target was too noisy on this validation set.
- **Tuning.** Equal-budget random search per arm with seed-paired scoring by FLOPs-to-target (D-014, D-017); a second, independent tuning of the baseline as a noise floor [C-10].
- **Pre-registered test (Study 1).** Hypotheses, thresholds, seeds and the three-way rule (supported, refuted, inconclusive; margin 15%) were fixed and tagged before any final seed (D-019).
- **Exploratory work** is labelled as such everywhere; any exploratory lead is restated from a new pre-registered run before it is called a result (D-015).
- **Compute used.** To be filled from the logs.

## 5. Results
Status: draft | Depends on: F-009 to F-012, E-008 to E-010 | Checked: 2026-10-25

### 5.1 The meeting loss against a tuned baseline
The meeting loss needed about 36% more FLOPs than the tuned baseline to reach 97.0% and 97.5% validation accuracy, with paired-bootstrap intervals that exclude any gain [C-09]. The pre-registered outcome of the first hypothesis is "refuted". The result is conditional on the tuned configurations, AdamW and this model.

### 5.2 Other auxiliaries and controls
Deep supervision did not beat the tuned baseline at the primary targets; its selection-level lead did not replicate [C-08]. Control F is nearly inert by construction [C-12]. The autoencoder auxiliary costs more than any other arm per step [C-07].

### 5.3 Deep plain network with SGD
Planned (check C): [C-20].

### 5.4 Selective backward
Status of this subsection: pilot only. A 3-seed pilot with the baseline's tuned hyperparameters reached the targets at a fraction of the baseline's FLOPs, and random rows of the same count did worse [C-13]; the highest-loss rows carry nearly all of the gradient norm after the early phase [C-14]. These are exploratory. Three further exploratory looks (3 seeds each) qualify the pilot: a plain smaller batch does not clearly beat batch 128, and selective backward keeps an advantage over the best small batch that is smaller than its advantage over the baseline [C-22]; the per-step wall-clock saving on one CPU is much smaller than the FLOPs saving [C-23]; and the FLOPs-to-target is flat across the kept fractions 0.1 to 0.3 [C-26]. The pre-registered test [C-21] and a second setting [C-24] are planned. Do not state any selective-backward result as established until [C-21] has a supported outcome.

### 5.5 Tuning noise and protocol lessons
Re-tuning the baseline gave a configuration that tied at selection but differed on fresh seeds [C-10]; rankings depend on how "reaching a target" is defined [C-11].

## 6. Analysis (mechanism)
Status: skeleton | Depends on: F-IDs of Stage 4, checks A, L, G0, B, C, D | Checked: 2026-10-25
- The auxiliary gradient is positively aligned with the task gradient [C-03], yet gives no gain [C-09]: the working explanation is redundancy (an interpretation until tested).
- Geometry of the signal: parallel and orthogonal parts, Gauss-Newton direction [C-15].
- Link to selective backward: is the meeting gradient already concentrated on the rows backprop uses? [C-16]
- Dose-response at fixed hyperparameters [C-17]; oracle target [C-18]; sufficiency with backpropagation cut [C-19].
- Ablations and variants V1 to V3 (all listed in the appendix).

## 7. Limitations
Status: draft | Checked: 2026-10-25
- One MLP on MNIST, AdamW, 3,000 steps; the deep network, SGD and a second setting are separate, planned checks [C-20] [C-24].
- Conditional on tuned configurations; two tunings of the baseline already differ noticeably on fresh seeds [C-10].
- FLOPs, not wall-clock, unless stated [C-23].
- Exploratory checks do not carry the weight of the pre-registered test (D-015).
- Selective backward is an existing idea (Jiang et al. 2019); the contribution is the test, not the method.

## 8. Conclusion
Status: skeleton

## References
Use `paper/references.bib`; every entry is verified before submission.

## Appendix
- A. All hyperparameters and search ranges.
- B. All variants and arms tried, including failures (arms A to F, V1 to V3, S variants).
- C. Extra plots (FIG-01 to FIG-10 as they are made).
- D. Reproduction commands and code link.
