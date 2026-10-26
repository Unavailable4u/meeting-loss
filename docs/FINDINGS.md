# Findings

Curated conclusions, each with evidence, scope, and confidence. A finding is **not** a raw result: raw results go in `EXPERIMENT_LOG.md`, and findings are what we currently believe because of them. Newest at the bottom. Template: `docs/templates/finding-entry.md`.

Rules:
- Every finding cites evidence IDs (`E-...`).
- Status values: `tentative`, `supported`, `refuted`, `superseded by F-0XX`.
- When a finding changes status, check `paper/CLAIMS.md` and the paper draft (drift protocol step 4).
- Include negative and inconclusive findings.

---

## F-001: The FLOP counter matches the analytic count for arm A
- Type: verification (deterministic check; tests no hypothesis)
- Status: supported
- Confidence: high (exact match)
- Evidence: E-001 (logged 155,058,176 FLOPs/step at batch size 128). Analytic count for the 784-256-128-256-10 MLP: forward 2*128*268,800 = 68,812,800; weight gradients 68,812,800; input gradients for layers 2 to 4 only, 2*128*68,096 = 17,432,576 (layer 1 needs none); total 155,058,176.
- Scope: arm A, this architecture, batch size 128, optimizer FLOPs excluded.
- Implications: FLOPs-to-target comparisons across arms can use `src/flops.py`. Satisfies the intent of Stage 1's "FLOP counter verified" exit criterion and Stage 2 test 5.
- Found / last re-verified: 2026-10-21

## F-002: Untuned baseline is reproducible; 0.97 and 0.975 are usable targets, 0.98 is too noisy on a 5,000-sample validation set
- Type: exploratory
- Status: tentative
- Confidence: medium-low (untuned baseline only, 5 seeds)
- Evidence: E-001. Final validation accuracy 0.9769 (sd 0.0022, 5 seeds). Steps to 0.97: 900, 1200, 950, 900, 850 (mean 960, sd 124). Steps to 0.975: 1700, 1750, 1600, 1550, 1400 (mean 1600, sd 123). Steps to 0.98: 4300, 5750, never, 4000, 2200 (4 of 5 seeds). Binomial sampling error of accuracy at 0.975 with n = 5,000 is about 0.22 points, the same size as the seed-to-seed spread.
- Scope: arm A, constant learning rate 1e-3, AdamW, batch 128, ReLU MLP, fixed 55k/5k split.
- Implications: D-003 should use 0.97 and 0.975 (and report 0.98 only as secondary). Consider whether a larger validation split (for example 50k/10k) is worth a new decision superseding part of D-008, because a 0.2-point sampling error limits what differences can be resolved. Re-check after arm A is tuned (D-014).
- Found / last re-verified: 2026-10-21

## F-003: Per-step FLOP overhead of each arm relative to arm A
- Type: verification (deterministic count)
- Status: supported
- Confidence: high (counted, not sampled; independent of batch size because all costs scale linearly with it)
- Evidence: `scripts/flops_overhead.py`, output `results/processed/flops_overhead_v1.1.csv`. Overhead vs arm A: deepsup +0.63%, ae +115.64%, meet +22.91%, meet with sigma_inv +34.57%, meet_ao +44.55%, meet_frozen +11.67%, meet_grad +5.83%.
- Scope: this MLP (784-256-128-256-10, `d_aux` 128), one training step including all extra forwards and backwards, optimizer FLOPs excluded.
- Implications: to win at equal FLOPs, arm D must reduce steps-to-target by more than about 19% (1 - 1/1.2291) just to break even on FLOPs. The autoencoder arm C is costly because it reconstructs the 784-dimensional input. The frozen-G variant V3 is the cheapest meeting-loss variant. This sharpens risk R-02.
- Found / last re-verified: 2026-10-21

## F-004: In short untuned diagnostic runs the meeting loss neither collapses nor helps at equal steps; alignment is positive for D, weaker for V2, and about zero for the frozen G
- Type: exploratory
- Status: tentative
- Confidence: low (one seed per arm, 2,000 steps, untuned loss weights, validation sampling error about 0.2 points)
- Evidence: E-003. No non-finite values and no collapse of `h_var` in any of the three runs. Alignment after warm-up is positive at 100% of evaluations for meet (median 0.50) and meet_ao (median 0.22), and at 33% for meet_frozen (median -0.02). Validation accuracy tracks arm A seed 0 within about 0.4 points (last-five mean: A 0.9755, meet 0.9740, meet_ao 0.9703, meet_frozen 0.9757). `h_var` grows (meet 4.3 to 13.2, meet_ao 4.0 to 58.2, meet_frozen 4.1 to 22.9) and the V2 inverse loss rises from 1.6 to 41.3.
- Scope: MNIST MLP, AdamW, constant learning rate 1e-3, `lam_meet` 0.1, `lam_inv` 1.0, warm-up 500, ReLU, seed 0.
- Implications: the implementation is usable for the sweeps. At equal steps there is no visible gain while the arms cost 12% to 45% more FLOPs per step (F-003), so any gain has to come out of tuning (risk R-02). The frozen G behaves as a near-null control. Not evidence for or against H1 or H3. Recheck V2 over a longer run, because its inverse loss is still rising at step 2000.
- Found / last re-verified: 2026-10-22

## F-005: Control F (`meet_grad`) is nearly inert: its gradient on the middle state is 2*lam_meet*eta_hat/d times the task gradient
- Type: verification (derivation plus measurement; tests no hypothesis)
- Status: supported
- Confidence: high (analytic and measured, agree to five digits)
- Evidence: With `L_meet = mean((h - b)^2)` over B*d elements and `b = sg(h - eta_hat*B*g_h)`, where `g_h` is the gradient of the batch-mean task loss, the gradient of `lam_meet*L_meet` on `h` is `2*lam_meet*eta_hat*g_h/d`, parallel to `g_h` and smaller by that factor. Measured on a random batch with `src/model.py` (d = 128, eta_hat = 0.1): ratio of norms 0.00006 at lam_meet = 0.0382 (the t1 winner), 0.00156 at 1.0, 0.01375 at 8.8, against 0.00006, 0.00156 and 0.01375 predicted. E-004: F's winner scores 1.10 times arm A's under first crossing and 0.99 times under two consecutive evaluations (F-006), which is what arm A plus 5.8% FLOPs overhead (F-003) and tuning noise would give.
- Scope: this implementation, d_mid = 128, eta_hat = 0.1, `lam_meet` anywhere in the D-014 range (at most 1.4% of the task gradient at lam_meet = 8.8).
- Implications: F cannot serve as an upper bound or as an informative reference for gradient alignment (H3) or for H5; report it as a negative control. Because the factor is tiny for any optimizer, the D-013 remark that F is informative under SGD with momentum does not hold at these settings. A meaningful F needs a larger effective step (a rescaled `eta_hat`, or a sum instead of a mean in `L_meet`); that changes the method, so it needs its own decision before the SGD experiments (R-19).
- Found / last re-verified: 2026-10-22

## F-006: Rankings among the top arms depend on the crossing rule; the meeting loss (D) is behind A under all three rules in the t1 selection sweep
- Type: exploratory
- Status: tentative
- Confidence: low (selection seeds 1000 and 1001 only, 25 trials per arm, 3,000 steps, 5,000-sample validation set)
- Evidence: E-004 re-scored with `scripts/rescore_sensitivity.py` (`results/processed/sensitivity_t1.csv`). Ratio of an arm's winning score to arm A's, in-sample / held-out (best trial picked on one seed, scored on the other seed, averaged over both directions), under first crossing, two consecutive (adopted, D-018) and a 3-evaluation mean:
  D meet 1.31 / 1.17, 1.19 / 1.29, 1.25 / 1.25. B deepsup 1.03 / 0.82, 0.82 / 0.77, 0.89 / 0.79. F meet_grad 1.10 / 1.08, 0.99 / 1.01, 1.06 / 0.98. E base_lowlr 1.38 / 1.22, 1.18 / 1.88, 1.25 / 1.61. C ae 2.07 / 1.87, 1.80 / 1.99, 1.90 / 2.03.
  Arm A's own score is 186 GFLOPs in-sample under first crossing and 233 under two consecutive; its held-out score is 233 and 248, so in-sample scores are optimistic by 7% to 25%. 118 of 182 crossings were single-evaluation flukes (D-018).
- Scope: this sweep only; selection results, not confirmatory evidence (D-015).
- Implications: D is behind A by 17% to 31% on every rule and view, and its best trials use a small `lam_meet` (E-004), so there is no sign of a gain at these settings. B is ahead of A by 11% to 23% under the two smoothed rules, and by 18% held-out but only level in-sample (1.03) under first crossing, and tuning noise of this size is not yet measured, so the B-versus-A difference is not a result until the re-tuning noise floor (tag t2) says how far arm A's own score moves between two tunings (D-014, R-17). E's held-out score is unstable (1.22 to 1.88), which confirms that E needs a better tuning protocol before H5 (to be decided).
- Found / last re-verified: 2026-10-22
- Update (2026-10-22, E-005, F-007): the re-tuning pass t2 shows no shift in arm A's winning score (0.0% under the adopted rule), so the B-versus-A difference above exceeds the selection-level noise floor; it is still to be confirmed with final seeds.

## F-007: Re-tuning arm A with independent seeds reproduces its selection score; deep supervision (B) is ahead of both tunings at the selection level
- Type: exploratory
- Status: tentative
- Confidence: low to medium (selection seeds only; one pair of tunings of A; 25 trials per arm; scores on a 50-step grid)
- Evidence: E-005 and E-004 under the adopted rule (D-018). Arm A: both tunings score 232.6 GFLOPs (mean 1500 steps) although their winning hyperparameters differ widely; the top six trials of each tuning lie between 232.6 and 263.6. Arm B: the winner t10 scores 191.2 (steps 1200 and 1250), the runner-up 218.5, the third 245.8. B's t10 is also the best trial when picked on either seed alone (held-out score 191.2, ratio 0.77 to A's held-out 248.1). A's winners have a larger seed spread (steps 1250 and 1750; 1600 and 1400).
- Scope: MNIST MLP, selection seeds 1000, 1001, 2000, 2001, 3,000-step runs.
- Implications: the tuned baseline is stable at the selection level, which lowers the worry that arm A is a weak or lucky baseline (R-17), though the final-seed noise floor of D-014 is still to be measured. B ahead of A by about 18% (6% for its runner-up) is a lead worth testing with final seeds; it is not a result. If it holds, B is the arm that D has to beat for H2, and D (1.19 times A) has a larger gap to close.
- Found / last re-verified: 2026-10-22

## F-008: At the selection level no meeting-loss variant beats tuned arm A on FLOPs; alignment is high for D yet its best trials use a small weight, and the frozen-G variant that ties A has about zero alignment
- Type: exploratory
- Status: tentative
- Confidence: low to medium (selection seeds only, 25 trials per arm, 3,000 steps, 5,000-sample validation set)
- Evidence: E-004, E-005 and E-006 under the adopted rule (D-018). Ratio of the winning score to arm A's 232.6 GFLOPs (in-sample / held-out): D 1.19 / 1.29, V1 1.44 / 1.41, V2 1.28 / 1.20, V3 0.97 / 0.91. Mean steps: A 1500, D 1450, V2 1325, V3 1300, V1 1600. Alignment of the winners after step 1000: D +0.75, V1 +0.81, V2 +0.20, V3 -0.003 (median). D: 1 of 6 trials with `lam_meet` above 0.5 reached 0.975, against 11 of 19 at 0.5 or below; D's winner has `lam_meet` 0.0051.
- Scope: MNIST MLP, AdamW, relu or tanh, selection seeds 1000 and 1001 (A also 2000 and 2001), 3,000-step runs.
- Implications: the expected outcome of the Stage 3 test of H1 is a null or negative result, and H1 can only be settled by the confirmatory runs. The ranking among A, B-code, F and V3 (within 4%) is inside the tuning noise (the top six trials of A span 12%). That D's alignment is high while the tuned weight is small is consistent with, but does not show, the idea that a target aligned with the true gradient adds little next to backpropagation (METHOD: an exact-gradient component only rescales the gradient); testing that is a Stage 4 question. Do not read V3's tie as a meeting effect (its alignment is zero). If H1 fails in Stage 3, the gate in GUIDE Section 6 applies.
- Found / last re-verified: 2026-10-22

## F-009: The meeting loss (arm D) needs about 36% more FLOPs than tuned arm A to reach 97.0% and 97.5% validation accuracy; H1 is refuted, while gradient alignment is high (H3)
- Type: confirmatory
- Status: supported (as a null result: the pre-registered outcome of H1 is "refuted")
- Confidence: high within its scope (20 paired seeds, two targets, intervals far from both zero and a 15% gain)
- Evidence: E-008. D against A: +36.9% [+26.4, +47.4] at 0.97 and +35.5% [+26.9, +44.4] at 0.975 (97.5% paired bootstrap intervals), mean steps 1025 and 1510 against 920 and 1370. Against the second tuning of A: +2.2% [-4.7, +9.9] and +15.6% [+9.5, +22.5]. Alignment of D with the task gradient: +0.75 to +0.79 in every run, positive at 100% of evaluations (H3 threshold met).
- Scope: MNIST, 784-256-128-256-10 MLP split at the 128-d middle layer, AdamW, 25-trial random-search tuning per arm (D-014, D-017, D-018), 3,000 steps, 55,000 / 5,000 split, one tuned configuration per arm.
- Implications: the paper's main result is a null (negative) result on this setting; H2 and H5 are not tested (gated). That alignment is high while the gain is absent supports the view that a target aligned with the true gradient adds little next to backpropagation, but this is an interpretation, not a tested mechanism; testing it is a Stage 4 question. The Stage 3 gate row "no gain or worse" applies (GUIDE Section 6). It does not say anything about SGD, other tasks or other layers (H4 is untested), and control F (negative control) cannot address the exact-gradient question (F-005).
- Found / last re-verified: 2026-10-23

## F-010: Deep supervision (arm B) does not beat tuned arm A at the primary targets; the selection-level lead of 18% did not replicate (H6 refuted)
- Type: confirmatory (the hypothesis was generated from the exploratory finding F-007 and pre-registered in D-019 before any final seed)
- Status: supported (as a null result: the pre-registered outcome of H6 is "refuted")
- Confidence: medium to high for the comparison with A; the comparison with A2 gives another answer, see F-011
- Evidence: E-008. B against A: +2.3% [-4.8, +10.4] at 0.97 and +1.7% [-5.0, +9.2] at 0.975. Exploratory: against A2, -23.7% [-27.8, -19.1] and -13.2% [-18.7, -7.2]; at 0.98 against A -19.2% [-28.0, -9.0] and against A2 -14.1% [-21.7, -6.9], with B missing 0.98 in 1 of 20 seeds against 8 for A and 3 for A2.
- Scope: as F-009.
- Implications: the pre-registered conclusion is that B is not more efficient than A at 0.97 or 0.975. B being ahead of A2 and, at 0.98, ahead of both is a lead for a later confirmatory run (planned in advance on fresh seeds, D-015), not a claim. Selection on 2 seeds overstated B's advantage (191.2 at selection, 216.1 on 20 seeds).
- Found / last re-verified: 2026-10-23

## F-011: Two tunings of arm A tie at selection but differ by 17% to 34% on the final seeds; selection-level differences from 2 seeds are not reliable at this size
- Type: exploratory
- Status: tentative
- Confidence: medium (one pair of tunings, 20 seeds each)
- Evidence: E-008. A2 against A: +34.0% [+26.5, +42.0] at 0.97, +17.2% [+6.7, +27.7] at 0.975, -6.0% [-13.8, +2.1] at 0.98. Mean GFLOPs to 0.975, selection to final: A 232.6 to 212.4, A2 232.6 to 248.9, B 191.2 to 216.1, D 276.3 to 287.8. The across-seed SD of log steps-to-0.975 on the final runs is only 0.08 to 0.12.
- Scope: MNIST MLP, arm A, 25 trials per tuning, 2 selection seeds, target 0.975.
- Implications: the dominant noise is in the tuned configuration (2-seed selection among 25 trials, with the minimum of a noisy score chosen), not in the training seed; F-007's 0.0% re-tuning shift was a selection-level tie that did not carry over. The D-019 rule uses the shift only to block a "supported" verdict, so it did not matter for the two "refuted" verdicts. H1 holds against either tuning of A; H6 does not (F-010). Any future comparison should pre-register which baseline is the reference, and either select with more seeds or re-evaluate the top few trials on fresh seeds before choosing. Statements about arms must say they are conditional on the tuned configurations (R-21).
- Found / last re-verified: 2026-10-23

## F-012: A selective backward pass (backward only for the highest-loss 30% of each batch) reached 97.0% and 97.5% at about 0.61 to 0.63 of tuned arm A's FLOPs in a 3-seed pilot; random rows of the same size did worse than A
- Type: exploratory
- Status: tentative
- Confidence: low to medium (3 seeds, one tuning of A, FLOPs not wall-clock)
- Evidence: E-010 (pilot GO under the rule written before the run); E-009 (gradient concentration). S30 / S50 / R30 mean FLOPs ratio vs A: 0.634 / 0.768 / 1.413 at 0.97 and 0.612 / 0.707 / 1.181 at 0.975; no missed seeds; every individual seed of S30 and S50 is below the 0.85 GO line at both targets (highest: S50 seed 601 at 0.975, 0.809).
- Scope: MNIST MLP, hyperparameters inherited from `t1_base_two.yaml`, 3,000 steps, seeds 600 to 602, sel_warmup 100, FLOPs counted for the forward pass of the full batch plus the backward pass of the selected rows (optimizer and top-k ignored).
- Implications: a lead for a pre-registered test, not a claim (D-015). Most of the saving follows from the per-step cost, so the question for a confirmatory test is whether the step count stays near A's across seeds, tunings and a second setting. R30 shows that the selection by loss is what keeps the step count low.
- Found / last re-verified: 2026-10-25

## F-013: The meeting gradient is mostly parallel to the task gradient and lives on the same high-loss rows that selective backward keeps; at the tuned weight it is a small addition (checks A and L)
- Type: exploratory
- Status: tentative
- Confidence: low to medium (3 seeds, arm D at one tuned configuration, first-order local measurements, one weak probe)
- Evidence: E-012. Lower-layer cosine of the meeting and task gradients 0.75 to 0.80 at steps 500 to 2000; 57% to 64% of the meeting gradient's squared norm is parallel to the task gradient; its norm is 16% to 21% of the task gradient's before the weight 0.005. The meeting gradient has 97% to 99% of its norm in the top 30% of rows by loss (the task gradient: 97% to 99%, arm A 98% to 99.7%), with correlation 0.993 between a row's loss and the norm of its meeting direction. In h-space the meeting direction is closer to backprop's (cosine 0.60 to 0.68) than to the perfect-local-inverse direction (0.25 to 0.31), and backprop and that direction differ (0.37 to 0.44). Pre-registered verdicts: Q1 NOT REDUNDANT, Q2 NO, Q3 GN DIFFERS, L ALREADY SELECTIVE.
- Scope: MNIST MLP, arm D at `t1_meet_two` (tanh, `lam_meet` 0.005), AdamW, seeds 600 to 602, 3,000 steps, 16 fixed training batches of 128 per checkpoint.
- Implications: consistent with the reading that the meeting signal adds little to backpropagation in this setup (same rows, mostly the same direction, small weight), not a test of it; G0 and B decide it. The Q1 verdict rests on a one-step probe that overshoots late in training (the task step raises the loss in 5 of 6 seed-steps at 1000 and 2000), so it should not be quoted as evidence that the orthogonal part carries information. The learned G does not approximate the curvature direction, while the curvature direction itself differs from backprop's, which keeps check B's oracle arms open as a different signal. The hybrid arm (meeting loss on the selected rows) is not added to `prereg-v2` (rule of check L). Any paper statement needs the scope above (C-15, C-16).
- Found / last re-verified: 2026-10-25

## F-014: At arm A's tuned settings, adding the learned meeting signal with its strength set relative to the task gradient slows training at every tested dose, and the larger doses blow up the middle representation (check G0)
- Type: exploratory
- Status: tentative
- Confidence: medium for "harms at rho 0.1 and above" (10 paired seeds, intervals far from 0, paired design removes tuning noise); low for any statement about doses below 0.1 or about the cause
- Evidence: E-013. Steps-to-target versus rho = 0: +76% / +60% (rho 0.1), +186% / +114% (0.3), +208% / +114% (1 and 3) at 0.97 / 0.975, all intervals above +40%; verdict DOSE HARMS by the rule fixed in advance. Median `h_var` after step 1000 is 7 at rho 0 and 1.2e3 to 1.3e5 at the tested doses, with the inverse loss following it; final validation accuracy falls from 0.980 to 0.976 (0.1), 0.955 (0.3), 0.78 and 0.83 (1 and 3).
- Scope: MNIST MLP, arm A's tuned settings (relu, AdamW, cosine, batch 128, 3,000 steps), learned `G` as in arm D with its tuned `lam_inv`, rho in {0, 0.1, 0.3, 1, 3}, seeds 1100 to 1109, with the ramp of 200 steps. The smallest dose is about 100 times arm D's effective dose (F-013), so this says nothing about doses near arm D's.
- Implications: the E-008 null for arm D is not a case of "a too weak signal that more strength would rescue": more strength made it clearly worse here. It does not show that the signal is harmful at arm D's own dose. A low-dose extension or an oracle target with bounded scale (check B) is needed before saying anything about the small-dose regime or about target quality. Feeds claim C-17.
- Found / last re-verified: 2026-10-26 (recomputed from the raw logs on a fresh clone at ba5e89d)

## F-015: With the same fixed hyperparameters, no dose of the learned meeting signal speeds up training in either activation; relu makes the harm grow with the dose, tanh keeps it flat up to rho 1 and breaks at rho 3 (checks G0, G1, G2)
- Type: exploratory
- Status: tentative
- Confidence: medium for "no dose helped by 10% or more" (three checks, 150 paired runs, intervals about +-7% wide at 10 seeds); low for the mechanism of the instability
- Evidence: E-013 (relu, rho 0.1 to 3: +60% to +208% steps), E-015 (relu, rho 0.001 to 0.03: -1% to +9%, harm appearing at 0.03), E-014 (tanh, rho 0.1 to 1: -4% to +6%, intervals include 0; rho 3: +79% to +130%). In tanh the meeting gradient is strongly aligned with the task gradient (+0.8) and the scale of `h` stays at its baseline up to rho 1; in relu `h_var` grows with the dose from the lowest dose on (2.6 times at rho 0.001, 10^2 to 10^4 times at rho 0.3 to 3).
- Scope: MNIST MLP, arm A's tuned `lr`, `wd`, schedule and batch size (not retuned for tanh), learned `G` with arm D's `lam_inv`, 3,000 steps, ramp of 200 steps, one meeting-loss form (linear in `h`, strength relative to the task gradient at `h`), seeds 1100 to 1129 in three blocks.
- Implications: (1) Giving the signal more strength does not rescue the idea in this setup: the E-008 result is not a too-weak-signal artifact. (2) The signal is not wrong in direction (alignment +0.8 in tanh); it is redundant with backpropagation at these settings, which is what F-013 suggested from geometry alone. (3) The one clear design lesson: a meeting term that is linear in `h` with the strength tied to the task gradient is unstable when the activation does not saturate; a method that uses such a signal needs a bound or a normalisation of the middle state's scale. (4) The additive use of the signal is exhausted by these checks; what remains open is whether the signal can replace part of backpropagation at a lower cost (check D), and whether a better target changes the picture (check B). The Gate 3 condition of `mechanism-overview.md` is not met. **Correction to F-014's reasoning:** F-014's implication that an unbounded `h` is the cause stands as a hypothesis; the claim in the G1/G2 plan that tanh "bounds `h`" was wrong (see E-014).
- Found / last re-verified: 2026-10-26 (recomputed from the raw logs on a fresh clone at 5fee443)

## F-016: On this CPU the selective backward arm saves much less wall-clock time per step than FLOPs: 17.5% at one thread and 8.6% at eight, against 38.7%
- Type: exploratory
- Status: tentative
- Confidence: low to medium (three repeats; the one-thread verdict is within the noise of its 0.85 threshold, per-repeat ratios 0.61 to 0.92)
- Evidence: E-016 (time per step S30 over A: 0.825 at one thread, 0.914 at eight; exact FLOPs ratio 0.613)
- Scope: one CPU (WSL2), MNIST MLP at batch 128, A's tuned hyperparameters, a hand-written backward on the plain MLP; no GPU, no wider model, data loading not timed.
- Implications: (1) The FLOPs saving overstates the speed saving at this model size: top-k selection, the row gather and per-step Python overhead are not in FLOPs. (2) Combined with the step ratios of E-017 and E-018 (about 0.98 to 1.20), a derived and unmeasured estimate puts the time to target at about 0.8 to 1.0 of A's at one thread and 0.9 to 1.1 at eight, so a wall-clock gain is not established. (3) Any wall-clock statement in the paper must come from a measurement of time to target and cover more than one thread setting and, ideally, a larger width or a GPU.
- Found / last re-verified: 2026-10-26 (recomputed from `results/processed/timing_sw1.csv`)

## F-017: A plain smaller batch does not clearly beat batch 128, and against the best small batch the margin of selective backward shrinks to 12% at 0.97 and 29% at 0.975 (3 seeds)
- Type: exploratory
- Status: tentative
- Confidence: low (3 evaluation seeds; the 0.97 value of the best small batch rests on one seed; the small-batch arms had a lighter tuning than arm A)
- Evidence: E-017 (verdict PARTIAL; b48 0.889 and 0.921 of A's FLOPs, S30 0.778 and 0.652, S30 over b48 0.876 and 0.708; per-seed values in the entry); E-018 and E-010 for S30's advantage over A on other seeds
- Scope: MNIST MLP, learning rate tuned only (8 trials, 2 seeds) for batch 32, 48 and 64 with arm A's other settings, horizons of equal samples, S30 with A's hyperparameters, one CPU's worth of FLOP counting (exact).
- Implications: (1) The baseline-in-batch-size question is not settled: it is neither a clear problem nor clearly fine. By the rule the best small-batch arm joins the reference set of `prereg-v2`. (2) The earlier statement in the prep note that the expected effect is far from a 15% margin holds against arm A (ratios 0.65 to 0.78 here) but **not against the best small batch at 0.97** (0.876, one seed at parity); a 15% margin against the best of all references would be at risk of an inconclusive outcome. (3) S30's advantage over A at 0.97 varies by seed group (0.78 here, 0.65 in E-018, about 0.6 in E-010), so the 0.97 target is the noisier of the two. (4) Options for `prereg-v2`, for the researcher to decide: keep 15% against the best of all references and accept the risk; judge the primary comparison against A1 and report the best small batch as a secondary reference; tune the small-batch arms as fully as A before the test (which makes the reference harder); or lower the margin (a decision with a cost for the claim). None is chosen here.
- Found / last re-verified: 2026-10-26 (recomputed from the raw logs on a fresh clone at c4d101b)

## F-018: The FLOPs-to-target of selective backward is flat across the kept fraction 0.1 to 0.3 (0.58 to 0.63 of A at 0.975) and worse at 0.5; the knee rule lands on the lowest tested value
- Type: exploratory
- Status: tentative
- Confidence: low to medium (3 seeds; S10's step penalty at 0.97 is large and seed-dependent)
- Evidence: E-018 (ratios 0.693 / 0.584, 0.629 / 0.582, 0.646 / 0.626, 0.796 / 0.733 for sel_frac 0.1, 0.2, 0.3, 0.5; steps ratios 1.27 to 0.98 at 0.97); exact per-step cost 0.500, 0.557, 0.613, 0.722
- Scope: MNIST MLP, arm A's tuned hyperparameters, `sel_warmup` 100, top-loss rows, 3,000 steps, seeds 1210 to 1212.
- Implications: (1) Keeping fewer rows than 30% gives no clear extra saving: the smaller per-step cost is offset by more steps. (2) The knee rule's answer (0.1) is the edge of the grid; the position of the real optimum is open, and 0.2 looks at least as good as 0.1 on these seeds. (3) The forward pass is 44.4% of A's step, so the achievable per-step saving is bounded at about 56%; the measured best at 0.975 (0.58) is already close to what the per-step bound allows once the extra steps are paid.
- Found / last re-verified: 2026-10-26 (recomputed from the raw logs on a fresh clone at c4d101b)
