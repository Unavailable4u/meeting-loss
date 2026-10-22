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
