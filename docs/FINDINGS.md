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
