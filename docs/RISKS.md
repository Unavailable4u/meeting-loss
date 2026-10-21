# Risk Register

Review weekly. Update status and detection notes as things change. Template columns: ID, risk, detection, mitigation, status. Never delete a risk; mark it `closed` with the reason.

| ID | Risk | Detection | Mitigation | Status |
|---|---|---|---|---|
| R-01 | Prior work overlaps (difference target propagation, ladder networks, deep supervision) | Literature scan; reviewer-style novelty check | **Materialized 2026-10-21:** the loss is the SDTP/DTP target. Mitigation executed: contribution reframed as a compute-matched study (D-012); overlap stated openly; repeat the literature search before the lock and before submission | open (reframed, D-012) |
| R-02 | Extra compute from the backward network `G` cancels any speedup | FLOPs-to-target comparison | Smaller `G`; share layers; compare at equal FLOPs. Measured overhead (F-003): meet +22.9%, meet with noise +34.6%, meet_ao +44.6%, meet_frozen +11.7%. Arm D needs about a 19% step reduction to break even | open |
| R-03 | Collapse or noisy targets | Variance of `h`; alignment early in training | Warm-up, detach rules, smaller `lam_meet` | open |
| R-04 | Gains disappear against properly tuned baselines | Equal-budget tuning | Report honestly; reframe as a negative result | open |
| R-05 | Forking paths (too many variants) | Variant count in `PLAN.md` | Pre-listed variants only (max 3 per stage); report all | open |
| R-06 | Works only at small scale | Stage 6 scaling check | State the limit; do not generalize | open |
| R-07 | FLOP accounting error | Unit test on a known layer; cross-check with wall-clock | Fix and rerun | open |
| R-08 | Time overrun | Weekly review against `PLAN.md` | Cut Stage 5b or Stage 6 first; keep the core comparison rigorous | open |
| R-09 | Class labels carry too little information for useful per-example targets | Alignment near zero in Stage 3 | **Supported by Bartunov et al. 2018** (SDTP much worse than DTP on CIFAR; AO-SDTP recovers part). Mitigations: variant V2 (augmented output), V1 (noise), Stage 5b tasks where the result determines intermediate steps | open |
| R-10 | Starter code in `src/` may contain bugs | Unit tests | Partly mitigated: 17 tests pass on PyTorch 2.14.1 CPU (checked 2026-10-21 while preparing v1.1), and arm A ran end to end (E-001). Arm D and the variants have only been smoke-tested on synthetic data, not on MNIST; run `pytest` first in every environment | open |
| R-11 | Licensing: the DSN reference repository permits only educational, research and non-profit use; the paper notes a patent disclosure | Review before any commercial use | Arm B reimplemented from the paper, no code copied (D-016); check with the university before commercial use of DSN-style code or ideas | open |
| R-12 | The gain is just an effective learning-rate increase on the lower layers | Control arm E (`low_lr_mult`) and reference arm F | Tune E with the same budget as D (D-013, H5); note Adam normalizes gradient scale, so use SGD with momentum to make F informative | open |
| R-13 | ReLU is a poor fit for TP-style targets (Lee et al.: 3.15% vs 1.62% error) while backprop prefers it, so a fixed activation could bias the comparison | Compare arms under both activations | `act` in {relu, tanh} is a hyperparameter of every arm (D-013, D-014) | open |
| R-14 | Exploratory findings are reported as confirmed claims | Claim without a confirmatory or verification evidence entry | D-015: Type field on every log entry; claims only from confirmatory or verification evidence | open |
| R-15 | Comparison with Crowded Valley numbers is not apples to apples (different tasks, TensorFlow 1.15 DeepOBS) | Review of any comparison statement | Adopt their methodology; compare only to numbers we reproduce on the same tasks and protocol | open |
| R-16 | The 5,000-sample validation set limits resolution (about 0.22 points of sampling error near 97.5%) | F-002 | Use targets 0.97 and 0.975; consider a larger validation split by a new decision | open |
| R-17 | A weak baseline (constant learning rate, narrow search) inflates the apparent gain | Re-tuning noise floor; schedule as a hyperparameter | D-014: 25 trials, schedules, `act`, noise floor, 10 final seeds | open |
