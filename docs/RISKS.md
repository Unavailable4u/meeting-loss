# Risk Register

Review weekly. Update status and detection notes as things change. Template columns: ID, risk, detection, mitigation, status. Never delete a risk; mark it `closed` with the reason.

| ID | Risk | Detection | Mitigation | Status |
|---|---|---|---|---|
| R-01 | Prior work overlaps (difference target propagation, ladder networks, deep supervision) | Literature scan; reviewer-style novelty check | Pin the contribution to the auxiliary use next to backprop and to specific variants (surprise-weighted, multi-depth); state the overlap openly | open |
| R-02 | Extra compute from the backward network `G` cancels any speedup | FLOPs-to-target comparison | Smaller `G`; share layers; compare at equal FLOPs | open |
| R-03 | Collapse or noisy targets | Variance of `h`; alignment early in training | Warm-up, detach rules, smaller `lam_meet` | open |
| R-04 | Gains disappear against properly tuned baselines | Equal-budget tuning | Report honestly; reframe as a negative result | open |
| R-05 | Forking paths (too many variants) | Variant count in `PLAN.md` | Pre-listed variants only (max 3 per stage); report all | open |
| R-06 | Works only at small scale | Stage 6 scaling check | State the limit; do not generalize | open |
| R-07 | FLOP accounting error | Unit test on a known layer; cross-check with wall-clock | Fix and rerun | open |
| R-08 | Time overrun | Weekly review against `PLAN.md` | Cut Stage 5b or Stage 6 first; keep the core comparison rigorous | open |
| R-09 | Class labels carry too little information for useful per-example targets | Alignment near zero in Stage 3 | Move to tasks where the result determines intermediate steps (Stage 5b) | open |
| R-10 | Starter code in `src/` is untested and may contain bugs | Stage 2 unit tests | Run the tests first; fix before any result is trusted | open |
