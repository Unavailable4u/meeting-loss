# Check D: is the meeting signal sufficient to train the lower half? (replacement test, backprop cut at the middle state)

**Status:** plan and decision rule, written BEFORE the first real run. Type: `exploratory`. Do not edit the rule after seeing results; if it needs to change, add a dated note below instead. Part of the mechanism study (`mechanism-overview.md`). Run it after check B (it reuses B's oracle direction).

## The idea being checked

All meeting-loss runs so far kept full backprop, so the meeting signal was only ever an add-on. If it matches what backprop sends, the lower half should train on the meeting signal **alone**. That is also the PLAN's option (b) (weaker reliance on backprop). The replacement test asks: *sufficiency*. It does not promise a FLOPs win; it is the necessary step before any "decoupled" or asynchronous story.

## What is replaced

Cut the gradient from the top half into the lower half and give the lower half only the meeting signal:

```
h = F_low(x)
z = F_up(h.detach())                  # L_task trains F_up only
L_task  = CE(z, y)                    # F_up
L_inv   = || G(sg(p)) - sg(h) ||^2    # G only (as before)
b       = sg(h) + G(e_y) - G(sg(p))
L_meet  = || h - b ||^2               # F_low only; the ONLY loss that reaches F_low
```

Reduction of `L_meet`: **sum over the 128 dimensions, mean over the batch** (the current mean-over-dimensions form divides the gradient by 128 and makes AdamW's epsilon matter more than it should when `F_low` has no other signal). No warm-up ramp from zero: with no other signal the lower half would otherwise not train at all early on. The replacement of the lower half's gradient starts at step 0; a warm-start arm below tests the alternative.

## The arms

| Arm | Lower-half signal | Question |
|---|---|---|
| A | normal backprop (`t1_base_two.yaml`) | reference (and A2 `t2_base_two.yaml`) |
| R1 | learned `G` (arm D's `G`, replacement from step 0) | is the signal sufficient with a real `G`? |
| R1w | backprop for the first 300 steps, then cut and switch to R1 | can it carry training once started (separates "bootstrap" from "carry")? Exploratory sub-arm. |
| R2 | oracle `P` of check B (exact `(S J)^+ (e_y - p)`, `rho` = 1 in the units of check B) | is the signal sufficient with a **perfect** local inverse? |
| R4 | frozen random `G` (the V3 recipe, replacement) | the direct-feedback-alignment-like baseline |

Check on the code, not an arm: replacing the lower half's gradient by the **exact task gradient** (inject `-g_h`) must reproduce arm A bit for bit, because the gradient reaching `F_low` is then identical. Make this a unit test.

## Tuning and seeds

- R1 gets the same budget as the t1 sweep: 25 trials x 2 seeds (950, 951), same space as arm D (`act`, `lr`, `wd`, schedule, `lam_inv`, `warmup_steps` is replaced by the 300-step switch point in {0, 100, 300, 600} so R1w is the same arm with a non-zero switch), D-017 score and D-018 rule. R2 and R4 use arm A's tuned config; as a robustness check they are also run on R1's tuned config on seeds 900 to 902.
- Final runs: seeds 900 to 909, 3,000 steps, A, A2, R1, R1w, R2, R4 (60 runs). Tag `prereg-v4` before the first final seed (lock as in D-019 point 8).

## Decision rule (R1 versus A; steps-to-target at 0.97 and 0.975 under rule `two`; the share of runs reaching each target; final validation accuracy at step 3,000)

- **SUFFICIENT:** R1 reaches both targets in at least 8 of 10 seeds, **and** its paired mean steps-to-target is within +15% of A at both targets (point estimate; the 15% is the project's margin).
- **PARTIAL:** R1 reaches 0.97 in at least 8 of 10 seeds but fails the other conditions of SUFFICIENT.
- **INSUFFICIENT:** R1 reaches 0.97 in fewer than 5 of 10 seeds, or its final accuracy at step 3,000 is more than 1 percentage point below A's mean.
- **UNCLEAR:** anything else.

**Diagnosis from R2 and R4 (reported in every case):**
- R1 PARTIAL or INSUFFICIENT but R2 SUFFICIENT: the quality of `G` is the bottleneck (problem 5, the imperfect, moving inverse).
- R1 and R2 both PARTIAL or INSUFFICIENT: the local-inverse signal itself does not carry enough to train on.
- R1 SUFFICIENT: replacement is viable. It does **not** show a speedup (see below).
- R4 matching R1: `G` adds nothing over a random feedback matrix (consistent with `meet_frozen` having alignment about 0 and still tying A at selection).

## What each outcome leads to

- **SUFFICIENT:** next questions are wall-clock or parallel execution (layer-wise asynchronous updates), which this single-device repo cannot measure, and whether the cheaper `G` of check B's phase 3 still suffices. Plan them separately.
- **PARTIAL / R1w SUFFICIENT:** the signal helps once the network has started; look at warm-start designs as a new, separately pre-registered variant.
- **INSUFFICIENT with the oracle too:** the meeting signal is not a substitute for backprop on this network; closes this route.

## Cost accounting (exact, from F-001; batch 128)

Cutting backprop at `h` saves only the input gradients of the two top layers: `2 x 128 x (256 x 10 + 128 x 256)` = 9.04M FLOPs, which is **5.8%** of arm A's 155.1M per step. `G`'s forward and backward adds more than that (+6% to +23% depending on the variant), so the replacement arm costs *more* than arm A per step in FLOPs. Any benefit would have to come from parallel execution, not from FLOPs.

## What this check can NOT show

- A speedup of any kind: sufficiency only. Asynchronous or parallel execution is not measured on one device.
- That the oracle `P` is optimal; it is the first-order local inverse (`rho` is fixed at 1, no tuning of the oracle arm).
- Behaviour on deeper networks or other split points; MNIST MLP, AdamW, 3,000 steps, one split.
- Related work exists and should be read before any claim: fixed-weight and direct feedback alignment style training of hidden layers (Lillicrap et al. 2016; Nokland 2016; Shibuya et al., AAAI 2023, fixed-weight DTP) and decoupled or synthetic-gradient training (Jaderberg et al. 2017, cited from memory: verify). R4 is essentially a baseline from that literature.

## How to run (once the code exists)

```
python -m pytest -q            # includes the exact-gradient-replacement == arm A test
git add -A && git commit -m "..." && git push
python scripts/run_sweep.py --tag t4 --arms meet_replace ...      # check the CLI flags
git tag prereg-v4 && git push --tags
python scripts/run_final.py --plan configs/stage4_replacement_plan.yaml
```

Template lines; check the real flags in `scripts/run_sweep.py` and `scripts/run_final.py`. Log the sweep and the final runs as `exploratory` entries in `EXPERIMENT_LOG.md`.
