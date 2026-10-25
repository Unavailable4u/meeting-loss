# Checks G1 and G2: what G0 left open (low doses, and a bounded middle state)

**Status:** plan and decision rules, written BEFORE any code or run (2026-10-26). Type: `exploratory` (D-015). Follows E-013 and F-014 (check G0: DOSE HARMS at every tested dose, with the middle representation blowing up). Do not edit the rules after seeing results; add a dated note below instead. **Needs the researcher's go-ahead on the points marked DECIDE before any code is written.**

## What G0 left open

1. **The grid skipped arm D's own dose.** F-013 puts arm D's effective rho at about 0.001; the smallest dose in G0 was 0.1.
2. **G0 used arm A's activation (relu), arm D uses tanh.** In relu the middle state `h` is unbounded; in tanh it lies in (-1, 1). The meeting term is linear in `h`, so with relu nothing stops `h` from growing along the learned direction, and G0 shows exactly that: `h_var` 7 at rho 0, 10 times larger already at step 100 for rho 0.1, 10^2 to 10^4 times larger later, with the inverse loss following. This is a confound in how G0 should be read as a statement about the learned loss itself. It is a hypothesis, not a finding, that the unboundedness is the cause.

So the two natural questions are: does a low dose help (G1), and does the harm disappear when `h` cannot blow up (G2)?

## On "fixing" the system

G0 gives one concrete, testable lead (the unbounded `h`), so it is right to follow it. Three rules keep that honest:
- **Diagnose before fixing.** G2 tests the cause with the smallest possible change (the activation), not a new mechanism.
- **A fix is a new variant, not a rescue of H1.** H1 and H6 stay refuted (E-008). Any variant that works would need its own tuning under the D-014 and D-017 protocol, on both baselines, and a new pre-registered test on fresh seeds (`prereg-v3` or later). Nothing in G1 or G2 is a claim about efficiency.
- **No tuning on the final seeds.** The seeds below are new blocks used only for these checks.

## Check G1: low doses (relu, arm A's tuned config)

- **Same design as G0** (`mechanism-dose-response.md`, implementation notes of 2026-10-26): arm `meet_rho`, arm A's tuned config, `lam_inv` 0.105792, ramp 200 steps, 3,000 steps.
- **Doses:** rho in {0, 0.001, 0.003, 0.01, 0.03}. rho = 0.001 is arm D's effective dose; rho = 0 is the paired reference.
- **Seeds:** 1110 to 1119 (new, the same in every dose). 5 x 10 = 50 runs. Tag `mr2`.
- **Metric:** as G0 (steps-to-target at 0.97 and 0.975, D-018 rule, misses charged 3,000 steps, paired change against rho = 0, 95% bootstrap interval); also `h_var` and the inverse loss.
- **Decision rule (steps-to-target; both targets must agree):**
  - LOW DOSE HELPS: some rho > 0 has a point change of at most -10% at both targets and an upper interval end below 0 at both.
  - LOW DOSE HARMS: rho = 0.03 has a point change of at least +10% at both targets and the point changes do not decrease with rho over {0.003, 0.01, 0.03} at either target.
  - FLAT: every rho > 0 has a point change within +-5% at both targets and an interval that includes 0.
  - MIXED: anything else, and also when both HELPS and HARMS hold; report the curve.
- **Reading:** HELPS says the signal is useful once its strength is in arm D's range, and the E-008 loss came from the rest of arm D's configuration (tuned for tanh, sensitive to scale); then check B becomes the next step. FLAT says the signal at its natural dose does nothing (consistent with F-013) and strengthens the negative result. HARMS down to 0.03 says the instability starts well below the doses of G0 and G2 becomes more important.

## Check G2: the same doses as G0 with a bounded middle state (tanh)

- **Change from G0: only `act: tanh`** (arm D's activation). Everything else as G0: arm A's tuned `lr`, `wd`, schedule and batch size (not retuned for tanh), `lam_inv` 0.105792, ramp 200, 3,000 steps.
- **Doses:** rho in {0, 0.1, 0.3, 1, 3} (the G0 grid, so the two checks can be laid side by side). rho = 0 is the paired reference **in tanh**, so every comparison is within the same activation; G2 says nothing about tanh against relu.
- **Seeds:** 1120 to 1129 (new). 50 runs. Tag `mr3`.
- **Metric and decision rule:** exactly those of G0 (DOSE HELPS / DOSE HARMS / FLAT / MIXED), plus the `h_var` and inverse-loss diagnostics.
- **Reading:**
  - If DOSE HARMS again with `h_var` staying small, the blow-up of `h` is not the cause; the harm comes from the signal itself (direction quality, or the pull of the meeting term against the task loss). That would strengthen the negative result and weaken the case for "fix the scale".
  - If the harm shrinks or disappears (FLAT or HELPS), the unbounded `h` was the main cause. The next step would be a *designed* bounded-scale variant of the meeting loss (see DECIDE 3), tuned and tested under the full protocol.
  - If tanh without the meeting signal (rho = 0) is itself much slower than relu's, say so, but it does not affect the within-activation comparison.
- **Limit:** tanh with arm A's `lr` and `wd` is not a tuned baseline. Any statement about speed against a tuned arm A would need the protocol of D-014; G2 makes none.

## Cost

100 runs of 3,000 steps in total (G1 and G2), the same per-run cost as G0 (50 runs took about as long as the researcher's run of E-013; time one run first). One torch thread per worker (E-011).

## Code changes needed (after approval; none made yet)

- `exploratory/dose.py` / `scripts/run_dose_response.py`: a per-tag setting for the dose list, the activation and the seed block (a small tag table checked against the committed configs), and a second decision function for G1 (the rule above); G2 reuses `decide`. These are new files and arguments only: no change under `src/`, so no new drift entry beyond a short note in DR-010's follow-up (the locked arms and the `meet_rho` arm are untouched). Tests first: the G1 rule on synthetic tables, the config derivation for each tag, `act: tanh` with rho = 0 reproducing arm A in tanh (the existing test pattern).

## DECIDE (researcher)

1. **Go ahead with G1 and G2 as written?** (alternatives: only G2, since it tests the cause; or only G1.)
2. **G1 doses:** include rho = 0.001 (arm D's effective dose, but only an estimate from F-013) or replace it with 0.03/0.1 overlap? Current: {0, 0.001, 0.003, 0.01, 0.03}.
3. **After G2, if the bounded state fixes the harm:** do you want a designed variant (for example a meeting term bounded in `h`, or a penalty on the scale of `h`)? It would be a method change: it needs your decision, a note with its own rule, tuning under D-014, and a new pre-registered test. Not part of G1 or G2.

## Decisions and implementation notes (2026-10-26)

- **Researcher's decisions (2026-10-26):** G1 and G2 go ahead as written; rho = 0.001 stays in G1; the designed bounded-scale variant (DECIDE 3) is to be considered only after G2 has been read, with its own note and rule.
- **Tags:** `mr1` = G0, `mr2` = G1, `mr3` = G2 (`exploratory/dose.py` `TAGS`: doses, seeds, activation and rule per tag; the runner refuses an unregistered tag for real runs). Configs: `configs/dose/mr2_rho*.yaml` and `mr3_rho*.yaml`, generated and checked against their derivation (G2's differ from G0's only in `act` and `stage`; a test checks this). Raw logs in `results/raw/dose/<tag>/`.
- **G1 rule** is `decide_low` in `exploratory/dose.py`, as written above. "HARMS" needs rho = 0.03 at +10% or more at both targets and the point changes not decreasing over {0.003, 0.01, 0.03} at either target. HELPS and HARMS together give MIXED.
- **No change under `src/`**: the arm, the losses and the FLOP counter are those of DR-010. Tests: `tests/test_dose.py` (the tag table and seed blocks, the G2 config, the G1 rule, analysis by the rule of the tag, the unknown-tag refusal; the existing rho = 0 reproduces arm A test already runs in tanh).
- **Run order:** one tag at a time, not together (E-011: one torch thread per worker, laptop otherwise idle). Commands: `python scripts/run_dose_response.py --tag mr3 --workers 4 --threads 1` then `--tag mr2`. G2 first, because it tests the cause.

## Correction and results (2026-10-26, after the runs)

- **Correction of the premise.** The section "What G0 left open" says that in tanh the middle state `h` "lies in (-1, 1)". That is wrong: `h` is the output of the last linear layer of `low`; the activation is applied at the start of `up`. `h` is unbounded in both activations. What tanh changes is that it saturates, which shrinks the task gradient at `h` for large |h|; the dose of `meet_rho` is defined relative to that gradient, so it shrinks with it. G2 therefore tests "does the activation change the instability", not "does bounding `h` remove it". The text above is kept as written (rules are not edited); this note supersedes the premise.
- **Results:** G2 (tag `mr3`): MIXED by the rule, with no harm at rho 0.1 to 1 and a clear harm at rho 3 (E-014). G1 (tag `mr2`): MIXED by the rule, no help at any low dose, a small harm appearing at rho 0.03 (E-015). Summary: F-015.
- **DECIDE 3 (designed bounded-scale variant):** the G2 outcome does not by itself call for it, because no dose helped even where nothing blew up. A scale control would only matter for a variant that had a reason to be used at larger doses (for example check B's oracle targets or the replacement setting of check D). Left to the researcher; proposed to revisit it when check B or D is planned.
