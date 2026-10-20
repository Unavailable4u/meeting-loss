# Plan

**Last updated:** 2026-10-21
**Project start (week 1):** 2026-10-21. Set real calendar dates for each stage below once you know your availability. Weeks are counted from the start date and are rough (part-time pace).
**Current stage:** Stage 0 (project setup)

## Current focus

Finish Stage 0 and start the reading sprint and Stage 1 in parallel.

## Next three actions

1. Review and commit this repo skeleton; create the remote repository and push (record the choice of host and visibility in DECISIONS).
2. Start the reading sprint (five papers, below) and fill in `docs/LITERATURE.md` as you read.
3. Start Stage 1: set up the environment and build the plain MNIST baseline (arm A).

## Stage table

| Stage | Name | Target weeks | Status | Exit criteria (short) |
|---|---|---|---|---|
| 0 | Project setup and documents | 1 | **In progress** | All living docs exist; D-001 to D-011 recorded; repo pushed |
| 1 | Baseline reproduction | 1 | Not started | About 98% MNIST validation accuracy; reproducible across seeds; targets chosen (D-003); FLOP counter verified |
| 2 | Implement the meeting loss and test it | 1 to 2 | Not started | All unit tests pass; no collapse in diagnostic runs; METHOD matches code |
| 3 | Main comparison (arms A to D) | 2 to 3 | Not started | H1 and H2 evaluated against pre-registered thresholds; gate decision recorded |
| 4 | Diagnostics and ablations | 3 to 4 | Not started | Components that matter identified; H3 evaluated |
| 5 | Generalization (CIFAR-10, modular arithmetic) | 4 to 6 | Not started | H4 evaluated on at least one setting beyond MNIST |
| 6 | Robustness and scale | 6 to 7 | Not started | Table of where the method helps, does not help, and hurts |
| 7 | Write-up and release | 7 to 10 | Not started | Paper claims all supported or reworded; code reproduces one headline result on a clean machine |

Full task lists and gates for each stage are in `docs/GUIDE.md`, Section 6.

## Reading sprint (2 to 3 days, in parallel with Stages 0 and 1; finish before the pre-registration lock)

Read each in three passes (abstract and figures, then method, then experiments and limitations). After each, fill in its row in `docs/LITERATURE.md`.

| # | Paper | Why it matters |
|---|---|---|
| 1 | Lee et al. 2015, *Difference Target Propagation* | Closest relative of the design |
| 2 | Bartunov et al. 2018, on scalability of biologically motivated algorithms | Shows this method family struggling at scale |
| 3 | Rasmus et al. 2015, *Ladder Networks* | Combines bottom-up and top-down paths with matching |
| 4 | Lee et al. 2015, *Deeply-Supervised Nets* | The arm B baseline |
| 5 | Schmidt et al. 2021, *Descending through a Crowded Valley* | Evaluation rigor |

If any paper already does the auxiliary meeting loss in essentially the same form, that is a drift trigger: record it, and adjust the hypothesis before running the main experiments.

## Limits to prevent forking paths

- Maximum pre-listed variants per stage: **3** (log every variant tried, including failures).
- The pre-registration lock (git tag `prereg-v1`) happens **before** the Stage 3 main runs.

## What would make us stop or pivot

- The reading sprint shows the same auxiliary loss is already published in essentially the same form: reframe the contribution or pick the backup.
- Stage 3 gate outcomes (see `docs/GUIDE.md`, Section 6): no gain, or the gain is matched by arm B or C.
- Gradient alignment is near zero or negative after warm-up in Stage 2 or 3: diagnose before continuing.
- Compute overhead of the backward network cancels any gain at equal FLOPs.

**Designated backup plan:** the optimizer ideas 1 to 3 from `docs/reference/02-brain-vs-model-comparison.md`, Section 8 (surprise-gated learning rate, homeostatic update normalization, metaplastic per-parameter rates). Also a candidate for a companion study.

## Parked levers (from the proposal's lever map)

Optimizer, architecture (as primary focus), loss-function variants beyond the meeting loss, gradient computation (replacing backpropagation), and data. Revisit when a lead appears. Record leads in `docs/FINDINGS.md` as `tentative` or in the leads section of the experiment log.

## Status log

| Date | Note |
|---|---|
| 2026-10-21 | Repo created. Three guides added under `docs/`. D-001 to D-011 and DR-001 recorded. |
