# Meeting Loss: a bidirectional learning signal for faster neural network training

**Status:** Stage 1 in progress, method v1.1 (see `docs/PLAN.md` for the live status).
**Started:** 2026-10-21

## What this project is

A research project asking one question:

> Can a *meeting loss*, which compares the forward-computed middle state of a network with a middle state back-calculated from the known target, give lower layers a shorter and better learning signal, so a model reaches the same quality in fewer steps and FLOPs than a tuned baseline?

Version 1 adds the meeting loss as an **auxiliary** signal next to ordinary backpropagation, tests it at small scale (MNIST, then CIFAR-10, then a modular-arithmetic task), and measures efficiency as steps and FLOPs to a target accuracy. The goal is a rigorous, honest paper, whether the result is positive, mixed, or negative.

**Honest framing (D-012):** after reading the prior work, the loss turned out to be the simplified difference target propagation (SDTP) target applied at a middle layer. We do not claim a new loss. The contribution is a compute-matched study of using that target as an auxiliary signal next to backpropagation, with controls and a mechanism diagnosis. See `docs/LITERATURE.md`.

## Start here (reading order for a newcomer, about 30 minutes)

1. This README.
2. `docs/DOCUMENT_MAP.md`: **how all the documents connect** and which one to trust when they disagree. Read this before anything else in `docs/`.
3. `docs/PLAN.md`: where the project is right now and what to do next.
4. `docs/DECISIONS.md`: why things are the way they are (skim the titles, read D-009 in full).
5. `docs/GUIDE.md`: the full implementation guide (stages, method, rules). Read Sections 0 to 2 first.
6. Only if needed: `docs/reference/` (background strategy and the idea bank).

## The documents in one table

| Layer | File | Role |
|---|---|---|
| Strategy (background) | `docs/reference/01-proposal-and-starting-guide.md` | The original proposal: the map of five levers (optimizer, architecture, loss, gradient computation, data) and general research rigor. **Historical:** its Phase 1 focus was superseded by decision D-009. |
| Idea source (background) | `docs/reference/02-brain-vs-model-comparison.md` | How brains and models learn and remember, plus a bank of 12 mechanism ideas (Section 8). The place to go for the *next* idea. |
| Execution (authoritative) | `docs/GUIDE.md` | The working plan for the meeting-loss idea: method, stages, gates, the living-docs system, drift protocol, paper-draft system, starter code. |
| Living records (authoritative) | `docs/PLAN.md`, `HYPOTHESES.md`, `METHOD.md`, `DECISIONS.md`, `EXPERIMENT_LOG.md`, `FINDINGS.md`, `LITERATURE.md`, `RISKS.md`, `DRIFT_LOG.md` | The current truth about the project. Updated continuously. |
| Paper | `paper/PAPER_DRAFT.md`, `CLAIMS.md`, `FIGURES.md` | The living draft, with every claim tied to evidence. |

## Repository layout

```
meeting-loss/
├── README.md              you are here
├── CLAUDE.md              instructions for AI coding assistants working in this repo
├── docs/                  all planning, decisions, findings (see DOCUMENT_MAP.md)
│   ├── notes/             reading notes on papers and reference code
│   ├── reference/         background documents (historical; do not edit, supersede instead)
│   └── templates/         blank templates for every kind of entry
├── paper/                 draft, claims register, figure map, bibliography
├── src/                   model, losses, training loop, diagnostics, FLOP counting (starter code, untested until Stage 2)
├── configs/               one YAML per experiment family
├── scripts/               sweeps, analysis, figure generation
├── tests/                 unit tests (run these before trusting any result)
├── results/raw/           raw run logs: append-only, never edited by hand
├── results/processed/     produced from raw by scripts
└── notebooks/             exploration only; nothing cited from here
```

## The eight project rules (full version in `docs/GUIDE.md`, Section 0)

1. Docs before code.
2. One source of truth per kind of information.
3. Never delete; mark entries superseded.
4. Pre-register hypotheses and thresholds before the main runs (git tag `prereg-v1`).
5. Every number in the paper traces back to a logged run.
6. Record negative and inconclusive results in full.
7. Docs change in the same commit as the work that changed them.
8. Run the drift check at every stage gate and weekly.

## Resuming work in a new session

Follow the checklist in `docs/DOCUMENT_MAP.md`, Section 7. In short: read `docs/PLAN.md` (current stage and next actions), the latest entries in `docs/DECISIONS.md` and `docs/DRIFT_LOG.md`, and the open items in `docs/FINDINGS.md` and `docs/RISKS.md`. Do not rely on memory of earlier conversations; the repo is the record.

## Conventions

- IDs: `H` hypothesis, `D` decision, `E` experiment, `F` finding, `C` paper claim, `R` risk, `DR` drift entry, `FIG` figure or table.
- Dates are `YYYY-MM-DD`.
- Run names: `YYYYMMDD_stage_arm_seed`.

## License

To be decided before any public release (record the choice as a decision).
