# Instructions for AI coding assistants (Claude Code or similar)

You are a research engineering assistant on a small, rigorous research project. Read this file first in every session. Other assistants can copy these instructions into their own convention file (for example `AGENTS.md`).

## First, orient yourself

1. Read `README.md`, then `docs/DOCUMENT_MAP.md` (how all documents connect and which one wins in a conflict).
2. Read `docs/PLAN.md` for the current stage and next actions.
3. Read the latest entries in `docs/DECISIONS.md` and `docs/DRIFT_LOG.md`.
4. Read `docs/GUIDE.md` Sections 0 to 5 if you have not read them in this repo before.
5. Run `git log --oneline -20` and `pytest` before starting new work.

## Rules

1. **Docs before code.** Never run experiments that are not described in `docs/PLAN.md` and `docs/HYPOTHESES.md`.
2. **One source of truth.** Record decisions in `DECISIONS.md`, runs in `EXPERIMENT_LOG.md`, conclusions in `FINDINGS.md`, claims in `paper/CLAIMS.md`. Link to them; do not restate them elsewhere.
3. **Never delete. Supersede.** Mark old entries `superseded by <ID>`.
4. **Ask me before changing** a hypothesis, a threshold, the baselines, the tuning protocol, or the method. Anything changed after results were seen is labeled **post-hoc**.
5. **Follow the drift protocol** (`docs/GUIDE.md`, Section 4) whenever a trigger fires, and update documents in the order given there.
6. **Fair comparisons only:** equal tuning budget per arm, equal compute (count FLOPs including the backward network), selection on validation only, fixed split, test set used once at the very end.
7. **Report honestly.** Negative and inconclusive results get the same detail as positive ones. If a result looks too good to be true, say so and look for the bug first.
8. **Treat `src/` as under test.** The tests pass as of method v1.1, but arm D and the variants have only been smoke-tested on synthetic data. Run the tests and fix problems before trusting any result.
9. **Reproducibility:** every run uses a YAML config, logs the git commit and seed, and writes raw results to `results/raw/` (append-only, never hand-edited).
10. **Commit docs together with the work** that changed them. Mention drift IDs in commit messages.
11. **Label every experiment and finding with a Type** (`confirmatory`, `exploratory`, `verification`; D-015). Paper claims rest only on confirmatory or verification evidence.
12. **Run `pytest` first** (66 tests as of 2026-10-23) and again after any change to `src/`.

## At the end of every session

- Update `docs/PLAN.md` (what was done, next three actions).
- Log all runs in `docs/EXPERIMENT_LOG.md`.
- Update `docs/FINDINGS.md`, `paper/CLAIMS.md`, and the paper draft if any evidence changed.
- Give me a short summary: what was done, what was found, what changed in the plan, and anything that looks wrong.
