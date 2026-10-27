# tracks/: where the three lines of work keep their own files

Everything lives on `main` (D-024). A track is a folder and a plan note, not a branch.

| Track | Question | Folder | Plan notes (in `docs/notes/`) |
|---|---|---|---|
| Paper | Evaluation and negative-results paper: meeting-loss diagnosis, tuning noise, the controlled S30 confirmation | `paper/` (draft, claims) and `tracks/paper/` (scripts for figures, wall-clock analysis, a second setting) | to be written per item |
| Mechanism | Why does the meeting loss fail? Checks B, D, C (Gate 1 and 2 of `mechanism-overview.md`) | `tracks/mechanism/` | `mechanism-b-oracle-arm.md`, `mechanism-d-replacement-test.md`, `mechanism-c-deep-network-sgd.md` |
| Improve | Can a method beat Selective-Backprop under our protocol (for example a kept fraction that changes during training)? | `tracks/improve/` | to be written: the faithful Selective-Backprop arm, the headroom study, then a pre-registered plan |

Create a track folder when its first script or config is written; keep a short `README.md` in it saying what each file does.

Rules (full text in D-024):

1. `src/` is shared and is locked by the tag of the last confirmatory plan (`prereg-v2` today). Changes to it are additive only (a new arm or option; the behaviour of every existing arm stays bit-identical), need a drift entry, the regression check (`scripts/verify_regression.py`) and tests. Scripts, sweeps, analysis and configs of a track go in its folder.
2. Exploratory work follows a plan note written before the first run, uses selection seeds or declared budgets, and is logged as `exploratory`.
3. A confirmatory test has a plan file, hypotheses in `docs/HYPOTHESES.md`, a lock commit and a new tag (`prereg-v3`, ...). Its seeds are fresh and are not extended after the result.
4. Results of `prereg-v2` reproduce from the tag: `git checkout prereg-v2` (or `cc2830c`), then the commands in E-022.
