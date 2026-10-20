# Figure and Table Map

Every figure or table in the paper maps to a script and its data. Figures are regenerated from `results/raw/` by scripts in `scripts/`, never edited by hand. Status values: `planned`, `draft`, `final`.

| ID | What it shows | Script | Data source | Paper location | Status |
|---|---|---|---|---|---|
| FIG-01 | Validation accuracy vs steps and vs FLOPs for arms A to D, with confidence bands | scripts/make_figures.py (to write) | results/raw/ (Stage 3) | Sec. 5.1 | planned |
| FIG-02 | FLOPs-to-target per arm at several target accuracies (paired bootstrap CIs) | scripts/make_figures.py (to write) | results/raw/ (Stage 3) | Sec. 5.1, 5.2 | planned |
| FIG-03 | Gradient alignment between task and meeting gradients over training | scripts/make_figures.py (to write) | results/raw/ (arm D logs) | Sec. 6 | planned |
| FIG-04 | Ablation table (no L_inv, frozen random G, no warm-up, meeting depth, G size, distance) | scripts/make_figures.py (to write) | results/raw/ (Stage 4) | Sec. 6 | planned |
| FIG-05 | Generalization: CIFAR-10 and modular-arithmetic results | scripts/make_figures.py (to write) | results/raw/ (Stage 5) | Sec. 5.3 | planned |
| FIG-06 | Robustness and scale table (batch size, optimizer, width, depth, 4x scale) | scripts/make_figures.py (to write) | results/raw/ (Stage 6) | Sec. 5.4 | planned |
