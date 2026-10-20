# Literature Notes

For each paper: what it does, how it overlaps with our idea, what we do differently. Titles, years, and venues were written from memory and **must be verified** when you read each paper (fix them here if wrong). "Read" status values: `not read`, `skimmed`, `read in full`.

The five papers marked **SPRINT** should be read (three passes each) before the pre-registration lock. If any of them already does our auxiliary meeting loss in essentially the same form, that is a drift trigger (`GUIDE.md`, Section 4).

Last literature search for new overlapping work: not yet done.

## Target propagation and backward signals

| Priority | Citation (verify) | Read | What it does | Overlap with ours | What we do differently |
|---|---|---|---|---|---|
| **SPRINT** | Lee, Zhang, Fischer, Bengio, "Difference Target Propagation" (2015) | not read | | Closest relative: backward network `G`, target built from a difference | |
| | Bengio, "How auto-encoders could provide credit assignment in deep networks via target propagation" (2014) | not read | | | |
| | Meulemans et al., "A Theoretical Framework for Target Propagation" (2020) | not read | | | |
| **SPRINT** | Bartunov et al., "Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures" (2018) | not read | | Evidence on how this family scales | |
| | Lillicrap et al., "Random synaptic feedback weights support error backpropagation for deep learning" (2016) | not read | | | |

## Auxiliary and bidirectional objectives

| Priority | Citation (verify) | Read | What it does | Overlap with ours | What we do differently |
|---|---|---|---|---|---|
| **SPRINT** | Lee et al., "Deeply-Supervised Nets" (2015) | not read | | Our arm B baseline | |
| **SPRINT** | Rasmus et al., "Semi-Supervised Learning with Ladder Networks" (2015) | not read | | Bottom-up and top-down paths with matching | |
| | Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017) | not read | | | |
| | Hinton, Dayan, Frey, Neal, "The wake-sleep algorithm for unsupervised neural networks" (1995) | not read | | | |

## Collapse prevention

| Priority | Citation (verify) | Read | What it does | Overlap with ours | What we do differently |
|---|---|---|---|---|---|
| | Grill et al., BYOL (2020) | not read | | | |
| | Chen and He, SimSiam (2021) | not read | | | |
| | Bardes et al., VICReg (2022) | not read | | | |

## Reversible computation (a different use of "backward")

| Priority | Citation (verify) | Read | What it does | Overlap with ours | What we do differently |
|---|---|---|---|---|---|
| | Gomez et al., "The Reversible Residual Network: Backpropagation Without Storing Activations" (2017) | not read | | | |

## LLM backward reasoning (inference-time relatives)

| Priority | Citation (verify) | Read | What it does | Overlap with ours | What we do differently |
|---|---|---|---|---|---|
| | Weng et al., "Large Language Models are Better Reasoners with Self-Verification" (2022) | not read | | | |
| | Berglund et al., "The Reversal Curse" (2023) | not read | | | |
| | Chen et al., "Reverse Thinking Makes LLMs Stronger Reasoners" (2024) | not read | | | |

## Evaluation rigor

| Priority | Citation (verify) | Read | What it does | Overlap with ours | What we do differently |
|---|---|---|---|---|---|
| **SPRINT** | Schmidt, Schneider, Hennig, "Descending through a Crowded Valley: Benchmarking Deep Learning Optimizers" (2021) | not read | | How to evaluate fairly | |

## Backup-plan papers (optimizer ideas; read only if we pivot)

See `docs/reference/02-brain-vs-model-comparison.md`, Section 11, and `docs/reference/01-proposal-and-starting-guide.md`, Section 10 (Adam, AdamW, Lion, Shampoo, Muon, LARS/LAMB).

## Novelty statement (update after the reading sprint)

Not yet written. Draft it here after the sprint, record it as a decision, and make sure `paper/PAPER_DRAFT.md` (Related Work and Introduction) matches it.
