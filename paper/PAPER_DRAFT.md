<!--
LIVING DRAFT RULES (docs/GUIDE.md, Section 7)
1. No numeric or comparative claim without a claim ID like [C-07].
2. Every claim ID has evidence IDs in paper/CLAIMS.md.
3. When a finding changes status, re-check every claim that cites it.
4. Each section keeps a status header (skeleton | draft | stable) and the date it was last checked against docs/FINDINGS.md.
5. Write stable parts early (Method, Setup); Results last.
6. Record limitations as soon as you know them.
Terminology must match docs/METHOD.md. Related Work must match the novelty statement in docs/LITERATURE.md.
-->

# <Working title: e.g. "Meeting in the Middle: A Bidirectional Target Loss for Faster Training">

Status: skeleton | Version: v0.1 | Last checked against FINDINGS: YYYY-MM-DD

## Abstract
Status: skeleton | Depends on: all sections
Problem (1 sentence) → idea (1) → key result with number [C-..] (1) → scope and limitation (1).

## 1. Introduction
Status: skeleton | Depends on: H1-H4, LITERATURE
- The cost of training and the credit-assignment path
- The idea in one paragraph
- Contributions (each one a claim ID): [C-01] ... [C-02] ...

## 2. Related work
Status: skeleton | Depends on: LITERATURE.md
- Target propagation and difference target propagation
- Auxiliary and deep-supervision losses
- Ladder networks, autoencoder and cycle-consistency objectives
- Predictive coding and local-learning rules
- Efficient optimizers
- How this work differs (must match novelty notes in DECISIONS)

## 3. Method
Status: skeleton | Depends on: METHOD.md
- Setup and notation
- The meeting loss and its three components
- Algorithm box
- Cost analysis (extra FLOPs, memory)

## 4. Experimental setup
Status: skeleton | Depends on: D-003..D-008
- Datasets and splits; models; arms A-D
- Tuning protocol and budgets; seeds; metrics (steps-to-target, FLOPs-to-target)
- Compute used

## 5. Results
Status: skeleton | Depends on: F-IDs from Stages 3, 5, 6
- 5.1 Main comparison [C-..]
- 5.2 Comparison with other auxiliary losses [C-..]
- 5.3 Generalization: CIFAR-10, modular arithmetic [C-..]
- 5.4 Robustness and scale [C-..]

## 6. Analysis
Status: skeleton | Depends on: F-IDs from Stage 4
- Ablations; gradient alignment (mechanism); failure cases

## 7. Limitations
Status: skeleton
- Scale; tasks tested; added compute; cases where it hurts; unresolved questions

## 8. Conclusion
Status: skeleton

## References
## Appendix
- A. All hyperparameters and search ranges
- B. All variants tried (including failures)
- C. Extra plots
- D. Reproduction commands and code link
