# Notes: Deeply-Supervised Nets (DSN) paper and reference code

Read 2026-10-21. Sources: the paper (arXiv:1409.5185, Lee, Xie, Gallagher, Zhang, Tu) and the reference repository `github.com/s9xie/DSN` (a Caffe fork). Purpose: specify arm B (decisions D-013, D-016). The code is **not** run or copied; arm B is reimplemented in PyTorch.

## What the paper says

- Companion objective on hidden layers: overall objective plus, for each hidden layer m, `alpha_m * [ ||w_m||^2 + loss_m(W, w_m) - gamma ]_+` with classifier heads `w_m` (SVM or softmax). The hinge at `gamma` makes the companion term vanish once it is small enough.
- Optional decay of the companion weight: `alpha_m * 0.1 * (1 - t/N)` with `t` the epoch and `N` the total number of epochs (reads as a linear decay to zero).
- Claims: faster convergence (loose theory assuming local strong convexity; the authors call the assumption very loose), more intuitive early-layer features, 4.55 times larger gradient variance in the first conv layer, better generalization, and larger gains with little training data (26% gain at 500 samples).
- Results: MNIST 0.39%, CIFAR-10 9.78% (8.22% with augmentation), CIFAR-100 34.57%, SVHN 1.92%. No compute accounting.

## What the reference code does (CIFAR-10, `examples/best_cifar10_9.65`)

- Network: Network-in-Network style with mlpconv blocks. Linear classifier heads ("ip_svm" layers) with **squared hinge loss** are attached after conv1, conv2 and conv3, plus the output loss.
- Solver: base learning rate 0.025, momentum 0.9, weight decay 5e-4, 50,000 iterations, step schedule.
- Companion heads use very small learning-rate multipliers (about 0.001 to 0.002 relative to the trunk).
- **Paper and code differ.** The prototxt has no explicit companion loss weights and no hinge threshold. The only decay found is a **learning-rate decay on the companion heads' parameters** (policy "power_inv": `lr * (1 + 1e-4 * iter)^-0.75`, with `localgamma`/`localpower` solver settings). So the paper's "decaying loss weight with threshold" and the code's "decaying head learning rate" are two different implementations of "the companion objective fades out".
- The repository README says softmax losses were used for CIFAR-100 and SVHN because hinge loss converged poorly with more than 10 classes.

## How we use it (D-013)

- **B-paper (primary):** cross-entropy companion head on the middle state, weight `lam_aux` decaying linearly to zero (`aux_sched: linear`), optional threshold (`aux_gamma`).
- **B-code (secondary baseline-implementation variant):** constant weight, with the companion head's learning rate decaying as in the reference code (`aux_lr_decay: power_inv`).
- Cross-entropy instead of squared hinge, because we use 10 classes with softmax everywhere else and the README reports hinge trouble beyond 10 classes.
- Both variants are logged; B-code does not count against the three-variant limit for the meeting loss.

## License and patent note (D-016, R-11)

The repository license grants permission to copy, modify and distribute the software for **educational, research and non-profit purposes** without fee. Any other use requires contacting the university's technology transfer office. The paper's first page also notes a patent disclosure (UC San Diego docket). This is fine for a research baseline implemented from the paper. Before any commercial use of DSN-style code or ideas, check with the university. Do not copy code from the repository into this project.
