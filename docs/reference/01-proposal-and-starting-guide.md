# Making Neural Network Training More Efficient: Proposal and Starting Guide

> **Status note (2026-10-21):** This is a frozen background document. Its **Phase 1 focus (optimizer and architecture first)** and its **12-week timeline** were superseded by decisions D-009 and D-010 (`docs/DECISIONS.md`, `docs/DRIFT_LOG.md` DR-001). The optimizer ideas are now the designated backup plan. The five-lever map, rigor rules, pitfalls list, paper outline, and reading list remain valid. See `docs/DOCUMENT_MAP.md` for how this document fits with the others.

*Working draft v1. Update it as the project evolves.*

## 1. Summary

This project investigates whether changing one part of the learning loop (predict, measure error, work out how to improve, nudge parameters) can make a model reach the same quality with less training: fewer steps, less compute, or less memory.

Phase 1 focuses on the two most accessible levers, the **optimizer (update rule)** and the **architecture**, tested at small scale with strict experimental discipline. The other levers (loss function, gradient computation, data) are kept in a backlog (Sections 3 and 9) so you can return to them when you find an idea or lead.

## 2. Problem and research question

Training large models is expensive. Most training today uses the same recipe: backpropagation to compute gradients, then Adam or AdamW to update parameters. Any improvement in how fast a model learns saves time, money and energy.

**Core question:** Can a modified rule in the training loop reach a target quality in fewer steps (or less compute) than a well-tuned standard baseline?

**Hypothesis template** (fill this in once you pick an idea):

> Changing [component] from [standard version] to [your version] will reduce [metric] by at least [X%] on [model + dataset], because [mechanism].

A good hypothesis is specific, measurable, and explains why it should work. The "because" is what turns a trial-and-error result into a paper.

## 3. The map of levers

| Area | What you could change | Existing work | Difficulty | Plan |
|---|---|---|---|---|
| **Optimizer (update rule)** | How the nudge is computed from the gradient: step size, momentum, per-parameter scaling, update direction | SGD, momentum, Adam, AdamW, Lion, Muon, Shampoo, LARS/LAMB | Medium | **Phase 1** |
| **Architecture (prediction)** | Attention variants, normalization, initialization, residual scaling, sparsity, mixture-of-experts | Transformers, LayerNorm/RMSNorm, MoE, state-space models (Mamba) | Medium | **Phase 1** |
| **Loss function** | What counts as "error" and how examples are weighted | Label smoothing, focal loss, contrastive losses | Medium | Backlog |
| **Gradient computation** | Replace or approximate backpropagation | Forward-forward, feedback alignment, local learning rules | Hard | Backlog |
| **Data** | Order and selection of training examples | Curriculum learning, data pruning, deduplication | Medium | Backlog |

## 4. Phase 1 focus: candidate ideas

Treat these as starting points to test, not as novel claims. Each may already exist in some form, so check the literature before committing (Step 3 of the guide).

**Optimizer ideas**

1. **Adaptive learning rate driven by training signals.** Adjust the step size from recent loss behavior (increase when loss falls steadily, decrease when it oscillates) instead of using a fixed schedule.
2. **Per-layer update rules.** Give different layers different step sizes or update normalization (related: LARS, LAMB).
3. **Update normalization.** Rescale each update so its size is controlled regardless of gradient magnitude (related: Lion's sign-based update, Muon's orthogonalized updates).
4. **Smarter use of gradient history.** Change how momentum averages past gradients, for example by trusting recent gradients more when they have been consistent.

**Architecture ideas**

5. Initialization and residual scaling that speed up early training.
6. Normalization variants (placement or form) that tolerate higher learning rates.
7. Cheaper attention or layer designs that keep quality at lower compute.

**How to choose:** pick the idea where you can explain the mechanism in two sentences and implement it in under 50 lines. Start with ideas 1 to 3. They only touch the optimizer, so the model and data stay fixed and comparisons are clean.

## 5. Experimental design

**Models and data (small and fast)**

- MLP or small CNN on MNIST, then CIFAR-10 (minutes per run)
- Tiny GPT (nanoGPT-style, roughly 1 to 10M parameters) on a small text dataset such as Tiny Shakespeare
- Later: a somewhat larger model, to test whether the gain holds as scale grows

**Baselines**

AdamW (and SGD with momentum for vision), each with its learning rate and weight decay properly swept, under the same compute budget as your method.

**Metrics** (choose one headline metric before running anything)

- Steps to reach a target validation loss
- Wall-clock time to that target
- Final loss or accuracy at a fixed compute budget
- Per-step cost and memory. A method that needs half the steps but costs twice as much per step is not a win.

**Rigor rules**

- Run 3 to 5 seeds per setting and report mean and spread
- Give the baseline the same tuning effort as your method
- Use fixed data splits and never tune on the test set
- Run ablations: remove each component to show what actually matters
- Log everything: config, seed, code version

## 6. Step-by-step starting guide

**Step 1: Set up.** Python, PyTorch, scripts or Jupyter, and a way to track runs (Weights & Biases, or simply CSV files). MNIST does not need a GPU, and free Colab or Kaggle GPUs cover the small-scale work. A university lab GPU is better if you have access. Use git from day one.

**Step 2: Reproduce a baseline.** Write a plain training loop (data, model, loss, backward, optimizer step) for MNIST, then CIFAR-10. Train with AdamW and plot the loss curves. The goal is that you understand every line and your numbers match known results (for example, an MLP above 98% on MNIST).

**Step 3: Literature scan (1 to 2 weeks, in parallel).** Read the papers in Section 10. For each of your ideas, search arXiv, Google Scholar and Semantic Scholar for similar work. Record what exists, how it was evaluated, and what gaps remain. The field moves fast, so check recent arXiv listings too.

**Step 4: Implement your own optimizer.** In PyTorch a custom optimizer is a small class. Here is a skeleton:

```python
import torch

class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9):
        super().__init__(params, dict(lr=lr, beta=beta))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                # momentum: running average of gradients
                m.mul_(group["beta"]).add_(p.grad, alpha=1 - group["beta"])
                # YOUR RULE GOES HERE: replace the line below
                p.add_(m, alpha=-group["lr"])
```

First confirm it behaves like SGD with momentum, then change the update rule.

**Step 5: Run the first comparison.** Your optimizer against tuned AdamW on MNIST and CIFAR-10, 3 seeds each. Plot validation loss against steps and against wall-clock time. Expect most first ideas to show no gain. That is normal and informative.

**Step 6: Iterate.** Look at why the idea did or did not help: gradient norms, update sizes, per-layer behavior, loss curves. Refine the hypothesis. Keep a lab notebook of every experiment, including the failures.

**Step 7: Scale and ablate.** If a signal holds, move to the tiny GPT, then a larger model. Run ablations and test sensitivity to learning rate and batch size.

**Step 8: Write.** Start drafting the method and experiment sections early. Writing exposes what is missing.

**Rough timeline** (adjust to your pace)

| Weeks | Focus |
|---|---|
| 1 to 2 | Environment, baseline, start reading |
| 3 to 4 | First implementation and first comparisons |
| 5 to 8 | Iterate on the idea, scale up |
| 9 to 12 | Ablations, sensitivity tests, writing |

## 7. Common pitfalls

- **Untuned baselines.** The most common reason claimed improvements do not replicate.
- **Single seed.** Training is noisy, so one lucky run proves nothing.
- **Unequal compute.** Compare at the same step count and the same cost.
- **Skipping the literature check.** Claiming novelty for something that already exists.
- **Hidden fragility.** A method that only works at one learning rate is weak.
- **Ignoring per-step cost.** Fewer steps is not the same as less compute.
- **Overclaiming from small scale.** State the limits honestly. Many small-scale wins vanish at scale.

## 8. Paper outline

1. Abstract
2. Introduction: the problem, the gap, your contribution
3. Related work
4. Method: formulas, intuition, algorithm
5. Experimental setup
6. Results: main comparison, ablations, sensitivity, scaling
7. Limitations and discussion
8. Conclusion

Appendix: hyperparameters, extra plots, link to code. Releasing clean code makes the paper more credible. A common path is an arXiv preprint first, then a workshop or conference submission.

## 9. Backlog: other levers to revisit

Keep a **leads log**: whenever something odd shows up during experiments, write down the date, the setup and the plot. Many research directions start as an unexplained curve.

**Loss function**

- Ideas: weight examples by difficulty, or design losses that give a stronger signal early in training.
- Revisit if: early progress is slow, or easy examples dominate training.

**Gradient computation**

- Ideas: local learning rules, cheaper gradient approximations to save memory.
- Hard, and alternatives to backpropagation rarely match it, so revisit only with a specific insight or a clear memory-saving goal.

**Data**

- Ideas: curriculum ordering, pruning redundant examples, importance sampling.
- Revisit once the optimizer work is stable. It often combines easily with it.

## 10. Starting reading list

- Kingma and Ba, "Adam: A Method for Stochastic Optimization" (2014)
- Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (AdamW, 2017)
- Schmidt, Schneider and Hennig, "Descending through a Crowded Valley: Benchmarking Deep Learning Optimizers" (2021). Essential for understanding evaluation rigor.
- Chen et al., "Symbolic Discovery of Optimization Algorithms" (Lion, 2023)
- Gupta, Koren and Singer, "Shampoo: Preconditioned Stochastic Tensor Optimization" (2018)
- Keller Jordan et al., the Muon optimizer (2024), write-up and repository
- You et al., LARS (2017) and LAMB (2019)
- Andrej Karpathy's nanoGPT and the "Neural Networks: Zero to Hero" video series
- Goodfellow, Bengio and Courville, *Deep Learning*, the chapter on optimization

This list may be out of date. Search for recent optimizer and training-efficiency papers on arXiv before settling on an idea.
