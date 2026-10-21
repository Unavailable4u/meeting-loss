# Hypotheses

**Pre-registration status:** NOT LOCKED. (Edits on 2026-10-21 under DR-002 were made before the lock and are not post-hoc.) Thresholds below are *proposed*. They are finalized in D-003 after Stage 1 (when the baseline numbers are known) and locked with git tag `prereg-v1` **before** the Stage 3 main runs. Any change after the lock is labeled **post-hoc** and logged in `DRIFT_LOG.md`.

**Null result (applies to all):** no difference, or a loss, once the backward network's compute is counted.

---

## H1 (primary): the meeting loss makes training more efficient

- **Statement:** Adding the meeting loss reduces the steps and FLOPs needed to reach a target validation accuracy, compared with a tuned baseline at equal total compute.
- **Metrics:** FLOPs-to-target and steps-to-target (targets chosen so the baseline reaches them in every seed).
- **Proposed success threshold:** mean FLOPs-to-target at least 15% lower than the tuned baseline; paired 95% bootstrap confidence interval across seeds excludes zero; holds at at least 2 different target accuracies. The gain must also exceed the re-tuning noise floor of arm A (D-014). Note: arm D costs about 23% more FLOPs per step (F-003), so it needs about a 19% step reduction just to break even.
- **Status:** open
- **Linked findings:** none yet
- **Paper sections:** Introduction, 5.1

## H2 (specificity): the backward-derived target is what matters

- **Statement:** The gain is not matched by other auxiliary losses at the same layer: DSN-style deep supervision (arm B, primary implementation B-paper; B-code reported as a secondary implementation) and autoencoder reconstruction (arm C). Costs are compared at equal FLOPs, not equal cost per step.
- **Metrics:** same as H1, arms A to D.
- **Proposed success threshold:** arm D beats both arm B and arm C under the same criterion as H1.
- **Status:** open
- **Linked findings:** none yet
- **Paper sections:** 5.2

## H3 (mechanism): the meeting gradient points the right way

- **Statement:** The gradient of the meeting loss on the lower layers is positively aligned with the true task gradient, and larger alignment goes with larger speedup.
- **Metrics:** cosine similarity between `g_task` and `g_meet` (see `docs/METHOD.md`); correlation of per-seed alignment with per-seed speedup.
- **Proposed success threshold:** median alignment positive at at least 90% of evaluation points after warm-up. Expectation from the literature: alignment is likely positive but well below 1 (Lee et al. 2015, Theorem 1 gives only a lower bound under perfect inverses; Meulemans et al. 2020 argue DTP follows Gauss-Newton-like targets).
- **Status:** open
- **Linked findings:** none yet
- **Paper sections:** 6

## H4 (generality): the effect is not specific to one setting

- **Statement:** The effect holds on CIFAR-10, on a modular-arithmetic task, and with more than one optimizer.
- **Metrics:** same as H1 per setting.
- **Proposed success threshold:** supported on at least one setting beyond MNIST, and with both AdamW and SGD with momentum on at least one setting.
- **Status:** open
- **Linked findings:** none yet
- **Paper sections:** 5.3, 5.4

## H5 (confound control): the gain is not just a lower-layer learning-rate effect

- **Statement:** Arm D's gain over arm A is not reproduced by control E (arm A with a tuned lower-layer learning-rate multiplier `low_lr_mult`, tuned with the same trial budget). Control F (exact-gradient target) is reported as a reference.
- **Metrics:** same as H1.
- **Proposed success threshold:** arm D beats arm E under the H1 criterion.
- **Status:** open
- **Linked findings:** none yet
- **Paper sections:** 5.2, 6
- **Note:** under AdamW, control F rescales the lower-layer gradient, which Adam largely normalizes away, so F is most informative under SGD with momentum (see D-013).

---

## Hypothesis template (for new entries)

```
## H<N>: <statement>
- Metric(s):
- Success threshold: <value>   (pre-registered on YYYY-MM-DD, git tag prereg-v1; post-hoc changes marked)
- Status: open | supported | refuted | inconclusive
- Linked findings: F-...
- Paper sections: ...
```
