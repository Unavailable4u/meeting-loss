# Paper plan: one paper, with explicit split triggers

**Status:** plan v1, 2026-10-25. Proposed together with D-020 and DR-008 (both marked proposed). The draft itself is `paper/PAPER_DRAFT.md`; claims live in `paper/CLAIMS.md`.

## Decision: one paper for now

The two lines (the auxiliary target loss, and the selective backward pass) answer the same question under the same protocol: *what reduces the compute needed to reach a target accuracy, measured fairly?* They share the setting (MNIST MLP), the baselines, the metric (FLOPs-to-target), the statistics and the lessons about tuning noise. The link between them is a measurement (check L). The claims do not strain each other, so there is no reason to divide the work.

## Why the paper does not depend on the selective backward result

| If the selective backward test ends | The paper is |
|---|---|
| SUPPORTED (and holds in a second setting, and wall-clock is reported) | A protocol, a negative result with a mechanism, and a conditional positive result |
| INCONCLUSIVE or REFUTED | A protocol, a negative result with a mechanism, and an honest second negative |
| Not finished by the deadline | A protocol and a negative result with a mechanism (the selective pilot goes in an appendix or future work, labelled tentative) |

The first study (D-019, pre-registered, confirmatory) already stands. Everything else strengthens it.

## What the paper can claim today (from `CLAIMS.md`)

- Supported: C-03 (alignment), C-07 (per-step overheads), C-09 (the meeting loss needs about 36% more FLOPs), C-12 (control F is nearly inert).
- Refuted, reported as null results: C-01, C-08 (deep supervision).
- Tentative, exploratory: C-10 (tuning noise), C-11 (crossing rule), C-13 (selective pilot), C-14 (gradient concentration).
- Planned, with provisional wording: the mechanism checks (C-15 to C-20), the selective test and its controls (C-21 to C-24).

## Split triggers (divide the paper only if one of these happens)

1. **Different headline.** The selective backward test is SUPPORTED in two settings, becomes the headline, and the meeting-loss material (null result plus mechanism checks) would push the paper past the length or focus of a single contribution. Then: paper 1 = selective backward (with the protocol), paper 2 = the meeting-loss negative result with its mechanism.
2. **A positive mechanism finding that needs its own confirmation.** For example check G0 ends DOSE HELPS or check B ends ORACLE HELPS, and a confirmatory study of the meeting loss becomes worthwhile. Then that study is a separate paper.
3. **No shared section.** If the claims register splits into two sets that share nothing except the protocol, and the link measurement (check L) turns out uninformative.

Without one of these, keep one paper. Re-check this list whenever a check or the selective test is logged.

## Writing order (GUIDE section 7)

Method and Setup now (stable). Related work as the reading is verified. Results as each finding is logged. Introduction and abstract last. Limitations continuously.

## Venue (a thought, not a decision)

A negative-results or reproducibility track, or a workshop on efficient training, fits the contribution. Revisit once the selective test and the second setting are known.

## Housekeeping

- Claims C-02, C-04 and C-06 depended on H1 being supported; they stay in the register (moot) and are not used in the text.
- FIG-05 and FIG-06 were planned for the positive-claim version; keep them as planned only if the selective line needs a second setting or robustness table.
