# Claims Register

Every claim the paper makes, mapped to evidence. A claim may only appear in `PAPER_DRAFT.md` with its ID. Status values: `planned` (no evidence yet), `supported`, `weak`, `refuted`. When a finding (`docs/FINDINGS.md`) changes status, re-check every claim that cites it and update the paper wording to match. Template: `docs/templates/claim-row.md`.

| ID | Claim (exact paper wording) | Type | Evidence (F-/E- IDs) | Status | Paper location | Last checked |
|---|---|---|---|---|---|---|
| C-01 | Adding a target-propagation-style auxiliary loss next to backpropagation reduces the FLOPs needed to reach a target validation accuracy compared with a tuned baseline at equal total compute. (provisional; final wording must match the locked H1 threshold and the noise floor) | empirical | none yet (H1) | planned | Sec. 5.1, Abstract | 2026-10-21 |
| C-02 | The gain is not matched by DSN-style deep supervision or an autoencoder-reconstruction auxiliary loss at the same layer. (provisional) | empirical | none yet (H2) | planned | Sec. 5.2 | 2026-10-21 |
| C-03 | The auxiliary loss's gradient on the lower layers is positively aligned with the true task gradient. (provisional) | empirical | none yet (H3) | planned | Sec. 6 | 2026-10-21 |
| C-04 | The effect holds beyond MNIST and across optimizers. (provisional; may be weakened or dropped depending on Stage 5 and 6 results) | empirical | none yet (H4) | planned | Sec. 5.3, 5.4 | 2026-10-21 |
| C-05 | The auxiliary target is the simplified difference target propagation (SDTP) target applied at a middle layer; we make no claim of a new loss. (final wording depends on the novelty statement in docs/LITERATURE.md) | background | docs/LITERATURE.md; D-012 | planned | Sec. 1, 2 | 2026-10-21 |
| C-06 | The gain is not explained by an effective learning-rate increase on the lower layers. (provisional) | empirical | none yet (H5) | planned | Sec. 5.2, 6 | 2026-10-21 |
| C-07 | Per-step compute overhead of each arm relative to the baseline (meeting loss about +23% FLOPs, deep supervision about +0.6%, autoencoder auxiliary about +116% for this MLP). | empirical (deterministic count) | F-003 | supported | Sec. 3 (cost analysis) | 2026-10-21 |
