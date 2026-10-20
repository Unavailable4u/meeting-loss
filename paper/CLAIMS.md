# Claims Register

Every claim the paper makes, mapped to evidence. A claim may only appear in `PAPER_DRAFT.md` with its ID. Status values: `planned` (no evidence yet), `supported`, `weak`, `refuted`. When a finding (`docs/FINDINGS.md`) changes status, re-check every claim that cites it and update the paper wording to match. Template: `docs/templates/claim-row.md`.

| ID | Claim (exact paper wording) | Type | Evidence (F-/E- IDs) | Status | Paper location | Last checked |
|---|---|---|---|---|---|---|
| C-01 | Adding the meeting loss reduces the FLOPs needed to reach a target validation accuracy compared with a tuned baseline at equal total compute. (wording provisional; final wording must match the locked H1 threshold) | empirical | none yet (H1) | planned | Sec. 5.1, Abstract | 2026-10-21 |
| C-02 | The gain is not matched by deep supervision or autoencoder-reconstruction auxiliary losses of similar cost. (provisional) | empirical | none yet (H2) | planned | Sec. 5.2 | 2026-10-21 |
| C-03 | The meeting loss's gradient on the lower layers is positively aligned with the true task gradient. (provisional) | empirical | none yet (H3) | planned | Sec. 6 | 2026-10-21 |
| C-04 | The effect holds beyond MNIST and across optimizers. (provisional; may be weakened or dropped depending on Stage 5 and 6 results) | empirical | none yet (H4) | planned | Sec. 5.3, 5.4 | 2026-10-21 |
| C-05 | The method is a variant of target-propagation-style learning signals used as an auxiliary loss next to backpropagation. (background; final wording depends on the novelty statement after the reading sprint) | background | docs/LITERATURE.md | planned | Sec. 2 | 2026-10-21 |
