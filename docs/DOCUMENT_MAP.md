# Document Map: how everything connects

**Purpose:** so that anyone, including you in a later session, can understand in about five minutes how the documents in this repo relate, which ones are authoritative, and where to look for what.

---

## 1. The big picture: three layers

| Layer | Document | What it is | Status |
|---|---|---|---|
| **Strategy** | `docs/reference/01-proposal-and-starting-guide.md` | The original proposal. Explains the goal (more efficient training), the **map of five levers** (optimizer, architecture, loss function, gradient computation, data), the general research rigor rules, and a first-paper plan. | Background. Phase 1 focus **superseded** by D-009 (see Section 5). |
| **Idea source** | `docs/reference/02-brain-vs-model-comparison.md` | How brains learn and remember compared with models, and a **bank of 12 mechanism ideas** (its Section 8) for improving training. | Background and ongoing idea bank. Consult it when you need the next idea. |
| **Execution** | `docs/GUIDE.md` | The working implementation guide for **one chosen idea** (the meeting loss): exact method, stages and gates, fairness rules, the living-docs system, the drift protocol, the paper-draft system, starter code. | **Authoritative** for how to run the project. |

Think of it as: **the proposal says where we might dig, the brain document says what might be buried there, and the guide says how we dig in one specific spot.**

---

## 2. How they synchronize: the flow

```
 Proposal (strategy, lever map) ──────┐
                                      ├──► pick ONE idea ──► HYPOTHESES.md + METHOD.md
 Brain comparison (idea bank) ────────┘                              │
                                                                     ▼
                                        GUIDE.md stages run it ──► EXPERIMENT_LOG.md
                                                                     │
                                                                     ▼
                                                                FINDINGS.md ──► paper/CLAIMS.md ──► paper/PAPER_DRAFT.md
                                                                     │
        results change the plan ◄──────── drift protocol (GUIDE Section 4) ◄──────┘
                │
                └──► new leads go back into the idea bank; "what worked or failed" goes back into the lever map
```

In words:

1. **Strategy and ideas feed a decision.** The proposal narrows the field to levers, the brain document supplies candidate mechanisms, and one idea is chosen and written down as a hypothesis (`HYPOTHESES.md`) and a method (`METHOD.md`). The choice and its reasons go in `DECISIONS.md`.
2. **The guide executes it.** The stages in `GUIDE.md` produce runs, which are logged in `EXPERIMENT_LOG.md`.
3. **Evidence becomes belief, then claims.** Analysis of runs produces `FINDINGS.md` entries (with confidence), which support or refute entries in `paper/CLAIMS.md`, which in turn govern what `paper/PAPER_DRAFT.md` is allowed to say.
4. **Drift flows backward.** When results, bugs, or new papers change anything, the **drift protocol** (`GUIDE.md`, Section 4) says exactly which documents to update, in which order. This keeps the plan, records, and paper consistent.
5. **The loop closes.** A failed or finished idea sends you back to the idea bank (`reference/02`, Section 8) and the lever map (`reference/01`, Section 3) to choose the next one.

---

## 3. Which document answers which question

| Question | Go to |
|---|---|
| What is this project, in one paragraph? | `README.md` |
| Where are we right now, and what do I do next? | `docs/PLAN.md` |
| Why did we choose X (or stop doing Y)? | `docs/DECISIONS.md` |
| What exactly are we testing, and what counts as success? | `docs/HYPOTHESES.md` |
| What exactly is the method (equations, which loss trains what)? | `docs/METHOD.md` (copied from `GUIDE.md`, Section 2, and kept current) |
| What have we run, and with what result? | `docs/EXPERIMENT_LOG.md` |
| What do we currently believe, and how sure are we? | `docs/FINDINGS.md` |
| What does related work say, and how do we differ? | `docs/LITERATURE.md` |
| What exactly did a specific paper or reference repo say (details, numbers, caveats)? | `docs/notes/` (for example `dsn-reference-notes.md`) and the rows in `docs/LITERATURE.md` |
| What could go wrong? | `docs/RISKS.md` |
| What changed from the original plan, and what did it touch? | `docs/DRIFT_LOG.md` |
| What is the paper saying, and is every claim backed? | `paper/PAPER_DRAFT.md` and `paper/CLAIMS.md` |
| What other ideas could we try? | `docs/reference/02-brain-vs-model-comparison.md`, Section 8 |
| What other parts of training could we work on later (loss, gradient computation, data, architecture)? | `docs/reference/01-proposal-and-starting-guide.md`, Section 3 (the lever table) and Section 9 (the backlog) |
| What are the rules for running experiments fairly? | `docs/GUIDE.md`, Section 5 |
| How do I keep everything consistent when things change? | `docs/GUIDE.md`, Section 4 |

---

## 4. Precedence: what to trust when documents disagree

From most to least authoritative:

1. **`DECISIONS.md`**: the latest *accepted* decision on a topic wins.
2. **`HYPOTHESES.md` and `METHOD.md`**: the current definition of what is tested and how.
3. **`PLAN.md`**: the current schedule and stage status.
4. **`GUIDE.md`**: the general procedure (it is updated through its changelog, and decisions can override its defaults).
5. **`docs/reference/*`**: historical background. These are **frozen**: do not edit their content. If something in them is out of date, record a decision and, if useful, add a status banner at the top.

If you find a disagreement that is not already explained, that is a drift event: open an entry in `DRIFT_LOG.md`.

---

## 5. The one known historical mismatch (already resolved)

The proposal's Phase 1 focus was the **optimizer and architecture** levers, with a 12-week timeline. When the meeting-loss idea was chosen, the work moved to the **loss function and gradient signal** levers (a loss added next to backpropagation, plus a small backward network). This was recorded as **D-009** and **DR-001**, and the 10-week timeline in `GUIDE.md` supersedes the 12-week one (**D-010**).

| Document | Still valid | Superseded |
|---|---|---|
| Proposal (`reference/01`) | The five-lever map, the rigor rules, the pitfalls list, the paper outline, the reading list | Phase 1 focus (optimizer/architecture first), the 12-week timeline. The optimizer ideas are now the **backup plan** and a possible companion study. |
| Brain comparison (`reference/02`) | Everything as an idea bank. Ideas 1 to 3 (surprise-gated learning rate, homeostatic normalization, metaplastic rates) are the **designated backup** if the meeting loss fails its gate. | Nothing |
| Guide (`GUIDE.md`) | Everything, as the current plan | Nothing (changes go to its changelog) |

---

## 6. The ID system

| Prefix | Meaning | Lives in |
|---|---|---|
| `H` | Hypothesis | `docs/HYPOTHESES.md` |
| `D` | Decision (bugs are logged as decisions titled "BUG: ...") | `docs/DECISIONS.md` |
| `E` | Experiment run group | `docs/EXPERIMENT_LOG.md` |
| `F` | Finding (curated conclusion) | `docs/FINDINGS.md` |
| `C` | Paper claim | `paper/CLAIMS.md` |
| `R` | Risk | `docs/RISKS.md` |
| `DR` | Drift entry | `docs/DRIFT_LOG.md` |
| `FIG` | Figure or table | `paper/FIGURES.md` |

The evidence trail for any statement in the paper is: **claim (C) → finding (F) → experiment (E) → raw results → git commit.**

---

## 7. Resuming work in a new session (checklist)

Do this at the start of every session, with any collaborator or AI assistant:

1. Read `README.md` and this file.
2. Read `docs/PLAN.md`: current stage, exit criteria, next three actions.
3. Read the last five entries of `docs/DECISIONS.md` and `docs/DRIFT_LOG.md`.
4. Skim open items in `docs/FINDINGS.md` (status `tentative`) and `docs/RISKS.md`.
5. Run `git log --oneline -20` to see recent work.
6. Run the tests (`pytest`). If they fail, fix that before anything else.
7. State out loud what you are about to do and which document it updates.
8. At the end of the session: update `PLAN.md` (next actions), log runs, and commit docs with the work.

**Never rely on memory of earlier conversations.** The repo is the record. If something important was only discussed, write it down as a decision or a finding.

---

## 8. Maintenance responsibilities

| Event | Update |
|---|---|
| A run finishes | `EXPERIMENT_LOG.md` |
| A run group is analyzed | `FINDINGS.md`, then `CLAIMS.md`, then the paper draft |
| A choice is made | `DECISIONS.md` |
| A stage starts or ends | `PLAN.md`, run the drift check (`GUIDE.md`, Appendix C) |
| Anything in the drift-trigger list happens (`GUIDE.md`, Section 4) | `DRIFT_LOG.md` and the propagation matrix |
| A new paper is read | `LITERATURE.md` |
| Weekly | Weekly review checklist (`GUIDE.md`, Section 4) |
