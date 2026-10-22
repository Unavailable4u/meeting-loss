"""When does a run "reach" a target accuracy? One definition, shared by training, tuning and analysis
(decision D-018, which supersedes the first-crossing definition used until then).

A single evaluation at or above the target is often a fluke: in the first tuning pass (E-004) about
two thirds of runs that crossed 0.975 were back below it at the next evaluation. The default rule
therefore asks for two consecutive evaluations at or above the target.

Rules (the position returned is an index into the list of evaluations):
  first  first evaluation with accuracy >= target (the pre-D-018 definition, kept for sensitivity checks)
  two    first evaluation i with accuracy >= target at i and at i + 1 (DEFAULT). The reported step and FLOPs
         are those of evaluation i. A run whose only evaluation at or above the target is the last one
         has not reached it.
  mean3  first evaluation i >= 2 whose mean accuracy over evaluations i - 2 .. i is >= target. The reported
         step and FLOPs are those of evaluation i (the end of the window). Sensitivity check only.
"""
CROSSINGS = ("first", "two", "mean3")
DEFAULT_CROSSING = "two"


def crossing_index(accs, target, rule=DEFAULT_CROSSING):
    """Index of the evaluation at which the run reaches `target` under `rule`; None if it never does."""
    if rule not in CROSSINGS:
        raise ValueError(f"unknown crossing rule {rule!r}; choose from {CROSSINGS}")
    n = len(accs)
    for i in range(n):
        if rule == "first":
            ok = accs[i] >= target
        elif rule == "two":
            ok = i + 1 < n and accs[i] >= target and accs[i + 1] >= target
        else:
            ok = i >= 2 and (accs[i - 2] + accs[i - 1] + accs[i]) / 3.0 >= target
        if ok:
            return i
    return None
