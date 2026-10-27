"""Writes the group configs and the plan of the Improve-track headroom study (docs/notes/improve-headroom-plan.md).

Every config is arm A1's tuned config (configs/tuned/t1_base_two.yaml) with only the keys named below changed, so all
arms share A1's hyperparameters (inherited, D-023 style). Run from the repository root:
    python tracks/improve/make_headroom_configs.py
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "tracks" / "improve" / "configs"
A1 = ROOT / "configs" / "tuned" / "t1_base_two.yaml"
SEEDS = "2100-2109"

S = dict(arm="selective", sel_warmup=100, sel_mode="top")
# name -> (role, changes to A1, note)
GROUPS = {
    "A1":      ("reference",   dict(),                                                         "arm A, A1's tuned config: the paired reference on the same seeds"),
    "S30":     ("control",     {**S, "sel_frac": 0.3},                                         "S as in prereg-v2 (kept 0.3, warm-up 100) on fresh seeds"),
    "S25":     ("descriptive", {**S, "sel_frac": 0.25},                                        "kept fraction 0.25"),
    "S20":     ("descriptive", {**S, "sel_frac": 0.2},                                         "kept fraction 0.2"),
    "S15":     ("descriptive", {**S, "sel_frac": 0.15},                                        "kept fraction 0.15"),
    "S10":     ("descriptive", {**S, "sel_frac": 0.1},                                         "kept fraction 0.1 (F-018: more steps at 0.97)"),
    "S20w0":   ("descriptive", {**S, "sel_frac": 0.2, "sel_warmup": 0},                        "kept 0.2, no warm-up"),
    "S20w25":  ("descriptive", {**S, "sel_frac": 0.2, "sel_warmup": 25},                       "kept 0.2, warm-up 25 steps"),
    "S20w50":  ("descriptive", {**S, "sel_frac": 0.2, "sel_warmup": 50},                       "kept 0.2, warm-up 50 steps"),
    "S20gn":   ("descriptive", {**S, "sel_frac": 0.2, "sel_score": "gradnorm"},                "kept 0.2, rows ranked by the last-layer gradient norm (Katharopoulos and Fleuret) instead of the loss"),
    "SOSGD":   ("descriptive", {**S, "sel_frac": 1.0, "sel_warmup": 0, "sel_schedule": "osgd"}, "Ordered-SGD rule for q: 1, 1/2, 1/4, 1/8, 1/16 of the batch at 80, 90, 95, 99.5 percent training accuracy"),
    "Sup":     ("descriptive", {**S, "sel_frac": 0.3, "sel_schedule": "linear", "sel_frac_start": 0.1, "sel_frac_end": 0.5},
                "kept fraction rising linearly from 0.1 to 0.5 after the warm-up (selection pressure falling, Loshchilov and Hutter)"),
    "SB2":     ("secondary",   dict(arm="sb", sb_beta=2.0, sb_history=1024),                   "Selective-Backprop as released, beta 2 (about 33% kept), one-epoch priming, second forward pass counted"),
    "SB2p100": ("secondary",   dict(arm="sb", sb_beta=2.0, sb_history=1024, sb_prime_steps=100), "Selective-Backprop, beta 2, priming 100 steps as in S"),
    "SB3":     ("descriptive", dict(arm="sb", sb_beta=3.0, sb_history=1024),                   "Selective-Backprop at the released default beta 3 (about 25% kept)"),
}


def main():
    base = yaml.safe_load(open(A1))
    OUT.mkdir(parents=True, exist_ok=True)
    plan_groups = []
    for name, (role, changes, note) in GROUPS.items():
        cfg = {**base, **changes, "stage": f"hr-{name}"}
        head = (f"# Improve-track headroom study (docs/notes/improve-headroom-plan.md), group {name}: {note}.\n"
                f"# Derived from configs/tuned/t1_base_two.yaml by tracks/improve/make_headroom_configs.py; do not edit.\n")
        (OUT / f"hr-{name}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
        plan_groups.append(f'  - {{name: {name}, config: tracks/improve/configs/hr-{name}.yaml, seeds: "{SEEDS}", role: {role}, note: "{note}"}}')
    plan = ("# Improve-track headroom study (exploratory; D-024, D-025; docs/notes/improve-headroom-plan.md).\n"
            "# Not a confirmatory plan: no lock tag. Run with: python tracks/improve/run_headroom.py --workers 4 --threads 1\n"
            "# Seeds 2100 to 2109 only; 2110 to 2199 stay free for follow-up looks (never reuse a seed across looks of the same question).\n"
            "horizon_steps: 3000\nseed_block: \"2100-2199\"\nout_dir: results/raw/improve_headroom\ngroups:\n" + "\n".join(plan_groups) + "\n")
    (ROOT / "tracks" / "improve" / "headroom_plan.yaml").write_text(plan)
    print(f"wrote {len(GROUPS)} configs and tracks/improve/headroom_plan.yaml")


if __name__ == "__main__":
    main()
