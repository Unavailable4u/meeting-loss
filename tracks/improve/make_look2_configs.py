"""Writes the group configs and the plan of look 2 of the Improve track (docs/notes/improve-look2-plan.md).

Every config is arm A1's tuned config with only the keys named below changed. Run from the repository root:
    python tracks/improve/make_look2_configs.py
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "tracks" / "improve" / "configs_look2"
A1 = ROOT / "configs" / "tuned" / "t1_base_two.yaml"
SEEDS = "2110-2119"

S = dict(arm="selective", sel_warmup=100, sel_mode="top")
GROUPS = {
    "A1":     ("reference",   dict(),                                                     "arm A, A1's tuned config: the paired reference on the same seeds"),
    "S30":    ("control",     {**S, "sel_frac": 0.3},                                     "S as in prereg-v2 (kept 0.3, warm-up 100)"),
    "S20":    ("descriptive", {**S, "sel_frac": 0.2},                                     "kept 0.2, warm-up 100: the reference of the warm-up and scale questions"),
    "S15":    ("descriptive", {**S, "sel_frac": 0.15},                                    "replication of S15 of look 1"),
    "S20w0":  ("descriptive", {**S, "sel_frac": 0.2, "sel_warmup": 0},                    "replication of S20 without warm-up (look 1: -17.8% / -14.6% against S20)"),
    "S15w0":  ("descriptive", {**S, "sel_frac": 0.15, "sel_warmup": 0},                   "combination: kept 0.15, no warm-up"),
    "S10w0":  ("descriptive", {**S, "sel_frac": 0.1, "sel_warmup": 0},                    "combination: kept 0.1, no warm-up"),
    "S20w5":  ("descriptive", {**S, "sel_frac": 0.2, "sel_warmup": 5},                    "warm-up dose: 5 steps"),
    "S20w10": ("descriptive", {**S, "sel_frac": 0.2, "sel_warmup": 10},                   "warm-up dose: 10 steps"),
    "S20w25": ("descriptive", {**S, "sel_frac": 0.2, "sel_warmup": 25},                   "warm-up dose: 25 steps"),
    "S20k":   ("descriptive", {**S, "sel_frac": 0.2, "sel_scale": "k"},                   "kept 0.2, warm-up 100, gradient divided by the number of kept rows (gradient-scale test)"),
    "S20w0k": ("descriptive", {**S, "sel_frac": 0.2, "sel_warmup": 0, "sel_scale": "k"},  "kept 0.2, no warm-up, gradient divided by the number of kept rows (gradient-scale test)"),
}


def main():
    base = yaml.safe_load(open(A1))
    OUT.mkdir(parents=True, exist_ok=True)
    plan_groups = []
    for name, (role, changes, note) in GROUPS.items():
        cfg = {**base, **changes, "stage": f"l2-{name}"}
        head = (f"# Improve-track look 2 (docs/notes/improve-look2-plan.md), group {name}: {note}.\n"
                f"# Derived from configs/tuned/t1_base_two.yaml by tracks/improve/make_look2_configs.py; do not edit.\n")
        (OUT / f"l2-{name}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False))
        plan_groups.append(f'  - {{name: {name}, config: tracks/improve/configs_look2/l2-{name}.yaml, seeds: "{SEEDS}", role: {role}, note: "{note}"}}')
    plan = ("# Improve-track look 2 (exploratory; D-024, D-025; docs/notes/improve-look2-plan.md).\n"
            "# Not a confirmatory plan: no lock tag. Run with: python tracks/improve/run_headroom.py --plan tracks/improve/look2_plan.yaml --workers 4 --threads 1\n"
            "# Seeds 2110 to 2119 (look 1 used 2100 to 2109); 2120 to 2199 stay free.\n"
            "horizon_steps: 3000\nseed_block: \"2100-2199\"\nout_dir: results/raw/improve_look2\ngroups:\n" + "\n".join(plan_groups) + "\n")
    (ROOT / "tracks" / "improve" / "look2_plan.yaml").write_text(plan)
    print(f"wrote {len(GROUPS)} configs and tracks/improve/look2_plan.yaml")


if __name__ == "__main__":
    main()
