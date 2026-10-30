"""Write configs/prereg_v3/{S20w0,S30,S20,S15,S20w10,R20w0}.yaml (A1's tuned config with the selective keys, D-026). The files must
equal src.prereg3.derive_config(name); tests/test_prereg3.py checks it. Existing files are never overwritten.

  python scripts/make_prereg3_configs.py
"""
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src.prereg3 import DERIVED, derive_config
from src.sweep import REPO

NOTE = {
    "S20w0": "S at sel_frac 0.2 with no warm-up, A1's hyperparameters (H8 primary, H9, H10)",
    "S30": "S at sel_frac 0.3, warm-up 100: the arm confirmed in prereg-v2 (reference of H9)",
    "S20": "S at sel_frac 0.2, warm-up 100 (descriptive: splits H9 into its two changes)",
    "S15": "S at sel_frac 0.15, warm-up 100 (descriptive: replication of the kept-fraction lead)",
    "S20w10": "S at sel_frac 0.2, warm-up 10 (descriptive: a short warm-up, no mechanism claim)",
    "R20w0": "random control: 20% of the rows chosen at random, no warm-up (descriptive, never decides)",
}

for name in DERIVED:
    path = REPO / "configs" / "prereg_v3" / f"{name}.yaml"
    want = derive_config(name)
    if path.exists():
        assert yaml.safe_load(open(path)) == want, f"{path} differs from its derivation"
        print(f"{path} exists and matches")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# prereg-v3 (D-026): {NOTE[name]}. Derived from configs/tuned/t1_base_two.yaml by src/prereg3.py; do not edit.\n"
                    + yaml.safe_dump(want, sort_keys=False))
    print(f"wrote {path}")
