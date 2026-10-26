"""Write configs/prereg_v2/{S30,S20,R30}.yaml (A1's tuned config with the selective keys, D-023). The files must equal
src.prereg2.derive_config(name); tests/test_prereg2.py checks it. Existing files are never overwritten.

  python scripts/make_prereg2_configs.py
"""
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from anywhere

from src.prereg2 import DERIVED, derive_config
from src.sweep import REPO

NOTE = {"S30": "S at sel_frac 0.3, A1's hyperparameters (H7a primary, H7b)", "S20": "S at sel_frac 0.2 (secondary, descriptive)",
        "R30": "random control: 30% of the rows chosen at random (descriptive, never decides)"}

for name in DERIVED:
    path = REPO / "configs" / "prereg_v2" / f"{name}.yaml"
    want = derive_config(name)
    if path.exists():
        assert yaml.safe_load(open(path)) == want, f"{path} differs from its derivation"
        print(f"{path} exists and matches")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# prereg-v2 (D-023): {NOTE[name]}. Derived from configs/tuned/t1_base_two.yaml by src/prereg2.py; do not edit.\n"
                    + yaml.safe_dump(want, sort_keys=False))
    print(f"wrote {path}")
