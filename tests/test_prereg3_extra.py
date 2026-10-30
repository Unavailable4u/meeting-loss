"""The independent recomputation of the prereg-v3 results (tracks/improve/prereg3_extra.py, E-027) must reproduce the committed table exactly."""
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "prereg_v3"
TABLE = ROOT / "results" / "processed" / "prereg_v3.csv"
spec = importlib.util.spec_from_file_location("prereg3_extra", ROOT / "tracks" / "improve" / "prereg3_extra.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

needs = pytest.mark.skipif(len(list(RAW.glob("*.jsonl"))) < 260 or not TABLE.exists(), reason="the 260 prereg-v3 logs or the table are not in this checkout")


def test_reach_is_the_two_consecutive_rule_and_a_last_evaluation_alone_does_not_count():
    assert pe.reach([0.9, 0.97, 0.98, 0.98], 0.97) == 1
    assert pe.reach([0.9, 0.98, 0.96, 0.98], 0.97) is None
    assert pe.reach([0.9, 0.9, 0.9, 0.98], 0.97) is None
    point, lo, hi = pe.pct(pe.np.array([2.0, 2.0, 2.0]), pe.np.array([1.0, 1.0, 1.0]), 0.05, n_boot=100)
    assert (point, lo, hi) == (-50.0, -50.0, -50.0)


@needs
def test_independent_recomputation_equals_the_committed_table_exactly(capsys):
    pe.main()
    out = capsys.readouterr().out
    diffs = [float(x) for x in re.findall(r"max abs difference ([0-9.e+-]+)", out)]
    assert len(diffs) == 6 and max(diffs) < 1e-9
    assert "worst difference of any number above 0.00e+00" in out
    assert out.count("-> supported") == 6
