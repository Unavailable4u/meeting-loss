"""Extra looks at look 2 (tracks/improve/look2_extra.py, E-025): reads the committed logs only, no torch needed."""
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "raw" / "improve_look2"
spec = importlib.util.spec_from_file_location("look2_extra", ROOT / "tracks" / "improve" / "look2_extra.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)

needs_logs = pytest.mark.skipif(len(list(LOGS.glob("*.jsonl"))) < 120, reason="the 120 look 2 logs are not in this checkout")


def test_pairs_name_only_groups_of_the_plan():
    import yaml
    names = {g["name"] for g in yaml.safe_load(open(ROOT / "tracks" / "improve" / "look2_plan.yaml"))["groups"]}
    assert all(a in names and b in names for a, b in le.PAIRS)


@needs_logs
def test_warmup_gap_equals_warmup_cost_plus_fewer_steps(capsys):
    """FLOPs are linear in steps after the warm-up, so the gap S20 minus S20w0 splits exactly into 100 x (full - selective step) and steps x selective step."""
    le.main()
    out = capsys.readouterr().out
    rows = re.findall(r"target (0\.97\d?): gap ([\d.]+) GFLOPs = warm-up's extra cost ([\d.]+) \+ fewer steps (\d+) x selective step ([\d.]+) \(sum ([\d.]+)\)", out)
    assert len(rows) == 2
    for _, gap, warm, _, step, total in rows:
        assert float(warm) + float(step) == pytest.approx(float(gap), abs=0.02)
        assert float(total) == pytest.approx(float(gap), abs=0.02)
