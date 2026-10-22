"""Tests for the crossing definition (src/crossing.py, decision D-018)."""
import pytest

from src.crossing import CROSSINGS, DEFAULT_CROSSING, crossing_index
from src.train import steps_to_target

FLUKE = [0.96, 0.976, 0.97, 0.976, 0.977, 0.98]      # a lone 0.976 at index 1, a real crossing at index 3


def test_default_is_two_consecutive():
    assert DEFAULT_CROSSING == "two"
    assert crossing_index(FLUKE, 0.975) == 3


def test_first_rule_is_fooled_by_the_fluke():
    assert crossing_index(FLUKE, 0.975, "first") == 1


def test_mean3_rule():
    # window means: i=2 -> 0.96867, i=3 -> 0.97400, i=4 -> 0.97433, i=5 -> 0.97767
    assert crossing_index(FLUKE, 0.975, "mean3") == 5
    assert crossing_index([0.99, 0.99], 0.975, "mean3") is None      # fewer than three evaluations


def test_target_equal_to_accuracy_counts_and_never_reaching_is_none():
    assert crossing_index([0.5, 0.975, 0.975], 0.975) == 1
    for rule in CROSSINGS:
        assert crossing_index([0.5, 0.6, 0.7, 0.8], 0.975, rule) is None
        assert crossing_index([], 0.975, rule) is None


def test_lone_final_evaluation_is_not_a_crossing_under_two():
    assert crossing_index([0.9, 0.976], 0.975, "two") is None
    assert crossing_index([0.9, 0.976], 0.975, "first") == 1


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError):
        crossing_index(FLUKE, 0.975, "median")


def test_steps_to_target_uses_the_same_definition_and_reports_the_first_of_the_pair():
    log = [dict(step=50 * (i + 1), flops=1000.0 * (i + 1), val_acc=a) for i, a in enumerate(FLUKE)]
    assert steps_to_target(log, 0.975) == (200, 4000.0)
    assert steps_to_target(log, 0.975, rule="first") == (100, 2000.0)
    assert steps_to_target(log, 2.0) is None
