import pytest

from jevpoc.eval import data, metrics


def test_metrics_known_values():
    assert metrics.accuracy([1, 2, 3], [1, 2, 0]) == pytest.approx(2 / 3)
    assert metrics.macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert metrics.auc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert metrics.ece([1.0, 1.0], [True, True]) == 0.0


def test_frozen_sets_have_disjoint_splits():
    for name in ("care_intent", "sim_change"):
        rows = data.load(name)
        assert {r["split"] for r in rows} == {"dev", "test"}
        assert len({r["id"] for r in rows}) == len(rows)


def test_frozen_sets_match_generator():
    assert data.load("care_intent") == data.care_intent()
    assert data.load("sim_change") == data.sim_change()


def test_harness_runs_on_mock(tmp_path, monkeypatch):
    from jevpoc.eval import harness

    monkeypatch.setattr(harness, "OUT", tmp_path)
    s = harness.main([])
    assert s["backend"] == "mock"
    assert (tmp_path / "report.html").exists()
    assert set(s["care_intent"]["arms"]) == {"baseline_rules", "jev_only", "jev_plus_rules"}
