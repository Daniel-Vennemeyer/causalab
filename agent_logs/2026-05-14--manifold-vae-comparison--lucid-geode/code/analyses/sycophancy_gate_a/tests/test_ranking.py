"""Unit tests for Gate A pure logic (no model)."""

import numpy as np

from analyses.sycophancy_gate_a.ranking import (
    agreement_rate,
    cycle_rate,
    gate_verdict,
    kendall_tau,
    pref_matrix,
    ranking_from_scores,
    win_scores,
)


def _total_order_comparisons(n):
    # item k more sycophantic than item m iff k > m (a clean 1-D order)
    comps = []
    for i in range(n):
        for j in range(i + 1, n):
            comps.append((i, j, max(i, j)))  # winner = higher index
    return comps


def test_win_scores_and_ranking_recover_order():
    n = 6
    comps = _total_order_comparisons(n)
    s = win_scores(n, comps)
    assert ranking_from_scores(s) == list(range(n))  # low->high


def test_cycle_rate_zero_for_total_order():
    n = 7
    P = pref_matrix(n, _total_order_comparisons(n))
    assert cycle_rate(P) == 0.0


def test_cycle_rate_high_for_rock_paper_scissors():
    # 3-cycle: 0>1, 1>2, 2>0
    comps = [(0, 1, 0), (1, 2, 1), (0, 2, 2)]
    P = pref_matrix(3, comps)
    assert cycle_rate(P) == 1.0


def test_agreement_rate():
    rep = {(0, 1): [1, 1, 1], (0, 2): [2, 2, 0], (1, 2): [1, 2, 1]}
    # 3/3, 2/3, 2/3 -> mean ~0.778
    assert abs(agreement_rate(rep) - (1.0 + 2 / 3 + 2 / 3) / 3) < 1e-9


def test_kendall_tau_identity_and_reverse():
    assert kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert kendall_tau([0, 1, 2, 3], [3, 2, 1, 0]) == -1.0


def test_gate_verdict_pass_fail():
    assert gate_verdict(0.9, 0.02)["passed"] is True
    assert gate_verdict(0.55, 0.02)["passed"] is False   # noisy judge
    assert gate_verdict(0.9, 0.4)["passed"] is False      # intransitive


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("ALL PASSED")
