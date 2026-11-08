"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): Reglergrenzen, Entartungen, Gleichstände."""

import math

import numpy as np
import pytest

import chs_evaluation as V
import chs_experiments as XP
import chs_game as G
import chs_play as PL
import chs_quantum as Q
from chs_scenario import Settings


def test_pure_product_state_with_bell_angles_is_below_the_classical_bound():
    """θ = 0 mit den Winkeln des Bell-Falls: E = cos2α·cos2β; die vier Korrelatoren sind cos45°, cos45°, 0, 0, also S = √2 und W = ½ + √2/8 = 0,6768."""
    a = V.analyse(Settings(theta_deg=0.0))
    assert a.w_quantum == pytest.approx(0.5 + math.sqrt(2) / 8) and a.w_quantum < 0.75
    assert a.w_best == pytest.approx(0.75)


def test_zero_visibility_wins_one_half_everywhere():
    a = V.analyse(Settings(vis_pct=0))
    assert a.w_quantum == pytest.approx(0.5) and a.w_best == pytest.approx(0.5)
    assert V.verdict_resource(a)[0] == "warning" and V.verdict_state(a)[0] == "warning"


def test_all_angles_equal_gives_exactly_the_classical_value():
    """Alle Winkel 0: beide messen 0 auf |00⟩-Anteil; der Bell-Zustand gibt gleiche Antworten: wie die Strategie „immer gleich“ = 3 von 4."""
    a = V.analyse(Settings(alpha0_deg=0.0, alpha1_deg=0.0, beta0_deg=0.0, beta1_deg=0.0))
    assert a.w_quantum == pytest.approx(0.75) and V.verdict_resource(a)[0] == "info"


def test_angle_limits_minus_90_and_90_are_the_same_measurement():
    lo = V.analyse(Settings(alpha0_deg=-90.0, beta1_deg=90.0))
    hi = V.analyse(Settings(alpha0_deg=90.0, beta1_deg=-90.0))
    assert lo.w_quantum == pytest.approx(hi.w_quantum)


@pytest.mark.parametrize("deg", [0.0, 45.0])
def test_state_angle_limits_run_through_the_whole_analysis(deg):
    a = V.analyse(Settings(theta_deg=deg, vis_pct=100))
    assert 0.5 <= a.w_quantum <= 1.0 and a.threshold <= 1.0


def test_single_round_and_minimum_rounds_play():
    for n in (1, 2, 20):
        p = PL.play(Settings(rounds=n))
        assert p.rounds == n and p.cumulative_rates().shape == (5, n)


def test_last_strategy_and_first_strategy_have_the_same_value():
    assert G.strategy_value(0) == G.strategy_value(15) == 0.75


def test_the_worst_strategies_win_one_quarter():
    assert min(G.all_strategy_wins()) == 1


def test_e4_with_one_repetition_and_one_round():
    cell = XP.e4_cell(Settings(), 1, 1)
    assert cell["more"] + cell["equal"] + cell["less"] == pytest.approx(1.0)


def test_visibility_exactly_at_the_threshold_is_not_above_three_quarters():
    th = math.pi / 4
    v = Q.advantage_threshold(th)
    assert Q.max_win_closed_form(th, v) == pytest.approx(0.75, abs=1e-12)


def test_tie_between_equal_strategies_is_resolved_by_the_documented_rule():
    """Gleichstand der 8 besten Strategien: die Zählung nennt die Zahl (8), die Auswahl in der Seitenleiste entscheidet, nicht die Reihenfolge."""
    assert G.best_classical_wins()[1] == 8
    assert np.array_equal(G.mixture_values(np.eye(16))[[0, 15]], [0.75, 0.75])
