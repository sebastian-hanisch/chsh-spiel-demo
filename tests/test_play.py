"""Spielverlauf: Fragen, Antworten der fünf Mannschaften, Stichproben gegen die Erwartung mit vier Standardfehlern Marge, Reproduzierbarkeit."""

import math

import numpy as np
import pytest

import chs_evaluation as V
import chs_play as PL
from chs_scenario import Settings


def test_questions_are_fair_and_reproducible():
    x, y = PL.draw_questions(5, 100_000)
    assert abs(x.mean() - 0.5) < 4 * 0.5 / math.sqrt(100_000) and abs(y.mean() - 0.5) < 4 * 0.5 / math.sqrt(100_000)
    assert abs(np.mean(x & y) - 0.25) < 4 * math.sqrt(0.25 * 0.75 / 100_000)
    x2, y2 = PL.draw_questions(5, 100_000)
    assert np.array_equal(x, x2) and np.array_equal(y, y2)
    assert not np.array_equal(x, PL.draw_questions(6, 100_000)[0])


def test_fixed_team_answers_with_the_strategy_bits():
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 1])
    a, b = PL.answers_fixed(6, x, y)            # a = (0, 1), b = (1, 0)
    assert a.tolist() == [0, 0, 1, 1] and b.tolist() == [1, 0, 1, 0]


def test_box_and_communication_win_every_round():
    x, y = PL.draw_questions(11, 5000)
    for a, b in (PL.answers_box(11, x, y), PL.answers_communication(11, x, y)):
        assert ((a ^ b) == (x & y)).all()
    a, _ = PL.answers_box(11, x, y)
    assert abs(a.mean() - 0.5) < 4 * 0.5 / math.sqrt(5000)          # jede einzelne Antwort bleibt ein fairer Münzwurf


def test_play_has_five_teams_on_the_same_questions():
    p = PL.play(Settings(rounds=200))
    assert p.a.shape == (5, 200) and p.won.shape == (5, 200) and p.rounds == 200
    assert p.won[3].all() and p.won[4].all()


def test_fixed_team_loses_exactly_the_both_one_rounds():
    p = PL.play(Settings(rounds=500, strategy=0))
    assert int((~p.won[0]).sum()) == int(np.sum((p.x == 1) & (p.y == 1)))


def test_entangled_rate_is_within_four_standard_errors_of_the_exact_value():
    n = 200_000
    a = V.analyse(Settings(rounds=n))
    rate = PL.play(Settings(rounds=n)).final_rates()[2]
    assert abs(rate - a.w_quantum) < 4 * V.standard_error(a.w_quantum, n)


def test_shared_direction_rate_is_within_four_standard_errors_of_its_formula():
    n = 200_000
    s = Settings(rounds=n, alpha0_deg=10.0, alpha1_deg=70.0, beta0_deg=-20.0, beta1_deg=33.0)
    a = V.analyse(s)
    rate = PL.play(s).final_rates()[1]
    assert abs(rate - a.w_shared) < 4 * V.standard_error(a.w_shared, n)


def test_entangled_answers_alone_are_fair_coins():
    """Keine Nachricht: Alices Antwort ist unabhängig von Bobs Frage ein Münzwurf (Bell-Zustand)."""
    n = 100_000
    p = PL.play(Settings(rounds=n))
    for y in (0, 1):
        sel = p.y == y
        assert abs(p.a[2][sel].mean() - 0.5) < 4 * 0.5 / math.sqrt(sel.sum())


def test_cumulative_rates_end_at_the_final_rates():
    p = PL.play(Settings(rounds=300))
    assert p.cumulative_rates()[:, -1] == pytest.approx(p.final_rates())
    assert p.cumulative_rates()[:, 0].tolist() == p.won[:, 0].astype(float).tolist()


def test_same_settings_give_the_same_play():
    p, q = PL.play(Settings(seed=9)), PL.play(Settings(seed=9))
    assert np.array_equal(p.a, q.a) and np.array_equal(p.won, q.won)
    assert not np.array_equal(p.a, PL.play(Settings(seed=10)).a)
