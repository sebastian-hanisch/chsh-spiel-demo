"""Eigenschaftstests: Aussagen, die für ganze Bereiche der Einstellungen gelten (feste Zufalls-Seeds)."""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import chs_evaluation as V
import chs_game as G
import chs_quantum as Q
from chs_scenario import Settings

ROOT = Path(__file__).resolve().parent.parent


def _random_case(rng):
    return rng.uniform(0, math.pi / 4), rng.uniform(0, 1), rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)


def test_quantum_model_sends_no_message_for_random_states_and_angles():
    """Die Antwortverteilung des einen hängt nicht von der Frage des anderen ab (größte Abweichung unter 1e-12 in 300 Zufallsfällen)."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        theta, v, al, be = _random_case(rng)
        assert G.marginal_dependence(Q.joint_table(Q.density_matrix(theta, v), al, be)) < 1e-12


def test_every_table_of_the_model_is_a_distribution():
    rng = np.random.default_rng(2)
    for _ in range(100):
        theta, v, al, be = _random_case(rng)
        t = Q.joint_table(Q.density_matrix(theta, v), al, be)
        assert t.min() > -1e-14 and np.allclose(t.sum(axis=(2, 3)), 1.0)


def test_noise_mixes_linearly_with_the_maximally_mixed_value():
    """W(v) = ½ + v · (W(1) − ½) für feste Winkel."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        theta, v, al, be = _random_case(rng)
        full = Q.win_probability(Q.density_matrix(theta, 1.0), al, be)
        assert Q.win_probability(Q.density_matrix(theta, v), al, be) == pytest.approx(0.5 + v * (full - 0.5), abs=1e-12)


def test_rotating_all_four_angles_by_the_same_amount_keeps_the_bell_value():
    rho = Q.density_matrix(math.pi / 4, 0.9)
    base = Q.win_probability(rho, Q.BELL_ALPHA, Q.BELL_BETA)
    for shift in (0.1, 0.7, -1.3, math.pi):
        assert Q.win_probability(rho, tuple(a + shift for a in Q.BELL_ALPHA), tuple(b + shift for b in Q.BELL_BETA)) == pytest.approx(base, abs=1e-12)


def test_swapping_alice_and_bob_keeps_the_value():
    rng = np.random.default_rng(4)
    for _ in range(50):
        theta, v, al, be = _random_case(rng)
        rho = Q.density_matrix(theta, v)
        assert Q.win_probability(rho, al, be) == pytest.approx(Q.win_probability(rho, be, al), abs=1e-12)


def test_angles_repeat_after_180_degrees():
    rho = Q.density_matrix(0.6, 0.9)
    assert Q.win_probability(rho, (0.1 + math.pi, 0.9), (0.2, -0.3 - math.pi)) == pytest.approx(Q.win_probability(rho, (0.1, 0.9), (0.2, -0.3)), abs=1e-12)


def test_no_setting_of_the_family_beats_the_bell_value():
    rng = np.random.default_rng(5)
    top = max(Q.win_probability(Q.density_matrix(*_random_case(rng)[:2]), *_random_case(rng)[2:]) for _ in range(300))
    assert top <= Q.BELL_VALUE + 1e-12


def test_more_entanglement_never_lowers_the_best_value():
    values = [Q.max_win(Q.density_matrix(math.radians(d), 1.0)) for d in range(0, 46)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_more_noise_never_raises_the_best_value():
    values = [Q.max_win(Q.density_matrix(math.pi / 4, v / 100)) for v in range(0, 101)]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))


def test_classical_strategies_are_a_fixed_set_whose_mixtures_stay_below_the_maximum():
    w = G.random_weights(500, 17)
    assert G.mixture_values(w).max() <= 0.75 + 1e-12


def test_analysis_is_consistent_with_its_parts():
    a = V.analyse(Settings(theta_deg=20.0, vis_pct=90, alpha0_deg=5.0, alpha1_deg=50.0))
    assert a.w_best >= a.w_quantum - 1e-12
    assert a.q_questions.mean() == pytest.approx(a.w_quantum)
    assert a.s_questions.mean() == pytest.approx(a.w_shared)


def test_same_scenario_in_a_second_process_with_other_hash_randomisation():
    """Gleiche Einstellungen und Zufalls-Seed geben in einem zweiten Prozess dieselben Gewinnquoten (PYTHONHASHSEED 1 und 2)."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import chs_play as P; from chs_scenario import Settings; "
            "print(P.play(Settings(seed=77, rounds=300)).final_rates().tolist())")
    outs = []
    for hs in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True, env=env, check=True, timeout=120).stdout)
    assert outs[0] == outs[1] and outs[0].startswith("[")
