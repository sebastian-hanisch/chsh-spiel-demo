"""Einheiten der Matrixrechnung an den Handrechnungen aus conftest.py."""

import math

import numpy as np
import pytest

import chs_quantum as Q


def test_state_vector_is_normalised_and_bell_at_45_degrees():
    for deg in (0, 10, 45):
        assert np.linalg.norm(Q.state_vector(math.radians(deg))) == pytest.approx(1.0)
    assert Q.state_vector(math.pi / 4) == pytest.approx([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert Q.state_vector(0.0).tolist() == [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("theta, v", [(0.0, 1.0), (math.pi / 4, 1.0), (0.4, 0.6), (0.4, 0.0)])
def test_density_matrix_is_a_state(theta, v):
    rho = Q.density_matrix(theta, v)
    assert np.trace(rho) == pytest.approx(1.0) and np.allclose(rho, rho.T)
    assert np.linalg.eigvalsh(rho).min() >= -1e-12


def test_density_matrix_without_visibility_is_the_maximally_mixed_state():
    assert np.allclose(Q.density_matrix(0.7, 0.0), np.eye(4) / 4)


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_visibility_outside_zero_to_one_is_refused(bad):
    with pytest.raises(ValueError):
        Q.density_matrix(0.5, bad)


def test_projectors_are_orthogonal_and_complete():
    p0, p1 = Q.projector(0.3, 0), Q.projector(0.3, 1)
    assert np.allclose(p0 + p1, np.eye(2)) and np.allclose(p0 @ p1, 0) and np.allclose(p0 @ p0, p0)
    assert Q.projector(0.0, 0).tolist() == [[1.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError):
        Q.projector(0.0, 2)


def test_bell_state_with_equal_angles_gives_equal_answers(bell):
    table = Q.joint_table(Q.density_matrix(math.pi / 4, 1.0), (0.0, 0.0), (0.0, 0.0))
    assert table[0, 0, 0, 0] == pytest.approx(bell["equal_angles"]["p00"])
    assert table[0, 0, 1, 1] == pytest.approx(bell["equal_angles"]["p11"])
    assert table[0, 0, 0, 1] == pytest.approx(0.0, abs=1e-15) and table[0, 0, 1, 0] == pytest.approx(0.0, abs=1e-15)


def test_bell_state_agreement_is_cos_squared_of_the_angle_difference():
    rho = Q.density_matrix(math.pi / 4, 1.0)
    for delta in (0.1, 0.5, 1.0, 2.0):
        table = Q.joint_table(rho, (delta, 0.0), (0.0, 0.0))
        assert table[0, 0, 0, 0] + table[0, 0, 1, 1] == pytest.approx(math.cos(delta) ** 2)


def test_joint_table_is_a_distribution_for_every_question_pair():
    table = Q.joint_table(Q.density_matrix(0.3, 0.8), (0.2, 1.1), (-0.7, 2.4))
    assert (table >= -1e-15).all() and np.allclose(table.sum(axis=(2, 3)), 1.0)


def test_bell_value_at_the_known_angles(bell):
    rho = Q.density_matrix(math.pi / 4, 1.0)
    assert Q.win_probability(rho, Q.BELL_ALPHA, Q.BELL_BETA) == pytest.approx(bell["value"], abs=1e-12)
    assert Q.BELL_VALUE == pytest.approx(0.5 + math.sqrt(2) / 4)


def test_product_state_with_all_angles_zero_wins_three_quarters():
    """Produktzustand |00⟩, alle Winkel 0: beide messen 0 (a = b = 0), also gleiche Antworten: wie die Strategie „beide immer 0“ = 3 von 4."""
    assert Q.win_probability(Q.density_matrix(0.0, 1.0), (0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.75)


def test_maximally_mixed_state_wins_one_half_whatever_the_angles():
    assert Q.win_probability(Q.density_matrix(0.7, 0.0), (0.3, 1.0), (2.0, -0.4)) == pytest.approx(0.5)


def test_correlation_matrix_of_the_bell_state():
    """Φ+ hat ⟨σx⊗σx⟩ = 1, ⟨σy⊗σy⟩ = −1, ⟨σz⊗σz⟩ = 1, alle anderen 0."""
    t = Q.correlation_matrix(Q.density_matrix(math.pi / 4, 1.0))
    assert t == pytest.approx(np.diag([1.0, -1.0, 1.0]), abs=1e-12)


def test_max_chsh_of_the_bell_state_is_two_root_two():
    """TᵀT = I: die zwei größten Eigenwerte sind 1 und 1, S = 2·√2 und W = ½ + 2√2/8."""
    rho = Q.density_matrix(math.pi / 4, 1.0)
    assert Q.max_chsh(rho) == pytest.approx(2 * math.sqrt(2))
    assert Q.max_win(rho) == pytest.approx(0.5 + math.sqrt(2) / 4)


def test_max_win_of_a_product_state_is_three_quarters():
    """T = diag(0, 0, 1): ein Eigenwert 1, der andere 0, S = 2, W = ½ + 2/8 = 0,75."""
    assert Q.max_win(Q.density_matrix(0.0, 1.0)) == pytest.approx(0.75)


def test_closed_form_and_threshold_agree_with_the_correlation_route():
    for deg in (0, 10, 25, 45):
        for v in (1.0, 0.8, 0.3):
            th = math.radians(deg)
            assert Q.max_win(Q.density_matrix(th, v)) == pytest.approx(Q.max_win_closed_form(th, v), abs=1e-12)
    assert Q.advantage_threshold(math.pi / 4) == pytest.approx(1 / math.sqrt(2))
    assert Q.advantage_threshold(0.0) == pytest.approx(1.0)
    assert Q.max_win_closed_form(math.radians(20), Q.advantage_threshold(math.radians(20))) == pytest.approx(0.75)


def test_fast_correlators_equal_the_matrix_route():
    rng = np.random.default_rng(3)
    for _ in range(20):
        th, v = rng.uniform(0, math.pi / 4), rng.uniform(0, 1)
        al, be = rng.uniform(-1.6, 1.6, 2), rng.uniform(-1.6, 1.6, 2)
        slow = Q.win_probability(Q.density_matrix(th, v), al, be)
        assert Q.win_from_correlators(th, v, al[0], al[1], be[0], be[1]) == pytest.approx(slow, abs=1e-12)


def test_win_from_correlators_broadcasts_over_arrays():
    g = np.radians(np.array([0.0, 45.0, 90.0]))
    out = Q.win_from_correlators(math.pi / 4, 1.0, g[:, None], 0.0, 0.0, 0.0)
    assert out.shape == (3, 1)


def test_sample_outcomes_inverts_the_distribution_function():
    """Tafel für (x, y) = (0, 0): P(0,0) = 0,1; P(0,1) = 0,2; P(1,0) = 0,3; P(1,1) = 0,4. Verteilungsfunktion 0,1 / 0,3 / 0,6 / 1,0."""
    table = np.zeros((2, 2, 2, 2))
    table[0, 0] = [[0.1, 0.2], [0.3, 0.4]]
    table[0, 1] = table[1, 0] = table[1, 1] = [[1.0, 0.0], [0.0, 0.0]]
    u = np.array([0.05, 0.15, 0.45, 0.95, 0.999999])
    a, b = Q.sample_outcomes(table, np.zeros(5, int), np.zeros(5, int), u)
    assert list(zip(a.tolist(), b.tolist())) == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]


def test_sample_outcomes_survives_a_rounding_remainder():
    """Summe knapp unter 1: eine Gleichverteilte oberhalb der letzten Summe fällt auf das letzte Paar, kein Indexfehler."""
    table = np.zeros((2, 2, 2, 2))
    table[:, :] = [[0.25, 0.25], [0.25, 0.25 - 1e-9]]
    a, b = Q.sample_outcomes(table, np.array([0]), np.array([0]), np.array([1.0 - 1e-12]))
    assert (a[0], b[0]) == (1, 1)
