"""Einheiten des Spiels und der klassischen Mannschaften an den Handrechnungen aus conftest.py."""

import math

import numpy as np
import pytest

import chs_game as G


def test_target_is_the_and_of_the_questions():
    assert [G.target(x, y) for x, y in G.QUESTIONS] == [0, 0, 0, 1]


def test_is_won_needs_a_xor_b_equal_to_the_target():
    assert G.is_won(0, 0, 1, 1) and G.is_won(0, 1, 0, 0) and G.is_won(1, 1, 0, 1)
    assert not G.is_won(1, 1, 0, 0) and not G.is_won(0, 0, 0, 1)


def test_winning_mask_has_two_winning_answer_pairs_per_question_pair():
    mask = G.winning_mask()
    assert mask.shape == (2, 2, 2, 2) and mask.sum() == 8
    assert np.array_equal(mask.sum(axis=(2, 3)), np.full((2, 2), 2.0))


def test_strategy_answers_read_the_bits_from_the_top():
    assert G.strategy_answers(0) == ((0, 0), (0, 0))
    assert G.strategy_answers(5) == ((0, 1), (0, 1))
    assert G.strategy_answers(15) == ((1, 1), (1, 1))
    assert G.strategy_answers(6) == ((0, 1), (1, 0))


@pytest.mark.parametrize("index", [-1, 16])
def test_strategy_index_outside_the_range_is_refused(index):
    with pytest.raises(ValueError):
        G.strategy_answers(index)


def test_strategy_wins_match_the_hand_values(strategies):
    for index, wins in strategies.values():
        assert G.strategy_wins(index) == wins


def test_eight_strategies_win_three_and_eight_win_one():
    wins = G.all_strategy_wins()
    assert sorted(wins) == [1] * 8 + [3] * 8
    assert G.best_classical_wins() == (3, 8)


def test_no_strategy_wins_all_four_pairs():
    assert max(G.all_strategy_wins()) < 4


def test_strategy_label_names_both_players():
    assert G.strategy_label(5) == "Alice 0/1, Bob 0/1"


def test_mixture_of_two_strategies_is_the_weighted_mean(strategies):
    """Strategie 0 hat Wert 3/4, Strategie 5 hat 1/4: Gewichte (0,7; 0,3) geben 0,7 · 0,75 + 0,3 · 0,25 = 0,6."""
    w = np.zeros((1, 16))
    w[0, 0], w[0, 5] = 0.7, 0.3
    assert G.mixture_values(w)[0] == pytest.approx(0.6)


def test_random_weights_are_distributions_and_reproducible():
    w = G.random_weights(50, 7)
    assert w.shape == (50, 16) and np.allclose(w.sum(axis=1), 1.0) and (w > 0).all()
    assert np.array_equal(w, G.random_weights(50, 7)) and not np.array_equal(w, G.random_weights(50, 8))


def test_reduce_angle_maps_to_half_open_interval():
    assert float(G.reduce_angle(math.pi)) == pytest.approx(0.0, abs=1e-12)
    assert float(G.reduce_angle(math.pi / 2)) == pytest.approx(-math.pi / 2)
    assert float(G.reduce_angle(-math.pi / 2)) == pytest.approx(-math.pi / 2)
    assert float(G.reduce_angle(0.3 + 5 * math.pi)) == pytest.approx(0.3)


def test_shared_direction_agreement_matches_the_hand_values(shared_direction):
    for delta, expected in shared_direction["agree"].items():
        assert float(G.shared_direction_agreement(delta)) == pytest.approx(expected)
        assert float(G.shared_direction_agreement(-delta)) == pytest.approx(expected)


def test_shared_direction_answer_is_zero_within_a_quarter_turn():
    lam = np.array([0.0, 0.7, 0.8, math.pi - 0.1])
    assert G.shared_direction_answer(lam, 0.0).tolist() == [0, 0, 1, 0]      # π/4 = 0,785…; π − 0,1 liegt 0,1 vor dem Umlauf


def test_shared_direction_with_the_bell_angles_wins_exactly_three_quarters(shared_direction):
    alpha, beta = (0.0, math.pi / 4), (math.pi / 8, -math.pi / 8)
    assert G.shared_direction_win(alpha, beta) == pytest.approx(shared_direction["value"])
    assert np.allclose(G.shared_direction_question_wins(alpha, beta), 0.75)


def test_no_signalling_box_is_a_distribution_that_always_wins(pr_box):
    box = G.no_signalling_box()
    assert np.allclose(box.sum(axis=(2, 3)), 1.0)
    assert G.win_probability_from_table(box) == pytest.approx(pr_box["win"])
    assert {(a, b) for a in (0, 1) for b in (0, 1) if box[0, 0, a, b] > 0} == set(pr_box["allowed_00"])
    assert {(a, b) for a in (0, 1) for b in (0, 1) if box[1, 1, a, b] > 0} == set(pr_box["allowed_11"])


def test_no_signalling_box_sends_no_message():
    assert G.marginal_dependence(G.no_signalling_box()) == pytest.approx(0.0)


def test_marginal_dependence_detects_a_message():
    """Eine Tabelle, in der Bobs Antwort die Frage von Alice verrät (b = x), hat Abhängigkeit 1."""
    table = np.zeros((2, 2, 2, 2))
    for x in (0, 1):
        for y in (0, 1):
            table[x, y, 0, x] = 1.0
    assert G.marginal_dependence(table) == pytest.approx(1.0)


def test_one_bit_of_communication_wins_every_pair():
    assert G.best_with_one_bit() == 4


def test_win_probability_of_a_deterministic_table(strategies):
    """Tafel der Strategie 0 (immer 0/0): gewinnt 3 von 4 Paaren, also 0,75."""
    table = np.zeros((2, 2, 2, 2))
    table[:, :, 0, 0] = 1.0
    assert G.win_probability_from_table(table) == pytest.approx(0.75)
    assert G.question_win_probabilities(table).tolist() == [[1.0, 1.0], [1.0, 0.0]]
