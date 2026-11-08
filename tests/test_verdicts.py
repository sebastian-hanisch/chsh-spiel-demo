"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np
import pytest

import chs_evaluation as ev
from chs_play import Play
from chs_scenario import Settings


def _analysis(w_quantum=0.8, w_best=0.85, threshold=0.7, settings=None, w_shared=0.7):
    s = settings or Settings(theta_deg=30.0, vis_pct=90)
    q = np.full((2, 2), w_quantum)
    return ev.Analysis(settings=s, classical_best=0.75, classical_best_count=8, strategy_value=0.75, w_shared=w_shared, w_quantum=w_quantum, w_best=w_best, threshold=threshold,
                       q_questions=q, s_questions=np.full((2, 2), w_shared), box=1.0, one_bit=1.0)


def _play(n, won_fixed, won_quantum):
    won = np.zeros((5, n), dtype=bool)
    won[0, :won_fixed] = True
    won[2, :won_quantum] = True
    won[3:] = True
    return Play(np.zeros(n, int), np.zeros(n, int), np.zeros((5, n), int), np.zeros((5, n), int), won)


def test_helpers_format_with_decimal_comma_and_real_minus():
    assert ev.de(0.8536, 2) == "0,85" and ev.de(-1.5, 1) == "−1,5" and ev.de(-0.0001, 2) == "0,00"
    assert ev.thousands(331776) == "331.776" and ev.pct(0.8536, 2) == "85,36 %" and ev.points(0.1036) == "10,36 Prozentpunkte"


def test_standard_error_of_a_rate():
    assert ev.standard_error(0.75, 100) == pytest.approx((0.75 * 0.25 / 100) ** 0.5)
    assert ev.standard_error(1.0, 100) == 0.0


def test_verdict_resource_success_names_value_gap_and_bound():
    kind, text = ev.verdict_resource(_analysis(w_quantum=0.85))
    assert kind == "success" and "**85,00 %**" in text and "10,00 Prozentpunkte mehr" in text and "75,00 %" in text


def test_verdict_resource_equal_is_info():
    kind, text = ev.verdict_resource(_analysis(w_quantum=0.75))
    assert kind == "info" and "genau so viel" in text and "75,00 %" in text


def test_verdict_resource_below_is_warning_with_the_gap():
    kind, text = ev.verdict_resource(_analysis(w_quantum=0.7))
    assert kind == "warning" and "5,00 Prozentpunkte weniger" in text and "**70,00 %**" in text


def test_verdict_resource_threshold_is_exclusive_at_one_billionth():
    assert ev.verdict_resource(_analysis(w_quantum=0.75 + 1e-10))[0] == "info"
    assert ev.verdict_resource(_analysis(w_quantum=0.75 + 1e-8))[0] == "success"


def test_verdict_state_with_advantage_mentions_the_reached_value_only_when_it_differs():
    kind, text = ev.verdict_state(_analysis(w_quantum=0.8, w_best=0.85))
    assert kind == "info" and "Zustandswinkel 30,0 Grad" in text and "Sichtbarkeit 90 %" in text and "**85,00 %**" in text and "10,00 Prozentpunkte über 75,00 %" in text
    assert "Die eingestellten Winkel erreichen 80,00 %" in text
    assert "erreichen" not in ev.verdict_state(_analysis(w_quantum=0.85, w_best=0.85))[1]


def test_verdict_state_product_state_says_no_entanglement():
    kind, text = ev.verdict_state(_analysis(w_quantum=0.7, w_best=0.75, threshold=1.0, settings=Settings(theta_deg=0.0)))
    assert kind == "warning" and "Produktzustand" in text and "75,00 %" in text


def test_verdict_state_noise_below_threshold_names_the_threshold():
    kind, text = ev.verdict_state(_analysis(w_quantum=0.74, w_best=0.7475, threshold=0.7071, settings=Settings(vis_pct=70)))
    assert kind == "warning" and "Sichtbarkeit 70 %" in text and "74,75 %" in text and "über 70,7 %" in text


def test_verdict_game_ahead_and_behind():
    a = _analysis(w_quantum=0.85)
    kind, text = ev.verdict_game(a, _play(100, 75, 85))
    assert kind == "success" and "Nach 100 Runden" in text and "ohne Ressource 75,0 %" in text and "Verschränkung 85,0 %" in text and "erwartet 85,0 % ± 3,6 %" in text
    assert "No-Signalling-Box 100,0 %" in text and "Absprache im Spiel 100,0 %" in text
    kind, text = ev.verdict_game(a, _play(100, 75, 70))
    assert kind == "info" and "nicht vor der festen Strategie" in text


def test_verdict_game_tie_is_not_ahead():
    assert ev.verdict_game(_analysis(), _play(100, 75, 75))[0] == "info"


def _rows():
    return [{"index": i, "label": f"s{i}", "wins": 3 if i < 8 else 1, "value": (3 if i < 8 else 1) / 4} for i in range(16)]


def test_verdict_e1_counts_and_mean():
    kind, text = ev.verdict_e1(_rows(), {"count": 20000, "max": 0.7499, "values": None})
    assert kind == "success" and "8 genau 3 von 4" in text and "75,00 %" in text and "8 nur 1" in text and "Mittel über alle 16: 50,0 %" in text
    assert "20.000 zufällige Mischungen" in text and "74,99 %" in text


def test_verdict_e1_flags_an_impossible_mixture_as_error():
    assert ev.verdict_e1(_rows(), {"count": 10, "max": 0.76, "values": None})[0] == "error"


def _counts(q_above, s_above):
    return {"cells": 331776, "points": 24, "quantum": {"above": q_above, "max": 0.8536, "at_max": 48}, "shared": {"above": s_above, "max": 0.75, "at_max": 100}}


def test_verdict_e2_three_branches():
    s = Settings()
    kind, text = ev.verdict_e2(_counts(24816, 0), s)
    assert kind == "success" and "331.776 Winkelquadrupeln (24 Richtungen" in text and "24.816 Zellen (7,5 %)" in text and "48 Zellen" in text and "0 Zellen über 0,75" in text
    assert ev.verdict_e2(_counts(0, 0), s)[0] == "warning"
    assert ev.verdict_e2(_counts(5, 3), s)[0] == "error"


def test_verdict_e3_counts_the_entangled_states_above_three_quarters():
    rows = [{"theta_deg": d, "best": b, "threshold": t} for d, b, t in ((0.0, 0.75, 1.0), (10.0, 0.7642, 0.946), (45.0, 0.8536, 0.7071))]
    kind, text = ev.verdict_e3(rows)
    assert kind == "info" and "in 2 von 2 verschränkten Zuständen" in text and "76,42 %" in text and "75,00 %" in text and "bei 10 Grad 94,6 %" in text and "(45 Grad) 70,7 %" in text


def test_verdict_e4_reports_first_round_count_with_95_percent():
    rows = [{"rounds": 25, "repetitions": 1000, "more": 0.7, "equal": 0.1, "less": 0.2}, {"rounds": 100, "repetitions": 1000, "more": 0.95, "equal": 0.02, "less": 0.03},
            {"rounds": 400, "repetitions": 1000, "more": 0.999, "equal": 0.0, "less": 0.001}]
    kind, text = ev.verdict_e4(rows)
    assert kind == "success" and "Bei 25 Runden" in text and "70,0 %" in text and "1.000 Wiederholungen" in text and "gleich viele: 10,0 %" in text and "bei 400 Runden in 99,9 %" in text
    assert "Ab 100 Runden sind es mindestens 95 %" in text


def test_verdict_e4_without_reaching_the_threshold():
    rows = [{"rounds": 25, "repetitions": 1000, "more": 0.5, "equal": 0.1, "less": 0.4}, {"rounds": 50, "repetitions": 1000, "more": 0.6, "equal": 0.1, "less": 0.3}]
    kind, text = ev.verdict_e4(rows)
    assert kind == "info" and "Bis 50 Runden erreicht der Anteil nicht 95 %" in text
