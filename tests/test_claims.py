"""Claims: jede Zahl der README (und der festen App-Texte) wird hier nachgerechnet; die Zeichenkette muss außerdem wörtlich in der README stehen.

Simulationen laufen mit festen Zufalls-Seeds (SplitMix64) und werden nur mit Marge von vier Standardfehlern verglichen; Simulationszahlen der README (Zufalls-Seed 35)
sind deterministisch und werden als Zeichenkette geprüft."""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

import chs_constants as C
import chs_evaluation as V
import chs_experiments as XP
import chs_game as G
import chs_play as PL
import chs_quantum as Q
from chs_scenario import Settings

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
APP = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
BELL = math.cos(math.pi / 8) ** 2


def readme_has(*texts):
    for t in texts:
        assert t in README, f"nicht in der README: {t!r}"


# --- H1: feste Strategien --------------------------------------------------------------------------------------------------------------------------------

def test_h1_sixteen_strategies_eight_and_eight():
    wins = G.all_strategy_wins()
    assert wins.count(3) == 8 and wins.count(1) == 8 and len(wins) == 16 and 4 not in wins
    assert sum(G.strategy_value(i) for i in range(16)) / 16 == pytest.approx(0.5)
    readme_has("gewinnen 8 drei von vier Fragepaaren (75,0 %), 8 nur eines (25,0 %); Mittel über alle 16: 50,0 %", "Keine gewinnt alle vier")


# --- H2: geteilter Zufall --------------------------------------------------------------------------------------------------------------------------------

def test_h2_mixtures_and_shared_direction():
    m = XP.e1_mixtures(20000, 35)
    assert V.pct(m["max"], 2) == "70,22 %" and V.pct(m["mean"], 2) == "50,10 %" and m["max"] <= 0.75
    readme_has("20 000 zufällige Mischungen", "größter Wert 70,22 %, Mittel 50,10 %")
    assert G.shared_direction_win(Q.BELL_ALPHA, Q.BELL_BETA) == pytest.approx(0.75)
    n = 1_000_000
    s = Settings(rounds=n)
    rate = PL.play(s).final_rates()[1]
    assert V.pct(float(rate), 2) == "75,02 %" and abs(rate - 0.75) < 4 * V.standard_error(0.75, n)
    assert V.pct(V.standard_error(0.75, n), 3) == "0,043 %"
    readme_has("exakt 75,0 %; in 1 000 000 Runden 75,02 % (Standardfehler 0,043 %)")
    g = XP.e2_grid_counts(s)
    assert g["cells"] == 331776 and g["shared"]["above"] == 0 and V.pct(g["shared"]["max"], 1) == "75,0 %"
    readme_has("(331 776 Quadrupel) liegt keines über 75 %, das Maximum ist 75,0 %")


# --- H3: Verschränkung -----------------------------------------------------------------------------------------------------------------------------------

def test_h3_bell_value_and_sample():
    a = V.analyse(Settings())
    assert a.w_quantum == pytest.approx(BELL) == pytest.approx(0.5 + math.sqrt(2) / 4)
    assert V.pct(a.w_quantum, 2) == "85,36 %" and f"{a.w_quantum:.6f}" == "0.853553"
    assert V.points(a.w_quantum - 0.75) == "10,36 Prozentpunkte"
    assert np.allclose(a.q_questions, BELL)
    readme_has("**85,36 %** = cos²(π/8) = ½ + √2/4 = 0,853553", "10,36 Prozentpunkte über dem klassischen Höchstwert", "Jedes der vier Fragepaare gewinnt mit derselben Wahrscheinlichkeit 85,36 %")
    n = 1_000_000
    rate = PL.play(Settings(rounds=n)).final_rates()[2]
    se = V.standard_error(BELL, n)
    assert abs(rate - BELL) < 4 * se
    assert V.pct(float(rate), 2) == "85,43 %" and V.pct(se, 3) == "0,035 %"
    readme_has("Born-Stichprobe aus 1 000 000 Runden (Zufalls-Seed 35): 85,43 % (Standardfehler 0,035 %)")


# --- H4: Winkelgitter ------------------------------------------------------------------------------------------------------------------------------------

def test_h4_grid_counts_and_equal_angles():
    g = XP.e2_grid_counts(Settings())
    assert g["cells"] == 331776 and g["points"] == 24
    assert g["quantum"]["above"] == 24816 and V.pct(g["quantum"]["above"] / g["cells"], 1) == "7,5 %" and g["quantum"]["at_max"] == 48
    assert V.pct(g["quantum"]["max"], 2) == "85,36 %"
    readme_has("**24 816 Zellen (7,5 %)** über 75 %", "das Maximum 85,36 % steht in 48 Zellen", "(24 Richtungen je Winkel, 7,5-Grad-Schritte, 331 776 Quadrupel)")
    zero = V.analyse(Settings(alpha0_deg=0.0, alpha1_deg=0.0, beta0_deg=0.0, beta1_deg=0.0))
    assert zero.w_quantum == pytest.approx(0.75)
    readme_has("Alle vier Winkel 0 Grad: genau 75,0 %")


# --- H5: Verschränkungsgrad ------------------------------------------------------------------------------------------------------------------------------

def test_h5_table_of_state_angles():
    expected = {0: "75,00", 5: "75,37", 10: "76,42", 15: "77,95", 20: "79,72", 30: "83,07", 45: "85,36"}
    for deg, text in expected.items():
        th = math.radians(deg)
        best = Q.max_win(Q.density_matrix(th, 1.0))
        assert V.de(100 * best, 2) == text
        assert best == pytest.approx(0.5 + 0.25 * math.sqrt(1 + math.sin(2 * th) ** 2))
    readme_has("Zustandswinkel 0 / 5 / 10 / 15 / 20 / 30 / 45 Grad: 75,00 / 75,37 / 76,42 / 77,95 / 79,72 / 83,07 / 85,36 %", "In 6 von 6 verschränkten Zuständen über 75 %")
    rows = XP.e3_rows()
    assert sum(1 for r in rows if r["theta_deg"] > 0 and r["best"] > 0.75 + 1e-9) == 6 == len([r for r in rows if r["theta_deg"] > 0])
    product = V.analyse(Settings(theta_deg=0.0))
    assert V.pct(product.w_best, 2) == "75,00 %" and V.pct(product.w_quantum, 2) == "67,68 %"
    readme_has("genau 75,0 % und mit den Bell-Winkeln nur 67,68 %")


# --- H6: Rauschen ----------------------------------------------------------------------------------------------------------------------------------------

def test_h6_noise_and_thresholds():
    for vis, text in {90: "81,82", 80: "78,28", 75: "76,52", 71: "75,10", 70: "74,75", 50: "67,68"}.items():
        best = Q.max_win(Q.density_matrix(math.pi / 4, vis / 100))
        assert V.de(100 * best, 2) == text and best == pytest.approx(0.5 + vis / 100 * math.sqrt(2) / 4)
    readme_has("Sichtbarkeit 90 / 80 / 75 / 71 / 70 / 50 %: 81,82 / 78,28 / 76,52 / 75,10 / 74,75 / 67,68 %", "W = ½ + v·√2/4", "**v über 1/√2 = 70,7 %**")
    assert V.pct(1 / math.sqrt(2), 1) == "70,7 %"
    for deg, text in {5: "98,5", 10: "94,6", 15: "89,4", 20: "84,1", 30: "75,6", 45: "70,7"}.items():
        assert V.de(100 * Q.advantage_threshold(math.radians(deg)), 1) == text
    readme_has("Zustandswinkel 5 / 10 / 15 / 20 / 30 / 45 Grad: 98,5 / 94,6 / 89,4 / 84,1 / 75,6 / 70,7 %")


# --- H7: weitere Sprossen und keine Nachricht -----------------------------------------------------------------------------------------------------------

def test_h7_box_communication_and_no_message():
    a = V.analyse(Settings())
    assert a.box == 1.0 and a.one_bit == 1.0
    assert len(list(itertools.product((0, 1), repeat=2))) * len(list(itertools.product((0, 1), repeat=4))) == 64                                          # 4 Strategien für Alice (a hängt von x ab), 16 für Bob (b hängt von x und y ab); die Aufzählung steht im Orakel
    readme_has("Aufzählung aller 64 Strategien mit einem Bit", "beide 100 %")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        th, v = rng.uniform(0, math.pi / 4), rng.uniform(0, 1)
        al, be = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
        worst = max(worst, G.marginal_dependence(Q.joint_table(Q.density_matrix(th, v), al, be)))
    assert worst < 1e-12 and G.marginal_dependence(G.no_signalling_box()) == 0.0
    readme_has("größte Abweichung unter 10⁻¹² in 300 Zufallsfällen; die Box: 0")


# --- H8: Rundenzahl --------------------------------------------------------------------------------------------------------------------------------------

def test_h8_exact_shares_and_simulation():
    stats = pytest.importorskip("scipy.stats")
    exact = {}
    for rounds in (25, 50, 100, 200):
        ks = np.arange(rounds + 1)
        pk = stats.binom.pmf(ks, rounds, 0.25)
        exact[rounds] = (float(np.sum(pk * stats.binom.sf(rounds - ks, rounds, BELL))), float(np.sum(pk * stats.binom.pmf(rounds - ks, rounds, BELL))))
    assert [V.de(100 * exact[r][0], 1) for r in (25, 50, 100, 200)] == ["77,4", "88,2", "96,1", "99,5"]
    assert V.de(100 * exact[25][1], 1) == "9,3"
    readme_has("bei 25 / 50 / 100 / 200 Runden: 77,4 / 88,2 / 96,1 / 99,5 %", "bei 25 Runden gewinnen beide gleich viele in 9,3 %")
    for rounds in (25, 50, 100, 200):
        cell = XP.e4_cell(Settings(), rounds, C.E4_REPETITIONS)
        se = math.sqrt(exact[rounds][0] * (1 - exact[rounds][0]) / C.E4_REPETITIONS)
        assert abs(cell["more"] - exact[rounds][0]) < 4 * se + 1e-9
    readme_has("(1 000 je Rundenzahl)")
    assert C.E4_REPETITIONS == 1000


# --- H9: ein Spiel ---------------------------------------------------------------------------------------------------------------------------------------

def test_h9_one_play():
    p = PL.play(Settings())
    r = p.final_rates()
    assert [V.pct(float(x), 2) for x in r] == ["75,75 %", "77,00 %", "84,75 %", "100,00 %", "100,00 %"]
    readme_has("ohne Ressource 75,75 %, geteilter Zufall 77,00 %, Verschränkung 84,75 %, Box und Absprache 100 %")
    assert V.pct(V.standard_error(0.75, 400), 1) == "2,2 %" and V.pct(V.standard_error(BELL, 400), 1) == "1,8 %"
    assert V.points(float(r[1]) - 0.75, 1) == "2,0 Prozentpunkte" and abs(r[1] - 0.75) < V.standard_error(0.75, 400)
    readme_has("um 2,0 Prozentpunkte über den erwarteten 75 %, innerhalb eines Standardfehlers", "die Standardfehler sind 2,2 % bzw. 1,8 %")


# --- Weitere Zahlen der README und der Texte ----------------------------------------------------------------------------------------------------------

def test_model_and_method_numbers():
    assert 24 ** 4 == 331776 and len(XP.heat_axis()) == 91 and C.HEAT_STEP_DEG == 2.0 and C.GRID_STEP_DEG == 7.5 and C.MIX_COUNT == 20000
    assert C.ROUNDS_MAX == 5000 and C.E4_ROUNDS[-1] == 1600 and len(C.E4_ROUNDS) == 7 and C.TOLERANCE == 1e-9 and C.STRATEGY_COUNT == 16
    readme_has("(91 × 91 Zellen, 2-Grad-Schritte)", "Slider-Maximum (5 000 Runden)", "sieben Rundenzahlen zu je 1 000 Wiederholungen, bis 1 600 Runden", "größer als 0,75 + 10⁻⁹", "(Zufalls-Seed · 8 + Nummer)")
    assert PL.STREAMS == 8
    assert Q.BELL_ALPHA == (0.0, math.pi / 4) and Q.BELL_BETA == (math.pi / 8, -math.pi / 8)
    readme_has("Port 9216", "Präfix `chs`")


def test_preset_help_texts_name_the_numbers_they_show():
    help_ = C.PRESET_HELP
    a = {n: V.analyse(Settings(theta_deg=p["theta"], vis_pct=p["vis"], alpha0_deg=p["a0"], alpha1_deg=p["a1"], beta0_deg=p["b0"], beta1_deg=p["b1"])) for n, p in C.PRESETS.items()}
    assert "0,8536" in help_["Optimale Winkel (Voreinstellung)"] and V.de(a["Optimale Winkel (Voreinstellung)"].w_quantum, 4) == "0,8536"
    assert "genau 0,75" in help_["Winkel ohne Vorteil"] and a["Winkel ohne Vorteil"].w_quantum == pytest.approx(0.75)
    assert "nur zu 0,75" in help_["Ohne Verschränkung"] and a["Ohne Verschränkung"].w_best == pytest.approx(0.75) and a["Ohne Verschränkung"].w_quantum < 0.75
    assert "bei 70,7 %" in help_["Rauschen 70 %"] and a["Rauschen 70 %"].w_quantum < 0.75 < Q.max_win(Q.density_matrix(math.pi / 4, 0.71))
    assert "Sichtbarkeit 70 %" in help_["Rauschen 70 %"] and "30 % weißem Rauschen" in help_["Rauschen 70 %"]


def test_app_text_numbers():
    """Feste Zahlen im Text der App: 85,36 %, drei von vier, 16 Strategien, Schwelle, Gitter."""
    a = V.analyse(Settings())
    assert V.pct(a.w_quantum, 2) == "85,36 %" and "85,36 %" in APP
    assert a.classical_best == 0.75 and "drei von vier" in APP and a.classical_best_count == 8
    assert "70,7 %" in APP and V.pct(Q.advantage_threshold(math.pi / 4), 1) == "70,7 %"
    assert "0,8536" in APP and V.de(BELL, 4) == "0,8536"
    assert "Gitter von −90 bis 82,5 Grad in 7,5-Grad-Schritten" in APP
    assert "Es gibt 16 solche Strategien" in APP and G.best_classical_wins() == (3, 8)
    assert "Gewinn 1" in APP or "gewinnt immer" in APP
