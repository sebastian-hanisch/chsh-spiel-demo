"""Orakel-Tests chsh-spiel-demo: jede Kernroutine gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2).

* Klassisches Maximum: Aufzählung in reinem Python mit ausgeschriebenen Bedingungen (nicht über die Bit-Zerlegung der App) und die CHSH-Algebra |S| ≤ 2 mit ±1-Ausgaben.
* Quantenwert: Zustandsvektor und explizite Basisvektoren (⟨ab|ψ⟩-Beträge) statt Dichtematrix und Projektoren; das Rauschen als Mischung W(v) = v·W(1) + (1 − v)/2; cos²(π/8);
  Eigenwert des Bell-Operators; Optimierung über alle vier Winkel (scipy, Nelder-Mead) gegen den Horodecki-Weg der App und gegen die geschlossene Formel.
* Geteilte Zufallsrichtung: Monte-Carlo mit eigenem Zufallsgenerator (numpy) gegen die Formel.
* E4: exakte Binomialfaltung (scipy) gegen die Wiederholungen der App."""

import itertools
import math

import numpy as np
import pytest

import chs_experiments as XP
import chs_game as G
import chs_quantum as Q
from chs_scenario import Settings


# --- Klassisch -----------------------------------------------------------------------------------------------------------------------------------------

def _brute_force_best():
    """Alle 16 Strategien als Tupel (a0, a1, b0, b1); ausgeschriebene Bedingung ohne Hilfsfunktionen der App."""
    best, count, values = 0, 0, []
    for a0, a1, b0, b1 in itertools.product((0, 1), repeat=4):
        won = 0
        for x in (0, 1):
            for y in (0, 1):
                a = a0 if x == 0 else a1
                b = b0 if y == 0 else b1
                if (a + b) % 2 == x * y:
                    won += 1
        values.append(won)
        if won > best:
            best, count = won, 1
        elif won == best:
            count += 1
    return best, count, values


def test_classical_maximum_by_brute_force():
    best, count, values = _brute_force_best()
    assert (best, count) == (3, 8) == G.best_classical_wins()
    assert sorted(values) == sorted(G.all_strategy_wins())


def _sign(bit):
    """Ausgabe ±1 zum Bit: (−1)^Bit."""
    return 1 - 2 * bit


def test_every_classical_strategy_obeys_the_chsh_algebra():
    """Mit Ausgaben ±1: S = a0·(b0 + b1) + a1·(b0 − b1) ist ±2 oder 0, also |S| ≤ 2, und die Gewinnwahrscheinlichkeit ist ½ + S/8 ≤ 3/4."""
    for a0, a1, b0, b1 in itertools.product((1, -1), repeat=4):
        s = a0 * b0 + a0 * b1 + a1 * b0 - a1 * b1
        assert abs(s) <= 2
    # Zuordnung zur App: Ausgabe ±1 = (−1)^Bit; der Gewinn aus S ist ½ + S/8
    for index in range(16):
        (a0, a1), (b0, b1) = G.strategy_answers(index)
        s = _sign(a0) * _sign(b0) + _sign(a0) * _sign(b1) + _sign(a1) * _sign(b0) - _sign(a1) * _sign(b1)
        assert G.strategy_value(index) == pytest.approx(0.5 + s / 8)


def test_best_with_one_bit_by_explicit_search():
    """Bob kennt x: b darf von (x, y) abhängen. Ausgeschrieben: für jedes (a(x), b(x, y)) zählen."""
    best = 0
    for a0, a1 in itertools.product((0, 1), repeat=2):
        for b in itertools.product((0, 1), repeat=4):
            won = sum(1 for x in (0, 1) for y in (0, 1) if ((a0 if x == 0 else a1) + b[2 * x + y]) % 2 == x * y)
            best = max(best, won)
    assert best == G.best_with_one_bit() == 4


# --- Quantenwert ---------------------------------------------------------------------------------------------------------------------------------------

def _pure_win(theta, alpha, beta):
    """Gewinnwahrscheinlichkeit des reinen Zustands cosθ|00⟩ + sinθ|11⟩ aus ⟨u ⊗ w|ψ⟩-Beträgen mit expliziten Basisvektoren."""
    psi = np.array([[math.cos(theta), 0.0], [0.0, math.sin(theta)]])            # Koeffizienten c[i][j] von |i⟩|j⟩
    total = 0.0
    for x in (0, 1):
        for y in (0, 1):
            u = {0: np.array([math.cos(alpha[x]), math.sin(alpha[x])]), 1: np.array([-math.sin(alpha[x]), math.cos(alpha[x])])}
            w = {0: np.array([math.cos(beta[y]), math.sin(beta[y])]), 1: np.array([-math.sin(beta[y]), math.cos(beta[y])])}
            for a in (0, 1):
                for b in (0, 1):
                    if (a + b) % 2 == x * y:
                        total += 0.25 * float(u[a] @ psi @ w[b]) ** 2
    return total


def _mixed_win(theta, v, alpha, beta):
    """Rauschen als Mischung: das maximal gemischte Paar gibt jedem Antwortpaar ¼, also Gewinn ½."""
    return v * _pure_win(theta, alpha, beta) + (1.0 - v) * 0.5


def test_bell_value_is_cos_squared_pi_over_eight():
    assert _pure_win(math.pi / 4, Q.BELL_ALPHA, Q.BELL_BETA) == pytest.approx(math.cos(math.pi / 8) ** 2, abs=1e-13)
    assert Q.win_probability(Q.density_matrix(math.pi / 4, 1.0), Q.BELL_ALPHA, Q.BELL_BETA) == pytest.approx(math.cos(math.pi / 8) ** 2, abs=1e-13)


def test_matrix_route_equals_the_vector_route_on_random_cases():
    rng = np.random.default_rng(11)
    for _ in range(60):
        theta, v = rng.uniform(0, math.pi / 4), rng.uniform(0, 1)
        al, be = rng.uniform(-math.pi / 2, math.pi / 2, 2), rng.uniform(-math.pi / 2, math.pi / 2, 2)
        assert Q.win_probability(Q.density_matrix(theta, v), al, be) == pytest.approx(_mixed_win(theta, v, al, be), abs=1e-12)


def test_bell_operator_has_largest_eigenvalue_two_root_two():
    """Bell-Operator B = A₀⊗(B₀ + B₁) + A₁⊗(B₀ − B₁) mit den Observablen cos2φ·σz + sin2φ·σx; der größte Eigenwert ist 2√2, und Φ⁺ erreicht ihn."""
    def obs(phi):
        return math.cos(2 * phi) * Q.SIGMA_Z + math.sin(2 * phi) * Q.SIGMA_X

    a0, a1 = obs(Q.BELL_ALPHA[0]), obs(Q.BELL_ALPHA[1])
    b0, b1 = obs(Q.BELL_BETA[0]), obs(Q.BELL_BETA[1])
    bell_op = np.kron(a0, b0 + b1) + np.kron(a1, b0 - b1)
    assert np.linalg.eigvalsh(bell_op).max() == pytest.approx(2 * math.sqrt(2))
    phi_plus = np.array([1, 0, 0, 1]) / math.sqrt(2)
    s = float(phi_plus @ bell_op @ phi_plus)
    assert s == pytest.approx(2 * math.sqrt(2)) and 0.5 + s / 8 == pytest.approx(math.cos(math.pi / 8) ** 2)


def test_bell_operator_bound_holds_for_random_observables():
    """Cirel'son-Grenze im Modell: für zufällige Ebenen-Messungen und zufällige Zustände der Familie nie über 2√2, also nie über cos²(π/8)."""
    rng = np.random.default_rng(5)
    top = 0.0
    for _ in range(300):
        theta, v = rng.uniform(0, math.pi / 4), rng.uniform(0, 1)
        al, be = rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2)
        top = max(top, _mixed_win(theta, v, al, be))
    assert top <= math.cos(math.pi / 8) ** 2 + 1e-12


@pytest.mark.parametrize("deg, v", [(0, 1.0), (15, 1.0), (30, 0.9), (45, 1.0), (45, 0.8), (20, 0.5)])
def test_optimum_over_all_four_angles_equals_horodecki_and_the_formula(deg, v):
    from scipy.optimize import minimize

    theta = math.radians(deg)
    rng = np.random.default_rng(100 + deg)
    best = 0.0
    for _ in range(8):
        x0 = rng.uniform(-math.pi / 2, math.pi / 2, 4)
        res = minimize(lambda p: -_mixed_win(theta, v, p[:2], p[2:]), x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 6000})
        best = max(best, -res.fun)
    rho = Q.density_matrix(theta, v)
    assert best == pytest.approx(Q.max_win(rho), abs=1e-8)
    assert best == pytest.approx(Q.max_win_closed_form(theta, v), abs=1e-8)
    assert best <= Q.max_win(rho) + 1e-12                      # die Optimierung kann den Bestwert nicht übertreffen


def test_threshold_is_where_the_optimum_crosses_three_quarters():
    from scipy.optimize import minimize

    theta = math.radians(20)
    v_star = Q.advantage_threshold(theta)

    def optimum(v):
        rng = np.random.default_rng(1)
        return max(-minimize(lambda p: -_mixed_win(theta, v, p[:2], p[2:]), rng.uniform(-1.5, 1.5, 4), method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 6000}).fun for _ in range(6))

    assert optimum(v_star - 0.01) < 0.75 - 1e-4 and optimum(v_star + 0.01) > 0.75 + 1e-4


def test_born_sample_matches_the_exact_table_by_chi_square_like_bound():
    """Je Fragepaar 100 000 Ziehungen aus der Tafel (eigener Zufallsgenerator): jeder Anteil innerhalb von vier Standardfehlern."""
    rho = Q.density_matrix(0.5, 0.9)
    al, be = (0.3, 1.2), (-0.4, 0.8)
    table = Q.joint_table(rho, al, be)
    rng = np.random.default_rng(21)
    n = 100_000
    for x, y in G.QUESTIONS:
        u = rng.random(n)
        a, b = Q.sample_outcomes(table, np.full(n, x), np.full(n, y), u)
        for aa, bb in G.ANSWERS:
            p = table[x, y, aa, bb]
            freq = np.mean((a == aa) & (b == bb))
            assert abs(freq - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-12


# --- Geteilte Zufallsrichtung ---------------------------------------------------------------------------------------------------------------------------

def test_shared_direction_formula_equals_a_monte_carlo():
    rng = np.random.default_rng(31)
    n = 400_000
    alpha, beta = (0.2, 1.3), (-0.5, 0.9)
    lam = rng.uniform(0, math.pi, n)

    def answer(phi):
        d = (lam - phi + math.pi / 2) % math.pi - math.pi / 2          # auf [−π/2, π/2) reduziert, eigene Zeile
        return (np.abs(d) >= math.pi / 4).astype(int)

    for x, y in G.QUESTIONS:
        sim = np.mean(((answer(alpha[x]) + answer(beta[y])) % 2) == x * y)
        assert abs(sim - G.shared_direction_question_wins(alpha, beta)[x, y]) < 4 * math.sqrt(0.25 / n)


def test_shared_direction_never_beats_the_classical_bound_on_random_angles():
    rng = np.random.default_rng(41)
    for _ in range(2000):
        al, be = rng.uniform(-math.pi, math.pi, 2), rng.uniform(-math.pi, math.pi, 2)
        assert G.shared_direction_win(al, be) <= 0.75 + 1e-12


# --- E4: exakte Binomialfaltung --------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rounds", [25, 100, 400])
def test_e4_matches_the_exact_distribution(rounds):
    stats = pytest.importorskip("scipy.stats")
    p = math.cos(math.pi / 8) ** 2
    ks = np.arange(rounds + 1)
    pk = stats.binom.pmf(ks, rounds, 0.25)                     # Zahl der Fragepaare (1,1); die Strategie „immer 0“ gewinnt rounds − k
    more = float(np.sum(pk * stats.binom.sf(rounds - ks, rounds, p)))
    equal = float(np.sum(pk * stats.binom.pmf(rounds - ks, rounds, p)))
    cell = XP.e4_cell(Settings(), rounds, 6000)
    assert abs(cell["more"] - more) < 4 * math.sqrt(more * (1 - more) / 6000) + 1e-9
    assert abs(cell["equal"] - equal) < 4 * math.sqrt(max(equal * (1 - equal), 1e-9) / 6000) + 1e-9
