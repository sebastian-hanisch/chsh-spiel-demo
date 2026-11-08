"""Das CHSH-Spiel und seine klassischen Mannschaften (nur Standardbibliothek und numpy).

Spiel: zwei Spieler ohne Verbindung, Fragen x, y ∈ {0, 1} gleichverteilt und unabhängig, Antworten a, b ∈ {0, 1}; gewonnen ist die Runde, wenn a ⊕ b = x ∧ y.
Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b)."""

import itertools
import math

import numpy as np

import chs_rng as R

QUESTIONS = tuple(itertools.product((0, 1), repeat=2))         # (x, y)
ANSWERS = tuple(itertools.product((0, 1), repeat=2))           # (a, b)


def target(x: int, y: int) -> int:
    """Das Ziel der Runde: a ⊕ b muss x ∧ y sein."""
    return x & y


def is_won(x: int, y: int, a: int, b: int) -> bool:
    return (a ^ b) == target(x, y)


def winning_mask() -> np.ndarray:
    """Maske M[x, y, a, b] = 1, wenn die Antworten (a, b) auf die Fragen (x, y) die Runde gewinnen."""
    mask = np.zeros((2, 2, 2, 2))
    for (x, y), (a, b) in itertools.product(QUESTIONS, ANSWERS):
        mask[x, y, a, b] = 1.0 if is_won(x, y, a, b) else 0.0
    return mask


def win_probability_from_table(table: np.ndarray) -> float:
    """Gewinnwahrscheinlichkeit aus einer Wahrscheinlichkeitstafel P[x, y, a, b] (Fragen gleichverteilt): ¼ · Σ_{x,y} Σ_{a⊕b = x∧y} P."""
    return float(0.25 * (np.asarray(table) * winning_mask()).sum())


def question_win_probabilities(table: np.ndarray) -> np.ndarray:
    """2×2-Feld: Gewinnwahrscheinlichkeit je Fragepaar (x, y) (bedingt auf die Fragen)."""
    return (np.asarray(table) * winning_mask()).sum(axis=(2, 3))


# --- Deterministische Strategien -----------------------------------------------------------------------------------------------------------------------

def strategy_answers(index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Strategie 0..15 als ((a(0), a(1)), (b(0), b(1))): die Bits von `index` von oben nach unten."""
    if not 0 <= index < 16:
        raise ValueError(f"Strategie {index} liegt außerhalb von 0 bis 15")
    return ((index >> 3) & 1, (index >> 2) & 1), ((index >> 1) & 1, index & 1)


def strategy_label(index: int) -> str:
    (a0, a1), (b0, b1) = strategy_answers(index)
    return f"Alice {a0}/{a1}, Bob {b0}/{b1}"


def strategy_wins(index: int) -> int:
    """Zahl der Fragepaare (von 4), die diese deterministische Strategie gewinnt."""
    fa, fb = strategy_answers(index)
    return sum(1 for x, y in QUESTIONS if is_won(x, y, fa[x], fb[y]))


def strategy_value(index: int) -> float:
    return strategy_wins(index) / 4


def all_strategy_wins() -> tuple[int, ...]:
    return tuple(strategy_wins(i) for i in range(16))


def best_classical_wins() -> tuple[int, int]:
    """(größte Zahl gewonnener Fragepaare, Zahl der Strategien mit diesem Wert) durch Aufzählung aller 16 Strategien."""
    wins = all_strategy_wins()
    best = max(wins)
    return best, wins.count(best)


def mixture_values(weights: np.ndarray) -> np.ndarray:
    """Gewinnwahrscheinlichkeit von Mischungen der 16 Strategien (Zeilen = Gewichte, summieren auf 1): gewichteter Mittelwert, also nie über dem Maximum.

    Privater und geteilter Zufall wirken beide so: jeder Zufall, der nur aus den eigenen Fragen und einer gemeinsamen Zufallsgröße Antworten macht, mischt feste Strategien."""
    values = np.array([strategy_value(i) for i in range(16)])
    return np.asarray(weights) @ values


def random_weights(count: int, seed: int) -> np.ndarray:
    """`count` Gewichtsvektoren, gleichverteilt auf dem Simplex (normierte Exponentialverteilte aus SplitMix64)."""
    u = R.uniform_block(seed, count * 16).reshape(count, 16)
    e = -np.log(1.0 - u)
    return e / e.sum(axis=1, keepdims=True)


# --- Geteilter Zufall als Richtung (klassisches Gegenstück zu den Messwinkeln) --------------------------------------------------------------------------

def reduce_angle(delta):
    """Winkeldifferenz auf [−π/2, π/2) (Periode π)."""
    return (np.asarray(delta, dtype=float) + math.pi / 2) % math.pi - math.pi / 2


def shared_direction_answer(lam, phi):
    """Geteilter Zufall: gemeinsame Richtung λ (gleichverteilt auf [0, π)); wer die Richtung φ hat, antwortet 0, wenn λ weniger als π/4 von φ entfernt ist, sonst 1."""
    return (np.abs(reduce_angle(np.asarray(lam) - phi)) >= math.pi / 4).astype(int)


def shared_direction_agreement(delta):
    """Wahrscheinlichkeit, dass zwei Richtungen mit Winkelabstand `delta` dieselbe Antwort geben: 1 − 2·|delta|/π (delta auf [−π/2, π/2) reduziert)."""
    return 1.0 - 2.0 * np.abs(reduce_angle(delta)) / math.pi


def shared_direction_question_wins(alpha, beta) -> np.ndarray:
    """2×2-Feld der Gewinnwahrscheinlichkeit je Fragepaar für das Modell „geteilte Zufallsrichtung“ mit Alices Winkeln `alpha[x]` und Bobs `beta[y]` (Bogenmaß)."""
    out = np.zeros((2, 2))
    for x, y in QUESTIONS:
        agree = float(shared_direction_agreement(alpha[x] - beta[y]))
        out[x, y] = agree if target(x, y) == 0 else 1.0 - agree
    return out


def shared_direction_win(alpha, beta) -> float:
    return float(shared_direction_question_wins(alpha, beta).mean())


# --- Weitere Sprossen ----------------------------------------------------------------------------------------------------------------------------------

def no_signalling_box() -> np.ndarray:
    """Die No-Signalling-Box (Definition): P(a, b | x, y) = ½, wenn a ⊕ b = x ∧ y, sonst 0. Ein Rechenmodell, keine Aussage über die Natur."""
    return 0.5 * winning_mask()


def marginal_dependence(table: np.ndarray) -> float:
    """Größte Abweichung, um die sich die Antwortverteilung des einen Spielers ändert, wenn nur die Frage des anderen wechselt (0 heißt: keine Nachricht kommt durch)."""
    t = np.asarray(table)
    alice = t.sum(axis=3)            # P(a | x, y) als [x, y, a]
    bob = t.sum(axis=2)              # P(b | x, y) als [x, y, b]
    return float(max(np.abs(alice[:, 0] - alice[:, 1]).max(), np.abs(bob[0] - bob[1]).max()))


def best_with_one_bit() -> int:
    """Beste Zahl gewonnener Fragepaare (von 4), wenn Bob zusätzlich Alices Frage x kennt (ein Bit Absprache im Spiel): Aufzählung aller 4 · 16 Strategien."""
    best = 0
    for fa in itertools.product((0, 1), repeat=2):                      # a = fa[x]
        for fb in itertools.product((0, 1), repeat=4):                  # b = fb[2x + y]
            wins = sum(1 for x, y in QUESTIONS if is_won(x, y, fa[x], fb[2 * x + y]))
            best = max(best, wins)
    return best
