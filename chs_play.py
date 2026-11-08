"""Spielverlauf Runde für Runde: fünf Mannschaften spielen dieselben Fragen (nur numpy, Zufall aus SplitMix64 mit festen Seeds).

Mannschaften: 0 keine Ressource (feste Strategie), 1 geteilter Zufall (gemeinsame Richtung), 2 Verschränkung (Born-Stichprobe), 3 No-Signalling-Box (Definition),
4 Absprache im Spiel (Bob kennt Alices Frage; im Spiel verboten, nur zum Vergleich)."""

import math
from dataclasses import dataclass

import numpy as np

import chs_game as G
import chs_quantum as Q
import chs_rng as R
from chs_scenario import Settings

TEAMS = 5
STREAMS = 8                      # Seeds seed·8 + k: ein eigener Strom je Zufallsquelle


def stream(seed: int, k: int, count: int) -> np.ndarray:
    """`count` Gleichverteilte aus dem k-ten Strom dieses Spiels."""
    return R.uniform_block(seed * STREAMS + k, count)


def draw_questions(seed: int, rounds: int) -> tuple[np.ndarray, np.ndarray]:
    """Fragen x, y je Runde, gleichverteilt und unabhängig."""
    return (stream(seed, 0, rounds) >= 0.5).astype(int), (stream(seed, 1, rounds) >= 0.5).astype(int)


def answers_fixed(index: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mannschaft 0: deterministische Strategie `index`."""
    fa, fb = G.strategy_answers(index)
    return np.array(fa)[x], np.array(fb)[y]


def answers_shared_direction(alpha, beta, seed: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mannschaft 1: dieselben Winkel wie die Verschränkung, aber eine gemeinsame Zufallsrichtung λ statt des Qubit-Paars."""
    lam = math.pi * stream(seed, 2, len(x))
    return G.shared_direction_answer(lam, np.array(alpha)[x]), G.shared_direction_answer(lam, np.array(beta)[y])


def answers_entangled(rho: np.ndarray, alpha, beta, seed: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mannschaft 2: Born-Stichprobe aus P[x, y, :, :] des Zustands."""
    return Q.sample_outcomes(Q.joint_table(rho, alpha, beta), x, y, stream(seed, 3, len(x)))


def answers_box(seed: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mannschaft 3: die Box gibt Alice ein gleichverteiltes Bit a und Bob b = a ⊕ (x ∧ y) (marginal gleichverteilt, gewinnt immer)."""
    a = (stream(seed, 4, len(x)) >= 0.5).astype(int)
    return a, a ^ (x & y)


def answers_communication(seed: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mannschaft 4: Alice antwortet mit einem Zufallsbit, Bob kennt zusätzlich x und rechnet b = a ⊕ (x ∧ y)."""
    a = (stream(seed, 5, len(x)) >= 0.5).astype(int)
    return a, a ^ (x & y)


@dataclass(frozen=True)
class Play:
    x: np.ndarray
    y: np.ndarray
    a: np.ndarray            # Form (TEAMS, Runden)
    b: np.ndarray
    won: np.ndarray          # bool, Form (TEAMS, Runden)

    @property
    def rounds(self) -> int:
        return len(self.x)

    def cumulative_rates(self) -> np.ndarray:
        """Gewinnquote nach t Runden, Form (TEAMS, Runden)."""
        return np.cumsum(self.won, axis=1) / np.arange(1, self.rounds + 1)

    def final_rates(self) -> np.ndarray:
        return self.won.mean(axis=1)


def play(settings: Settings) -> Play:
    """Alle fünf Mannschaften auf denselben `settings.rounds` Fragen."""
    x, y = draw_questions(settings.seed, settings.rounds)
    teams = (
        answers_fixed(settings.strategy, x, y),
        answers_shared_direction(settings.alpha, settings.beta, settings.seed, x, y),
        answers_entangled(settings.rho(), settings.alpha, settings.beta, settings.seed, x, y),
        answers_box(settings.seed, x, y),
        answers_communication(settings.seed, x, y),
    )
    a = np.array([t[0] for t in teams])
    b = np.array([t[1] for t in teams])
    won = (a ^ b) == (x & y)[None, :]
    return Play(x, y, a, b, won)
