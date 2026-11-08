"""Zellen und Reihen der Experimente (nur numpy). Jede Funktion bekommt alle Eingaben einschließlich des Zufalls-Seeds und liefert Zahlen, keine Texte."""

import math

import numpy as np

import chs_constants as C
import chs_game as G
import chs_quantum as Q
import chs_rng as R
from chs_scenario import Settings


# --- E1: alle 16 Strategien und Mischungen -------------------------------------------------------------------------------------------------------------

def e1_strategy_rows() -> list[dict]:
    return [{"index": i, "label": G.strategy_label(i), "wins": G.strategy_wins(i), "value": G.strategy_value(i)} for i in range(16)]


def e1_mixtures(count: int, seed: int) -> dict:
    """`count` zufällige Mischungen der 16 Strategien: Werte, Maximum, Mittel."""
    values = G.mixture_values(G.random_weights(count, seed))
    return {"values": values, "max": float(values.max()), "mean": float(values.mean()), "count": count}


# --- E2: Winkel-Landschaft ------------------------------------------------------------------------------------------------------------------------------

def heat_axis(step_deg: float = C.HEAT_STEP_DEG) -> np.ndarray:
    """Winkel −90 bis 90 Grad in `step_deg`-Schritten (inklusive beider Ränder), im Bogenmaß."""
    return np.radians(np.arange(C.ANGLE_MIN, C.ANGLE_MAX + step_deg / 2, step_deg))


def e2_heatmaps(settings: Settings, step_deg: float = C.HEAT_STEP_DEG) -> dict:
    """Gewinnwahrscheinlichkeit über (α₁, β₁) bei festen α₀ und β₀ aus den Einstellungen: Verschränkung (Zustand der Einstellungen) und geteilte Zufallsrichtung."""
    ax = heat_axis(step_deg)
    a1, b1 = np.meshgrid(ax, ax, indexing="ij")
    a0, b0 = settings.alpha[0], settings.beta[0]
    quantum = Q.win_from_correlators(settings.theta, settings.visibility, a0, a1, b0, b1)
    shared = 0.25 * (G.shared_direction_agreement(a0 - b0) + G.shared_direction_agreement(a0 - b1) + G.shared_direction_agreement(a1 - b0)
                     + (1.0 - G.shared_direction_agreement(a1 - b1)))
    return {"axis_deg": np.degrees(ax), "quantum": quantum, "shared": shared}


def e2_grid_counts(settings: Settings, step_deg: float = C.GRID_STEP_DEG) -> dict:
    """Zählung über alle vier Winkel auf dem Gitter (Schritt `step_deg`, −90 Grad bis knapp 90 Grad): Zellen über 0,75 und Maximum für die Verschränkung und für die geteilte Zufallsrichtung."""
    g = np.radians(np.arange(C.ANGLE_MIN, C.ANGLE_MAX, step_deg))
    a0, a1, b0, b1 = np.meshgrid(g, g, g, g, indexing="ij")
    quantum = Q.win_from_correlators(settings.theta, settings.visibility, a0, a1, b0, b1)
    shared = 0.25 * (G.shared_direction_agreement(a0 - b0) + G.shared_direction_agreement(a0 - b1) + G.shared_direction_agreement(a1 - b0)
                     + (1.0 - G.shared_direction_agreement(a1 - b1)))
    out = {"cells": int(quantum.size), "points": len(g)}
    for name, field in (("quantum", quantum), ("shared", shared)):
        top = float(field.max())
        out[name] = {"above": int((field > C.CLASSICAL_BOUND + C.TOLERANCE).sum()), "max": top, "at_max": int((field > top - C.TOLERANCE).sum())}
    return out


# --- E3: Zustand und Rauschen ---------------------------------------------------------------------------------------------------------------------------

E3_THETAS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0)
E3_VISIBILITIES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)


def e3_rows() -> list[dict]:
    """Bestwert (aus der Korrelationsmatrix, Horodecki) und Schwelle der Sichtbarkeit je Zustandswinkel bei reiner Sichtbarkeit."""
    rows = []
    for deg in E3_THETAS:
        th = math.radians(deg)
        rows.append({"theta_deg": deg, "best": Q.max_win(Q.density_matrix(th, 1.0)), "threshold": Q.advantage_threshold(th)})
    return rows


def e3_visibility_rows(theta_deg: float) -> list[dict]:
    th = math.radians(theta_deg)
    return [{"vis": v, "best": Q.max_win(Q.density_matrix(th, v))} for v in E3_VISIBILITIES]


def e3_curves(points: int = 91) -> dict:
    """Linien für die Abbildung: Bestwert gegen Zustandswinkel (v = 1) und Schwelle der Sichtbarkeit gegen Zustandswinkel."""
    deg = np.linspace(C.THETA_MIN, C.THETA_MAX, points)
    best = [Q.max_win(Q.density_matrix(math.radians(d), 1.0)) for d in deg]
    thr = [Q.advantage_threshold(math.radians(d)) for d in deg]
    return {"theta_deg": deg, "best": np.array(best), "threshold": np.array(thr)}


# --- E4: Wie viele Runden braucht man? -----------------------------------------------------------------------------------------------------------------

def e4_cell(settings: Settings, rounds: int, repetitions: int) -> dict:
    """`repetitions` Spiele zu je `rounds` Runden: Anteil, in dem die verschränkte Mannschaft strikt mehr, genau gleich viele und weniger Runden gewinnt als die klassische Strategie der Einstellungen
    (gleiche Fragen). Die verschränkte Mannschaft gewinnt je Fragepaar mit der Wahrscheinlichkeit aus der Born-Regel; die klassische Strategie ist fest."""
    q_win = G.question_win_probabilities(Q.joint_table(settings.rho(), settings.alpha, settings.beta))
    fa, fb = G.strategy_answers(settings.strategy)
    c_win = np.array([[1.0 if G.is_won(x, y, fa[x], fb[y]) else 0.0 for y in (0, 1)] for x in (0, 1)])
    seed = settings.seed * 1000 + rounds % 1000
    cat = np.minimum((R.uniform_block(seed * 2, repetitions * rounds) * 4).astype(int), 3).reshape(repetitions, rounds)
    u = R.uniform_block(seed * 2 + 1, repetitions * rounds).reshape(repetitions, rounds)
    x, y = cat // 2, cat % 2
    q_wins = (u < q_win[x, y]).sum(axis=1)
    c_wins = c_win[x, y].sum(axis=1)
    return {"rounds": rounds, "repetitions": repetitions, "more": float((q_wins > c_wins).mean()), "equal": float((q_wins == c_wins).mean()), "less": float((q_wins < c_wins).mean()),
            "q_rate": float(q_wins.mean() / rounds), "c_rate": float(c_wins.mean() / rounds)}


def e4_rows(settings: Settings, rounds_list=C.E4_ROUNDS, repetitions: int = C.E4_REPETITIONS) -> list[dict]:
    return [e4_cell(settings, r, repetitions) for r in rounds_list]
