"""Gemeinsame Fixtures: die Mini-Fälle, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def strategies():
    """Strategie-Nummer = Bits (a(0), a(1), b(0), b(1)) von oben nach unten.
    0 = 0000: beide antworten immer 0, a ⊕ b = 0. Das Ziel ist x ∧ y: gewonnen bei (0,0), (0,1), (1,0), verloren bei (1,1): 3 von 4.
    5 = 0101: a = (0, 1), b = (0, 1). (0,0): 0 ⊕ 0 = 0, Ziel 0 gewonnen. (0,1): 0 ⊕ 1 = 1, Ziel 0 verloren. (1,0): 1 ⊕ 0 = 1, Ziel 0 verloren. (1,1): 1 ⊕ 1 = 0, Ziel 1 verloren: 1 von 4.
    6 = 0110: a = (0, 1), b = (1, 0). (0,0): 0 ⊕ 1 = 1 verloren. (0,1): 0 ⊕ 0 = 0 gewonnen. (1,0): 1 ⊕ 1 = 0 gewonnen. (1,1): 1 ⊕ 0 = 1 gewonnen: 3 von 4."""
    return {"all_zero": (0, 3), "diagonal": (5, 1), "cross": (6, 3)}


@pytest.fixture
def bell():
    """Bell-Zustand (θ = 45 Grad, v = 1): Φ+ = (|00⟩ + |11⟩)/√2.
    Gleiche Winkel α = β = 0: P(0,0) = |⟨00|Φ+⟩|² = ½, P(1,1) = ½, P(0,1) = P(1,0) = 0. Allgemein P(a = b) = cos²(α − β).
    Die bekannten Winkel (α = 0, π/4; β = π/8, −π/8): Abstände π/8, π/8, π/8 und 3π/8; gewonnen wird bei gleichem Ziel mit cos²(π/8) = (1 + cos(π/4))/2 = (1 + 0,70710678)/2 = 0,85355339,
    beim Paar (1,1) mit sin²(3π/8) = (1 − cos(3π/4))/2 = (1 + 0,70710678)/2 = 0,85355339 ebenfalls."""
    return {"equal_angles": {"p_equal": 1.0, "p00": 0.5, "p11": 0.5}, "value": (1 + math.sqrt(0.5)) / 2}


@pytest.fixture
def pr_box():
    """No-Signalling-Box: bei (x, y) = (0, 0) sind (0,0) und (1,1) mit je ½ erlaubt, bei (1,1) (0,1) und (1,0) mit je ½. Jede Antwort allein hat Wahrscheinlichkeit ½; der Gewinn ist 1."""
    return {"allowed_00": [(0, 0), (1, 1)], "allowed_11": [(0, 1), (1, 0)], "win": 1.0}


@pytest.fixture
def shared_direction():
    """Gemeinsame Richtung: Abstand 0 → gleiche Antwort mit 1 − 0 = 1, Abstand π/8 → 1 − 2·(π/8)/π = 3/4, Abstand π/4 → ½, Abstand π/2 → 0 (stets verschieden).
    Mit den bekannten Winkeln: drei Paare mit Abstand π/8 gewinnen mit 3/4, das Paar (1,1) mit Abstand 3π/8: Gleichheit 1 − 3/4 = ¼, also Verschiedenheit 3/4: Mittel 3/4."""
    return {"agree": {0.0: 1.0, math.pi / 8: 0.75, math.pi / 4: 0.5, math.pi / 2: 0.0}, "value": 0.75}
