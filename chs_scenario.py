"""Szenario: die Einstellungen der Demo (nur mit Schlüsselwörtern, damit ein eingefügtes Feld nie still Werte verschiebt)."""

import math
from dataclasses import dataclass

import chs_constants as C
import chs_quantum as Q


@dataclass(frozen=True, kw_only=True)
class Settings:
    theta_deg: float = C.DEFAULT_THETA          # Zustandswinkel in Grad (45 = Bell-Zustand)
    vis_pct: float = C.DEFAULT_VIS              # Sichtbarkeit in Prozent
    alpha0_deg: float = C.DEFAULT_ALPHA0        # Messwinkel in Grad: Alice bei Frage 0 / 1, Bob bei Frage 0 / 1
    alpha1_deg: float = C.DEFAULT_ALPHA1
    beta0_deg: float = C.DEFAULT_BETA0
    beta1_deg: float = C.DEFAULT_BETA1
    strategy: int = C.DEFAULT_STRATEGY          # klassische Strategie 0..15
    rounds: int = C.DEFAULT_ROUNDS
    seed: int = C.DEFAULT_SEED

    @property
    def theta(self) -> float:
        return math.radians(self.theta_deg)

    @property
    def visibility(self) -> float:
        return self.vis_pct / 100.0

    @property
    def alpha(self) -> tuple[float, float]:
        return math.radians(self.alpha0_deg), math.radians(self.alpha1_deg)

    @property
    def beta(self) -> tuple[float, float]:
        return math.radians(self.beta0_deg), math.radians(self.beta1_deg)

    def rho(self):
        return Q.density_matrix(self.theta, self.visibility)
