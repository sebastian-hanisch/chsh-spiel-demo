"""Das verschränkte Qubit-Paar als Matrixrechnung (nur numpy).

Zustand: ρ = v · |ψθ⟩⟨ψθ| + (1 − v) · I/4 mit |ψθ⟩ = cosθ |00⟩ + sinθ |11⟩ (θ = 45 Grad: Bell-Zustand, v = 1: rein, kleineres v: weißes Rauschen).
Messung: der Spieler mit Winkel φ misst in der Basis cosφ |0⟩ + sinφ |1⟩ gegen die dazu senkrechte; Ergebnis 0 bei der ersten, 1 bei der zweiten. Winkel im Bogenmaß.
Lehrbuchmodell (ideale projektive Messungen); keine Aussage über Geräte. Jede Regel ist eine eigene Funktion (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np

import chs_game as G

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]])
PAULIS = (SIGMA_X, SIGMA_Y, SIGMA_Z)
IDENTITY2 = np.eye(2)

BELL_ALPHA = (0.0, math.pi / 4)
BELL_BETA = (math.pi / 8, -math.pi / 8)
BELL_VALUE = math.cos(math.pi / 8) ** 2        # = 1/2 + √2/4 = 0,853553…


def state_vector(theta: float) -> np.ndarray:
    """|ψθ⟩ = cosθ |00⟩ + sinθ |11⟩ in der Basis 00, 01, 10, 11."""
    return np.array([math.cos(theta), 0.0, 0.0, math.sin(theta)])


def density_matrix(theta: float, visibility: float) -> np.ndarray:
    """ρ = v · |ψθ⟩⟨ψθ| + (1 − v) · I/4 mit v in [0, 1]."""
    if not 0.0 <= visibility <= 1.0:
        raise ValueError(f"Sichtbarkeit {visibility} liegt außerhalb von 0 bis 1")
    psi = state_vector(theta)
    return visibility * np.outer(psi, psi) + (1.0 - visibility) * np.eye(4) / 4.0


def projector(phi: float, outcome: int) -> np.ndarray:
    """Projektor auf die Basisrichtung cosφ |0⟩ + sinφ |1⟩ (Ergebnis 0) bzw. die dazu senkrechte (Ergebnis 1)."""
    v = np.array([math.cos(phi), math.sin(phi)])
    if outcome == 0:
        return np.outer(v, v)
    if outcome == 1:
        return IDENTITY2 - np.outer(v, v)
    raise ValueError("Ergebnis muss 0 oder 1 sein")


def joint_table(rho: np.ndarray, alpha, beta) -> np.ndarray:
    """Born-Regel: P[x, y, a, b] = Spur(ρ · (Π_a(α_x) ⊗ Π_b(β_y))). Die Zeilenreihenfolge von ρ ist (Alice, Bob)."""
    table = np.zeros((2, 2, 2, 2))
    for x in (0, 1):
        for y in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    table[x, y, a, b] = float(np.real(np.trace(rho @ np.kron(projector(alpha[x], a), projector(beta[y], b)))))
    return table


def win_probability(rho: np.ndarray, alpha, beta) -> float:
    return G.win_probability_from_table(joint_table(rho, alpha, beta))


def sample_outcomes(table: np.ndarray, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Born-Stichprobe: zu den Fragen (x, y) je Runde ein Antwortpaar (a, b), gezogen aus P[x, y, :, :] mit der Gleichverteilten u (Inversion der Verteilungsfunktion)."""
    cum = np.cumsum(np.asarray(table).reshape(2, 2, 4), axis=-1)
    k = (u[:, None] > cum[x, y]).sum(axis=1)
    k = np.minimum(k, 3)                              # Rundungsrest: die letzte Summe kann knapp unter 1 liegen
    return k // 2, k % 2


# --- Optimum über die Messungen (Horodecki) ------------------------------------------------------------------------------------------------------------

def correlation_matrix(rho: np.ndarray) -> np.ndarray:
    """T_ij = Spur(ρ · σ_i ⊗ σ_j) für die drei Pauli-Matrizen."""
    return np.array([[float(np.real(np.trace(rho @ np.kron(si, sj)))) for sj in PAULIS] for si in PAULIS])


def max_chsh(rho: np.ndarray) -> float:
    """Größter CHSH-Wert S = E00 + E01 + E10 − E11 über alle Messungen: 2·√(Summe der zwei größten Eigenwerte von TᵀT) (Horodecki 1995, zweiter Rechenweg zur Optimierung)."""
    t = correlation_matrix(rho)
    eig = np.sort(np.linalg.eigvalsh(t.T @ t))
    return 2.0 * math.sqrt(max(0.0, float(eig[-1] + eig[-2])))


def max_win(rho: np.ndarray) -> float:
    """Größte Gewinnwahrscheinlichkeit über alle Messungen: ½ + S/8."""
    return 0.5 + max_chsh(rho) / 8.0


def max_win_closed_form(theta: float, visibility: float) -> float:
    """Geschlossene Formel für die Zustandsfamilie: ½ + (v/4) · √(1 + sin²2θ)."""
    return 0.5 + visibility / 4.0 * math.sqrt(1.0 + math.sin(2.0 * theta) ** 2)


def advantage_threshold(theta: float) -> float:
    """Kleinste Sichtbarkeit, ab der der Zustand 0,75 überschreiten kann: v* = 1/√(1 + sin²2θ) (für θ = 0 gleich 1: dort nie)."""
    return 1.0 / math.sqrt(1.0 + math.sin(2.0 * theta) ** 2)


# --- Schnelle Fassung für Gitter und Heatmaps ----------------------------------------------------------------------------------------------------------

def correlator(theta: float, visibility: float, a, b):
    """E(a, b) = Erwartung von (±1)-Ergebnissen beider Spieler: v · (cos2a cos2b + sin2θ · sin2a sin2b). Verbreitet sich über Arrays."""
    return visibility * (np.cos(2 * a) * np.cos(2 * b) + math.sin(2 * theta) * np.sin(2 * a) * np.sin(2 * b))


def win_from_correlators(theta: float, visibility: float, a0, a1, b0, b1):
    """W = ½ + ⅛ · (E00 + E01 + E10 − E11): Gewinnwahrscheinlichkeit aus den vier Korrelatoren; verbreitet sich über Arrays."""
    def e(a, b):
        return correlator(theta, visibility, a, b)

    return 0.5 + (e(a0, b0) + e(a0, b1) + e(a1, b0) - e(a1, b1)) / 8.0
