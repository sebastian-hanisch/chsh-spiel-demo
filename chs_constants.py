"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Alle Rechnungen dieser Demo sind winzig (4×4-Matrizen, Gitter bis 24⁴ = 331 776 Zellen, Wiederholungen bis 1,6 Millionen Zufallszahlen); das Rechenbudget am
Slider-Maximum steht bei den Grenzen."""

THETA_MIN, THETA_MAX, DEFAULT_THETA = 0.0, 45.0, 45.0         # Zustandswinkel in Grad: 0 = Produktzustand, 45 = Bell-Zustand
VIS_MIN, VIS_MAX, DEFAULT_VIS = 0, 100, 100                   # Sichtbarkeit v in Prozent (Anteil des reinen Zustands, Rest ist weißes Rauschen)
ANGLE_MIN, ANGLE_MAX = -90.0, 90.0                            # Messwinkel in Grad (Hilbert-Raum-Winkel; Periode 180 Grad)
DEFAULT_ALPHA0, DEFAULT_ALPHA1 = 0.0, 45.0                    # Alice: Winkel bei Frage 0 und 1
DEFAULT_BETA0, DEFAULT_BETA1 = 22.5, -22.5                    # Bob: Winkel bei Frage 0 und 1 (die bekannten Winkel für den Bell-Zustand)
STRATEGY_COUNT, DEFAULT_STRATEGY = 16, 0                      # 16 deterministische Strategien; 0 = beide antworten immer 0
ROUNDS_MIN, ROUNDS_MAX, DEFAULT_ROUNDS = 20, 5000, 400       # Runden des Spielverlaufs: O(Runden) für fünf Mannschaften
SEED_MAX, DEFAULT_SEED = 999999, 35

CLASSICAL_BOUND = 0.75                                        # klassisches Maximum; die App rechnet es aus der Aufzählung, der Wert hier ist nur die Vergleichslinie
TOLERANCE = 1e-9                                              # Gleitkommagrenze für „größer als 0,75“ in Gitterzählungen

SHOWN_ROUNDS = 12                                             # Zeilen der Protokolltabelle
GRID_STEP_DEG = 7.5                                           # Gitter der Zählung: 24 Winkel je Messrichtung, 24⁴ = 331 776 Zellen (enthält 22,5 und 45 Grad)
HEAT_STEP_DEG = 2.0                                           # Heatmap: 91 × 91 Zellen
MIX_COUNT = 20000                                             # Zufallsmischungen im Experiment E1
E4_ROUNDS = (25, 50, 100, 200, 400, 800, 1600)                # Rundenzahlen in E4 (Wiederholungen · Runden ≤ 1,6 Millionen Zufallszahlen)
E4_REPETITIONS = 1000

TEAM_NAMES = ("Keine Ressource", "Geteilter Zufall", "Verschränkung", "No-Signalling-Box", "Absprache im Spiel")
TEAM_COLORS = ("#7f7f7f", "#4c78a8", "#2F6B65", "#b279a2", "#e45756")
COLOR_ALICE, COLOR_BOB = "#4c78a8", "#e68a00"
COLOR_LINE = "#4c78a8"

PRESETS = {
    "Optimale Winkel (Voreinstellung)": {"theta": DEFAULT_THETA, "vis": DEFAULT_VIS, "a0": DEFAULT_ALPHA0, "a1": DEFAULT_ALPHA1, "b0": DEFAULT_BETA0, "b1": DEFAULT_BETA1,
                                         "strategy": DEFAULT_STRATEGY, "rounds": DEFAULT_ROUNDS, "seed": DEFAULT_SEED},
    "Winkel ohne Vorteil": {"theta": DEFAULT_THETA, "vis": DEFAULT_VIS, "a0": 0.0, "a1": 0.0, "b0": 0.0, "b1": 0.0,
                            "strategy": DEFAULT_STRATEGY, "rounds": DEFAULT_ROUNDS, "seed": DEFAULT_SEED},
    "Ohne Verschränkung": {"theta": 0.0, "vis": DEFAULT_VIS, "a0": DEFAULT_ALPHA0, "a1": DEFAULT_ALPHA1, "b0": DEFAULT_BETA0, "b1": DEFAULT_BETA1,
                           "strategy": DEFAULT_STRATEGY, "rounds": DEFAULT_ROUNDS, "seed": DEFAULT_SEED},
    "Rauschen 70 %": {"theta": DEFAULT_THETA, "vis": 70, "a0": DEFAULT_ALPHA0, "a1": DEFAULT_ALPHA1, "b0": DEFAULT_BETA0, "b1": DEFAULT_BETA1,
                      "strategy": DEFAULT_STRATEGY, "rounds": DEFAULT_ROUNDS, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Optimale Winkel (Voreinstellung)": "Bell-Zustand, rein (Sichtbarkeit 100 %), Messwinkel 0 und 45 Grad für Alice, 22,5 und −22,5 Grad für Bob: die bekannten Winkel, bei denen die Verschränkung 0,8536 erreicht.",
    "Winkel ohne Vorteil": "Gleicher Bell-Zustand, aber alle vier Messwinkel auf 0 Grad: die Verschränkung gewinnt dann genau 0,75, so viel wie die beste klassische Strategie. Der Zustand allein genügt nicht, die Messungen müssen passen.",
    "Ohne Verschränkung": "Produktzustand (Zustandswinkel 0) mit den Winkeln des Bell-Falls: die Mannschaft liegt unter 0,75. Auch mit den besten Winkeln für diesen Zustand reicht es nur zu 0,75.",
    "Rauschen 70 %": "Bell-Zustand mit Sichtbarkeit 70 %, also 30 % weißem Rauschen, und den optimalen Winkeln: der Wert liegt knapp unter 0,75, weil die Schwelle bei 70,7 % liegt.",
}
