"""Kennzahlen und Verdikte. Jedes Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`."""

import math
from dataclasses import dataclass

import numpy as np

import chs_constants as C
import chs_game as G
import chs_quantum as Q
from chs_play import Play
from chs_scenario import Settings


def de(x: float, nd: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und echtem Minuszeichen (−)."""
    text = f"{x:.{nd}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")
    return text.replace(".", ",").replace("-", "−")


def thousands(n: int | float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x: float, nd: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma und Leerzeichen vor dem Prozentzeichen."""
    return f"{de(100.0 * x, nd)} %"


def points(x: float, nd: int = 2) -> str:
    """Differenz zweier Anteile in Prozentpunkten."""
    return f"{de(100.0 * x, nd)} Prozentpunkte"


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    classical_best: float        # beste deterministische Strategie, durch Aufzählung
    classical_best_count: int    # Zahl der Strategien mit diesem Wert
    strategy_value: float        # die in der Seitenleiste gewählte Strategie
    w_shared: float              # geteilter Zufall mit den eingestellten Winkeln (gemeinsame Richtung)
    w_quantum: float             # Verschränkung mit den eingestellten Winkeln (Born-Regel)
    w_best: float                # Bestwert dieses Zustands über alle Messungen
    threshold: float             # Sichtbarkeit, ab der dieser Zustandswinkel 0,75 überschreiten kann
    q_questions: np.ndarray      # 2×2: Gewinnwahrscheinlichkeit je Fragepaar, Verschränkung
    s_questions: np.ndarray      # 2×2: dasselbe für die gemeinsame Richtung
    box: float                   # No-Signalling-Box (Definition)
    one_bit: float               # mit Absprache im Spiel (Aufzählung)


def analyse(settings: Settings) -> Analysis:
    best_wins, best_count = G.best_classical_wins()
    rho = settings.rho()
    table = Q.joint_table(rho, settings.alpha, settings.beta)
    return Analysis(
        settings=settings, classical_best=best_wins / 4, classical_best_count=best_count, strategy_value=G.strategy_value(settings.strategy),
        w_shared=G.shared_direction_win(settings.alpha, settings.beta), w_quantum=G.win_probability_from_table(table), w_best=Q.max_win(rho),
        threshold=Q.advantage_threshold(settings.theta), q_questions=G.question_win_probabilities(table),
        s_questions=G.shared_direction_question_wins(settings.alpha, settings.beta), box=G.win_probability_from_table(G.no_signalling_box()),
        one_bit=G.best_with_one_bit() / 4)


def standard_error(p: float, n: int) -> float:
    """Standardfehler einer Gewinnquote aus n unabhängigen Runden mit Gewinnwahrscheinlichkeit p."""
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


# --- Verdikte ------------------------------------------------------------------------------------------------------------------------------------------

def verdict_resource(a: Analysis) -> tuple[str, str]:
    """Kernfrage: Schlägt die Verschränkung mit den eingestellten Messungen die beste klassische Strategie?"""
    bound = a.classical_best
    diff = a.w_quantum - bound
    if diff > C.TOLERANCE:
        return "success", (f"Die verschränkte Mannschaft gewinnt mit Wahrscheinlichkeit **{pct(a.w_quantum, 2)}**, das sind {points(diff)} mehr als der klassische Höchstwert "
                           f"von {pct(bound, 2)} (beste von 16 festen Strategien; geteilter Zufall mischt nur diese Strategien und erreicht nicht mehr).")
    if abs(diff) <= C.TOLERANCE:
        return "info", (f"Die verschränkte Mannschaft gewinnt mit Wahrscheinlichkeit **{pct(a.w_quantum, 2)}**, genau so viel wie der klassische Höchstwert von {pct(bound, 2)}: "
                        "mit diesen Messrichtungen bringt die Verschränkung hier keinen Vorsprung.")
    return "warning", (f"Die verschränkte Mannschaft gewinnt nur mit Wahrscheinlichkeit **{pct(a.w_quantum, 2)}**, {points(-diff)} weniger als der klassische Höchstwert von {pct(bound, 2)}: "
                       "diese Messrichtungen und dieser Zustand sind schlechter als eine feste Absprache.")


def verdict_state(a: Analysis) -> tuple[str, str]:
    """Was ist für diesen Zustand überhaupt möglich (Bestwert über alle Messungen)?"""
    s = a.settings
    bound = a.classical_best
    gap = a.w_best - bound
    if gap > C.TOLERANCE:
        reach = "" if a.w_quantum >= a.w_best - C.TOLERANCE else f" Die eingestellten Winkel erreichen {pct(a.w_quantum, 2)}."
        return "info", (f"Für diesen Zustand (Zustandswinkel {de(s.theta_deg, 1)} Grad, Sichtbarkeit {de(s.vis_pct, 0)} %) gibt es Messungen mit **{pct(a.w_best, 2)}**, "
                        f"{points(gap)} über {pct(bound, 2)}.{reach}")
    if a.threshold >= 1.0 - C.TOLERANCE:
        return "warning", (f"Dieser Zustand ist ein Produktzustand: kein Messwinkel bringt mehr als {pct(a.w_best, 2)}, die Verschränkung fehlt. "
                           "Nicht einmal reine Sichtbarkeit würde den klassischen Höchstwert überschreiten.")
    return "warning", (f"Bei Sichtbarkeit {de(s.vis_pct, 0)} % gibt es für diesen Zustand keine Messung über {pct(bound, 2)} (Bestwert {pct(a.w_best, 2)}). "
                       f"Dafür müsste die Sichtbarkeit über {pct(a.threshold, 1)} liegen.")


def verdict_game(a: Analysis, play: Play) -> tuple[str, str]:
    """Der Spielverlauf: Gewinnquoten der fünf Mannschaften auf denselben Fragen, mit Standardfehler der verschränkten Mannschaft."""
    n = play.rounds
    rates = play.final_rates()
    se = standard_error(a.w_quantum, n)
    ahead = rates[2] > rates[0]
    text = (f"Nach {thousands(n)} Runden: ohne Ressource {pct(rates[0], 1)}, geteilter Zufall {pct(rates[1], 1)}, Verschränkung {pct(rates[2], 1)} "
            f"(erwartet {pct(a.w_quantum, 1)} ± {pct(se, 1)} Standardfehler), No-Signalling-Box {pct(rates[3], 1)}, Absprache im Spiel {pct(rates[4], 1)}. ")
    if ahead:
        text += "In dieser Stichprobe liegt die Verschränkung vor der festen Strategie; wie oft das bei so vielen Runden zutrifft, zeigt Experiment E4."
        return "success", text
    text += "In dieser Stichprobe liegt die Verschränkung nicht vor der festen Strategie; bei wenigen Runden kann der Zufall die Reihenfolge drehen (Experiment E4)."
    return "info", text


def verdict_e1(rows: list[dict], mixtures: dict) -> tuple[str, str]:
    top = max(r["wins"] for r in rows)
    n_top = sum(1 for r in rows if r["wins"] == top)
    n_low = sum(1 for r in rows if r["wins"] == min(r["wins"] for r in rows))
    mean = sum(r["value"] for r in rows) / len(rows)
    text = (f"Von den {len(rows)} festen Strategien gewinnen {n_top} genau {top} von 4 Fragepaaren (**{pct(top / 4, 2)}**), {n_low} nur {min(r['wins'] for r in rows)}; Mittel über alle {len(rows)}: {pct(mean, 1)}. "
            f"{thousands(mixtures['count'])} zufällige Mischungen: größter Wert {pct(mixtures['max'], 2)}.")
    if mixtures["max"] <= top / 4 + C.TOLERANCE:
        return "success", text + " Keine Mischung liegt über dem Maximum der Strategien, wie es sein muss: eine Mischung ist ein gewichteter Mittelwert."
    return "error", text + " Eine Mischung liegt über dem Maximum der Strategien: das kann nicht sein, die Rechnung ist fehlerhaft."


def verdict_e2(counts: dict, settings: Settings) -> tuple[str, str]:
    q, s = counts["quantum"], counts["shared"]
    share = q["above"] / counts["cells"]
    text = (f"Auf dem Gitter mit {thousands(counts['cells'])} Winkelquadrupeln ({counts['points']} Richtungen je Winkel) liegen für die Verschränkung (Zustandswinkel {de(settings.theta_deg, 1)} Grad, "
            f"Sichtbarkeit {de(settings.vis_pct, 0)} %) {thousands(q['above'])} Zellen ({pct(share, 1)}) über 0,75, Maximum {pct(q['max'], 2)} in {thousands(q['at_max'])} Zellen. "
            f"Mit geteilter Zufallsrichtung sind es {thousands(s['above'])} Zellen über 0,75, Maximum {pct(s['max'], 2)}.")
    if q["above"] > 0 and s["above"] == 0:
        return "success", text
    if q["above"] == 0 and s["above"] == 0:
        return "warning", text + " Für diesen Zustand gibt es auf dem Gitter keine Messung mit Vorsprung."
    return "error", text + " Die geteilte Zufallsrichtung liegt über 0,75: das widerspricht der Grenze für klassische Modelle."


def verdict_e3(rows: list[dict]) -> tuple[str, str]:
    full = next(r for r in rows if r["theta_deg"] == max(x["theta_deg"] for x in rows))
    weak = next(r for r in rows if r["theta_deg"] == 10.0)
    zero = next(r for r in rows if r["theta_deg"] == 0.0)
    entangled = [r for r in rows if r["theta_deg"] > 0]
    above = sum(1 for r in entangled if r["best"] > C.CLASSICAL_BOUND + C.TOLERANCE)
    text = (f"Bei reiner Sichtbarkeit liegt der Bestwert in {above} von {len(entangled)} verschränkten Zuständen über 0,75 (Zustandswinkel {de(weak['theta_deg'], 0)} Grad: {pct(weak['best'], 2)}), "
            f"beim Produktzustand nicht ({pct(zero['best'], 2)}). "
            f"Die Schwelle der Sichtbarkeit sinkt mit der Verschränkung: bei {de(weak['theta_deg'], 0)} Grad {pct(weak['threshold'], 1)}, beim Bell-Zustand ({de(full['theta_deg'], 0)} Grad) {pct(full['threshold'], 1)}. "
            "Darunter erreicht in dieser Zustandsfamilie keine Messung mehr als 0,75.")
    return "info", text


def verdict_e4(rows: list[dict], threshold: float = 0.95) -> tuple[str, str]:
    reach = next((r for r in rows if r["more"] >= threshold), None)
    first, last = rows[0], rows[-1]
    text = (f"Bei {first['rounds']} Runden gewinnt die verschränkte Mannschaft in {pct(first['more'], 1)} der {thousands(first['repetitions'])} Wiederholungen strikt mehr Runden als die feste Strategie "
            f"(gleich viele: {pct(first['equal'], 1)}), bei {last['rounds']} Runden in {pct(last['more'], 1)}. ")
    if reach is not None:
        return "success", text + f"Ab {reach['rounds']} Runden sind es mindestens {pct(threshold, 0)}."
    return "info", text + f"Bis {last['rounds']} Runden erreicht der Anteil nicht {pct(threshold, 0)}."
