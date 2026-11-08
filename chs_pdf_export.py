"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Kennzahlen und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import chs_constants as C
import chs_evaluation as V
import chs_game as G

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "α": "alpha", "θ": "theta", "ρ": "rho", "λ": "lambda", "φ": "phi", "π": "pi", "ψ": "psi", "Φ": "Phi", "Δ": "Delta", "×": "x", "≈": "~",
            "⊕": "xor", "∧": "und", "⟩": ">", "⟨": "<", "½": "1/2", "…": "...", "‚": ",", "‘": "'", "’": "'", "”": '"', "µ": "mu"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: V.Analysis, play, verdicts: list[tuple[str, str]]) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Das CHSH-Spiel: Was geteilter Zufall und Verschränkung bringen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Zustandswinkel {V.de(s.theta_deg, 1)} Grad, Sichtbarkeit {V.de(s.vis_pct, 0)} %; Messwinkel Alice {V.de(s.alpha0_deg, 1)} / {V.de(s.alpha1_deg, 1)} Grad, "
                                f"Bob {V.de(s.beta0_deg, 1)} / {V.de(s.beta1_deg, 1)} Grad; feste Strategie: {G.strategy_label(s.strategy)}; {s.rounds} Runden, Zufalls-Seed {s.seed}."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Beste feste Strategie und geteilter Zufall (beste Mischung): {V.pct(a.classical_best, 2)}; geteilter Zufall mit den eingestellten Winkeln: {V.pct(a.w_shared, 2)}; "
                                f"Verschränkung mit den eingestellten Winkeln: {V.pct(a.w_quantum, 2)}; Bestwert dieses Zustands: {V.pct(a.w_best, 2)}; "
                                f"No-Signalling-Box (Definition): {V.pct(a.box, 0)}; Absprache im Spiel: {V.pct(a.one_bit, 0)}."), new_x="LMARGIN", new_y="NEXT")
    rates = play.final_rates()
    pdf.multi_cell(0, 5, _latin(f"Gewinnquoten nach {play.rounds} Runden: " + "; ".join(f"{name} {V.pct(float(r), 1)}" for name, r in zip(C.TEAM_NAMES, rates)) + "."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Lehrbuchmodell: ideale projektive Messungen an einem Qubit-Paar, gleichverteilte Fragen, unabhängige Runden. Keine Aussage über Geräte oder Experimente. "
                                "Alle Zahlen sind eigene Rechnung, durch Aufzählung, zweite Rechenwege und Simulation gestützt."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
