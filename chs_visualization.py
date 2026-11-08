"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import chs_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def _de(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def build_ladder(a):
    """Die Sprossen der Ressourcenleiter als waagerechte Balken von 0 bis 100 % (Achse nicht abgeschnitten) mit der Linie des klassischen Höchstwerts."""
    rows = [
        ("Keine Ressource (beste feste Strategie)", a.classical_best, C.TEAM_COLORS[0]),
        ("Geteilter Zufall (beste Mischung)", a.classical_best, C.TEAM_COLORS[1]),
        ("Geteilter Zufall, Ihre Winkel", a.w_shared, C.TEAM_COLORS[1]),
        ("Verschränkung, Ihre Winkel", a.w_quantum, C.TEAM_COLORS[2]),
        ("Verschränkung, Bestwert dieses Zustands", a.w_best, C.TEAM_COLORS[2]),
        ("No-Signalling-Box (Definition)", a.box, C.TEAM_COLORS[3]),
        ("Absprache im Spiel (verboten)", a.one_bit, C.TEAM_COLORS[4]),
    ]
    fig = go.Figure(go.Bar(
        y=[r[0] for r in rows], x=[100 * r[1] for r in rows], orientation="h", marker_color=[r[2] for r in rows], text=[f"{_de(100 * r[1], 2)} %" for r in rows], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
    fig.add_vline(x=100 * a.classical_best, line_dash="dash", line_color="rgba(120,120,120,0.9)")
    fig.update_xaxes(title_text="Gewinnwahrscheinlichkeit (%)", range=[0, 112])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=10, r=40, t=10, b=10))
    return _base(fig, height=330, legend=False).update_layout(margin=dict(l=10, r=40, t=10, b=10))


def build_angles(settings):
    """Die vier Messrichtungen als Strahlen im Einheitskreis (Winkel wie in den Reglern); gleicher Maßstab über zwei unsichtbare Eckpunkte."""
    fig = go.Figure()
    t = np.linspace(0, 2 * math.pi, 181)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="rgba(150,150,150,0.5)", width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[-1.25, 1.25], y=[-1.15, 1.15], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    items = [("Alice, Frage 0", settings.alpha0_deg, C.COLOR_ALICE, "solid"), ("Alice, Frage 1", settings.alpha1_deg, C.COLOR_ALICE, "dot"),
             ("Bob, Frage 0", settings.beta0_deg, C.COLOR_BOB, "solid"), ("Bob, Frage 1", settings.beta1_deg, C.COLOR_BOB, "dot")]
    for name, deg, color, dash in items:
        r = math.radians(deg)
        fig.add_trace(go.Scatter(x=[-math.cos(r), math.cos(r)], y=[-math.sin(r), math.sin(r)], mode="lines", line=dict(color=color, width=3, dash=dash),
                                 name=f"{name} ({_de(deg, 1)} Grad)", hovertemplate=f"{name}: {_de(deg, 1)} Grad<extra></extra>"))
    fig.update_xaxes(visible=False, scaleanchor="y")
    fig.update_yaxes(visible=False)
    return _base(fig, height=330)


def build_cumulative(play, a):
    """Laufende Gewinnquote der fünf Mannschaften auf denselben Fragen, dazu die Erwartung der Verschränkung und der klassische Höchstwert."""
    rates = play.cumulative_rates()
    t = np.arange(1, play.rounds + 1)
    fig = go.Figure()
    for k, name in enumerate(C.TEAM_NAMES):
        fig.add_trace(go.Scatter(x=t, y=100 * rates[k], mode="lines", name=name, line=dict(color=C.TEAM_COLORS[k], width=2 if k not in (3, 4) else 1.5, dash="solid" if k < 3 else "dot"),
                                 hovertemplate=name + ": %{y:.1f} % nach %{x} Runden<extra></extra>"))
    fig.add_hline(y=100 * a.classical_best, line_dash="dash", line_color="rgba(120,120,120,0.9)")
    fig.add_hline(y=100 * a.w_quantum, line_dash="dash", line_color=C.TEAM_COLORS[2])
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="Gewinnquote bis zur Runde (%)", range=[0, 102])
    return _base(fig, height=380)


def _heat(grid, axis_deg, title):
    colorscale = [[0.0, "#3b4a6b"], [0.5, "#aab4c8"], [0.75, "#f1efe6"], [1.0, "#2F6B65"]]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=100 * grid, x=axis_deg, y=axis_deg, zmin=0, zmax=100, colorscale=colorscale, colorbar=dict(title=dict(text="Gewinn (%)", side="top"), tickvals=[0, 25, 50, 75, 100], len=0.9),
                             hovertemplate="Bob, Frage 1: %{x:.0f} Grad<br>Alice, Frage 1: %{y:.0f} Grad<br>Gewinn: %{z:.2f} %<extra></extra>"))
    fig.add_trace(go.Contour(z=100 * grid, x=axis_deg, y=axis_deg, showscale=False, contours=dict(start=75, end=75, size=1, coloring="none"), line=dict(color="#000000", width=2),
                             hoverinfo="skip", name="75 %"))
    fig.update_xaxes(title_text="Bobs Winkel bei Frage 1 (Grad)", range=[-90, 90], tickvals=[-90, -45, 0, 45, 90])
    fig.update_yaxes(title_text="Alices Winkel bei Frage 1 (Grad)", range=[-90, 90], tickvals=[-90, -45, 0, 45, 90])
    return _base(fig, height=360, legend=False)


def build_heat_quantum(heat):
    return _heat(heat["quantum"], heat["axis_deg"], "Verschränkung")


def build_heat_shared(heat):
    return _heat(heat["shared"], heat["axis_deg"], "Geteilte Zufallsrichtung")


def build_mix_hist(values, bound, quantum):
    fig = go.Figure(go.Histogram(x=100 * np.asarray(values), xbins=dict(start=45, end=80, size=0.5), marker_color=C.TEAM_COLORS[1], hovertemplate="%{x:.1f} %: %{y} Mischungen<extra></extra>", name="Mischungen"))
    fig.add_vline(x=100 * bound, line_dash="dash", line_color="rgba(120,120,120,0.9)")
    fig.add_vline(x=100 * quantum, line_dash="dash", line_color=C.TEAM_COLORS[2])
    fig.update_xaxes(title_text="Gewinnwahrscheinlichkeit der Mischung (%)", range=[45, 90])
    fig.update_yaxes(title_text="Zahl der Mischungen")
    return _base(fig, height=300, legend=False)


def build_e3_best(curves):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves["theta_deg"], y=100 * curves["best"], mode="lines", line=dict(color=C.TEAM_COLORS[2], width=3), name="Bestwert (reine Sichtbarkeit)",
                             hovertemplate="Zustandswinkel %{x:.1f} Grad: %{y:.2f} %<extra></extra>"))
    fig.add_hline(y=75, line_dash="dash", line_color="rgba(120,120,120,0.9)")
    fig.update_xaxes(title_text="Zustandswinkel θ (Grad); 45 = Bell-Zustand", range=[0, 45])
    fig.update_yaxes(title_text="Bestwert über alle Messungen (%)", range=[70, 90])
    return _base(fig, height=320, legend=False)


def build_e3_threshold(curves):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves["theta_deg"], y=100 * curves["threshold"], mode="lines", line=dict(color=C.TEAM_COLORS[2], width=3), fill="tozeroy", fillcolor="rgba(150,150,150,0.2)",
                             name="Schwelle", hovertemplate="Zustandswinkel %{x:.1f} Grad: Schwelle %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Zustandswinkel θ (Grad)", range=[0, 45])
    fig.update_yaxes(title_text="Sichtbarkeit, ab der mehr als 75 % möglich sind (%)", range=[60, 102])
    return _base(fig, height=320, legend=False)


def build_e4(rows):
    xs = [r["rounds"] for r in rows]
    fig = go.Figure()
    for key, name, color in (("more", "verschränkt strikt vorn", C.TEAM_COLORS[2]), ("equal", "gleich viele", "#999999"), ("less", "feste Strategie vorn", C.TEAM_COLORS[0])):
        fig.add_trace(go.Scatter(x=xs, y=[100 * r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=3),
                                 hovertemplate=name + ": %{y:.1f} % bei %{x} Runden<extra></extra>"))
    fig.update_xaxes(title_text="Runden je Spiel", type="log", tickvals=xs, ticktext=[str(x) for x in xs])
    fig.update_yaxes(title_text="Anteil der Wiederholungen (%)", range=[0, 102])
    return _base(fig, height=340)
