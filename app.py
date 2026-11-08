"""Das CHSH-Spiel: Was geteilter Zufall und Verschränkung bringen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der Linie Teamrätsel (Arbeitstitel), ein optionales Randstück. Zwei Spieler ohne Verbindung beantworten je eine Frage; gewonnen wird, wenn a XOR b gleich x UND y ist. Die Demo vergleicht
die Ressourcen: keine, geteilter Zufall, ein verschränktes Qubit-Paar (Matrixrechnung, Born-Regel) und, nur als Definition, die No-Signalling-Box.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import chs_constants as C
import chs_evaluation as V
import chs_experiments as XP
import chs_game as G
import chs_play as PL
import chs_visualization as Z
from chs_layout import dunkles_design, kopf_css
from chs_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from chs_scenario import Settings

st.set_page_config(page_title="Das CHSH-Spiel: Was geteilter Zufall und Verschränkung bringen – Sebastian Hanisch", layout="wide")


# --- Zwischenspeicher (Schlüssel = alle Eingaben inklusive Zufalls-Seed) --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _analysis(s):
    return V.analyse(s)


@st.cache_data(show_spinner=False)
def _play(s):
    return PL.play(s)


@st.cache_data(show_spinner=False)
def _mixtures(seed):
    return XP.e1_mixtures(C.MIX_COUNT, seed)


@st.cache_data(show_spinner=False)
def _heat(s):
    return XP.e2_heatmaps(s)


@st.cache_data(show_spinner=False)
def _grid(s):
    return XP.e2_grid_counts(s)


@st.cache_data(show_spinner=False)
def _e3_curves():
    return XP.e3_curves()


@st.cache_data(show_spinner=False)
def _e4(s):
    return XP.e4_rows(s)


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


def signed_points(diff: float) -> str:
    """Differenz in Prozentpunkten mit ASCII-Vorzeichen, damit Streamlit die Farbe (delta_color) richtig setzt."""
    sign = "-" if diff < -C.TOLERANCE else "+"
    return f"{sign}{V.de(abs(100.0 * diff), 2)} Prozentpunkte"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in chs_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎲 Das CHSH-Spiel: Was geteilter Zufall und Verschränkung bringen")
st.markdown(
    """
**Stück 5 der Linie „Teamrätsel“ (Arbeitstitel) und ein Randstück: optional, ohne Logistik-Kulisse, ein Lehrbuchmodell aus der Physik.** Zwei Spieler sprechen vorher alles ab, dürfen danach aber nicht mehr miteinander reden.
Jeder bekommt eine Frage (0 oder 1, per Münzwurf) und antwortet mit 0 oder 1. Gewonnen wird, wenn die Antworten **gleich** sind, außer beide Fragen waren 1: dann müssen sie **verschieden** sein.
**Ohne Hilfsmittel gewinnt man höchstens in drei von vier Fällen, und geteilter Zufall ändert daran nichts. Ein verschränktes Qubit-Paar erreicht im Modell 85,36 %.**
Die Demo stellt die Ressourcen als Leiter nebeneinander, spielt sie Runde für Runde auf denselben Fragen und zeigt, welche Messrichtungen, wie viel Verschränkung und wie viel Rauschen der Vorsprung verträgt.
Gerechnet, nicht behauptet; **kein Satz über Geräte oder Experimente**, nur das Modell.

Abgrenzung: `korreliert-demo` lässt einen Vermittler eine Zuordnung ziehen und jedem Lkw nur seinen Teil mitteilen, mit Eigennutz und Gleichgewichtsbedingungen; hier haben beide **dasselbe Ziel**, und die Frage ist, wie weit eine Ressource das Ziel erreichen lässt.
Die Koordinationsspiele (`fokalpunkt-demo`, `schwaechstes-glied-demo`) handeln von Eigennutz und Lernen. Weiter unten: fünf **🔬 Experimente** in Tabs, ein Ausblick auf die No-Signalling-Box, die Abschnitte **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Spiel", expanded=True):
    st.markdown(
        """
1. **Vor dem Spiel** dürfen Alice und Bob eine Strategie und gemeinsame Ressourcen festlegen. **Im Spiel** gibt es keine Verbindung.
2. **Eine Runde:** Alice bekommt die Frage x, Bob die Frage y (beide 0 oder 1, gleichverteilt und unabhängig). Jeder gibt eine Antwort a bzw. b (0 oder 1). **Gewonnen**, wenn a ⊕ b = x ∧ y: gleiche Antworten, außer bei x = y = 1.
3. **Feste Strategie:** jeder antwortet je Frage mit einem festen Bit. Es gibt 16 solche Strategien; man kann sie alle durchzählen (Experiment E1).
4. **Warum Zufall nichts bringt:** Wer vor dem Spiel eine Zufallszahl teilt, wählt damit nur eine der 16 Strategien aus. Der Gewinn ist ein gewichteter Mittelwert, also nie größer als der der besten Strategie.
5. **Verschränkung:** Ein Qubit-Paar im Bell-Zustand liefert Antwortpaare, deren Verteilung von beiden Messrichtungen abhängt (Born-Regel). Das ist **keine Nachricht**: die Antwortverteilung jedes Spielers hängt nicht von der Frage des anderen ab (in den Tests für zufällige Zustände und Winkel geprüft); beim Bell-Zustand ist jede einzelne Antwort ein fairer Münzwurf.
        """
    )

with st.expander("Erklärtabelle: die Ressourcen der Leiter"):
    st.markdown(
        """
| Ressource | Was die Spieler haben | Höchster Gewinn | Wie die Demo es bestimmt |
|---|---|---|---|
| **Keine** | nur eine feste Absprache | 3/4 | Aufzählung aller 16 festen Strategien |
| **Geteilter Zufall** (privater Zufall ist ein Sonderfall) | eine gemeinsam gezogene Zufallsgröße | 3/4 | Mischungen der 16 Strategien; außerdem ein Modell mit gemeinsamer Zufallsrichtung und denselben Winkeln wie die Verschränkung |
| **Verschränktes Qubit-Paar** | ein Paar im Zustand cosθ\\|00⟩ + sinθ\\|11⟩, mit Rauschen | bis cos²(π/8) ≈ 0,8536 (θ = 45 Grad, rein) | 4×4-Matrixrechnung, Born-Regel, Stichprobe |
| **No-Signalling-Box** | ein Rechenmodell mit Gewinn 1, ohne Nachricht | 1 | nur Definition, Tabelle im Ausblick |
| **Absprache im Spiel** | Bob kennt Alices Frage (im Spiel verboten) | 1 | Aufzählung aller Strategien mit einem Bit |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚛️ Verschränktes Paar")
    theta = st.slider("Zustandswinkel θ (Grad)", *bounds("theta_slider"), key="theta_slider", step=0.5,
                      help="Zustand cosθ|00⟩ + sinθ|11⟩. 45 Grad ist der Bell-Zustand (maximal verschränkt), 0 Grad ein Produktzustand ohne Verschränkung.")
    vis = st.slider("Sichtbarkeit v (%)", *bounds("vis_slider"), key="vis_slider", step=1,
                    help="Anteil des reinen Zustands; der Rest ist weißes Rauschen (Zustand v·|ψ⟩⟨ψ| + (1−v)·I/4). 100 % ist rein. Das Rauschen ist eine Modellannahme, keine Messung.")
    st.header("📐 Messrichtungen")
    a0 = st.slider("Alice, Frage 0 (Grad)", *bounds("a0_slider"), key="a0_slider", step=0.5, help="Messbasis cosφ|0⟩ + sinφ|1⟩ mit dem Winkel φ; Periode 180 Grad.")
    a1 = st.slider("Alice, Frage 1 (Grad)", *bounds("a1_slider"), key="a1_slider", step=0.5, help="Winkel der Messung, wenn Alice die Frage 1 bekommt.")
    b0 = st.slider("Bob, Frage 0 (Grad)", *bounds("b0_slider"), key="b0_slider", step=0.5, help="Winkel der Messung, wenn Bob die Frage 0 bekommt.")
    b1 = st.slider("Bob, Frage 1 (Grad)", *bounds("b1_slider"), key="b1_slider", step=0.5, help="Winkel der Messung, wenn Bob die Frage 1 bekommt. Die bekannten Bestwerte für den Bell-Zustand sind 0 und 45 Grad (Alice), 22,5 und −22,5 Grad (Bob).")
    st.header("🃏 Klassische Mannschaft")
    strategy = st.selectbox("Feste Strategie", list(range(C.STRATEGY_COUNT)), key="strategy_select", format_func=G.strategy_label,
                            help="Antwort auf Frage 0 / Antwort auf Frage 1, für Alice und für Bob. Sie gilt für die Mannschaft „Keine Ressource“ im Spielverlauf und im Experiment E4.")
    st.header("🎲 Spielverlauf")
    rounds = st.slider("Runden", *bounds("rounds_slider"), key="rounds_slider", step=20, help="Wie viele Runden alle Mannschaften auf denselben Fragen spielen.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"theta_slider": float(theta), "vis_slider": int(vis), "a0_slider": float(a0), "a1_slider": float(a1), "b0_slider": float(b0), "b1_slider": float(b1),
                   "strategy_select": int(strategy), "rounds_slider": int(rounds), "seed_input": int(seed)})

settings = Settings(theta_deg=float(theta), vis_pct=float(vis), alpha0_deg=float(a0), alpha1_deg=float(a1), beta0_deg=float(b0), beta1_deg=float(b1),
                    strategy=int(strategy), rounds=int(rounds), seed=int(seed))
a = _analysis(settings)
play = _play(settings)
verdicts = []   # Verdikte für den PDF-Export

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 📊 Wie weit kommt jede Ressource?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Beste klassische Strategie", V.pct(a.classical_best, 2), delta="gilt auch mit geteiltem Zufall", delta_color="off",
          help=f"Durch Aufzählung aller 16 festen Strategien: {a.classical_best_count} von 16 gewinnen drei von vier Fragepaaren. Geteilter Zufall mischt diese Strategien.")
m2.metric("Geteilter Zufall, Ihre Winkel", V.pct(a.w_shared, 2), delta=signed_points(a.w_shared - a.classical_best), delta_color="inverse" if a.w_shared > a.classical_best + C.TOLERANCE else "off",
          help="Dieselben vier Winkel wie die Verschränkung, aber statt des Qubit-Paars eine gemeinsame Zufallsrichtung. Delta: gegen den klassischen Höchstwert.")
m3.metric("Verschränkung, Ihre Winkel", V.pct(a.w_quantum, 2), delta=signed_points(a.w_quantum - a.classical_best), delta_color="normal" if abs(a.w_quantum - a.classical_best) > C.TOLERANCE else "off",
          help="Gewinnwahrscheinlichkeit des verschränkten Paars mit Ihrem Zustand und Ihren vier Messwinkeln (Born-Regel). Delta: gegen den klassischen Höchstwert.")
m4.metric("Bestwert dieses Zustands", V.pct(a.w_best, 2), delta=signed_points(a.w_best - a.classical_best), delta_color="normal" if abs(a.w_best - a.classical_best) > C.TOLERANCE else "off",
          help="Größter Wert über alle Messungen für den eingestellten Zustand (Korrelationsmatrix, Horodecki). Delta: gegen den klassischen Höchstwert.")

st.markdown("#### ❓ Kernfrage: Bringt die Ressource einen Vorsprung?")
v_res = V.verdict_resource(a)
show(v_res)
verdicts.append(v_res)
v_state = V.verdict_state(a)
show(v_state)
verdicts.append(v_state)

col_ladder, col_angles = st.columns([3, 2])
with col_ladder:
    st.plotly_chart(Z.build_ladder(a), width="stretch", key="ladder_chart")
    st.caption("Die Leiter der Ressourcen (gestrichelt: klassischer Höchstwert). Die Absprache im Spiel ist verboten und nur zum Vergleich da; die Box ist eine Definition.")
with col_angles:
    st.plotly_chart(Z.build_angles(settings), width="stretch", key="angles_chart")
    st.caption("Die vier Messrichtungen (Winkel in Grad wie in der Seitenleiste; eine Richtung und ihre Gegenrichtung sind dieselbe Messung).")

st.markdown("**Gewinnwahrscheinlichkeit je Fragepaar**")
fa, fb = G.strategy_answers(settings.strategy)
table([{"Frage x (Alice)": x, "Frage y (Bob)": y, "Ziel": "verschieden" if G.target(x, y) else "gleich",
        "Verschränkung": V.pct(float(a.q_questions[x, y]), 2), "Geteilte Zufallsrichtung": V.pct(float(a.s_questions[x, y]), 2),
        "Feste Strategie der Seitenleiste": "gewinnt" if G.is_won(x, y, fa[x], fb[y]) else "verliert"} for x, y in G.QUESTIONS])
st.caption(f"Das Mittel über die vier Fragepaare ist die Gewinnwahrscheinlichkeit. Feste Strategie der Seitenleiste: {G.strategy_label(settings.strategy)} ({G.strategy_wins(settings.strategy)} von 4 Paaren).")

# --- Spielverlauf --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 🎮 Runde für Runde")
st.caption(f"Alle fünf Mannschaften bekommen dieselben {V.thousands(play.rounds)} Fragen (Zufalls-Seed {settings.seed}). Die verschränkte Mannschaft zieht ihre Antworten aus der Born-Regel, der geteilte Zufall aus einer gemeinsamen Richtung, die Box und die Absprache aus ihrer Definition.")
rate_rows = play.final_rates()
r1, r2, r3, r4, r5 = st.columns(5)
for col, name, rate, i in zip((r1, r2, r3, r4, r5), C.TEAM_NAMES, rate_rows, range(5)):
    col.metric(name, V.pct(float(rate), 1), help="Gewinnquote über alle gespielten Runden." + (f" Fest: {G.strategy_label(settings.strategy)}." if i == 0 else ""))
v_game = V.verdict_game(a, play)
show(v_game)
verdicts.append(v_game)
st.plotly_chart(Z.build_cumulative(play, a), width="stretch", key="cumulative_chart")
st.caption("Laufende Gewinnquote. Gestrichelt grau: klassischer Höchstwert; gestrichelt grün: Erwartung der Verschränkung mit Ihren Einstellungen. Am Anfang streuen die Kurven stark.")
shown = min(C.SHOWN_ROUNDS, play.rounds)
def mark(won) -> str:
    return "✓" if won else "✗"


table([{"Runde": t + 1, "x": int(play.x[t]), "y": int(play.y[t]), "Ziel": "verschieden" if int(play.x[t]) & int(play.y[t]) else "gleich",
        **{name: f"a={int(play.a[k, t])}, b={int(play.b[k, t])} {mark(play.won[k, t])}" for k, name in enumerate(C.TEAM_NAMES)}} for t in range(shown)])
st.caption(f"Die ersten {shown} Runden im Protokoll (a, b: die Antworten; ✓ gewonnen, ✗ verloren).")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["1 Alle 16 Strategien", "2 Winkel-Landschaft", "3 Zustand und Rauschen", "4 Wie viele Runden?"])

with tab1:
    st.subheader("🔬 Experiment E1: Alle festen Strategien und alle Mischungen")
    st.caption("Jede feste Strategie gewinnt genau die Fragepaare, in denen ihre Antworten das Ziel treffen. Geteilter Zufall mischt diese 16 Strategien mit beliebigen Gewichten; die Mischung ist ein gewichteter Mittelwert.")
    rows1 = XP.e1_strategy_rows()
    st.dataframe([{"Nr.": r["index"], "Strategie (Antwort auf Frage 0 / 1)": r["label"], "gewonnene Fragepaare": r["wins"], "Gewinn (%)": 100 * r["value"]} for r in rows1],
                 hide_index=True, width="stretch", column_config={"Gewinn (%)": st.column_config.NumberColumn(format="%.1f %%")})
    mix = _mixtures(settings.seed)
    st.plotly_chart(Z.build_mix_hist(mix["values"], a.classical_best, a.w_quantum), width="stretch", key="e1_hist")
    st.caption(f"{V.thousands(mix['count'])} zufällige Mischungen (gleichverteilt auf dem Simplex, Zufalls-Seed {settings.seed}). Grau gestrichelt: klassischer Höchstwert; grün gestrichelt: die Verschränkung mit Ihren Einstellungen.")
    v1 = V.verdict_e1(rows1, mix)
    show(v1)
    verdicts.append(v1)

with tab2:
    st.subheader("🔬 Experiment E2: Welche Messrichtungen lohnen sich?")
    st.caption("Links die Verschränkung mit Ihrem Zustand, rechts dieselben Winkel mit geteilter Zufallsrichtung. Variiert werden Alices und Bobs Winkel bei Frage 1; die Winkel bei Frage 0 stehen wie in der Seitenleiste. "
               "Schwarze Linie: 75 %. Rechts liegt nirgends etwas darüber.")
    heat = _heat(settings)
    hl, hr = st.columns(2)
    with hl:
        st.markdown("**Verschränkung**")
        st.plotly_chart(Z.build_heat_quantum(heat), width="stretch", key="e2_heat_quantum")
    with hr:
        st.markdown("**Geteilte Zufallsrichtung**")
        st.plotly_chart(Z.build_heat_shared(heat), width="stretch", key="e2_heat_shared")
    grid = _grid(settings)
    table([{"Modell": "Verschränkung", "Zellen über 75 %": V.thousands(grid["quantum"]["above"]), "Anteil": V.pct(grid["quantum"]["above"] / grid["cells"], 1),
            "Maximum": V.pct(grid["quantum"]["max"], 2), "Zellen im Maximum": V.thousands(grid["quantum"]["at_max"])},
           {"Modell": "Geteilte Zufallsrichtung", "Zellen über 75 %": V.thousands(grid["shared"]["above"]), "Anteil": V.pct(grid["shared"]["above"] / grid["cells"], 1),
            "Maximum": V.pct(grid["shared"]["max"], 2), "Zellen im Maximum": V.thousands(grid["shared"]["at_max"])}])
    st.caption(f"Zählung über alle vier Winkel: Gitter von −90 bis 82,5 Grad in 7,5-Grad-Schritten, {grid['points']} Werte je Winkel, {V.thousands(grid['cells'])} Zellen.")
    v2 = V.verdict_e2(grid, settings)
    show(v2)
    verdicts.append(v2)

with tab3:
    st.subheader("🔬 Experiment E3: Wie viel Verschränkung, wie viel Rauschen?")
    st.caption("Für jeden Zustand gibt es einen besten Satz von Messungen; sein Wert folgt aus der Korrelationsmatrix des Zustands (Horodecki) und wird im Test gegen eine Optimierung über alle vier Winkel geprüft.")
    curves = _e3_curves()
    c3a, c3b = st.columns(2)
    with c3a:
        st.markdown("**Bestwert gegen Zustandswinkel (reine Sichtbarkeit)**")
        st.plotly_chart(Z.build_e3_best(curves), width="stretch", key="e3_best_chart")
    with c3b:
        st.markdown("**Schwelle der Sichtbarkeit gegen Zustandswinkel**")
        st.plotly_chart(Z.build_e3_threshold(curves), width="stretch", key="e3_thr_chart")
    rows3 = XP.e3_rows()
    table([{"Zustandswinkel (Grad)": V.de(r["theta_deg"], 0), "Bestwert (reine Sichtbarkeit)": V.pct(r["best"], 2),
            "Schwelle der Sichtbarkeit": "nie" if r["threshold"] >= 1.0 - C.TOLERANCE else V.pct(r["threshold"], 1)} for r in rows3])
    st.markdown(f"**Rauschen beim eingestellten Zustandswinkel ({V.de(settings.theta_deg, 1)} Grad)**")
    table([{"Sichtbarkeit": V.pct(r["vis"], 0), "Bestwert": V.pct(r["best"], 2), "über 75 %": "ja" if r["best"] > C.CLASSICAL_BOUND + C.TOLERANCE else "nein"} for r in XP.e3_visibility_rows(settings.theta_deg)])
    v3 = V.verdict_e3(rows3)
    show(v3)
    verdicts.append(v3)

with tab4:
    st.subheader("🔬 Experiment E4: Wie viele Runden braucht man, um den Vorsprung zu sehen?")
    st.caption(f"Die verschränkte Mannschaft (Ihr Zustand, Ihre Winkel) gegen die feste Strategie aus der Seitenleiste auf denselben Fragen. {V.thousands(C.E4_REPETITIONS)} Wiederholungen je Rundenzahl; "
               "gezählt wird, wer am Ende mehr Runden gewonnen hat. Der Vorsprung im Mittel hängt von Zustand, Winkeln und fester Strategie ab (Bell-Zustand mit besten Winkeln gegen die beste feste Strategie: 85,36 % gegen 75,0 % der Runden); bei wenigen Runden streuen einzelne Spiele stark.")
    if st.button("▶️ Wiederholungen rechnen", key="e4_button", help=f"{len(C.E4_ROUNDS)} Rundenzahlen mit je {V.thousands(C.E4_REPETITIONS)} Wiederholungen, bis {C.E4_ROUNDS[-1]} Runden; rechnet rund eine Sekunde."):
        st.session_state["e4_signature"] = settings
    if st.session_state.get("e4_signature") == settings:
        with st.spinner("Rechne die Wiederholungen (rund eine Sekunde) …"):
            rows4 = _e4(settings)
        st.plotly_chart(Z.build_e4(rows4), width="stretch", key="e4_chart")
        table([{"Runden": r["rounds"], "verschränkt strikt vorn": V.pct(r["more"], 1), "gleich viele": V.pct(r["equal"], 1), "feste Strategie vorn": V.pct(r["less"], 1),
                "Gewinnquote verschränkt": V.pct(r["q_rate"], 1), "Gewinnquote fest": V.pct(r["c_rate"], 1)} for r in rows4])
        v4 = V.verdict_e4(rows4)
        show(v4)
        verdicts.append(v4)

with st.expander("Ausblick: die No-Signalling-Box (nur eine Definition)"):
    box = G.no_signalling_box()
    st.markdown("Ein **Rechenmodell**, kein Befund: eine Box, die auf die Fragen (x, y) ein Antwortpaar (a, b) mit **gleicher Wahrscheinlichkeit** aus allen Paaren zieht, die das Ziel treffen. Jede einzelne Antwort bleibt ein fairer Münzwurf, es kommt keine Nachricht durch, und die Box gewinnt immer.")
    table([{"x": x, "y": y, "Ziel": "verschieden" if G.target(x, y) else "gleich", **{f"P(a={ab[0]}, b={ab[1]})": V.de(float(box[x, y, ab[0], ab[1]]), 2) for ab in G.ANSWERS}} for x, y in G.QUESTIONS])
    st.caption(f"Gewinn der Box: {V.pct(a.box, 0)}. Größte Abweichung der Antwortverteilung eines Spielers, wenn nur die Frage des anderen wechselt: {V.de(G.marginal_dependence(box), 1)}. "
               "Die Demo behauptet nichts darüber, ob es in der Natur so etwas gibt; die Quelle der Box steht in der README.")

# --- Verwandte Demos -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(
    """
- **`korreliert-demo`**: ein Vermittler zieht eine Zuordnung und teilt jedem Lkw nur dessen Teil mit (gemeinsames Signal); dort mit Eigennutz und Gleichgewichtsbedingung, hier mit gemeinsamem Ziel und einer Obergrenze für alle Mischungen.
- **`fokalpunkt-demo`**, **`schwaechstes-glied-demo`**: Koordinationsspiele mit Eigennutz und Lernen; hier gewinnt das Team gemeinsam oder nicht.
- **Informationsleiter der Linie** (Arbeitstitel `informationsleiter-demo`): dort steigt der erreichbare Gewinn mit mehr Wissen über die Angebote, hier mit mehr Ressourcen zwischen den Spielern bei gleichbleibenden Fragen.
"""
)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Lehrbuchmodell, keine Aussage über Geräte** | Ideale projektive Messungen an einem Qubit-Paar, kein Detektorverlust, kein Zeitfenster. Wie weit echte Aufbauten das erreichen, ist hier weder gerechnet noch behauptet. | – |
| **Fragen gleichverteilt und unabhängig** | Bei anderer Verteilung der Fragen ändern sich der klassische Höchstwert und der Quantenwert; gerechnet ist nur der Fall ¼ je Fragepaar. | – |
| **Eine bestimmte Zustandsfamilie** | Gerechnet sind cosθ\\|00⟩ + sinθ\\|11⟩ mit weißem Rauschen, Messungen in einer Ebene. Der Bestwert dieser Familie folgt aus der Korrelationsmatrix; andere Zustände sind nicht gerechnet. | – |
| **Der Rauschanteil ist eine Annahme** | Echte Störungen sind selten weißes Rauschen; die Schwelle 70,7 % gilt für diese Rauschform. | – |
| **Runden unabhängig, Zufall aus SplitMix64** | Zufallszahlen sind festgelegt durch den Zufalls-Seed; ein anderer Seed gibt andere Stichproben, nie andere Erwartungswerte. | – |
| **Die Box ist eine Definition** | Ihr Gewinn 100 % folgt aus der Tabelle, nicht aus einer Messung; ob sie in der Natur vorkommt, wird nicht behauptet. | – |
| **Kein Logistik-Fall** | Das Stück hat keine Kulisse aus Transport oder Lager; es gehört als Randstück zur Linie und erhebt keinen Praxisanspruch. | – |
"""
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind zwei Spieler ohne Verbindung, Fragen $x, y \in \{0, 1\}$ (gleichverteilt, unabhängig) und Antworten $a, b \in \{0, 1\}$; gesucht ist die größte Gewinnwahrscheinlichkeit je Ressource.
Alle Herleitungen sind eigene Rechnung, durch Aufzählung, zweite Rechenwege und Simulation gestützt; die Lehrbuchresultate (CHSH 1969, Cirel'son 1980, Horodecki 1995) sind in der README genannt.

| Symbol | Bedeutung |
|---|---|
| $x,\ y,\ a,\ b$ | Fragen und Antworten von Alice und Bob |
| $W$ | Gewinnwahrscheinlichkeit des Spiels |
| $\theta,\ v$ | Zustandswinkel und Sichtbarkeit des Qubit-Paars |
| $\alpha_x,\ \beta_y$ | Messwinkel von Alice bei Frage $x$ und von Bob bei Frage $y$ |
| $E_{xy}$ | Korrelator der Antworten, $+1$ bei gleichen, $-1$ bei verschiedenen Antworten |
| $T,\ \lambda_1,\ \lambda_2$ | Korrelationsmatrix des Zustands und die zwei größten Eigenwerte von $T^{\top}T$ |

**Gewinnbedingung:**

$$a \oplus b = x \wedge y$$

**Gewinnwahrscheinlichkeit** bei gleichverteilten Fragen:

$$W = \frac14 \sum_{x,y} \Pr\left[a \oplus b = x \wedge y \,\middle|\, x, y\right]$$

**Klassisch.** Eine feste Strategie $(f_A, f_B)$ gewinnt das Paar $(x, y)$ genau dann, wenn $f_A(x) \oplus f_B(y) = x \wedge y$. Die Aufzählung aller 16 Strategien ergibt höchstens drei gewonnene Paare, und jeder (auch geteilte) Zufall ist eine Mischung mit Gewichten $p_s$:

$$W = \sum_s p_s\, W_s \le \max_s W_s = \frac34$$

**Zustand und Messung.** Zustand mit Rauschen und die Born-Regel für Projektoren $\Pi_a^{\varphi}$ auf $\cos\varphi\,|0\rangle + \sin\varphi\,|1\rangle$ (Ergebnis 0) bzw. die dazu senkrechte Richtung (Ergebnis 1):

$$\rho = v\,|\psi_\theta\rangle\langle\psi_\theta| + (1 - v)\,\frac{I}{4}, \qquad |\psi_\theta\rangle = \cos\theta\,|00\rangle + \sin\theta\,|11\rangle$$

$$\Pr[a, b \mid x, y] = \operatorname{Tr}\!\left[\rho\,\left(\Pi_a^{\alpha_x} \otimes \Pi_b^{\beta_y}\right)\right]$$

**Korrelator und Gewinn** für diese Zustandsfamilie und Messungen in einer Ebene:

$$E(\alpha, \beta) = v\left(\cos 2\alpha \cos 2\beta + \sin 2\theta \,\sin 2\alpha \sin 2\beta\right)$$

$$W = \frac12 + \frac18\left(E_{00} + E_{01} + E_{10} - E_{11}\right)$$

**Bell-Zustand** ($\theta = \pi/4$, $v = 1$): $E(\alpha, \beta) = \cos(2\alpha - 2\beta)$. Mit $\alpha_0 = 0$, $\alpha_1 = \pi/4$, $\beta_0 = \pi/8$, $\beta_1 = -\pi/8$ gewinnt jedes der vier Fragepaare mit derselben Wahrscheinlichkeit:

$$W = \cos^2\frac{\pi}{8} = \frac12 + \frac{\sqrt 2}{4} \approx 0{,}8536$$

**Bestwert über alle Messungen** (Horodecki 1995; zweiter Rechenweg zur Optimierung über die Winkel), mit $T_{ij} = \operatorname{Tr}[\rho\,\sigma_i \otimes \sigma_j]$:

$$S_{\max} = 2\sqrt{\lambda_1 + \lambda_2}, \qquad W_{\max} = \frac12 + \frac{S_{\max}}{8}$$

Für die Zustandsfamilie ergibt das $W_{\max} = \frac12 + \frac v4\sqrt{1 + \sin^2 2\theta}$; er überschreitet $\frac34$ genau dann, wenn die Sichtbarkeit über der Schwelle liegt:

$$v^{*} = \frac{1}{\sqrt{1 + \sin^2 2\theta}}$$

**Geteilte Zufallsrichtung.** Gemeinsame Richtung $\lambda$, gleichverteilt auf $[0, \pi)$; wer den Winkel $\varphi$ hat, antwortet 0, wenn $\lambda$ weniger als $\pi/4$ von $\varphi$ entfernt ist. Für den Abstand $\Delta$ (auf $[-\pi/2, \pi/2)$ reduziert):

$$\Pr[a = b] = 1 - \frac{2\,|\Delta|}{\pi}$$

**No-Signalling-Box** (Definition):

$$\Pr[a, b \mid x, y] = \frac12\,\mathbf 1\left[a \oplus b = x \wedge y\right]$$

**Literatur.** Die Quellen stehen in der README mit Stand der Prüfung; Formeln und Zahlen dieser Demo sind eigene Rechnung.
        """
    )

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from chs_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(a, play, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="chsh-spiel.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
