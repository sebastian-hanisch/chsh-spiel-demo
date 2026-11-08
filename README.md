# Das CHSH-Spiel: Was geteilter Zufall und Verschränkung bringen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-chsh-spiel-demo.streamlit.app/)** -->

Stück 5 der Linie **„Teamrätsel“** (Arbeitstitel) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, und ehrlich gesagt ein **Randstück (optional)**: Es hat keine Logistik-Kulisse, es ist ein Lehrbuchmodell aus der Physik und liegt weit vom Rest der Linie. Gebraucht wird es als **Gegenstück zur Informationsleiter** der Linie: dort steigt der Gewinn mit mehr Wissen über die Aufgabe, hier mit mehr Ressourcen zwischen den Spielern bei gleichbleibenden Fragen.

Das Spiel: zwei Spieler sprechen vorher alles ab und dürfen dann nicht mehr miteinander reden. Jeder bekommt eine Frage x bzw. y (0 oder 1, gleichverteilt und unabhängig), antwortet mit a bzw. b (0 oder 1), und die Runde ist **gewonnen, wenn a ⊕ b = x ∧ y**: gleiche Antworten, außer bei x = y = 1. Ohne Ressource und mit geteiltem Zufall liegt das Maximum bei **3/4**; ein **verschränktes Qubit-Paar** (Bell-Zustand, Messwinkel 0 und 45 Grad gegen 22,5 und −22,5 Grad) erreicht in diesem Modell **cos²(π/8) = 0,8536**. Die Demo vergleicht die Sprossen der Leiter auf denselben Fragen, zeigt, **welche Messrichtungen, wie viel Verschränkung und wie viel Rauschen** der Vorsprung braucht, und fragt, **wie viele Runden** man spielen muss, um ihn zu sehen.

Abgrenzung: `korreliert-demo` lässt einen Vermittler eine Zuordnung ziehen und jedem Lkw nur dessen Teil mitteilen (ein gemeinsames Signal, Eigennutz, Gleichgewichtsbedingungen als LP); hier haben beide Spieler **dasselbe Ziel**, und gefragt ist, wie weit eine Ressource kommt, wenn jede Mischung fester Strategien höchstens 3/4 erreicht. `fokalpunkt-demo` und `schwaechstes-glied-demo` (Koordinationsspiele) handeln von Eigennutz und Lernen. **Lehrbuchmodell, keine Aussage über Geräte oder Experimente.**

## Kernfrage

*Was bringt dem Team eine Ressource, wenn es nach der Absprache nicht mehr reden darf: ein fester Plan, geteilter Zufall, ein verschränktes Paar?* Eigenes dieser Demo sind (1) der **Vergleich aller Sprossen auf denselben Fragen**, Runde für Runde, (2) das **Modell „geteilte Zufallsrichtung“** mit denselben Winkeln wie die Verschränkung (gibt genau 3/4 statt 0,8536), (3) die **Zählung der Winkelquadrupel** auf einem Gitter, (4) die **Schwelle des Rauschens** je Zustand und (5) die **Rundenzahl**, ab der der Vorsprung sichtbar wird. Die Einzelresultate (CHSH-Ungleichung, Cirel'son-Grenze, Horodecki-Formel) sind Lehrbuch.

## Modell

- **Spiel:** Fragen x, y ∈ {0, 1} gleichverteilt (je Fragepaar ¼), Antworten a, b ∈ {0, 1}, gewonnen bei a ⊕ b = x ∧ y. Die Gewinnwahrscheinlichkeit W ist das Mittel über die vier Fragepaare.
- **Feste Strategie:** jeder antwortet je Frage mit einem festen Bit, 16 Strategien (`chs_game.py`). **Geteilter Zufall:** eine vor dem Spiel gezogene Zufallsgröße wählt eine dieser Strategien aus; der Gewinn ist ein gewichteter Mittelwert, nie größer als der der besten. Privater Zufall ist ein Sonderfall.
- **Geteilte Zufallsrichtung** (nur ein Beispiel für geteilten Zufall, `chs_game.py`): eine gemeinsame Richtung λ, gleichverteilt auf [0, π); wer den Winkel φ hat, antwortet 0, wenn λ weniger als π/4 von φ entfernt ist. Die Wahrscheinlichkeit gleicher Antworten ist 1 − 2·|Δ|/π für den Winkelabstand Δ.
- **Verschränktes Paar** (`chs_quantum.py`): Zustand ρ = v·|ψθ⟩⟨ψθ| + (1 − v)·I/4 mit |ψθ⟩ = cosθ|00⟩ + sinθ|11⟩ (θ = 45 Grad: Bell-Zustand; v: Sichtbarkeit, der Rest ist weißes Rauschen). Gemessen wird in der Basis cosφ|0⟩ + sinφ|1⟩ (Ergebnis 0) bzw. der dazu senkrechten (Ergebnis 1); die Antworten kommen aus der Born-Regel P(a, b | x, y) = Spur(ρ · Π ⊗ Π). Der Bestwert über alle Messungen folgt aus der Korrelationsmatrix T des Zustands: S = 2·√(λ₁ + λ₂) (die zwei größten Eigenwerte von TᵀT), W = ½ + S/8.
- **No-Signalling-Box** (nur Definition): P(a, b | x, y) = ½, wenn a ⊕ b = x ∧ y, sonst 0. **Absprache im Spiel** (verboten, zum Vergleich): Bob kennt zusätzlich Alices Frage.
- **Zufall:** SplitMix64 mit festem Zufalls-Seed, vektorisiert in numpy (`chs_rng.py`).

Alle Herleitungen sind eigene Rechnung, durch Aufzählung, zweite Rechenwege und Simulation gestützt; die Lehrbuchresultate sind in den Quellen genannt.

## Befunde und Korrekturen gegenüber dem Plan

Alle Zahlen sind in dieser Demo neu gemessen oder exakt gerechnet (`tests/test_claims.py`). Simulationen haben feste Zufalls-Seeds (SplitMix64) und werden nur mit Marge von vier Standardfehlern verglichen. Die Rechnungen sind klein (4×4-Matrizen, Gitter bis 24⁴ Zellen), es gibt keine vorgerechneten Messreihen.

| Nr. | Frage | Antwort | Beleg |
|---|---|---|---|
| H1 | Wie viel schafft eine feste Absprache? | **Höchstens 3/4.** Von den 16 festen Strategien gewinnen 8 drei von vier Fragepaaren (75,0 %), 8 nur eines (25,0 %); Mittel über alle 16: 50,0 %. Keine gewinnt alle vier. | Aufzählung in der App; unabhängig in reinem Python mit ausgeschriebenen Bedingungen und über die CHSH-Algebra \|S\| ≤ 2 (`test_h1_*`) |
| H2 | Hilft geteilter Zufall? | **Nein.** 20 000 zufällige Mischungen der 16 Strategien (Zufalls-Seed 35): größter Wert 70,22 %, Mittel 50,10 %; keine über 75 %, wie es für gewichtete Mittelwerte sein muss. Das Modell „geteilte Zufallsrichtung“ mit den Winkeln 0 und 45 Grad (Alice), 22,5 und −22,5 Grad (Bob) gewinnt exakt 75,0 %; in 1 000 000 Runden 75,02 % (Standardfehler 0,043 %). Auf dem Winkelgitter (331 776 Quadrupel) liegt keines über 75 %, das Maximum ist 75,0 % | `test_h2_*` |
| H3 | Was schafft die Verschränkung? | **85,36 %** = cos²(π/8) = ½ + √2/4 = 0,853553 (Bell-Zustand rein, Winkel wie oben): 10,36 Prozentpunkte über dem klassischen Höchstwert. Born-Stichprobe aus 1 000 000 Runden (Zufalls-Seed 35): 85,43 % (Standardfehler 0,035 %). Jedes der vier Fragepaare gewinnt mit derselben Wahrscheinlichkeit 85,36 % | Matrixrechnung gegen Zustandsvektor, Eigenwert des Bell-Operators und Formel (`test_h3_*`, `test_oracle_*`) |
| H4 | Auf welche Winkel kommt es an? | Auf dem Gitter (24 Richtungen je Winkel, 7,5-Grad-Schritte, 331 776 Quadrupel) liegen **24 816 Zellen (7,5 %)** über 75 %, das Maximum 85,36 % steht in 48 Zellen. Alle vier Winkel 0 Grad: genau 75,0 %, wie die Strategie „beide immer gleich“ | `test_h4_*` |
| H5 | Wie viel Verschränkung braucht man? | Für cosθ\|00⟩ + sinθ\|11⟩ bei reiner Sichtbarkeit ist der Bestwert ½ + ¼·√(1 + sin²2θ). Zustandswinkel 0 / 5 / 10 / 15 / 20 / 30 / 45 Grad: 75,00 / 75,37 / 76,42 / 77,95 / 79,72 / 83,07 / 85,36 %. In 6 von 6 verschränkten Zuständen über 75 %, der Produktzustand erreicht mit den besten Winkeln genau 75,0 % und mit den Bell-Winkeln nur 67,68 % | Bestwert aus der Korrelationsmatrix gegen Optimierung über alle vier Winkel (Nelder-Mead, 8 Startpunkte) und die Formel, Tabelle `test_h5_*` |
| H6 | Wie viel Rauschen verträgt der Vorsprung? | Bell-Zustand mit Sichtbarkeit v: W = ½ + v·√2/4. Über 75 % nur für **v über 1/√2 = 70,7 %**: Sichtbarkeit 90 / 80 / 75 / 71 / 70 / 50 %: 81,82 / 78,28 / 76,52 / 75,10 / 74,75 / 67,68 %. Die Schwelle sinkt mit der Verschränkung: Zustandswinkel 5 / 10 / 15 / 20 / 30 / 45 Grad: 98,5 / 94,6 / 89,4 / 84,1 / 75,6 / 70,7 % (nur für diese Rauschform) | `test_h6_*` |
| H7 | Wie weit gehen die Sprossen darüber hinaus? | No-Signalling-Box (Definition) und Absprache im Spiel (Aufzählung aller 64 Strategien mit einem Bit): beide 100 %. Im Modell des Qubit-Paars hängt die Antwortverteilung eines Spielers nicht von der Frage des anderen ab: größte Abweichung unter 10⁻¹² in 300 Zufallsfällen; die Box: 0 | `test_h7_*` |
| H8 | Wie viele Runden braucht der Vorsprung? | Verschränkte Mannschaft (Bell, optimale Winkel) gegen die feste Strategie „beide immer 0“ auf denselben Fragen, wer am Ende **strikt mehr** Runden gewonnen hat. Exakt (Faltung von Binomialverteilungen) bei 25 / 50 / 100 / 200 Runden: 77,4 / 88,2 / 96,1 / 99,5 %; bei 25 Runden gewinnen beide gleich viele in 9,3 %. Die Wiederholungen der App (1 000 je Rundenzahl) liegen innerhalb von vier Standardfehlern der exakten Werte | `test_h8_*` |
| H9 | Wie sieht ein Spiel aus? | Zufalls-Seed 35, 400 Runden, alle Mannschaften auf denselben Fragen: ohne Ressource 75,75 %, geteilter Zufall 77,00 %, Verschränkung 84,75 %, Box und Absprache 100 %. **Ein einzelner Lauf**, kein Befund: die Standardfehler sind 2,2 % bzw. 1,8 %, der geteilte Zufall liegt hier um 2,0 Prozentpunkte über den erwarteten 75 %, innerhalb eines Standardfehlers | `test_h9_*` |

### Korrekturen gegenüber dem Plan

- **Zahl der besten Strategien:** der Plan nannte nur „höchstens 3/4“; gemessen sind es genau 8 von 16 Strategien mit 3/4 und 8 mit 1/4 (Mittel 50,0 %).
- **Neu gegenüber dem Plan:** das Modell „geteilte Zufallsrichtung“ (gibt bei den Bell-Winkeln genau 3/4), die Schwelle der Sichtbarkeit (70,7 %), der Bestwert teilverschränkter Zustände und die Rundenzahl (H8).
- **No-Signalling-Box:** der Plan nannte „nicht gerechnet“; hier steht die Definition (Tabelle) und die Prüfung, dass sie nichts überträgt und immer gewinnt. Das ist eine Folge der Definition, keine Aussage über die Natur.
- **Name und Port:** Ordner `chsh-spiel-demo` (ohne „ressourcen“, weil die Streamlit Cloud App-Namen mit „sso“ ablehnt), Präfix `chs`, Port 9216.

## Methodik

- **Matrixrechnung:** `chs_quantum.py` rechnet mit 4×4-Matrizen (Kronecker-Produkt der Projektoren); eine schnelle Fassung für Gitter und Heatmaps nutzt die geschlossene Form der Korrelatoren E(α, β) = v·(cos2α·cos2β + sin2θ·sin2α·sin2β), die im Test gegen die Matrixrechnung geprüft wird.
- **Stichproben:** Born-Regel per Inversion der Verteilungsfunktion mit Zufallszahlen aus SplitMix64 (`chs_rng.py`, wörtliche Kopie aus `ab-test-demo`, Kopie-Test); Fragen, geteilte Richtung, Box und Absprache haben je einen eigenen Strom (Zufalls-Seed · 8 + Nummer).
- **Gitter:** Winkel von −90 bis 82,5 Grad in 7,5-Grad-Schritten (24 Werte), alle vier Winkel unabhängig; „über 75 %“ heißt größer als 0,75 + 10⁻⁹. Die Heatmaps (91 × 91 Zellen, 2-Grad-Schritte) variieren Alices und Bobs Winkel bei Frage 1; die Winkel bei Frage 0 stehen wie in der Seitenleiste.
- **E4:** je Rundenzahl 1 000 Wiederholungen; die verschränkte Mannschaft gewinnt je Fragepaar mit der Wahrscheinlichkeit aus der Born-Regel, die feste Strategie ist deterministisch. „Strikt vorn“, „gleich viele“, „feste Strategie vorn“ zählen die Wiederholungen.
- **Rechenbudget:** am Slider-Maximum (5 000 Runden) rechnet der Spielverlauf aller Mannschaften in unter 0,01 s; die größte Rechnung ist E4 (sieben Rundenzahlen zu je 1 000 Wiederholungen, bis 1 600 Runden) mit rund 0,16 s, die Gitterzählung (331 776 Zellen) braucht rund 0,07 s (einmal unter Windows gemessen, nicht als Test festgeschrieben).

## Ehrliche Grenzen

- **Lehrbuchmodell, keine Aussage über Geräte oder Experimente:** ideale projektive Messungen an einem Qubit-Paar, kein Detektorverlust, kein Zeitfenster. Wie weit echte Aufbauten diese Werte erreichen, ist weder gerechnet noch behauptet.
- **Fragen gleichverteilt:** bei anderer Verteilung ändern sich der klassische Höchstwert und der Quantenwert.
- **Eine Zustandsfamilie:** gerechnet sind cosθ|00⟩ + sinθ|11⟩ mit weißem Rauschen und Messungen in einer Ebene. Dass dies für diese Familie der Bestwert ist, folgt aus der Horodecki-Formel (Literatur, hier an der Optimierung über alle vier Winkel geprüft); andere Zustände sind nicht gerechnet. Die Cirel'son-Grenze für **alle** Quantenstrategien wird nicht selbst bewiesen.
- **Rauschen ist weißes Rauschen:** die Schwelle 70,7 % gilt nur für diese Rauschform.
- **Das Modell „geteilte Zufallsrichtung“ ist ein Beispiel,** kein Beweis für alle geteilten Zufallsgrößen; der Beweis ist das Mittelwert-Argument (H1, H2).
- **Die Box ist eine Definition,** ob es sie in der Natur gibt, wird nicht behauptet.
- **H9 ist ein Lauf.** Stichproben mit 400 Runden unterscheiden die Sprossen nicht sicher (H8: bei 25 Runden ist die Verschränkung in 77,4 % der Spiele strikt vorn, nicht in allen).
- **Randstück:** ohne Logistik-Kulisse, ohne Praxisanspruch.

## Quellen

Die bibliografischen Angaben sind per Websuche bestätigt (2026-11-08); der Inhalt ist nur laut Suchtreffer wiedergegeben, **keine der Arbeiten wurde im Volltext gelesen**.

- **Clauser, Horne, Shimony und Holt (1969), Physical Review Letters 23, 880–884:** laut Suchtreffer eine Verallgemeinerung von Bells Satz mit einem Vorschlag für einen Test lokaler Theorien mit verborgenen Variablen. Die Spielform (a ⊕ b = x ∧ y) ist hier Lehrbuchwissen und nicht an dieser Quelle geprüft.
- **Cirel'son (1980), Letters in Mathematical Physics 4, 93–100:** laut Suchtreffer die Schranke 2√2 für die CHSH-Kombination bei beliebigen Quantenobservablen. Die Demo rechnet nur die genannte Familie.
- **Popescu und Rohrlich (1994), Foundations of Physics 24, 379–385:** laut Suchtreffer Korrelationen jenseits der Quantenmechanik, die die Kausalität der Relativitätstheorie achten und CHSH stärker verletzen. Die Box der Demo ist eine eigene Definition im Sinne dieser Gedankenmodelle.
- **Horodecki, Horodecki und Horodecki (1995), Physics Letters A 200, 340–344:** laut Suchtreffer ein notwendiges und hinreichendes Kriterium dafür, wann ein gemischter Zwei-Qubit-Zustand die CHSH-Ungleichung verletzt, in Termen der Korrelationsmatrix. Die Formel S = 2·√(λ₁ + λ₂) ist hier nicht am Text geprüft, sondern an der Optimierung über alle Winkel in der gerechneten Familie.

## Verifikation

- **Handrechnung** (`tests/conftest.py`): Strategien 0, 5 und 6 (3, 1 und 3 von 4), Bell-Zustand mit gleichen Winkeln (P(0,0) = P(1,1) = ½), cos²(π/8) = (1 + cos45°)/2, gemeinsame Richtung (Abstand π/8 gibt 3/4), die Box.
- **Orakel** (`tests/test_oracle_chsh_spiel.py`): klassisches Maximum durch Aufzählung mit ausgeschriebenen Bedingungen und über die CHSH-Algebra; Quantenwert über Zustandsvektor und explizite Basisvektoren (Rauschen als Mischung), über den größten Eigenwert des Bell-Operators und über cos²(π/8); Optimierung über alle vier Winkel (scipy, Nelder-Mead) gegen den Horodecki-Weg der App und die Formel; Born-Stichprobe mit eigenem Generator; geteilte Zufallsrichtung per Monte-Carlo; E4 gegen die exakte Binomialfaltung.
- **Eigenschaftstests** (`tests/test_eigenschaften_chsh_spiel.py`): keine Nachricht (300 Zufallsfälle), jede Tafel ist eine Verteilung, Rauschen mischt linear, gemeinsame Drehung aller Winkel und Vertauschen von Alice und Bob ändern den Wert nicht, 180 Grad Periode, Monotonie in Verschränkung und Rauschen, gleicher Zufalls-Seed in einem zweiten Prozess mit anderer Hash-Randomisierung.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README, gemessen oder exakt gerechnet; die Hilfetexte der Presets.
- **Zweig- und Randfalltests:** jedes Preset meldet das Verdikt, das sein Text nennt; Produktzustand, Sichtbarkeit 0, alle Winkel gleich, ±90 Grad, eine Runde; Verdikte als reine Funktionen an handgesetzten Werten (`tests/test_verdicts.py`).
- **Oberfläche:** AppTest (jeder Knopf, jeder Regler am Minimum und Maximum, alle Regler zugleich, jedes Preset, Strategien, Permalink, PDF), Layout und Footer.
- **Plattform:** Simulationen mit festen Seeds und vier Standardfehlern Marge, Gleitkommaaggregate mit `approx`; die Optimierung (Nelder-Mead) wird mit Toleranz 10⁻⁸ geprüft. Die Linux-CI ist hier nicht lokal nachgestellt (kein Docker).

## Tests

187 Tests (unter Windows, Python 3.14 im Test-venv). Zusammensetzung: Spiel und klassische Einheiten, Matrixrechnung, Spielverlauf, Experimente, Orakel, Eigenschaften, Claims, Verdikte, Randfälle, Presets, Layout, AppTest-Rauchtests, Visualisierung, Footer, Laufzeit-Importe, SplitMix64-Kopie. Laufzeit der vollen Suite rund 30 Sekunden seriell (unter Windows gemessen).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Hauptansicht, Spielverlauf, vier Experimente, Ausblick, Grenzen, Formeln, PDF |
| `chs_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets mit Hilfetexten |
| `chs_game.py` | Spiel, 16 Strategien, Mischungen, geteilte Zufallsrichtung, Box, Absprache |
| `chs_quantum.py` | Zustand, Messung, Born-Regel, Bestwert (Horodecki), schnelle Korrelatoren |
| `chs_play.py` | Spielverlauf der fünf Mannschaften auf denselben Fragen |
| `chs_scenario.py` | Einstellungen (nur mit Schlüsselwörtern) |
| `chs_experiments.py` | Zellen und Reihen der Experimente E1 bis E4 |
| `chs_evaluation.py` | Kennzahlen und Verdikte als reine Funktionen |
| `chs_visualization.py` | Plotly-Abbildungen (Achsen fest) |
| `chs_presets.py` | Permalink, Presets, Zufalls-Seed-Button |
| `chs_layout.py` | fester Kopf |
| `chs_pdf_export.py` | PDF-Export |
| `chs_rng.py` | SplitMix64 (wörtliche Kopie aus `ab-test-demo/abt_rng.py`) |
| `tests/` | alle Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Beweis der Cirel'son-Grenze für alle Quantenstrategien** und andere Zustände als die gerechnete Familie.
- **Mehr als zwei Spieler, mehr Fragen und Antworten** (andere Spiele aus der Familie der Bell-Spiele).
- **Hardware, Detektorverluste, Zeitfenster** und jede Aussage über Experimente.
- **Andere Rauschformen** als weißes Rauschen.
- **Ungleichverteilte Fragen.**

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in der App-Einstellung wählen (getestet mit 3.12 in der CI). <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2; die Gegenproben in den Tests brauchen zusätzlich SciPy (`requirements-dev.txt`).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
