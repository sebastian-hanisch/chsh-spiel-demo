"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def _extreme_app(edge: str) -> AppTest:
    """Alle Regler zugleich am Minimum bzw. am Maximum (Eckfall-Matrix, DEMO-PLAYBOOK §1d)."""
    at = _fresh_app()
    for s in at.sidebar.slider:
        value = s.min if edge == "min" else s.max
        s.set_value(int(value) if isinstance(s.value, int) else value)
    at.run(timeout=TIMEOUT)
    return at


def test_all_sliders_at_their_minima_and_maxima_together():
    for edge in ("min", "max"):
        at = _extreme_app(edge)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_every_preset_loads_and_renders():
    names = [b.label for b in _fresh_app().sidebar.button if b.key and b.key.startswith("preset_")]
    assert len(names) == 4
    for name in names:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])


def test_every_strategy_of_the_selectbox_renders():
    for index in (0, 5, 6, 15):
        at = _fresh_app()
        at.sidebar.selectbox(key="strategy_select").set_value(index).run(timeout=TIMEOUT)
        assert not at.exception, (index, [str(e) for e in at.exception])


def test_experiment_button_e4_shows_a_chart_and_a_verdict():
    at = _fresh_app()
    at.button(key="e4_button").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Wiederholungen" in s.value and "Runden" in s.value for s in list(at.success) + list(at.info))


def test_pdf_button_makes_a_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["pdf_bytes"][:5] == b"%PDF-"


def test_permalink_values_are_clamped_and_garbage_is_ignored():
    at = AppTest.from_file(APP_PATH)
    at.query_params["a0"] = "500"
    at.query_params["theta"] = "abc"
    at.query_params["rounds"] = "1"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["a0_slider"] == 90.0 and at.session_state["theta_slider"] == 45.0 and at.session_state["rounds_slider"] == 20
