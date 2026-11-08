"""Experimente E1 bis E4: Form der Ergebnisse, Zählungen und Zweige (jede Variante weicht auf mindestens einer Einstellung ab)."""

import math

import numpy as np
import pytest

import chs_constants as C
import chs_experiments as XP
from chs_scenario import Settings


def test_e1_rows_list_all_sixteen_strategies():
    rows = XP.e1_strategy_rows()
    assert [r["index"] for r in rows] == list(range(16))
    assert all(r["value"] == r["wins"] / 4 for r in rows)


def test_e1_mixtures_never_exceed_the_best_strategy():
    m = XP.e1_mixtures(2000, 3)
    assert m["count"] == 2000 and len(m["values"]) == 2000
    assert m["max"] <= 0.75 + 1e-12 and 0.45 < m["mean"] < 0.55


def test_heat_axis_runs_from_minus_90_to_90_degrees():
    ax = np.degrees(XP.heat_axis())
    assert ax[0] == pytest.approx(-90.0) and ax[-1] == pytest.approx(90.0) and len(ax) == 91


def test_e2_heatmaps_have_the_grid_shape_and_the_bell_peak():
    h = XP.e2_heatmaps(Settings())
    assert h["quantum"].shape == (91, 91) == h["shared"].shape
    assert 0.853 < h["quantum"].max() <= math.cos(math.pi / 8) ** 2 + 1e-12       # (α₁, β₁) = (45, −22,5) liegt nicht auf dem 2-Grad-Raster, ein Nachbarpunkt kommt sehr nahe
    assert h["shared"].max() <= 0.75 + 1e-12


def test_e2_grid_counts_for_the_bell_state():
    g = XP.e2_grid_counts(Settings())
    assert g["points"] == 24 and g["cells"] == 24 ** 4
    assert g["quantum"]["above"] == 24816 and g["quantum"]["at_max"] == 48
    assert g["quantum"]["max"] == pytest.approx(math.cos(math.pi / 8) ** 2)
    assert g["shared"]["above"] == 0 and g["shared"]["max"] == pytest.approx(0.75)


def test_e2_product_state_has_no_cell_above_three_quarters():
    g = XP.e2_grid_counts(Settings(theta_deg=0.0))
    assert g["quantum"]["above"] == 0 and g["quantum"]["max"] == pytest.approx(0.75)


def test_e2_noise_below_the_threshold_removes_every_cell():
    assert XP.e2_grid_counts(Settings(vis_pct=70))["quantum"]["above"] == 0
    assert XP.e2_grid_counts(Settings(vis_pct=71))["quantum"]["above"] > 0       # 0,71 > 1/√2 = 0,7071


def test_e3_rows_follow_the_theta_list():
    rows = XP.e3_rows()
    assert [r["theta_deg"] for r in rows] == list(XP.E3_THETAS)
    assert rows[0]["best"] == pytest.approx(0.75) and rows[0]["threshold"] == pytest.approx(1.0)
    assert all(rows[i]["best"] < rows[i + 1]["best"] for i in range(len(rows) - 1))
    assert all(rows[i]["threshold"] > rows[i + 1]["threshold"] for i in range(len(rows) - 1))


def test_e3_visibility_rows_fall_with_the_noise():
    rows = XP.e3_visibility_rows(45.0)
    assert [r["vis"] for r in rows] == list(XP.E3_VISIBILITIES)
    assert all(rows[i]["best"] > rows[i + 1]["best"] for i in range(len(rows) - 1))
    assert rows[0]["best"] == pytest.approx(math.cos(math.pi / 8) ** 2)


def test_e3_curves_cover_the_whole_range():
    c = XP.e3_curves(31)
    assert c["theta_deg"][0] == 0.0 and c["theta_deg"][-1] == 45.0 and len(c["best"]) == 31 == len(c["threshold"])


def test_e4_cell_shares_add_up_and_rise_with_the_rounds():
    s = Settings()
    a, b = XP.e4_cell(s, 25, 500), XP.e4_cell(s, 400, 500)
    for r in (a, b):
        assert r["more"] + r["equal"] + r["less"] == pytest.approx(1.0)
    assert b["more"] > a["more"] and b["less"] < a["less"]
    assert XP.e4_cell(s, 25, 500) == a                                   # gleiche Eingabe, gleiche Zahlen


def test_e4_branches_differ_with_the_state():
    """Zweig-Test: bei Rauschen unter der Schwelle gewinnt die verschränkte Mannschaft selten strikt mehr Runden als die feste Strategie, im Bell-Fall fast immer."""
    good = XP.e4_cell(Settings(), 400, 400)["more"]
    bad = XP.e4_cell(Settings(vis_pct=70), 400, 400)["more"]
    assert good > 0.99 and bad < 0.5


def test_e4_uses_the_strategy_of_the_settings():
    weak = XP.e4_cell(Settings(strategy=5), 100, 300)       # Strategie 5 gewinnt nur 1 von 4 Paaren
    strong = XP.e4_cell(Settings(strategy=0), 100, 300)
    assert weak["c_rate"] < 0.35 < 0.7 < strong["c_rate"] and weak["more"] > strong["more"]


def test_e4_rows_follow_the_rounds_list():
    rows = XP.e4_rows(Settings(), rounds_list=(25, 50), repetitions=50)
    assert [r["rounds"] for r in rows] == [25, 50] and all(r["repetitions"] == 50 for r in rows)
    assert C.E4_ROUNDS[0] == 25
