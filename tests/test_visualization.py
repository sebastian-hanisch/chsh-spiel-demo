"""Abbildungen: jede baut sich, alle Achsen sind gesperrt, die Daten sind endlich und liegen im festen Bereich."""

import numpy as np
import pytest

import chs_evaluation as V
import chs_experiments as XP
import chs_play as PL
import chs_visualization as Z
from chs_scenario import Settings


def _figures():
    s = Settings()
    a = V.analyse(s)
    p = PL.play(s)
    heat = XP.e2_heatmaps(s)
    curves = XP.e3_curves(21)
    return {
        "ladder": Z.build_ladder(a), "angles": Z.build_angles(s), "cumulative": Z.build_cumulative(p, a), "heat_q": Z.build_heat_quantum(heat), "heat_s": Z.build_heat_shared(heat),
        "hist": Z.build_mix_hist(XP.e1_mixtures(500, 1)["values"], 0.75, a.w_quantum), "e3_best": Z.build_e3_best(curves), "e3_thr": Z.build_e3_threshold(curves),
        "e4": Z.build_e4(XP.e4_rows(s, rounds_list=(25, 50), repetitions=50)),
    }


@pytest.fixture(scope="module")
def figures():
    return _figures()


def test_every_axis_is_locked(figures):
    for name, fig in figures.items():
        for axis in list(fig.select_xaxes()) + list(fig.select_yaxes()):
            assert axis.fixedrange is True, name


def test_ladder_has_seven_bars_and_the_bell_value(figures):
    bar = figures["ladder"].data[0]
    assert len(bar.y) == 7 and max(bar.x) == pytest.approx(100.0) and bar.x[3] == pytest.approx(85.3553, abs=1e-3)


def test_angle_sketch_draws_four_rays_and_keeps_the_scale(figures):
    fig = figures["angles"]
    assert sum(1 for t in fig.data if t.name and ("Alice" in t.name or "Bob" in t.name)) == 4
    assert fig.layout.xaxis.scaleanchor == "y"


def test_cumulative_has_five_teams_and_the_data_is_finite(figures):
    fig = figures["cumulative"]
    assert len(fig.data) == 5 and all(np.isfinite(np.array(t.y, dtype=float)).all() for t in fig.data)


def test_heatmaps_have_a_75_percent_contour(figures):
    for key in ("heat_q", "heat_s"):
        fig = figures[key]
        assert fig.data[1].contours.start == 75 and fig.data[0].z.shape == (91, 91)
