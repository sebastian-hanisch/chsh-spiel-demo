"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, die Zahlen der Hilfetexte, Permalink-Begrenzung."""

import math

import chs_constants as C
import chs_evaluation as ev
import chs_presets as P
from chs_scenario import Settings


def _settings(preset):
    return Settings(theta_deg=preset["theta"], vis_pct=preset["vis"], alpha0_deg=preset["a0"], alpha1_deg=preset["a1"], beta0_deg=preset["b0"], beta1_deg=preset["b1"],
                    strategy=preset["strategy"], rounds=preset["rounds"], seed=preset["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Optimale Winkel (Voreinstellung)"]) == Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_are_pairwise_different():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_each_preset_shows_the_effect_its_help_text_names():
    """Verdikt je Preset (DEMO-PLAYBOOK §11a, Punkt 7): ein Preset ohne Effekt darf keinen Effekt melden."""
    a = {name: ev.analyse(_settings(p)) for name, p in C.PRESETS.items()}
    opt = a["Optimale Winkel (Voreinstellung)"]
    assert math.isclose(opt.w_quantum, math.cos(math.pi / 8) ** 2) and ev.verdict_resource(opt)[0] == "success"
    zero = a["Winkel ohne Vorteil"]
    assert math.isclose(zero.w_quantum, 0.75) and ev.verdict_resource(zero)[0] == "info"
    prod = a["Ohne Verschränkung"]
    assert prod.w_quantum < 0.75 and math.isclose(prod.w_best, 0.75) and ev.verdict_resource(prod)[0] == "warning" and "Produktzustand" in ev.verdict_state(prod)[1]
    noisy = a["Rauschen 70 %"]
    assert noisy.w_quantum < 0.75 and noisy.w_best < 0.75 and ev.verdict_resource(noisy)[0] == "warning"
    assert 0.70 < noisy.threshold < 0.71


def test_permalink_values_are_clamped_to_the_slider_bounds():
    spec = P.SETTING_SPECS["a0_slider"]
    assert max(spec.lo, min(spec.hi, spec.caster("500"))) == 90.0
    assert max(spec.lo, min(spec.hi, spec.caster("-500"))) == -90.0
