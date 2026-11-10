"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink; Aussagen der Hilfetexte gegen die Messung stehen in tests/test_claims.py."""

import znf_constants as C
import znf_presets as P
import znf_scenario as S


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_the_options():
    for name, p in C.PRESETS.items():
        assert C.M_MIN <= p["m"] <= C.M_MAX and (p["m"] - C.M_MIN) % C.M_STEP == 0, name
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0 and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["tau"] in C.TAU_OPTIONS and p["form"] in C.FORMS, name


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert S.World(m=first["m"], n=first["n"], form=first["form"], seed=first["seed"]) == S.World()
    assert first["tau"] == C.DEFAULT_TAU


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_grid_and_series_are_inside_the_slider_bounds():
    assert all(C.M_MIN <= m <= C.M_MAX for m in C.GRID_M) and all(t in C.TAU_OPTIONS for t in C.GRID_TAU)
    assert C.BOOST_SHARES[0] == 0.0 and all(0.0 <= b <= 1.0 for b in C.BOOST_SHARES)
    assert C.DEFAULT_M in C.GRID_M                      # die Standardwelt ist eine Zeile der Kostentabelle: der Cache trifft


def test_presets_differ_from_each_other_in_at_least_one_setting():
    keys = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(keys)) == len(keys)


def test_permalink_werte_werden_begrenzt_und_auswahlen_eingerastet():
    spec_tau, spec_m, spec_form = P.SETTING_SPECS["tau_select"], P.SETTING_SPECS["m_slider"], P.SETTING_SPECS["form_radio"]
    assert P._clean(spec_tau, 0.9000000000000001) == 0.9 and P._clean(spec_tau, 0.77) == C.DEFAULT_TAU
    assert P._clean(spec_m, 5000) == C.M_MAX and P._clean(spec_m, -3) == C.M_MIN
    assert P._clean(spec_form, "gamma") == "gamma" and P._clean(spec_form, "weibull") == C.DEFAULT_FORM
