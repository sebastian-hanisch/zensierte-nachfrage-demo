"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, alle ausverkauft, nie ausverkauft, Gleichstand von Verkauf und Bestand,
Eingabeprüfung, Grenzen der Regler, Meldungen statt stiller Zahlen."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import znf_constants as C
import znf_core as K
import znf_estimators as E
import znf_model as M
import znf_scenario as S


def test_eine_periode_und_genau_zwei_perioden():
    assert E.km_quantile_fast([3.0], [True], 0.5) == 3.0 and math.isnan(E.km_quantile_fast([3.0], [False], 0.5))
    assert E.sales_quantile([3.0], 0.9) == 3.0
    fit = E.lognormal_fit([3.0], [True])
    assert not fit["ok"] and math.isnan(fit["mu"]) and "mindestens zwei verschiedene" in fit["message"]
    # zwei genaue Werte: das Maximum-Likelihood-Lognormal hat μ̂ = Mittel der Logarithmen und σ̂ = halber Abstand
    fit = E.lognormal_fit([2.0, 8.0], [True, True])
    assert fit["ok"] and fit["mu"] == pytest.approx(math.log(4.0), abs=1e-6) and fit["sigma"] == pytest.approx(math.log(2.0), abs=1e-5)
    t = M.hand_table([(5, 3)])
    assert [r["s"] for r in t["rows"]] == [F(0)] and M.hand_quantile(t, F(1, 2)) == 3
    t = M.hand_table([(5, 3), (5, 4)])
    assert [r["s"] for r in t["rows"]] == [F(1, 2), F(0)] and M.hand_quantile(t, F(1, 2)) == 3 and M.hand_sales_quantile(t, F(1, 2)) == F(7, 2)


def test_alle_ausverkauft_kein_quantil_keine_anpassung_und_die_verkaeufe_sind_der_bestand():
    smp = S.simulate(300, 1.0, "lognormal", 0.0, 4)          # Bestand um 1: ausverkauft in fast jeder Periode
    assert (~smp.observed).mean() > 0.99
    if not smp.observed.any():
        assert math.isnan(E.km_quantile_fast(smp.sales, smp.observed, 0.5))
        assert not E.lognormal_fit(smp.sales, smp.observed)["ok"]
    km = K.kaplan_meier(smp.sales, np.zeros(smp.sales.size, bool))
    assert km["t"].size == 0 and math.isnan(K.quantile(km, 0.5)) and K.survival_at(km, 10.0) == 1.0
    st = M.run_study(S.World(m=1.0, n=100), (0.9,), reps=3)
    assert np.isnan(st.q["km"]).all() and not st.ln_ok.any() and np.isnan(st.q["lognormal"]).all()
    assert (st.q["km_fallback"] == np.array([[S.make_sample(S.World(m=1.0, n=100), r).sales.max()] for r in range(3)])).all()
    c = M.summarize(st, 0.9, "km")
    assert c.defined == 0 and math.isnan(c.q_mean) and math.isnan(c.cost_mean)
    assert np.isfinite(M.summarize(st, 0.9, "sales").cost_mean)


def test_nie_ausverkauft_kaplan_meier_ist_die_empirische_verteilung():
    smp = S.simulate(200, 1e7, "lognormal", 0.0, 6)
    assert smp.observed.all()
    for tau in (0.5, 0.9, 0.95):
        assert E.km_quantile_fast(smp.sales, smp.observed, tau) == float(np.quantile(smp.sales, tau, method="inverted_cdf"))
    fit = E.lognormal_fit(smp.sales, smp.observed)
    assert fit["ok"] and fit["sigma"] == pytest.approx(np.log(smp.sales).std(), rel=1e-5)


def test_gleichstand_verkauf_gleich_bestand_zaehlt_als_ausverkauft():
    # Konvention: Verkauf = Bestand heißt ausverkauft (Nachfrage ≥ Bestand), auch wenn die Nachfrage zufällig genau so groß war
    t = M.hand_table([(8, 8), (9, 7), (9, 9)])
    assert [(r["value"], r["exact"], r["censored"]) for r in t["rows"]] == [(7, 1, 0), (8, 0, 1), (9, 0, 1)]


def test_gleichstaende_in_den_werten_kern_und_schnelle_fassung_stimmen_ueberein():
    t = [5.0, 5.0, 5.0, 7.0, 7.0, 9.0]
    for ev in ([1, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]):
        km = K.kaplan_meier(t, ev)
        for tau in (0.1, 0.5, 0.9):
            a, b = E.km_quantile_fast(t, ev, tau), K.quantile(km, tau)
            assert (math.isnan(a) and math.isnan(b)) or a == b


def test_eingabepruefung_meldet_statt_stiller_zahlen():
    for bad_sales in ([0.0, 1.0], [-1.0, 2.0], [float("nan"), 1.0], [float("inf"), 1.0]):
        with pytest.raises(K.SurvivalError):
            E.km_quantile_fast(bad_sales, [True, True], 0.5)
        with pytest.raises(K.SurvivalError):
            E.lognormal_fit(bad_sales, [True, True])
    with pytest.raises(K.SurvivalError):
        E.km_quantile_fast([1.0, 2.0], [True], 0.5)                 # Länge
    with pytest.raises(K.SurvivalError):
        E.sales_quantile([], 0.5)
    with pytest.raises(K.SurvivalError):
        E.sales_quantile([1.0, float("nan")], 0.5)
    for bad in ((0, 5, "lognormal", 0.0, 1), (10, 0.0, "lognormal", 0.0, 1), (10, 100, "lognormal", 1.5, 1), (10, 100, "lognormal", -0.1, 1)):
        with pytest.raises(K.SurvivalError):
            S.simulate(*bad)


def test_lognormal_entartete_daten_geben_status_oder_endliche_zahlen():
    fit = E.lognormal_fit([5.0, 5.0, 5.0], [True, True, True])      # nur ein verschiedener genauer Wert
    assert not fit["ok"] and "zwei verschiedene" in fit["message"]
    x = [1.0, 2.0] + [100.0] * 50                                    # zwei genaue Werte, sonst nur Ausverkäufe weit darüber
    fit = E.lognormal_fit(x, [True, True] + [False] * 50)
    assert (not fit["ok"]) or (math.isfinite(fit["mu"]) and math.isfinite(fit["sigma"]) and fit["sigma"] > 0)
    assert math.isnan(E.lognormal_quantile(E.lognormal_fit([5.0], [True]), 0.5))


def test_extreme_bestellmengen_geben_endliche_mehrkosten():
    for form in C.FORMS:
        out = M.extra_cost(np.array([1e-9, 1e-3, 1e6]), 0.9, form)
        assert np.all(np.isfinite(out)) and np.all(out > 0)
        assert float(M.extra_cost(0.0, 0.9, form)) == pytest.approx(float(M.extra_cost(1e-12, 0.9, form)), rel=1e-9)     # q = 0 wird auf eine winzige Menge geklemmt


def test_reglergrenzen_rechnen_ohne_fehler_und_mit_endlichen_verkaufsquantilen():
    for m in (C.M_MIN, C.M_MAX):
        for n in (C.N_MIN, C.N_MAX):
            for form in C.FORMS:
                w = S.World(m=float(m), n=n, form=form, boost=0.3)
                st = M.run_study(w, C.TAU_OPTIONS, reps=2)
                assert np.isfinite(st.q["sales"]).all() and np.isfinite(st.q["km_fallback"]).all() and (st.q["sales"] > 0).all()
                for tau in C.TAU_OPTIONS:
                    for meth in M.METHODS:
                        M.summarize(st, tau, meth)       # wirft nicht


def test_hand_tabelle_alle_muster_haben_aufsteigende_werte_und_monotones_s():
    for name, pairs in S.PATTERNS.items():
        t = M.hand_table(pairs)
        values = [r["value"] for r in t["rows"]]
        assert values == sorted(set(values)), name
        s = [r["s"] for r in t["rows"]]
        assert all(a >= b for a, b in zip(s, s[1:])) and all(0 <= x <= 1 for x in s), name
        assert sum(r["exact"] + r["censored"] for r in t["rows"]) == len(pairs), name
        assert t["rows"][0]["at_risk"] == len(pairs), name
