"""Einheiten des Modells: Handtabellen in Brüchen (von Hand gerechnet), schnelles Kaplan–Meier-Quantil gegen den Kern, Lognormal-Gradient, Kosten, Welt, Läufe."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import znf_constants as C
import znf_core as K
import znf_estimators as E
import znf_model as M
import znf_scenario as S

# ------------------------------------------------------------------------------------------------ Handtabellen (alle Brüche von Hand)


def _rows(table):
    return [(r["value"], r["exact"], r["censored"], r["at_risk"], r["s_prev"], r["s"]) for r in table["rows"]]


def test_handtabelle_handbeispiel(hand_pairs):
    """Sechs Perioden: genau 6, 7, 9; ausverkauft 8, 10, 12. Im Risiko bei 6: 6, bei 7: 5, bei 9: 3 (9, 10, 12). S = 5/6, 5/6·4/5 = 2/3, 2/3·2/3 = 4/9."""
    t = M.hand_table(hand_pairs)
    assert _rows(t) == [
        (6, 1, 0, 6, F(1), F(5, 6)),
        (7, 1, 0, 5, F(5, 6), F(2, 3)),
        (8, 0, 1, 4, F(2, 3), F(2, 3)),
        (9, 1, 0, 3, F(2, 3), F(4, 9)),
        (10, 0, 1, 2, F(4, 9), F(4, 9)),
        (12, 0, 1, 1, F(4, 9), F(4, 9)),
    ]
    assert t["sales"] == [6, 7, 8, 9, 10, 12] and t["events"] == 3
    assert M.hand_quantile(t, F(1, 2)) == 9            # kleinster genauer Wert mit S ≤ 1/2
    assert M.hand_quantile(t, F(4, 5)) is None         # S ≤ 1/5 wird nie erreicht: nicht schätzbar
    assert M.hand_quantile(t, F(5, 9)) == 9            # 1 − 5/9 = 4/9: genau erreicht (Grenzfall)
    assert M.hand_quantile(t, F(56, 100)) is None      # knapp darüber: nicht erreicht
    assert M.hand_sales_quantile(t, F(1, 2)) == F(17, 2)          # Median der Verkäufe 8,5
    assert F(sum(t["sales"]), 6) == F(26, 3)                      # Mittel 8,67 (steht im Plan)


def test_handtabelle_nie_ausverkauft():
    """Verkäufe 6, 7, 5, 9, 11, 8: alle genau. Sortiert 5, 6, 7, 8, 9, 11, S = 5/6, 2/3, 1/2, 1/3, 1/6, 0; Median (S ≤ 1/2) bei 7, Quantil 0,8 (S ≤ 1/5) bei 9."""
    t = M.hand_table(S.PATTERNS["Nie ausverkauft"])
    assert [r["s"] for r in t["rows"]] == [F(5, 6), F(2, 3), F(1, 2), F(1, 3), F(1, 6), F(0)]
    assert [r["at_risk"] for r in t["rows"]] == [6, 5, 4, 3, 2, 1]
    assert M.hand_quantile(t, F(1, 2)) == 7 and M.hand_quantile(t, F(4, 5)) == 9
    assert M.hand_sales_quantile(t, F(1, 2)) == F(15, 2)           # Mittel der beiden mittleren Verkäufe 7 und 8
    assert M.hand_sales_quantile(t, F(4, 5)) == 9


def test_handtabelle_immer_ausverkauft():
    """Alle sechs Perioden ausverkauft (Verkäufe 10, 8, 12, 9, 10, 11): kein genauer Wert, S bleibt 1, kein Quantil schätzbar; aus den Verkäufen Median 10."""
    t = M.hand_table(S.PATTERNS["Immer ausverkauft"])
    assert t["events"] == 0 and all(r["s"] == 1 and r["exact"] == 0 for r in t["rows"])
    assert [(r["value"], r["censored"], r["at_risk"]) for r in t["rows"]] == [(8, 1, 6), (9, 1, 5), (10, 2, 4), (11, 1, 2), (12, 1, 1)]
    assert M.hand_quantile(t, F(1, 2)) is None and M.hand_quantile(t, F(1, 10)) is None
    assert M.hand_sales_quantile(t, F(1, 2)) == 10


def test_handtabelle_gleichstand_genau_und_ausverkauft_bei_demselben_wert():
    """Perioden (12, 9) genau 9 und (9, 9) ausverkauft bei 9: Bei 9 sind im Risiko 9, 9, 10, 12 = 4 (der Ausverkaufte bleibt im Risiko, das Ereignis zählt davor)."""
    t = M.hand_table(S.PATTERNS["Gleichstand: ein Wert ist genau und zugleich Bestand"])
    assert _rows(t) == [
        (6, 1, 0, 6, F(1), F(5, 6)),
        (7, 1, 0, 5, F(5, 6), F(2, 3)),
        (9, 1, 1, 4, F(2, 3), F(1, 2)),
        (10, 0, 1, 2, F(1, 2), F(1, 2)),
        (12, 0, 1, 1, F(1, 2), F(1, 2)),
    ]
    assert M.hand_quantile(t, F(1, 2)) == 9 and M.hand_quantile(t, F(4, 5)) is None
    assert M.hand_sales_quantile(t, F(1, 2)) == 9


def test_handtabelle_gleicht_dem_kern_in_jedem_muster():
    for name, pairs in S.PATTERNS.items():
        t = M.hand_table(pairs)
        km = K.kaplan_meier([v for _, v in pairs], [1 if v < b else 0 for b, v in pairs])
        assert km["t"].tolist() == [r["value"] for r in t["rows"] if r["exact"]], name
        assert km["s"].tolist() == pytest.approx([float(r["s"]) for r in t["rows"] if r["exact"]], abs=1e-15), name
        for tau in (F(1, 2), F(4, 5)):
            want = K.quantile(km, float(tau))
            got = M.hand_quantile(t, tau)
            assert (got is None and math.isnan(want)) or got == want, (name, tau)
            assert float(M.hand_sales_quantile(t, tau)) == pytest.approx(float(np.quantile([v for _, v in pairs], float(tau))), abs=1e-12), (name, tau)


def test_handtabelle_eingabepruefung():
    for bad in ([], [(10, 11)], [(10, 0)], [(10, -1)]):
        with pytest.raises(K.SurvivalError):
            M.hand_table(bad)


# ------------------------------------------------------------------------------------------------ schnelles Kaplan–Meier-Quantil gegen den Kern


def test_km_quantile_fast_gleicht_dem_kern_auch_mit_gleichstaenden():
    rng = K.SplitMix64(11)
    for r in range(150):
        n = int(rng.uniform(1)[0] * 40) + 2
        t = np.ceil(rng.exponential(n) * 4) + 1                       # ganze Zahlen: viele Gleichstände
        e = rng.uniform(n) < 0.6
        km = K.kaplan_meier(t, e)
        for tau in (0.1, 0.5, 0.8, 0.95):
            a, b = E.km_quantile_fast(t, e, tau), K.quantile(km, tau)
            assert (math.isnan(a) and math.isnan(b)) or a == b, (r, tau)


def test_km_quantile_fast_ohne_ereignis_ist_nan():
    assert math.isnan(E.km_quantile_fast([1, 2, 3], [0, 0, 0], 0.5))


# ------------------------------------------------------------------------------------------------ Lognormal: Gradient und Spezialfälle


def test_lognormal_gradient_stimmt_mit_finiter_differenz():
    rng = K.SplitMix64(5)
    logt = np.log(S.draw_demand(rng, 60, "lognormal"))
    ev = rng.uniform(60) < 0.5
    for p in ([4.5, -0.7], [4.0, -0.2], [5.2, 0.4]):
        p = np.array(p)
        _, g = E._lognormal_nll(p, logt, ev)
        num = np.zeros(2)
        for i in range(2):
            h = np.zeros(2)
            h[i] = 1e-6
            num[i] = (E._lognormal_nll(p + h, logt, ev)[0] - E._lognormal_nll(p - h, logt, ev)[0]) / 2e-6
        assert g == pytest.approx(num, rel=1e-6, abs=1e-6)


def test_lognormal_ohne_zensierung_ist_mittel_und_standardabweichung_der_logarithmen():
    """Ohne Ausverkauf ist die Maximum-Likelihood-Lösung geschlossen: μ̂ = Mittel von ln x, σ̂² = Mittel der quadrierten Abweichungen (durch n, nicht n − 1)."""
    x = np.array([3.0, 5.0, 8.0, 13.0, 21.0, 34.0])
    fit = E.lognormal_fit(x, np.ones(6, bool))
    lx = np.log(x)
    assert fit["ok"] and fit["mu"] == pytest.approx(lx.mean(), abs=1e-6) and fit["sigma"] == pytest.approx(lx.std(), abs=1e-6)


def test_lognormal_quantil_ist_exp_mu_plus_sigma_mal_z():
    fit = {"mu": math.log(100.0), "sigma": 0.5, "ok": True}
    assert E.lognormal_quantile(fit, 0.5) == pytest.approx(100.0, rel=1e-12)
    assert E.lognormal_quantile(fit, 0.8) == pytest.approx(100.0 * math.exp(0.5 * 0.8416212335729143), rel=1e-12)   # Φ⁻¹(0,8) = 0,8416212335729143
    assert math.isnan(E.lognormal_quantile({"mu": float("nan"), "sigma": float("nan"), "ok": False}, 0.5))


# ------------------------------------------------------------------------------------------------ Kosten


def test_kosten_lognormal_von_hand():
    """Die Formel des Expanders an drei Punkten, von Hand: E[(D − q)⁺] ist bei q → 0 gleich E[D] = 100·e^{0,125}, bei sehr großem q gleich 0, und beim Median (q = 100)
    E[D]·Φ(σ) − 100·Φ(0) = E[D]·Φ(0,5) − 50."""
    mean = 100.0 * math.exp(0.125)
    assert M.expected_shortage(1e-9, "lognormal") == pytest.approx(mean, rel=1e-9)
    assert M.expected_shortage(1e9, "lognormal") == pytest.approx(0.0, abs=1e-12)
    assert M.demand_mean("lognormal") == pytest.approx(mean, rel=1e-12)
    # Bei q = Median: Φ((μ + σ² − ln q)/σ) = Φ(σ) = Φ(0,5), Φ((μ − ln q)/σ) = Φ(0) = ½
    phi_half = 0.6914624612740131      # Φ(0,5)
    assert M.expected_shortage(100.0, "lognormal") == pytest.approx(mean * phi_half - 100.0 * 0.5, rel=1e-12)


def _cdf(form, x):
    """Verteilungsfunktion der wahren Nachfrage aus SciPy (anderer Weg als die Formeln der Demo)."""
    from scipy.stats import gamma, lognorm

    if form == "lognormal":
        return lognorm.cdf(x, C.DEMAND_SIGMA, scale=C.DEMAND_MEDIAN)
    return gamma.cdf(x, S.gamma_shape(), scale=S.gamma_scale())


def test_beste_bestellmenge_hat_die_geringsten_kosten_und_ableitung_f_minus_tau():
    """C'(q) = F(q) − τ: die numerische Ableitung der erwarteten Kosten stimmt mit F(q) − τ aus SciPy überein; sie ist 0 am Quantil, vorher negativ, danach positiv."""
    for form in C.FORMS:
        for tau in (0.5, 0.8, 0.95):
            q = M.true_quantile(form, tau)
            c0 = float(M.expected_cost(q, tau, form))
            assert c0 > 0
            for dq in (-30.0, -5.0, 5.0, 30.0):
                assert float(M.expected_cost(q + dq, tau, form)) > c0, (form, tau, dq)
            h = 1e-3
            for x in (q - 40.0, q - 10.0, q, q + 10.0, q + 40.0):
                slope = (float(M.expected_cost(x + h, tau, form)) - float(M.expected_cost(x - h, tau, form))) / (2 * h)
                assert slope == pytest.approx(float(_cdf(form, x)) - tau, abs=1e-6), (form, tau, x)


def test_mehrkosten_null_an_der_besten_menge_und_nan_bleibt_nan():
    for form in C.FORMS:
        q = M.true_quantile(form, 0.9)
        assert float(M.extra_cost(q, 0.9, form)) == pytest.approx(0.0, abs=1e-12)
        out = M.extra_cost(np.array([q, float("nan"), q * 0.5]), 0.9, form)
        assert out[0] == pytest.approx(0.0, abs=1e-12) and math.isnan(out[1]) and out[2] > 0


def test_gamma_form_und_skala_geben_median_100_und_gleichen_variationskoeffizienten():
    from scipy.stats import gamma

    k, th = S.gamma_shape(), S.gamma_scale()
    assert gamma.median(k, scale=th) == pytest.approx(100.0, rel=1e-9)
    assert 1.0 / k == pytest.approx(math.exp(0.25) - 1.0, rel=1e-12)              # Variationskoeffizient² von Gamma = 1/k = e^σ² − 1 (das Lognormal)
    assert k == pytest.approx(3.5208, abs=1e-4)                                    # 1/(1,2840254 − 1)


def test_falsche_verteilung_wird_gemeldet():
    for fn in (lambda: M.true_quantile("normal", 0.5), lambda: M.expected_shortage(1.0, "normal"), lambda: S.simulate(10, 100, "normal", 0.0, 1)):
        with pytest.raises(K.SurvivalError):
            fn()


# ------------------------------------------------------------------------------------------------ Welt


def test_simulate_ist_bestimmt_und_hat_die_beschriebene_struktur():
    a = S.simulate(500, 100.0, "lognormal", 0.0, 7)
    b = S.simulate(500, 100.0, "lognormal", 0.0, 7)
    assert np.array_equal(a.sales, b.sales) and np.array_equal(a.observed, b.observed)
    assert np.array_equal(a.sales, np.minimum(a.demand, a.stock))
    assert np.array_equal(a.observed, a.demand < a.stock)
    assert np.all(a.sales > 0) and np.all(a.demand > 0)
    assert a.sales[a.observed] == pytest.approx(a.demand[a.observed], abs=0)
    assert a.sales[~a.observed] == pytest.approx(a.stock[~a.observed], abs=0)


def test_gepaarte_welten_teilen_nachfrage_und_zufall_des_bestands():
    lo, hi = S.simulate(300, 70.0, "lognormal", 0.0, 3), S.simulate(300, 140.0, "lognormal", 0.0, 3)
    assert np.array_equal(lo.demand, hi.demand)
    assert hi.stock == pytest.approx(2.0 * lo.stock, rel=1e-12)             # derselbe Zufall, doppelter Median
    boosted = S.simulate(300, 70.0, "lognormal", 0.2, 3)
    assert np.array_equal(boosted.demand, lo.demand)
    changed = boosted.stock != lo.stock
    assert 0.1 < changed.mean() < 0.3                                          # etwa 20 % Testperioden
    assert boosted.stock[changed] == pytest.approx(lo.stock[changed] * C.BOOST_STOCK_MEDIAN / 70.0, rel=1e-12)


def test_verteilungen_der_welt_treffen_median_und_streuung_bei_grossem_n():
    big = S.simulate(200000, 100.0, "lognormal", 0.0, 21)
    assert np.median(big.demand) == pytest.approx(100.0, rel=0.01) and np.std(np.log(big.demand)) == pytest.approx(0.5, rel=0.01)
    assert np.std(np.log(big.stock)) == pytest.approx(0.3, rel=0.01)
    gam = S.simulate(200000, 100.0, "gamma", 0.0, 21)
    assert np.median(gam.demand) == pytest.approx(100.0, rel=0.01)
    assert np.mean(gam.demand) == pytest.approx(M.demand_mean("gamma"), rel=0.01)
    assert np.var(gam.demand) / np.mean(gam.demand) ** 2 == pytest.approx(math.exp(0.25) - 1.0, rel=0.03)


def test_run_seed_trennt_die_laeufe():
    assert S.run_seed(35, 0) == 35 and S.run_seed(35, 2) == 35 + 2 * 1000003


def test_run_study_form_und_kennzahlen():
    w = S.World(m=100.0, n=200)
    st = M.run_study(w, (0.5, 0.9), reps=20)
    assert st.reps == 20 and st.taus == (0.5, 0.9)
    for name in M.METHODS:
        assert st.q[name].shape == (20, 2)
    assert st.ln_ok.all() and 0.3 < st.sold_out.mean() < 0.7
    assert np.all(st.q["sales"] > 0) and np.all(st.q["km_fallback"] > 0)
    # Rückfall: gleich Kaplan–Meier, wo es eine Zahl gibt; sonst der größte Verkauf
    km, fb = st.q["km"], st.q["km_fallback"]
    assert np.array_equal(km[np.isfinite(km)], fb[np.isfinite(km)])
    for r in np.where(~np.isfinite(km[:, 1]))[0]:
        assert fb[r, 1] == S.make_sample(w, int(r)).sales.max()
    ohne = M.run_study(w, (0.5,), reps=3, with_lognormal=False)
    assert ohne.ln_ok is None and np.isnan(ohne.q["lognormal"]).all()


def test_summarize_und_defined_share():
    w = S.World(m=70.0, n=300)
    st = M.run_study(w, (0.5, 0.9), reps=30)
    c = M.summarize(st, 0.9, "km")
    assert c.reps == 30 and c.defined == int(np.sum(np.isfinite(st.q["km"][:, 1]))) and M.defined_share(st, 0.9) == c.defined / 30
    s = M.summarize(st, 0.9, "sales")
    assert s.defined == 30 and s.cost_mean == pytest.approx(float(np.mean(M.extra_cost(st.q["sales"][:, 1], 0.9, "lognormal"))), rel=1e-12)
