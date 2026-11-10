"""Orakel-Tests: die Kernroutinen gegen einen unabhängigen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2).

* Kaplan–Meier (Kurve, Quantil, schnelle Fassung, Handtabelle): gegen die **Umverteilung nach rechts** (Efron) in exakten Brüchen, ein anderer Algorithmus als das
  Produkt-Grenzwert-Verfahren: jede ausverkaufte Periode gibt ihre Masse zu gleichen Teilen an alle späteren Werte weiter.
* Zensiertes Lognormal: gegen `scipy.optimize.minimize` (Nelder–Mead, ohne Gradient) auf einer selbst geschriebenen Likelihood mit `scipy.stats.norm`, gegen eine
  **Rastersuche** um den Schätzer und gegen den geschlossenen Fall ohne Zensierung.
* Erwartete Kosten und beste Menge: gegen numerische **Integration** der Dichte (`scipy.integrate.quad`, `scipy.stats`) und gegen eine **Simulation** mit Band.
* Simulation mit bekannter Wahrheit: Kaplan–Meier und Lognormal treffen das wahre Quantil in einem sehr großen Datensatz, die Verkäufe nicht.
"""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import znf_constants as C
import znf_core as K
import znf_estimators as E
import znf_model as M
import znf_scenario as S

scipy_optimize = pytest.importorskip("scipy.optimize")
scipy_stats = pytest.importorskip("scipy.stats")
scipy_integrate = pytest.importorskip("scipy.integrate")


# ------------------------------------------------------------------------------------------------ Umverteilung nach rechts (Bruch-Orakel)


def redistribute_survival(values, exact):
    """Umverteilung nach rechts (Verfahren nach Efron, Selbstkonsistenz; Quelle aus dem Gedächtnis genannt, nicht eingesehen): jede Periode hat zunächst die Masse 1/n. Der Reihe nach (aufsteigender Wert, bei gleichem Wert zuerst die genauen) gibt eine ausverkaufte
    Periode ihre Masse zu gleichen Teilen an alle **späteren** Perioden der Reihenfolge weiter (hat sie keine, bleibt die Masse bei ihr). S(x) = 1 − Summe der Massen der genauen Werte ≤ x.
    Gibt {genauer Wert: S(Wert)} in exakten Brüchen zurück."""
    n = len(values)
    order = sorted(range(n), key=lambda i: (values[i], 0 if exact[i] else 1, i))
    w = {i: F(1, n) for i in range(n)}
    for pos, i in enumerate(order):
        if not exact[i]:
            later = order[pos + 1:]
            if later:
                share = w[i] / len(later)
                for j in later:
                    w[j] += share
                w[i] = F(0)
    out, cum = {}, F(0)
    for x in sorted({values[i] for i in range(n) if exact[i]}):
        cum += sum(w[i] for i in range(n) if exact[i] and values[i] == x)
        out[x] = 1 - cum
    return out


def _random_censored(rng, n, p_exact):
    """n ganzzahlige Werte aus 1..9 (viele Gleichstände) mit zufälliger Zensierung."""
    values = [int(rng.uniform(1)[0] * 9) + 1 for _ in range(n)]
    exact = [bool(rng.uniform(1)[0] < p_exact) for _ in range(n)]
    return values, exact


def _oracle_quantile(surv, tau: F):
    """Kleinster genauer Wert mit S ≤ 1 − τ, exakt in Brüchen; None = nicht schätzbar."""
    for x, s in surv.items():
        if s <= 1 - tau:
            return x
    return None


def test_kaplan_meier_gegen_umverteilung_nach_rechts_in_bruechen():
    rng = K.SplitMix64(2024)
    checked = 0
    for r in range(250):
        n = int(rng.uniform(1)[0] * 12) + 2
        values, exact = _random_censored(rng, n, 0.3 + 0.6 * float(rng.uniform(1)[0]))
        if not any(exact):
            continue
        surv = redistribute_survival(values, exact)
        km = K.kaplan_meier(values, exact)
        assert km["t"].tolist() == list(surv), r
        assert km["s"].tolist() == pytest.approx([float(s) for s in surv.values()], abs=1e-12), r
        # Quantile: Kern, schnelle Fassung und Handtabelle gegen das Bruch-Orakel
        pairs = [(v, v) if not e else (v + 1, v) for v, e in zip(values, exact)]      # Bestand: Verkauf = Bestand heißt ausverkauft
        table = M.hand_table(pairs)
        for tau in (F(1, 2), F(4, 5), F(9, 10)):
            want = _oracle_quantile(surv, tau)
            got_core = K.quantile(km, float(tau))
            got_fast = E.km_quantile_fast(values, exact, float(tau))
            got_hand = M.hand_quantile(table, tau)
            for got in (got_core, got_fast):
                assert (math.isnan(got) and want is None) or got == want, (r, tau)
            assert got_hand == want, (r, tau)
        checked += 1
    assert checked == 237          # von 250 gezogenen Datensätzen haben 237 mindestens einen genauen Wert (README nennt beide Zahlen)


def test_handtabelle_der_demo_gegen_umverteilung_nach_rechts():
    for name, pairs in S.PATTERNS.items():
        values = [v for _, v in pairs]
        exact = [v < b for b, v in pairs]
        t = M.hand_table(pairs)
        if not any(exact):
            assert t["events"] == 0, name
            continue
        surv = redistribute_survival(values, exact)
        got = {r["value"]: r["s"] for r in t["rows"] if r["exact"]}
        assert got == surv, name


# ------------------------------------------------------------------------------------------------ zensiertes Lognormal


def _loglik(x, ev, m, s):
    """Log-Likelihood mit scipy.stats.norm geschrieben (nicht der Weg der Demo): Dichte der Lognormalverteilung für genaue, Überlebensfunktion für ausverkaufte Werte."""
    from scipy.stats import norm

    lx = np.log(x)
    z = (lx - m) / s
    return float(np.sum(norm.logpdf(z[ev]) - math.log(s) - lx[ev]) + np.sum(norm.logsf(z[~ev])))


def _scipy_fit(x, ev):
    from scipy.optimize import minimize

    lx = np.log(x)
    res = minimize(lambda p: -_loglik(x, ev, p[0], math.exp(p[1])), [float(np.mean(lx)) + 0.3, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 6000, "maxfev": 12000})
    return float(res.x[0]), math.exp(float(res.x[1])), -float(res.fun)


def test_lognormal_gegen_scipy_optimize_auf_selbst_geschriebener_likelihood():
    worlds = [S.World(m=m, n=n, seed=1000 + 17 * i) for i, (m, n) in enumerate([(70, 100), (100, 100), (100, 300), (150, 300), (200, 500), (60, 500), (100, 1000)])]
    checked = 0
    for w in worlds:
        for r in range(3):
            smp = S.make_sample(w, r)
            fit = E.lognormal_fit(smp.sales, smp.observed)
            assert fit["ok"]
            mu, sg, ll = _scipy_fit(smp.sales, smp.observed)
            assert fit["mu"] == pytest.approx(mu, abs=2e-5) and fit["sigma"] == pytest.approx(sg, rel=2e-5), (w, r)
            assert fit["loglik"] == pytest.approx(ll, abs=1e-7), (w, r)
            assert fit["loglik"] == pytest.approx(_loglik(smp.sales, smp.observed, fit["mu"], fit["sigma"]), abs=1e-9)   # die Demo-Likelihood ist die des Orakels
            checked += 1
    assert checked == 21


def test_lognormal_rastersuche_um_den_schaetzer_findet_nichts_besseres():
    for seed in (5, 6, 7):
        smp = S.make_sample(S.World(m=80.0, n=200, seed=seed), 0)
        fit = E.lognormal_fit(smp.sales, smp.observed)
        ms = fit["mu"] + np.linspace(-0.25, 0.25, 21)
        ss = fit["sigma"] * np.exp(np.linspace(-0.3, 0.3, 21))
        grid = np.array([[_loglik(smp.sales, smp.observed, m, s) for s in ss] for m in ms])
        assert fit["loglik"] >= grid.max() - 1e-9
        i, j = np.unravel_index(np.argmax(grid), grid.shape)
        assert abs(ms[i] - fit["mu"]) <= 0.0251 and abs(math.log(ss[j] / fit["sigma"])) <= 0.0301      # der beste Rasterpunkt liegt neben dem Schätzer (ein Rasterschritt)


def test_lognormal_gradient_am_schaetzer_ist_null_ausser_rundung():
    smp = S.make_sample(S.World(m=100.0, n=500, seed=3), 0)
    fit = E.lognormal_fit(smp.sales, smp.observed)
    _, g = E._lognormal_nll(np.array([fit["mu"], math.log(fit["sigma"])]), np.log(smp.sales), smp.observed)
    assert np.max(np.abs(g)) / smp.sales.size < 5e-7


def test_lognormal_ohne_ausverkauf_gleich_dem_geschlossenen_fall_in_vielen_datensaetzen():
    for r in range(20):
        smp = S.make_sample(S.World(m=1e6, n=80, seed=40), r)         # Bestand so hoch, dass nie ausverkauft ist
        assert smp.observed.all()
        fit = E.lognormal_fit(smp.sales, smp.observed)
        lx = np.log(smp.sales)
        assert fit["mu"] == pytest.approx(lx.mean(), abs=1e-5) and fit["sigma"] == pytest.approx(lx.std(), rel=1e-5), r


# ------------------------------------------------------------------------------------------------ Kosten gegen Integration und Simulation


def _dist(form):
    from scipy.stats import gamma, lognorm

    return lognorm(C.DEMAND_SIGMA, scale=C.DEMAND_MEDIAN) if form == "lognormal" else gamma(S.gamma_shape(), scale=S.gamma_scale())


def _quad_cost(q, tau, form):
    from scipy.integrate import quad

    d = _dist(form)
    over = quad(lambda x: (x - q) * d.pdf(x), q, math.inf, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
    under = quad(lambda x: (q - x) * d.pdf(x), 0.0, q, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
    return tau * over + (1 - tau) * under


def test_erwartete_kosten_gegen_numerische_integration():
    for form in C.FORMS:
        for tau in (0.5, 0.8, 0.9):
            for q in (60.0, 100.0, 140.0, 190.0, 260.0):
                assert float(M.expected_cost(q, tau, form)) == pytest.approx(_quad_cost(q, tau, form), rel=1e-7), (form, tau, q)


def test_beste_menge_gegen_numerische_minimierung_der_integrierten_kosten():
    from scipy.optimize import minimize_scalar

    for form in C.FORMS:
        for tau in (0.5, 0.8, 0.95):
            res = minimize_scalar(lambda q: _quad_cost(q, tau, form), bounds=(20.0, 500.0), method="bounded", options={"xatol": 1e-6})
            assert res.x == pytest.approx(M.true_quantile(form, tau), rel=2e-4), (form, tau)
            assert M.true_quantile(form, tau) == pytest.approx(float(_dist(form).ppf(tau)), rel=1e-10)


def test_erwartete_kosten_gegen_simulation_mit_band():
    for form in C.FORMS:
        d = S.draw_demand(K.SplitMix64(99), 1_000_000, form)
        for tau, q in ((0.8, 120.0), (0.9, 185.0)):
            sample_cost = np.mean(tau * np.maximum(d - q, 0.0) + (1 - tau) * np.maximum(q - d, 0.0))
            se = np.std(tau * np.maximum(d - q, 0.0) + (1 - tau) * np.maximum(q - d, 0.0)) / math.sqrt(d.size)
            assert abs(sample_cost - float(M.expected_cost(q, tau, form))) < 4 * se, (form, tau)


# ------------------------------------------------------------------------------------------------ Simulation mit bekannter Wahrheit


def test_grosser_datensatz_kaplan_meier_und_lognormal_treffen_die_wahrheit_die_verkaeufe_nicht():
    smp = S.simulate(400_000, 100.0, "lognormal", 0.0, 77)
    assert 0.45 < 1 - smp.observed.mean() < 0.55
    for tau in (0.5, 0.8):
        q_true = M.true_quantile("lognormal", tau)
        assert E.km_quantile_fast(smp.sales, smp.observed, tau) == pytest.approx(q_true, rel=0.01), tau
        assert E.sales_quantile(smp.sales, tau) < 0.85 * q_true, tau
    fit = E.lognormal_fit(smp.sales, smp.observed)
    assert fit["mu"] == pytest.approx(math.log(100.0), abs=0.01) and fit["sigma"] == pytest.approx(0.5, abs=0.01)
    assert E.lognormal_quantile(fit, 0.9) == pytest.approx(M.true_quantile("lognormal", 0.9), rel=0.02)


def test_grosser_datensatz_gamma_kaplan_meier_trifft_das_lognormal_verfehlt_die_form():
    smp = S.simulate(400_000, 100.0, "gamma", 0.0, 78)
    q_true = M.true_quantile("gamma", 0.9)
    assert E.km_quantile_fast(smp.sales, smp.observed, 0.9) == pytest.approx(q_true, rel=0.02)      # 400 000 Perioden: das Quantil 0,9 ist schätzbar
    fit = E.lognormal_fit(smp.sales, smp.observed)
    assert fit["ok"] and abs(E.lognormal_quantile(fit, 0.9) / q_true - 1) > 0.03                  # auch mit unendlich vielen Daten bleibt die falsche Form falsch
