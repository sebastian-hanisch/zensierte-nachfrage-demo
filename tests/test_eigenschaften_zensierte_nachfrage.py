"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Aussagen, die für das Verfahren gelten müssen, ohne dass das richtige Ergebnis bekannt ist.

Jede Eigenschaft nennt ihre Voraussetzung; sie sind Aussagen über das Verfahren (Kaplan–Meier mit Ausverkauf als Zensierung, Verkaufsquantil, zensiertes Lognormal), nicht
nur über das heutige Verhalten des Codes. Je Eigenschaft 30 bis 60 Datensätze mit festem Zufalls-Seed."""

import math

import numpy as np
import pytest

import znf_core as K
import znf_estimators as E
import znf_model as M
import znf_scenario as S

TAUS = (0.5, 0.8, 0.9)


def _worlds(n_worlds=12, n=150):
    """Verschiedene Welten (Bestandsniveau, Verteilung, Zufalls-Seed), je ein Datensatz."""
    out = []
    for i in range(n_worlds):
        w = S.World(m=[70.0, 100.0, 150.0][i % 3], n=n, form=["lognormal", "gamma"][i % 2], seed=500 + i)
        out.append((w, S.make_sample(w, 0)))
    return out


def test_ohne_ausverkauf_ist_die_kurve_die_empirische_verteilung_und_das_quantil_das_untere_quantil():
    """Voraussetzung: nie ausverkauft (Bestand weit über jeder Nachfrage). Dann ist Kaplan–Meier gleich 1 − empirische Verteilungsfunktion der Nachfrage, und das Quantil ist
    das untere empirische Quantil (kleinster Wert mit F ≥ τ, `inverted_cdf`); das Verkaufsquantil mit Interpolation liegt zwischen zwei benachbarten Werten."""
    for r in range(40):
        smp = S.simulate(60 + 3 * r, 1e6, "lognormal", 0.0, 900 + r)
        assert smp.observed.all()
        km = K.kaplan_meier(smp.sales, smp.observed)
        for x in np.quantile(smp.sales, [0.1, 0.5, 0.9]):
            assert K.survival_at(km, float(x)) == pytest.approx(float(np.mean(smp.sales > x)), abs=1e-12)
        for tau in TAUS:
            assert E.km_quantile_fast(smp.sales, smp.observed, tau) == float(np.quantile(smp.sales, tau, method="inverted_cdf"))
            lo, hi = np.quantile(smp.sales, tau, method="lower"), np.quantile(smp.sales, tau, method="higher")
            assert lo <= E.sales_quantile(smp.sales, tau) <= hi


def test_ohne_ausverkauf_haengt_die_kurve_nicht_vom_bestand_ab():
    """Voraussetzung: bei beiden Bestandsniveaus nie ausverkauft. Dann sind die Verkäufe die Nachfrage, und die Kurve ändert sich nicht, wenn der Bestand noch höher gesetzt wird."""
    for r in range(30):
        a = S.simulate(100, 1e6, "lognormal", 0.0, 700 + r)
        b = S.simulate(100, 1e9, "lognormal", 0.0, 700 + r)
        assert a.observed.all() and b.observed.all()
        ka, kb = K.kaplan_meier(a.sales, a.observed), K.kaplan_meier(b.sales, b.observed)
        assert np.array_equal(ka["t"], kb["t"]) and np.array_equal(ka["s"], kb["s"])


def test_kaplan_meier_quantil_liegt_nie_ueber_dem_groessten_beobachteten_wert():
    for w, smp in _worlds(24, 120):
        exact_max = float(smp.sales[smp.observed].max()) if smp.observed.any() else float("nan")
        for tau in (0.5, 0.8, 0.9, 0.95, 0.99):
            q = E.km_quantile_fast(smp.sales, smp.observed, tau)
            assert math.isnan(q) or q <= exact_max
            assert math.isnan(q) or q <= float(smp.sales.max())


def test_quantil_aus_verkaeufen_liegt_nie_ueber_dem_quantil_der_nachfrage_im_selben_datensatz():
    """Voraussetzung: Verkauf = min(Nachfrage, Bestand) ≤ Nachfrage in jeder Periode. Dann gilt das für jedes empirische Quantil (Ordnungsstatistiken sind monoton)."""
    for w, smp in _worlds(30, 200):
        for tau in (0.1, 0.5, 0.8, 0.95):
            assert E.sales_quantile(smp.sales, tau) <= float(np.quantile(smp.demand, tau)) + 1e-12


def test_quantile_wachsen_mit_tau_und_nicht_schaetzbar_bleibt_nicht_schaetzbar():
    taus = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    for w, smp in _worlds(24, 150):
        fit = E.lognormal_fit(smp.sales, smp.observed)
        sales = [E.sales_quantile(smp.sales, t) for t in taus]
        km = [E.km_quantile_fast(smp.sales, smp.observed, t) for t in taus]
        ln = [E.lognormal_quantile(fit, t) for t in taus]
        assert sales == sorted(sales) and ln == sorted(ln)
        finite = [q for q in km if math.isfinite(q)]
        assert finite == sorted(finite)
        # einmal nicht schätzbar, dann für jedes größere τ nicht schätzbar
        first_nan = next((i for i, q in enumerate(km) if math.isnan(q)), len(km))
        assert all(math.isnan(q) for q in km[first_nan:])


def test_skalieren_von_bestand_und_nachfrage_skaliert_alle_quantile():
    """Voraussetzung: alle Mengen mit c > 0 multipliziert (andere Einheit, z. B. Paletten statt Stück). Kaplan–Meier und Verkaufsquantil skalieren mit c, das Lognormal mit μ + ln c."""
    for w, smp in _worlds(18, 150):
        for c in (0.01, 3.0, 40.0):
            s2 = smp.sales * c
            for tau in TAUS:
                assert E.sales_quantile(s2, tau) == pytest.approx(c * E.sales_quantile(smp.sales, tau), rel=1e-12)
                a, b = E.km_quantile_fast(smp.sales, smp.observed, tau), E.km_quantile_fast(s2, smp.observed, tau)
                assert (math.isnan(a) and math.isnan(b)) or b == pytest.approx(c * a, rel=1e-12)
            f1, f2 = E.lognormal_fit(smp.sales, smp.observed), E.lognormal_fit(s2, smp.observed)
            assert f2["mu"] == pytest.approx(f1["mu"] + math.log(c), abs=2e-6) and f2["sigma"] == pytest.approx(f1["sigma"], rel=1e-5)


def test_reihenfolge_der_perioden_aendert_nichts():
    rng = K.SplitMix64(31)
    for w, smp in _worlds(18, 150):
        perm = np.argsort(rng.uniform(smp.sales.size))
        s2, o2 = smp.sales[perm], smp.observed[perm]
        for tau in TAUS:
            a, b = E.km_quantile_fast(smp.sales, smp.observed, tau), E.km_quantile_fast(s2, o2, tau)
            assert (math.isnan(a) and math.isnan(b)) or a == b
            assert E.sales_quantile(s2, tau) == E.sales_quantile(smp.sales, tau)
        assert E.lognormal_fit(s2, o2)["mu"] == pytest.approx(E.lognormal_fit(smp.sales, smp.observed)["mu"], abs=1e-6)


def test_hoeherer_bestand_gepaart_senkt_den_ausverkauf_und_hebt_das_verkaufsquantil():
    """Voraussetzung: gepaarte Welten (gleiche Nachfrage, gleicher Zufall im Bestand, nur der Median des Bestands wächst). Dann steigt jeder Verkauf min(D, B) schwach, also auch jedes
    Verkaufsquantil, und der Anteil ausverkaufter Perioden sinkt schwach."""
    for seed in range(40):
        prev_sold, prev_q = 2.0, {t: -1.0 for t in TAUS}
        for m in (60.0, 80.0, 100.0, 130.0, 170.0):
            smp = S.simulate(200, m, "lognormal", 0.0, 3000 + seed)
            sold = 1.0 - smp.observed.mean()
            assert sold <= prev_sold
            for t in TAUS:
                q = E.sales_quantile(smp.sales, t)
                assert q >= prev_q[t]
                prev_q[t] = q
            prev_sold = sold


def test_gleicher_zufalls_seed_gleiche_ausgabe_und_laeufe_sind_unabhaengig_von_der_reihenfolge():
    w = S.World(m=100.0, n=120, seed=8)
    a = M.run_study(w, (0.8, 0.9), reps=6)
    b = M.run_study(w, (0.8, 0.9), reps=6)
    for name in M.METHODS:
        assert np.array_equal(a.q[name], b.q[name], equal_nan=True)
    # Lauf 3 allein gerechnet ist derselbe wie Lauf 3 im Block
    smp = S.make_sample(w, 3)
    assert E.sales_quantile(smp.sales, 0.8) == a.q["sales"][3, 0]
    assert E.km_quantile_fast(smp.sales, smp.observed, 0.9) == pytest.approx(a.q["km"][3, 1], nan_ok=True)


def test_mehrkosten_sind_nie_negativ_und_wachsen_mit_dem_abstand_zur_besten_menge():
    for form in ("lognormal", "gamma"):
        for tau in (0.5, 0.8, 0.9):
            q = M.true_quantile(form, tau)
            below = [float(M.extra_cost(q * f, tau, form)) for f in (0.9, 0.8, 0.6, 0.4)]
            above = [float(M.extra_cost(q * f, tau, form)) for f in (1.1, 1.3, 1.6, 2.0)]
            assert min(below + above) > 0 and below == sorted(below) and above == sorted(above)
