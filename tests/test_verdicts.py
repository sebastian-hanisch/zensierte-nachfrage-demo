"""Verdikte (reine Funktionen in znf_evaluation.py) an handgesetzten Werten: Schwellen, jeder Zweig und jede Zahl im Text (DEMO-PLAYBOOK §12)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest
from conftest import HAND_PAIRS

import znf_core as K
import znf_evaluation as ev
import znf_model as M
import znf_scenario as S


def _study(form="lognormal", tau=0.9, sales=None, km=None, fb=None, ln=None, sold=0.5, boost=0.0):
    """Ein Study mit einem τ und handgesetzten Bestellmengen (je Lauf eine); fehlende Verfahren bekommen in jedem Lauf die beste Menge. Zahl der Läufe: Länge der ersten angegebenen Liste (sonst 4)."""
    q_true = M.true_quantile(form, tau)
    reps = len(next((v for v in (sales, km, fb, ln) if v is not None), [0] * 4))

    def col(values):
        return np.array([q_true] * reps if values is None else list(values), dtype=float).reshape(-1, 1)

    q = {"sales": col(sales), "km": col(km), "km_fallback": col(fb), "lognormal": col(ln)}
    return M.Study(S.World(m=100.0, form=form, boost=boost), (tau,), np.array([q_true]), q, np.full(reps, sold), np.ones(reps, bool), np.zeros(reps))


# ------------------------------------------------------------------------------------------------ Formatierung


def test_formatierung():
    assert ev.share_pct(583, 1000) == "58 %" and ev.share_pct(995, 1000) == "99,5 %" and ev.share_pct(4, 1000) == "0,4 %"
    assert ev.share_pct(0, 1000) == "0 %" and ev.share_pct(1000, 1000) == "100 %"
    assert ev.signed_pct(-0.368) == "−37 %" and ev.signed_pct(0.012) == "+1 %" and ev.signed_pct(0.0003) == "±0 %"
    assert ev.pct(0.5) == "50,0 %" and ev.num(119.246) == "119,2" and ev.tau_text(0.95) == "0,95" and ev.tau_text(0.5) == "0,5"
    assert ev.frac(F(4, 9)) == "4/9" and ev.frac(F(2, 1)) == "2" and ev.frac_dec(F(4, 9)) == "4/9 ≈ 0,44" and ev.frac_dec(F(9)) == "9"


# ------------------------------------------------------------------------------------------------ Experiment 1


def test_verdict_hand_quantil_schaetzbar():
    kind, text = ev.verdict_hand(M.hand_table(HAND_PAIRS), F(1, 2))
    assert kind == "info"
    assert "Quantil 0,5 auf **9**" in text and "17/2 ≈ 8,50" in text and "3 von 6 Perioden ausverkauft" in text


def test_verdict_hand_quantil_nicht_schaetzbar_nennt_plateau_und_erreichbares_tau():
    kind, text = ev.verdict_hand(M.hand_table(HAND_PAIRS), F(4, 5))
    assert kind == "warning" and "**nicht schätzbar**" in text
    assert "bis 4/9 ≈ 0,44" in text and "τ = 5/9 ≈ 0,56" in text and "Aus den Verkäufen ergibt sich 10 (3 von 6 Perioden ausverkauft)" in text


def test_verdict_hand_alle_ausverkauft():
    kind, text = ev.verdict_hand(M.hand_table(S.PATTERNS["Immer ausverkauft"]), F(1, 2))
    assert kind == "error" and "Alle 6 Perioden sind ausverkauft" in text and "**nicht schätzbar**" in text and "ergibt sich 10" in text


def test_verdict_hand_nie_ausverkauft():
    kind, text = ev.verdict_hand(M.hand_table(S.PATTERNS["Nie ausverkauft"]), F(1, 2))
    assert kind == "success" and "Keine Periode ist ausverkauft" in text and "**7**" in text and "15/2 ≈ 7,50" in text


# ------------------------------------------------------------------------------------------------ Experiment 2


def test_verdict_quantile_warnung_bei_wenigen_schaetzbaren_laeufen():
    q = M.true_quantile("lognormal", 0.9)
    st = _study(sales=[120.0] * 4, km=[180.0, 190.0, math.nan, math.nan])
    kind, text = ev.verdict_quantile(st, 0.9)
    assert kind == "warning"
    assert "Bei 50 % ausverkauften Perioden" in text and "**120,0**" in text and f"Wahrheit {ev.num(q)}" in text and "(−37 %)" in text
    assert "in 2 von 4 Läufen (50 %) eine Zahl" in text and "185,0 (−3 %)" in text and "erreicht die Kurve das Niveau nicht" in text


def test_verdict_quantile_erfolg_wenn_alles_schaetzbar_und_nah():
    q = M.true_quantile("lognormal", 0.5)
    st = _study(tau=0.5, sales=[q * 0.98] * 4, km=[q] * 4)
    kind, text = ev.verdict_quantile(st, 0.5)
    assert kind == "success" and "in 4 von 4 Läufen (100 %) eine Zahl" in text


def test_verdict_quantile_ohne_schaetzbaren_lauf():
    st = _study(sales=[100.0] * 4, km=[math.nan] * 4)
    kind, text = ev.verdict_quantile(st, 0.9)
    assert kind == "warning" and "in keinem Lauf erreicht die Kurve das Niveau" in text and "in 0 von 4 Läufen (0 %) eine Zahl" in text


# ------------------------------------------------------------------------------------------------ Experiment 3


def _plateau(floor, last_exact, sold_out_max, sold=0.5, n=500):
    return {"floor": floor, "last_exact": last_exact, "largest": 300.0, "largest_sold_out": sold_out_max, "sold_out": sold, "n": n}


def test_verdict_plateau_zweige():
    kind, text = ev.verdict_plateau(_plateau(0.12, 250.0, True), 0.9)
    assert kind == "warning" and "bis S = 0,120 (beim Wert 250,0)" in text and "nur Quantile bis τ = 0,880" in text and "τ = 0,9 liegt darüber" in text and "**nicht schätzbar**" in text
    assert "Der größte Verkauf ist ein Ausverkauf" in text and "500 Perioden, 50 % ausverkauft" in text
    kind, text = ev.verdict_plateau(_plateau(0.12, 250.0, True), 0.8)
    assert kind == "success" and "das gewählte τ = 0,8 gehört dazu" in text
    kind, text = ev.verdict_plateau(_plateau(0.0, 212.7, False), 0.95)
    assert kind == "success" and "jedes Quantil schätzbar" in text and "genau beobachtete Nachfrage" in text
    kind, text = ev.verdict_plateau(_plateau(1.0, math.nan, True, sold=1.0), 0.5)
    assert kind == "error" and "Keine Periode hat die Nachfrage genau gezeigt" in text


def test_verdict_plateau_grenzfall_tau_genau_am_plateau():
    # S = 0,1 und τ = 0,9: 1 − τ = 0,1 gleicht dem Plateau, das Quantil ist erreichbar (Kern: S ≤ 1 − τ)
    kind, _ = ev.verdict_plateau(_plateau(0.1, 250.0, True), 0.9)
    assert kind == "success"


def test_plateau_info_aus_einem_datensatz():
    pairs = HAND_PAIRS
    sales = [v for _, v in pairs]
    observed = [v < b for b, v in pairs]
    km = K.kaplan_meier(sales, observed)
    info = ev.plateau_info(km, sales, observed)
    assert info["floor"] == pytest.approx(4 / 9) and info["last_exact"] == 9.0 and info["largest"] == 12.0 and info["largest_sold_out"] is True
    assert info["sold_out"] == 0.5 and info["n"] == 6


def test_verdict_boundary_schwellen():
    for km, kind, needle in (
        ([100.0] * 1000, "success", "praktisch jedem Lauf"),
        ([math.nan] * 1000, "error", "praktisch keinem Lauf"),
        ([100.0] * 500 + [math.nan] * 500, "warning", "bleibt die Kurve über 1 − τ stehen"),
    ):
        st = _study(sales=[100.0] * 1000, km=km, boost=0.1)
        k, text = ev.verdict_boundary(st, 0.9, 0.1)
        assert k == kind and needle in text and "Bei 10 % Testperioden" in text
    k, text = ev.verdict_boundary(_study(sales=[100.0] * 1000, km=[100.0] * 995 + [math.nan] * 5), 0.9, 0.0)
    assert k == "success" and "995 von 1000 Läufen (99,5 %)" in text          # 99,5 % ist die Schwelle „praktisch jeder“


# ------------------------------------------------------------------------------------------------ Experiment 4


def test_verdict_costs_lognormal_welt_mit_gewinner():
    q = M.true_quantile("lognormal", 0.9)
    st = _study(sales=[0.5 * q] * 4, km=[q, q, math.nan, math.nan], fb=[1.1 * q] * 4, ln=[q] * 4)
    kind, text = ev.verdict_costs(st, 0.9)
    sales_cost = float(M.extra_cost(0.5 * q, 0.9, "lognormal"))
    assert kind == "warning" and f"**{ev.pct(sales_cost, 1)}**" in text and f"das {ev.num(1 + sales_cost)}-Fache" in text
    assert "Nachfrage Lognormal, 50 % der Perioden ausverkauft" in text and "zensiertes Lognormal: 0,0 %" in text
    assert "nur in 50 % der Läufe schätzbar (0,0 % in diesen)" in text and "seine Form passt zur erzeugten Nachfrage" in text


def test_verdict_costs_gamma_welt_falsche_form():
    q = M.true_quantile("gamma", 0.9)
    st = _study(form="gamma", sales=[0.5 * q] * 4, km=[q] * 4, fb=[q * 1.02] * 4, ln=[q * 0.7] * 4)
    kind, text = ev.verdict_costs(st, 0.9)
    assert kind == "warning" and "Nachfrage Gamma (gleiche Streuung)" in text and "Die Form passt hier nicht" in text and "bleibt es günstiger" in text and "nicht günstiger als die Verkäufe" not in text
    assert "seine Form passt zur erzeugten Nachfrage" not in text


def test_verdict_costs_info_bei_kleinen_mehrkosten_und_ohne_schaetzbaren_lauf():
    q = M.true_quantile("lognormal", 0.5)
    st = _study(tau=0.5, sales=[q * 0.99] * 4, km=[math.nan] * 4, fb=[q] * 4, ln=[q * 1.001] * 4)
    kind, text = ev.verdict_costs(st, 0.5)
    assert kind == "info" and "keine Zahl in diesen" in text and "nur in 0 % der Läufe schätzbar" in text


def test_example_run_waehlt_den_ersten_lauf_mit_plateau_sonst_lauf_null():
    w = S.World(m=100.0, n=300)
    r = ev.example_run(w)
    smp = S.make_sample(w, r)
    assert not smp.observed[int(np.argmax(smp.sales))]                      # der größte Verkauf ist ein Ausverkauf: die Kurve bleibt auf einem Plateau stehen
    assert all(S.make_sample(w, k).observed[int(np.argmax(S.make_sample(w, k).sales))] for k in range(r))       # alle früheren Läufe haben keins
    info = ev.plateau_info(K.kaplan_meier(smp.sales, smp.observed), smp.sales, smp.observed)
    assert info["floor"] > 0.0 and info["largest_sold_out"] is True
    assert ev.example_run(S.World(m=1e6, n=300)) == 0                          # nie ausverkauft: es gibt kein Plateau, Rückfall auf Lauf 0


def test_verdict_costs_gamma_lognormal_nicht_guenstiger_als_die_verkaeufe_und_ohne_anpassung():
    q = M.true_quantile("gamma", 0.9)
    st = _study(form="gamma", sales=[q * 0.9] * 4, km=[q] * 4, fb=[q * 1.02] * 4, ln=[q * 0.5] * 4)
    kind, text = ev.verdict_costs(st, 0.9)
    assert kind == "warning" and "und nicht günstiger als die Verkäufe" in text and "bleibt es günstiger" not in text
    st = _study(sales=[q * 0.5] * 4, km=[q] * 4, fb=[q] * 4, ln=[math.nan] * 4)
    kind, text = ev.verdict_costs(st, 0.9)
    assert "zensiertes Lognormal: keine Anpassung möglich" in text and "am günstigsten" not in text
