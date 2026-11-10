"""Jede Zahl der README steht in einer Messung (`tools/measure_zensierte_nachfrage_result.json`, eingefroren, feste Zufalls-Seeds) oder in einem Test.

Dieser Test prüft viererlei:
  1. Alle Messzellen werden mit dem Code der Demo neu gerechnet und stimmen mit der eingefrorenen Datei überein (Band 1e-6, Zählungen exakt).
  2. Die Tabellen der README stehen wörtlich so in der README, wie `tools/measure_zensierte_nachfrage.tables()` sie aus der Datei bildet.
  3. Die Prosa-Zahlen der README (Mittel, Zählungen, Spannen) folgen aus denselben Zahlen, jeweils mit ihren Voraussetzungen; die Zahlen der Vormessung stehen als Vergleich.
  4. Die Aussagen der Preset-Hilfen, der Grenzen-Tabelle der App und des Annahmen-Textes gelten gegen die Messzellen; der Annahmen-Text steht gleich in App und README."""

import json
from fractions import Fraction as F
from pathlib import Path

import measure_zensierte_nachfrage as MC
import pytest

import znf_constants as C
import znf_evaluation as ev
import znf_model as M
import znf_scenario as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
DATA = json.loads((ROOT / "tools" / "measure_zensierte_nachfrage_result.json").read_text(encoding="utf-8"))
CELLS = DATA["cells"]
STD = CELLS["standard"]


def _same(a, b, path=""):
    """Rekursiver Vergleich: Zahlen mit Band, Ganzzahlen und Texte exakt."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert b == pytest.approx(a, rel=1e-6, abs=1e-9), (path, a, b)
    else:
        assert a == b, (path, a, b)


def _m(cell, tau, method):
    return cell["taus"][f"{tau:g}"]["methods"][method]


def _cost(cell, tau, method):
    return _m(cell, tau, method)["cost_mean"]


@pytest.fixture(scope="module")
def fresh():
    return MC.measure()


# ------------------------------------------------------------------------------------------------ 1. neu gerechnet


def test_alle_messzellen_neu_gerechnet_stimmen_mit_der_eingefrorenen_datei(fresh):
    _same(DATA, fresh)


# ------------------------------------------------------------------------------------------------ 2. Tabellen und Texte


def test_alle_tabellen_der_readme_stehen_woertlich_in_der_readme():
    for name, table in MC.tables(DATA).items():
        assert table in README, name


def test_annahmen_text_steht_gleich_in_app_und_readme():
    assert C.ASSUMPTIONS_MARKDOWN.rstrip() in README
    assert "C.ASSUMPTIONS_MARKDOWN" in APP


def test_readme_nennt_die_annahme_des_unabhaengigen_bestands_und_die_linie():
    assert "**Annahme in dieser Demo, sichtbar in der App und hier: Der Bestand einer Periode ist unabhängig von ihrer Nachfrage.**" in README and "**Überlebenszeit**" in README
    assert "Der Bestand einer Periode ist unabhängig von ihrer Nachfrage" in APP


def test_nachbarn_sind_nur_verlinkt_und_zahlen_der_nachbarn_stehen_nicht_hier():
    for repo in ("forecast-inventory-demo", "newsvendor-direkt-demo", "informatives-zensieren-demo"):
        assert f"](https://github.com/sebastian-hanisch/{repo})" in README and repo in APP
    assert "import forecast" not in APP and "import newsvendor" not in APP


# ------------------------------------------------------------------------------------------------ 3. Prosa-Zahlen


def test_standardwelt_zahlen_des_abschnitts_warum():
    assert (STD["m"], STD["n"], STD["form"], STD["boost"], STD["seed"], STD["reps"]) == (100, 500, "lognormal", 0.0, 35, 1000)
    assert ev.pct(STD["sold_out_mean"]) == "50,0 %"
    t = STD["taus"]["0.9"]
    assert ev.num(t["q_true"]) == "189,8" and ev.num(t["methods"]["sales"]["q_mean"]) == "119,2" and ev.signed_pct(t["methods"]["sales"]["q_mean"] / t["q_true"] - 1) == "−37 %"
    km = t["methods"]["km"]
    assert km["defined"] == 583 and ev.share_pct(km["defined"], STD["reps"]) == "58 %" and ev.num(km["q_mean"]) == "179,9" and ev.signed_pct(km["q_mean"] / t["q_true"] - 1) == "−5 %"
    assert ev.num(t["methods"]["sales"]["q_se"], 2) == "0,08" and ev.num(km["q_se"], 2) == "0,78"
    assert ev.pct(_cost(STD, 0.9, "sales")) == "56,0 %" and ev.num(1 + _cost(STD, 0.9, "sales")) == "1,6"
    m70 = CELLS["m70"]
    assert ev.pct(m70["sold_out_mean"]) == "73,0 %" and ev.pct(_cost(m70, 0.9, "sales")) == "128,9 %" and ev.num(1 + _cost(m70, 0.9, "sales")) == "2,3"
    assert ev.pct(_cost(STD, 0.9, "km_fallback")) == "3,0 %" and ev.pct(_cost(STD, 0.9, "lognormal")) == "0,6 %"
    for needle in ("Zufalls-Seed 35", "50,0 %", "189,8", "119,2 heraus (−37 %)", "583 von 1000 Läufen (58 %)", "179,9 (−5 %)", "56,0 % mehr", "das 1,6-Fache", "128,9 % (das 2,3-Fache)", "73,0 %"):
        assert needle in README, needle
    taus = STD["taus"]
    gaps = [taus[f"{t:g}"]["methods"]["sales"]["q_mean"] / taus[f"{t:g}"]["q_true"] - 1 for t in C.TAU_OPTIONS]
    assert ev.signed_pct(min(gaps)) == "−42 %" and ev.signed_pct(max(gaps)) == "−19 %" and gaps == sorted(gaps, reverse=True)
    assert "zwischen −19 % und −42 %" in README


def test_die_verkaufsregel_ist_in_jeder_zelle_unter_dem_wahren_quantil_und_teurer_als_alle_anderen():
    for key, cell in CELLS.items():
        for tau in C.TAU_OPTIONS:
            t = cell["taus"][f"{tau:g}"]
            assert t["methods"]["sales"]["q_mean"] < t["q_true"], (key, tau)
    lognormal_cells = [CELLS["m70"], STD, CELLS["m150"]]
    for cell in lognormal_cells + [CELLS["gamma"]]:
        for tau in C.GRID_TAU:
            assert all(_cost(cell, tau, "sales") > _cost(cell, tau, other) for other in ("km", "km_fallback", "lognormal")), (cell["m"], tau)
    assert "Die Verkaufsregel ist in allen 9 Zellen die teuerste" in README


def test_lognormal_ist_in_der_lognormal_welt_in_jeder_zelle_am_guenstigsten():
    n_best = 0
    for cell in (CELLS["m70"], STD, CELLS["m150"]):
        for tau in C.GRID_TAU:
            costs = {m: _cost(cell, tau, m) for m in M.METHODS}
            n_best += costs["lognormal"] == min(costs.values())
    assert n_best == 9
    max_se = max(_m(c, t, "lognormal")["cost_se"] for c in (CELLS["m70"], STD, CELLS["m150"]) for t in C.GRID_TAU)
    assert ev.num(100 * max_se, 2) == "0,06"
    assert "in allen 9 am günstigsten (Standardfehler seiner Mehrkosten höchstens 0,06 Prozentpunkte)" in README
    assert "in allen 9 Zellen am günstigsten" in README and "in 9 von 9 Zellen am günstigsten" in README


def test_kosten_wachsen_mit_ausverkauf_und_tau_bei_der_verkaufsregel():
    sold = [(CELLS[k]["sold_out_mean"], k) for k in ("m150", "standard", "m70")]
    assert sold == sorted(sold)
    for tau in C.GRID_TAU:
        costs = [_cost(CELLS[k], tau, "sales") for k in ("m150", "standard", "m70")]
        assert costs == sorted(costs)                                  # mehr Ausverkauf, mehr Mehrkosten
    for key in ("m70", "standard", "m150"):
        costs = [_cost(CELLS[key], tau, "sales") for tau in C.GRID_TAU]
        assert costs == sorted(costs)                                  # höheres τ, mehr Mehrkosten
    assert ev.num(100 * _cost(CELLS["m70"], 0.5, "sales")) == "25,6" and ev.num(100 * _cost(CELLS["m70"], 0.9, "sales")) == "128,9" and ev.num(100 * _cost(STD, 0.5, "sales")) == "6,9"
    assert "von 25,6 % bei $\\tau$ = 0,5 auf 128,9 % bei $\\tau$ = 0,9" in README and "Bei $\\tau$ = 0,5 und Bestand 100 sind es 6,9 %" in README


def test_gamma_welt_lognormal_ist_bei_hohem_tau_teurer_als_kaplan_meier_mit_rueckfall():
    g = CELLS["gamma"]
    assert g["form"] == "gamma" and ev.pct(g["sold_out_mean"]) == "49,2 %"
    ln_worse = [tau for tau in C.GRID_TAU if _cost(g, tau, "lognormal") > _cost(g, tau, "km_fallback")]
    assert ln_worse == [0.8, 0.9]
    assert [ev.num(100 * _cost(g, t, "lognormal")) for t in (0.8, 0.9)] == ["3,4", "10,6"] and [ev.num(100 * _cost(g, t, "km_fallback")) for t in (0.8, 0.9)] == ["2,2", "3,2"]
    assert [ev.num(100 * _cost(g, t, "sales")) for t in (0.8, 0.9)] == ["31,9", "62,5"] and [ev.num(100 * _cost(g, t, "km")) for t in (0.8, 0.9)] == ["1,5", "3,1"]
    assert ev.num(_cost(g, 0.9, "lognormal") / _cost(g, 0.9, "km_fallback")) == "3,3"             # „das 3,3-Fache“ in der README
    assert "das 3,3-Fache der Mehrkosten von Kaplan–Meier mit Rückfall (10,6 % gegen 3,2 %)" in README and "bei τ = 0,8 und 0,9 hinter Kaplan–Meier mit Rückfall" in APP
    assert all(_cost(g, t, "lognormal") < _cost(g, t, "sales") / 5 for t in C.GRID_TAU)            # weit unter der Verkaufsregel: höchstens ein Fünftel
    assert _cost(g, 0.5, "lognormal") < _cost(g, 0.5, "km_fallback") and max(_cost(g, 0.5, m) for m in ("km", "km_fallback", "lognormal")) < 0.003
    assert (round(100 * _cost(g, 0.5, "lognormal"), 2), round(100 * _cost(g, 0.5, "km_fallback"), 2)) == (0.19, 0.22)
    for needle in ("(0,19 % gegen 0,22 %, Unterschied etwa 3 Standardfehler)", "3,4 % gegen 2,2 % und 10,6 % gegen 3,2 %", "(31,9 % und 62,5 %)", "49,2 % ausverkauft", "1,5 % und 3,1 %", "bei $\\tau$ = 0,5 liegen beide gleichauf (0,19 % gegen 0,22 %, Unterschied etwa 3 Standardfehler)","(höchstens ein Fünftel ihrer Mehrkosten)"):
        assert needle in README, needle


def test_anteil_schaetzbarer_laeufe_je_bestandsniveau_und_tau():
    by_m = [[_m(CELLS[k], t, "km")["defined"] for t in (0.9, 0.95)] for k in ("m70", "standard", "m150")]
    assert [x[0] for x in by_m] == [251, 583, 995] and [x[1] for x in by_m] == [226, 330, 813]
    assert "in 251 von 1000 Läufen (Bestand 70),\n583 (Bestand 100) und 995 (Bestand 150) schätzbar; bei $\\tau$ = 0,95 in 226, 330 und 813 von 1000" in README
    for key in ("m70", "standard", "m150"):                    # schätzbar wird mit höherem τ nie häufiger
        d = [_m(CELLS[key], t, "km")["defined"] for t in C.TAU_OPTIONS]
        assert d == sorted(d, reverse=True)


def test_reihe_ueber_n_anteil_steigt_kosten_der_verkaufsregel_bleiben():
    rows = DATA["n_series"]
    assert [r["n"] for r in rows] == [100, 250, 500, 1000]
    d = [_m(r, 0.9, "km")["defined"] / r["reps"] for r in rows]
    assert d == sorted(d) and [ev.share_pct(_m(r, 0.9, "km")["defined"], r["reps"]) for r in rows] == ["42 %", "46 %", "58 %", "70 %"]
    sales = [100 * _cost(r, 0.9, "sales") for r in rows]
    assert ev.num(min(sales)) == "55,8" and ev.num(max(sales)) == "57,3"
    assert "bei 55,8 % bis 57,3 %" in README


def test_testperioden_gegenmittel():
    rows = DATA["boost_series"]
    assert [r["boost"] for r in rows] == list(C.BOOST_SHARES)
    defined = [_m(r, 0.9, "km")["defined"] for r in rows]
    assert defined == [583, 966, 998, 1000, 1000]
    fb = [100 * _cost(r, 0.9, "km_fallback") for r in rows]
    assert [ev.num(x) for x in fb] == ["3,0", "4,0", "2,5", "1,6", "1,1"]
    assert fb[1] > fb[0] and fb[2] < fb[0] and fb[3] < fb[2] and fb[4] < fb[3]            # erst ab 10 % unter dem Wert ohne Testperioden
    left = [r["leftover_mean"] for r in rows]
    assert left == sorted(left) and ev.num(left[0]) == "20,9" and ev.num(left[-1]) == "75,3"
    sold = [r["sold_out_mean"] for r in rows]
    assert sold == sorted(sold, reverse=True)
    for needle in ("von 58 % auf 97 %", "(4,0 % bei 5 %, 2,5 % bei 10 %, 1,1 % bei 30 %; vorher 3,0 %)", "(20,9 Stück ohne, 75,3 Stück bei 30 % Testperioden)", "von 20,9 auf 75,3 Stück", "97 % der Läufe schätzbar (vorher 58 %)"):
        assert needle in README, needle


def test_stabilitaet_ueber_drei_zufalls_seeds():
    st = DATA["stability"]
    assert [r["seed"] for r in st] == [35, 101, 202] and all(r["reps"] == 1000 for r in st)
    assert [_m(r, 0.9, "km")["defined"] for r in st] == [583, 592, 602]
    assert ev.share_pct(min(_m(r, 0.9, "km")["defined"] for r in st), 1000) == "58 %" and ev.share_pct(max(_m(r, 0.9, "km")["defined"] for r in st), 1000) == "60 %"
    assert "bei 58 % bis 60 %" in README
    assert st[0]["taus"] == STD["taus"]                         # der Seed der Standardwelt ist der erste Eintrag


def test_alle_lognormal_anpassungen_gelungen():
    cells = list(CELLS.values()) + DATA["stability"] + DATA["n_series"]
    assert sum(c["reps"] for c in cells) == 11000 and all(c["lognormal_failed"] == 0 for c in cells)
    # Doppelte zählen nicht mit: Stabilität (Zufalls-Seed 35) und Perioden-Reihe (500) sind die Standardzelle
    distinct = {json.dumps(c["taus"], sort_keys=True) for c in cells}
    assert len(distinct) == 9 and 1000 * len(distinct) == 9000
    assert "In allen 9 000 verschiedenen angepassten Datensätzen" in README and "(0 Fehlschläge" in README
    assert all(c["lognormal_failed"] is None for c in DATA["boost_series"])      # dort wird das Lognormal nicht angepasst


def test_standardfehler_und_laufzahl_stehen_in_der_readme():
    assert "Standardfehler von 3,5 Punkten" in README
    assert round(100 * (0.58 * 0.42 / 200) ** 0.5, 1) == 3.5


# ------------------------------------------------------------------------------------------------ Vergleich mit der Vormessung (Plan)

# Zahlen der Vormessung V19 (200 Läufe, numpy-Zufall, `out_uz_v16_bis_v19.txt`): (aus Verkäufen, KM nur wo schätzbar, Anteil schätzbar in %, KM mit Rückfall, Lognormal), Mehrkosten in %
PLAN_V19 = {
    ("m70", 0.5): (25.5, 0.8, 100, 1.1, 0.3), ("m70", 0.8): (72.9, 3.8, 42, 2.8, 1.0), ("m70", 0.9): (127.9, 16.9, 26, 17.3, 1.5),
    ("standard", 0.5): (6.8, 0.2, 100, 0.2, 0.1), ("standard", 0.8): (28.8, 1.4, 94, 2.4, 0.4), ("standard", 0.9): (55.4, 3.0, 62, 3.0, 0.7),
    ("m150", 0.5): (0.4, 0.2, 100, 0.2, 0.1), ("m150", 0.8): (5.2, 0.4, 100, 0.4, 0.2), ("m150", 0.9): (13.7, 1.2, 100, 1.4, 0.4),
}


def test_abstand_zur_vormessung_v19():
    cost_diffs, share_diffs = [], []
    for (key, tau), (sales, km, share, fb, ln) in PLAN_V19.items():
        ours_share = 100 * _m(CELLS[key], tau, "km")["defined"] / 1000
        cost_diffs += [100 * _cost(CELLS[key], tau, "sales") - sales, 100 * _cost(CELLS[key], tau, "km") - km, 100 * _cost(CELLS[key], tau, "km_fallback") - fb, 100 * _cost(CELLS[key], tau, "lognormal") - ln]
        share_diffs.append(ours_share - share)
    assert ev.num(max(abs(d) for d in cost_diffs)) == "1,9" and ev.num(max(abs(d) for d in share_diffs)) == "4,6"
    worst = max(PLAN_V19, key=lambda c: abs(100 * _cost(CELLS[c[0]], c[1], "km") - PLAN_V19[c][1]))
    assert worst == ("m70", 0.9) and (PLAN_V19[worst][1], ev.num(100 * _cost(CELLS["m70"], 0.9, "km"))) == (16.9, "18,8")
    assert ev.share_pct(_m(CELLS["m70"], 0.8, "km")["defined"], 1000) == "47 %" and PLAN_V19[("m70", 0.8)][2] == 42
    for needle in ("um höchstens 1,9 Prozentpunkte", "16,9 gegen 18,8", "um höchstens 4,6 Punkte", "42 % gegen 47 %"):
        assert needle in README, needle


def test_abstand_zu_den_zahlen_der_messung_v8_und_v19b():
    """Die Zahlen der Vormessung (V8: 119,5 und 183,0, 54 %; V19: 62 %) stehen nur als Vergleich in der README; die Demo-Zahlen weichen davon ab und gelten."""
    t = STD["taus"]["0.9"]
    sales, km_mean, km_share = t["methods"]["sales"]["q_mean"], t["methods"]["km"]["q_mean"], t["methods"]["km"]["defined"] / 1000
    assert abs(sales - 119.5) < 0.5                        # die Verkaufsregel stimmt mit der Vormessung überein (119,5 ± 0,2)
    assert km_mean != pytest.approx(183.0, abs=1.0) and abs(km_mean - 183.0) < 4.0       # Kaplan–Meier bedingt auf schätzbar: 179,9 gegen 183,0, größer als ein Punkt Abstand
    assert abs(km_share - 0.54) < 0.05 and abs(km_share - 0.62) < 0.05                  # der Anteil liegt zwischen den beiden Vormessungen (54 % und 62 %)
    for needle in ("aus Verkäufen 119,5 ± 0,2 (−37 %), aus Kaplan–Meier 183,0 ± 2,0 (−4 %), Kaplan–Meier in 54 % der Läufe definiert", "Die Vormessung V19 selbst gab für dieselbe Welt 62 % an"):
        assert needle in README, needle


# ------------------------------------------------------------------------------------------------ 4. Hilfetexte, Grenzen-Tabelle, Handbeispiel


def test_preset_hilfen_gelten_gegen_die_messung():
    h = C.PRESET_HELP
    std, m70, m150, g = STD, CELLS["m70"], CELLS["m150"], CELLS["gamma"]
    assert "50,0 % ausverkauft, Kaplan–Meier in 58 % der Läufe schätzbar" in h["Standardwelt (Bestand 100)"]
    assert ev.pct(std["sold_out_mean"]) == "50,0 %" and ev.share_pct(_m(std, 0.9, "km")["defined"], 1000) == "58 %"
    assert "73,0 %" in h["Starker Ausverkauf (Bestand 70)"] and "128,9 %" in h["Starker Ausverkauf (Bestand 70)"] and "25 %" in h["Starker Ausverkauf (Bestand 70)"]
    assert ev.pct(m70["sold_out_mean"]) == "73,0 %" and ev.num(100 * _cost(m70, 0.9, "sales")) == "128,9" and ev.share_pct(_m(m70, 0.9, "km")["defined"], 1000) == "25 %"
    text = h["Wenig Ausverkauf (Bestand 150)"]
    assert "24,3 %" in text and "99,5 %" in text and "13,7 %" in text and "teuerste der vier" in text
    assert ev.pct(m150["sold_out_mean"]) == "24,3 %" and ev.share_pct(_m(m150, 0.9, "km")["defined"], 1000) == "99,5 %" and ev.num(100 * _cost(m150, 0.9, "sales")) == "13,7"
    assert _cost(m150, 0.9, "sales") == max(_cost(m150, 0.9, m) for m in M.METHODS)
    text = h["Mittleres Quantil (τ = 0,5)"]
    t5 = std["taus"]["0.5"]
    assert "81,2 gegen 100,0, −19 %" in text and ev.num(t5["methods"]["sales"]["q_mean"]) == "81,2" and ev.num(t5["q_true"]) == "100,0"
    assert ev.signed_pct(t5["methods"]["sales"]["q_mean"] / t5["q_true"] - 1) == "−19 %" and _m(std, 0.5, "km")["defined"] == 1000 and "in allen 1000 Läufen" in text
    text = h["Falsche Form (Gamma)"]
    assert "10,6 %" in text and "3,2 %" in text and ev.num(100 * _cost(g, 0.9, "lognormal")) == "10,6" and ev.num(100 * _cost(g, 0.9, "km_fallback")) == "3,2"
    text = h["Kurze Historie (n = 100)"]
    assert "nur in 42 % der Läufe" in text and ev.share_pct(_m(DATA["n_series"][0], 0.9, "km")["defined"], 1000) == "42 %"


def test_grenzen_tabelle_der_app_nennt_nur_gemessene_zahlen():
    g = CELLS["gamma"]
    assert "10,6 % gegen 3,2 % Mehrkosten, Bestand 100, 500 Perioden, 1000 Läufe" in APP
    assert ev.num(100 * _cost(g, 0.9, "lognormal")) == "10,6" and ev.num(100 * _cost(g, 0.9, "km_fallback")) == "3,2" and g["n"] == 500 and g["reps"] == 1000
    assert "in 42 % der Läufe schätzbar, mit 1000 Perioden in 70 %" in APP
    n = DATA["n_series"]
    assert (n[0]["n"], n[-1]["n"]) == (100, 1000) and ev.share_pct(_m(n[0], 0.9, "km")["defined"], 1000) == "42 %" and ev.share_pct(_m(n[-1], 0.9, "km")["defined"], 1000) == "70 %"


def test_handbeispiel_in_readme_und_json():
    hand = DATA["hand"]["Sechs Perioden (Handbeispiel)"]
    assert [(r["value"], r["exact"], r["censored"], r["at_risk"], r["s_prev"], r["s"]) for r in hand["rows"]] == [
        (6, 1, 0, 6, "1", "5/6"), (7, 1, 0, 5, "5/6", "2/3"), (8, 0, 1, 4, "2/3", "2/3"), (9, 1, 0, 3, "2/3", "4/9"), (10, 0, 1, 2, "4/9", "4/9"), (12, 0, 1, 1, "4/9", "4/9")]
    assert hand["quantiles"]["0.5"] == {"km": 9, "sales": "17/2"} and hand["quantiles"]["0.8"] == {"km": None, "sales": "10"}
    assert hand["sales"] == [6, 7, 8, 9, 10, 12] and F(sum(hand["sales"]), 6) == F(26, 3)
    assert M.hand_table(S.PATTERNS["Sechs Perioden (Handbeispiel)"])["rows"][-1]["s"] == F(4, 9)
    for needle in ("| 9 | 1 | 0 | 3 | 2/3 → 4/9 |", "Das Quantil 0,5 ist **9**", "**nicht schätzbar**: Die Kurve bleibt bei 4/9 stehen, erreichbar sind nur Quantile bis $\\tau = 5/9$", "Median von 8,5 (Mittel 26/3 ≈ 8,67)",
                   "(10, 6), (10, 10), (8, 8), (12, 7), (12, 12), (10, 9)"):
        assert needle in README, needle
    assert set(DATA["hand"]) == set(S.PATTERNS)
    for name, pairs in S.PATTERNS.items():                       # alle Muster der App stehen mit ihren Brüchen in der JSON-Datei
        rows = [(r["value"], r["exact"], r["censored"], r["at_risk"], str(r["s_prev"]), str(r["s"])) for r in M.hand_table(pairs)["rows"]]
        assert rows == [(r["value"], r["exact"], r["censored"], r["at_risk"], r["s_prev"], r["s"]) for r in DATA["hand"][name]["rows"]], name
