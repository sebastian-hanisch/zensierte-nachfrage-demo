"""Messskript der Demo: jede Zahl der README steht in `tools/measure_zensierte_nachfrage_result.json` (eingefroren, feste Zufalls-Seeds).

Aufruf (Repo-Wurzel):  _venvs/test/Scripts/python.exe -X utf8 tools/measure_zensierte_nachfrage.py            schreibt die JSON-Datei
                       _venvs/test/Scripts/python.exe -X utf8 tools/measure_zensierte_nachfrage.py --print    zeigt die Tabellen, schreibt nichts

Alle Messungen laufen mit dem Code der Demo (SplitMix64, znf_model); Läufe je Zelle: C.REPS = 1000, Lauf r hat den Zufalls-Seed `run_seed(Zufalls-Seed, r)`.
Die Funktionen sind rein; tests/test_claims.py rechnet Teile davon neu und prüft, dass die Tabellen der README wörtlich aus der Datei stammen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import znf_constants as C  # noqa: E402
import znf_evaluation as ev  # noqa: E402
import znf_model as M  # noqa: E402
import znf_scenario as S  # noqa: E402

RESULT = Path(__file__).with_name("measure_zensierte_nachfrage_result.json")
TAUS = C.TAU_OPTIONS
SEEDS_STABILITY = (35, 101, 202)       # drei verschiedene Zufalls-Seeds für die Stabilitätsprobe der Standardzelle
N_SERIES = (100, 250, 500, 1000)


def cell(study: M.Study) -> dict:
    """Eine Welt: Kennzahlen der Läufe, je τ die vier Verfahren (Läufe mit Zahl, Mittel und Standardfehler der Bestellmenge und der Mehrkosten)."""
    w = study.world
    out = {
        "m": w.m, "n": w.n, "form": w.form, "boost": w.boost, "seed": w.seed, "reps": study.reps,
        "sold_out_mean": float(study.sold_out.mean()),
        "leftover_mean": float(study.leftover.mean()),
        "lognormal_failed": int(np.sum(~study.ln_ok)) if study.ln_ok is not None else None,
        "taus": {},
    }
    for tau in study.taus:
        j = study.taus.index(tau)
        methods = {}
        for m in M.METHODS:
            if m == "lognormal" and study.ln_ok is None:
                continue
            c = M.summarize(study, tau, m)
            methods[m] = {"defined": c.defined, "q_mean": c.q_mean, "q_se": c.q_se, "cost_mean": c.cost_mean, "cost_se": c.cost_se}
        out["taus"][f"{tau:g}"] = {"tau": tau, "q_true": float(study.q_true[j]), "methods": methods}
    return out


def measure() -> dict:
    def study(**kw) -> M.Study:
        with_ln = kw.pop("with_lognormal", True)
        return M.run_study(S.World(**kw), TAUS, C.REPS, with_lognormal=with_ln)

    cells = {
        "standard": cell(study()),
        "m70": cell(study(m=70)),
        "m150": cell(study(m=150)),
        "gamma": cell(study(form="gamma")),
    }
    return {
        "cells": cells,
        "stability": [cell(study(seed=s)) for s in SEEDS_STABILITY],
        "n_series": [cell(study(n=n)) for n in N_SERIES],
        "boost_series": [cell(study(boost=b, with_lognormal=False)) for b in C.BOOST_SHARES],
        "hand": hand_numbers(),
    }


def hand_numbers() -> dict:
    """Die Handtabellen aller Muster (kein Zufall) als Brüche in Textform."""
    from fractions import Fraction

    out = {}
    for name, rows in S.PATTERNS.items():
        t = M.hand_table(rows)
        out[name] = {
            "rows": [{"value": r["value"], "exact": r["exact"], "censored": r["censored"], "at_risk": r["at_risk"], "s_prev": str(r["s_prev"]), "s": str(r["s"])} for r in t["rows"]],
            "sales": t["sales"],
            "quantiles": {f"{float(tau):g}": {"km": M.hand_quantile(t, tau), "sales": str(M.hand_sales_quantile(t, tau))} for tau in (Fraction(1, 2), Fraction(4, 5))},
        }
    return out


# ------------------------------------------------------------------------------------------------ Tabellen der README (aus der JSON-Datei)


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(headers, rows) -> str:
    return "\n".join([_row(headers), "|" + "|".join("---" for _ in headers) + "|"] + [_row(r) for r in rows])


def _c(cellv: dict, tau: float, method: str) -> dict:
    return cellv["taus"][f"{tau:g}"]["methods"][method]


def _km_text(cellv: dict, tau: float) -> str:
    """Mehrkosten von Kaplan–Meier mit dem Anteil der Läufe, in denen die Zahl existiert."""
    m = _c(cellv, tau, "km")
    return f"{ev.num(100 * m['cost_mean'])} ({ev.share_pct(m['defined'], cellv['reps'])} schätzbar)" if m["defined"] else "keine Zahl (0 % schätzbar)"


def tables(data: dict) -> dict[str, str]:
    """Die Tabellen der README, wörtlich aus den eingefrorenen Zahlen gebildet (tests/test_claims.py sucht sie in der README)."""
    out = {}
    p = ev.pct
    n = lambda x: ev.num(100 * x)   # noqa: E731  Mehrkosten als Prozentzahl ohne Zeichen
    std = data["cells"]["standard"]
    rows = []
    for tau in TAUS:
        t = std["taus"][f"{tau:g}"]
        qs, qk = t["methods"]["sales"], t["methods"]["km"]
        rows.append([ev.tau_text(tau), ev.num(t["q_true"]), f"{ev.num(qs['q_mean'])} ({ev.signed_pct(qs['q_mean'] / t['q_true'] - 1)})",
                     f"{ev.num(qk['q_mean'])} ({ev.signed_pct(qk['q_mean'] / t['q_true'] - 1)})" if qk["defined"] else "keine Zahl", ev.share_pct(qk["defined"], std["reps"])])
    out["quantile"] = _table(["τ", "Wahrheit (Quantil der Nachfrage)", "aus den Verkäufen (Mittel)", "aus Kaplan–Meier (Mittel über die Läufe mit Zahl)", "Läufe mit Zahl"], rows)

    rows = []
    for key in ("m70", "standard", "m150"):
        c = data["cells"][key]
        for tau in C.GRID_TAU:
            rows.append([int(c["m"]), p(c["sold_out_mean"]), ev.tau_text(tau), n(_c(c, tau, "sales")["cost_mean"]), _km_text(c, tau),
                         n(_c(c, tau, "km_fallback")["cost_mean"]), n(_c(c, tau, "lognormal")["cost_mean"])])
    out["costs"] = _table(["Bestand (Median)", "ausverkauft", "τ", "aus den Verkäufen", "Kaplan–Meier (nur wo schätzbar)", "Kaplan–Meier mit Rückfall „größter Wert“", "zensiertes Lognormal"], rows)

    c = data["cells"]["gamma"]
    rows = [[ev.tau_text(tau), n(_c(c, tau, "sales")["cost_mean"]), _km_text(c, tau), n(_c(c, tau, "km_fallback")["cost_mean"]), n(_c(c, tau, "lognormal")["cost_mean"])] for tau in C.GRID_TAU]
    out["gamma"] = _table(["τ", "aus den Verkäufen", "Kaplan–Meier (nur wo schätzbar)", "Kaplan–Meier mit Rückfall „größter Wert“", "zensiertes Lognormal (falsche Form)"], rows)

    rows = []
    for c in data["n_series"]:
        t = c["taus"]["0.9"]
        rows.append([c["n"], ev.share_pct(t["methods"]["km"]["defined"], c["reps"]), n(t["methods"]["sales"]["cost_mean"]), n(t["methods"]["km_fallback"]["cost_mean"]), n(t["methods"]["lognormal"]["cost_mean"])])
    out["n_series"] = _table(["Perioden n", "Läufe mit Kaplan–Meier-Quantil", "aus den Verkäufen", "Kaplan–Meier mit Rückfall", "zensiertes Lognormal"], rows)

    rows = []
    for c in data["boost_series"]:
        t = c["taus"]["0.9"]
        km = t["methods"]["km"]
        rows.append([p(c["boost"], 0), p(c["sold_out_mean"]), ev.share_pct(km["defined"], c["reps"]), n(t["methods"]["km_fallback"]["cost_mean"]), ev.num(c["leftover_mean"])])
    out["boost"] = _table(["Testperioden (Anteil)", "ausverkauft", "Läufe mit Kaplan–Meier-Quantil", "Mehrkosten, Kaplan–Meier mit Rückfall (%)", "Überschuss je Periode (Stück)"], rows)

    rows = []
    for s in data["stability"]:
        t = s["taus"]["0.9"]["methods"]
        rows.append([s["seed"], p(s["sold_out_mean"]), n(t["sales"]["cost_mean"]), ev.share_pct(t["km"]["defined"], s["reps"]), n(t["km_fallback"]["cost_mean"]), n(t["lognormal"]["cost_mean"])])
    out["stability"] = _table(["Zufalls-Seed", "ausverkauft", "aus den Verkäufen", "Läufe mit Kaplan–Meier-Quantil", "Kaplan–Meier mit Rückfall", "zensiertes Lognormal"], rows)
    return out


def main() -> None:
    data = measure()
    if "--print" in sys.argv:
        for name, table in tables(data).items():
            print(f"\n== {name}\n{table}")
        print("\nstandard:", json.dumps(data["cells"]["standard"], indent=1)[:1500])
        print("hand:", json.dumps(data["hand"], indent=1, ensure_ascii=False)[:1500])
        return
    RESULT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("geschrieben:", RESULT)


if __name__ == "__main__":
    main()
