"""Formatierung (Dezimalkomma) und Verdikte. Jedes Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`,
tests/test_verdicts.py prüft Schwellen und jede Zahl im Text an handgesetzten Werten (DEMO-PLAYBOOK §12)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

import znf_constants as C
import znf_model as M
import znf_scenario as S

# ------------------------------------------------------------------------------------------------ Formatierung (Dezimalkomma)


def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma."""
    return f"{100.0 * x:.{digits}f}".replace(".", ",") + " %"


def share_pct(k: int, n: int) -> str:
    """Anteil k von n als ganze Prozent; ist k weder 0 noch n, wird nie auf 0 % oder 100 % gerundet (dann eine Nachkommastelle: 995 von 1000 sind 99,5 %, nicht 100 %)."""
    x = 100.0 * k / n
    text = f"{x:.0f}"
    if 0 < k < n and text in ("0", "100"):
        text = f"{x:.1f}"
    return text.replace(".", ",") + " %"


def signed_pct(x: float, digits: int = 0) -> str:
    """Relative Abweichung (−0,37 = −37 %) mit Vorzeichen und Dezimalkomma (echtes Minuszeichen)."""
    text = f"{100.0 * x:+.{digits}f}"
    if float(text) == 0.0:
        text = "±" + text[1:]      # auf null gerundet: weder + noch −
    return text.replace(".", ",").replace("-", "−") + " %"


def tau_text(tau: float) -> str:
    return f"{tau:g}".replace(".", ",")


def frac(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def frac_dec(x: Fraction, digits: int = 2) -> str:
    """Bruch mit Dezimalwert: `4/9 ≈ 0,44`; ganze Zahlen ohne Zusatz."""
    return frac(x) if x.denominator == 1 else f"{frac(x)} ≈ {num(float(x), digits)}"


def share_text(k: int, n: int) -> str:
    return f"{k} von {n}"


# ------------------------------------------------------------------------------------------------ Verdikte


def verdict_hand(table: dict, tau: Fraction) -> tuple[str, str]:
    """Experiment 1: Quantil der Handtabelle aus Kaplan–Meier gegen das aus den Verkäufen."""
    q_sales = M.hand_sales_quantile(table, tau)
    q_km = M.hand_quantile(table, tau)
    sold = sum(r["censored"] for r in table["rows"])
    total = table["events"] + sold
    tau_s = tau_text(float(tau))
    if sold == 0:
        return "success", (
            f"Keine Periode ist ausverkauft: Die Kurve ist die empirische Verteilung, und das Quantil {tau_s} ist **{frac(q_km)}**. "
            f"Aus den Verkäufen ergibt sich {frac_dec(q_sales)}; die Verkäufe sind hier die Nachfrage, der Unterschied kommt nur von der Interpolation zwischen zwei Werten."
        )
    if table["events"] == 0:
        return "error", (
            f"Alle {total} Perioden sind ausverkauft: Die Nachfrage ist nirgends genau bekannt, die Kurve bleibt bei 1, und das Quantil {tau_s} ist **nicht schätzbar**. "
            f"Aus den Verkäufen ergibt sich {frac_dec(q_sales)}, das ist nur ein Wert aus den Beständen."
        )
    if q_km is None:
        floor = table["rows"][-1]["s"]      # Höhe, auf der die Kurve stehen bleibt (nach dem letzten genauen Wert)
        return "warning", (
            f"Das Quantil {tau_s} ist **nicht schätzbar**: Die Kurve fällt nur bis {frac_dec(floor)} und bleibt dort stehen, denn der größte Wert ist ein Ausverkauf; "
            f"erreichbar wären nur Quantile bis τ = {frac_dec(1 - floor)}. Aus den Verkäufen ergibt sich {frac_dec(q_sales)} "
            f"({sold} von {total} Perioden ausverkauft); diese Zahl behandelt jeden Ausverkauf als genaue Nachfrage, obwohl er nur eine Untergrenze ist."
        )
    return "info", (
        f"Kaplan–Meier schätzt das Quantil {tau_s} auf **{frac(q_km)}**, aus den Verkäufen ergibt sich {frac_dec(q_sales)} ({sold} von {total} Perioden ausverkauft)."
    )


def verdict_quantile(study: M.Study, tau: float) -> tuple[str, str]:
    """Experiment 2: Quantil der Nachfrage aus den Verkäufen gegen Kaplan–Meier, gezählt über die Läufe."""
    j = study.taus.index(float(tau))
    q_true = float(study.q_true[j])
    sales = M.summarize(study, tau, "sales")
    km = M.summarize(study, tau, "km")
    sold = float(study.sold_out.mean())
    gap_sales = sales.q_mean / q_true - 1.0
    share = km.defined / km.reps
    share_str = share_pct(km.defined, km.reps)
    text = (
        f"Bei {pct(sold, 0)} ausverkauften Perioden liegt das Quantil {tau_text(tau)} aus den Verkäufen im Mittel bei **{num(sales.q_mean)}** gegen die Wahrheit {num(q_true)} "
        f"({signed_pct(gap_sales)}). Kaplan–Meier liefert in {share_text(km.defined, km.reps)} Läufen ({share_str}) eine Zahl"
    )
    if km.defined:
        text += f", im Mittel über diese Läufe {num(km.q_mean)} ({signed_pct(km.q_mean / q_true - 1.0)}); in den übrigen Läufen erreicht die Kurve das Niveau nicht."
    else:
        text += "; in keinem Lauf erreicht die Kurve das Niveau."
    kind = "warning" if share < 0.95 or abs(gap_sales) > 0.05 else "success"
    return kind, text


def example_run(world, max_runs: int = 60) -> int:
    """Nummer (ab 0) des ersten Laufs der Welt, in dem die Kurve auf einem Plateau stehen bleibt, also der größte Verkauf ein Ausverkauf ist; gibt es in den ersten `max_runs` Läufen keinen, ist es Lauf 0."""
    for r in range(max_runs):
        smp = S.make_sample(world, r)
        if not smp.observed[int(np.argmax(smp.sales))]:
            return r
    return 0


def plateau_info(km: dict, sales, observed) -> dict:
    """Wo die Kurve eines Datensatzes stehen bleibt: Höhe des Plateaus, größter genau beobachteter Wert, größter Verkauf und ob der größte Verkauf ein Ausverkauf ist."""
    sales = np.asarray(sales, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    i = int(np.argmax(sales))
    return {
        "floor": float(km["s"][-1]) if km["s"].size else 1.0,
        "last_exact": float(km["t"][-1]) if km["t"].size else float("nan"),
        "largest": float(sales[i]),
        "largest_sold_out": bool(not observed[i]),
        "sold_out": float(1.0 - observed.mean()),
        "n": int(sales.size),
    }


def verdict_plateau(info: dict, tau: float) -> tuple[str, str]:
    """Experiment 3, erster Teil: das Plateau eines Datensatzes gegen das gewählte τ."""
    floor = info["floor"]
    head = f"In diesem Datensatz ({info['n']} Perioden, {pct(info['sold_out'], 0)} ausverkauft) fällt die Kurve bis S = {num(floor, 3)}"
    if math.isnan(info["last_exact"]):
        return "error", head.replace("fällt die Kurve bis S = 1,000", "bleibt die Kurve bei S = 1,000") + ": Keine Periode hat die Nachfrage genau gezeigt, kein Quantil ist schätzbar."
    where = f" (beim Wert {num(info['last_exact'])})"
    ends = " Der größte Verkauf ist ein Ausverkauf, deshalb kann die Kurve nicht weiter fallen." if info["largest_sold_out"] else " Der größte Verkauf ist eine genau beobachtete Nachfrage."
    if floor <= 1e-12:
        return "success", f"{head}{where}, also bis 0: In diesem Datensatz ist jedes Quantil schätzbar, auch das gewählte τ = {tau_text(tau)}.{ends}"
    reach = f"Erreichbar sind in diesem Datensatz nur Quantile bis τ = {num(1.0 - floor, 3)}"
    if 1.0 - tau >= floor - 1e-12:
        return "success", f"{head}{where} und bleibt dort stehen. {reach}; das gewählte τ = {tau_text(tau)} gehört dazu.{ends}"
    return "warning", f"{head}{where} und bleibt dort stehen. {reach}; das gewählte τ = {tau_text(tau)} liegt darüber und ist hier **nicht schätzbar**.{ends}"


def verdict_boundary(study: M.Study, tau: float, boost: float) -> tuple[str, str]:
    """Experiment 3: Anteil der Läufe, in denen Kaplan–Meier das Quantil schätzen kann, bei dem gewählten Anteil an Testperioden."""
    share = M.defined_share(study, tau)
    k = int(round(share * study.reps))
    base = f"Bei {pct(boost, 0)} Testperioden mit hohem Bestand ist das Quantil {tau_text(tau)} in {share_text(k, study.reps)} Läufen ({share_pct(k, study.reps)}) schätzbar."
    if share >= 0.995:
        return "success", base + " Die Kurve erreicht das Niveau in praktisch jedem Lauf."
    if share <= 0.005:
        return "error", base + " Die Kurve erreicht das Niveau in praktisch keinem Lauf; ohne Form (Experiment 4) gibt es hier keine Zahl."
    return "warning", base + " In den übrigen Läufen bleibt die Kurve über 1 − τ stehen (der größte Wert ist ein Ausverkauf)."


def verdict_costs(study: M.Study, tau: float) -> tuple[str, str]:
    """Experiment 4: Mehrkosten der vier Bestellregeln in dieser Welt, mit Anteil der Läufe, in denen Kaplan–Meier eine Zahl liefert."""
    cells = {m: M.summarize(study, tau, m) for m in M.METHODS}
    sold = float(study.sold_out.mean())
    form = C.FORM_LABELS[study.world.form]
    sales, km, fb, ln = cells["sales"], cells["km"], cells["km_fallback"], cells["lognormal"]
    ranking = sorted((m for m in ("sales", "km_fallback", "lognormal") if cells[m].defined), key=lambda m: cells[m].cost_mean)
    factor = 1.0 + sales.cost_mean
    ln_text = pct(ln.cost_mean, 1) if ln.defined else "keine Anpassung möglich"
    text = (
        f"Nachfrage {form}, {pct(sold, 0)} der Perioden ausverkauft, Bestellmenge zum Quantil {tau_text(tau)}: Aus den Verkäufen zu bestellen kostet im Mittel **{pct(sales.cost_mean, 1)}** mehr als die beste Menge "
        f"(das {num(factor)}-Fache der besten Kosten). Kaplan–Meier mit Rückfall: {pct(fb.cost_mean, 1)}, zensiertes Lognormal: {ln_text}; "
        f"Kaplan–Meier allein nur in {share_pct(km.defined, km.reps)} der Läufe schätzbar ({pct(km.cost_mean, 1) if km.defined else 'keine Zahl'} in diesen)."
    )
    if study.world.form == "gamma" and ln.defined and ln.cost_mean > fb.cost_mean:
        text += (
            f" Die Form passt hier nicht (Nachfrage gamma-verteilt): Das zensierte Lognormal ({pct(ln.cost_mean, 1)}) ist teurer als Kaplan–Meier mit Rückfall ({pct(fb.cost_mean, 1)}); "
            f"gemessen an den Verkäufen ({pct(sales.cost_mean, 1)}) bleibt es günstiger."
            if ln.cost_mean < sales.cost_mean else
            f" Die Form passt hier nicht (Nachfrage gamma-verteilt): Das zensierte Lognormal ({pct(ln.cost_mean, 1)}) ist teurer als Kaplan–Meier mit Rückfall ({pct(fb.cost_mean, 1)}) und nicht günstiger als die Verkäufe ({pct(sales.cost_mean, 1)})."
        )
        return "warning", text
    if ranking and ranking[0] == "lognormal" and study.world.form == "lognormal":
        text += " Das zensierte Lognormal ist hier am günstigsten; seine Form passt zur erzeugten Nachfrage."
    return ("warning" if sales.cost_mean > 0.10 else "info"), text
