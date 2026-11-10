"""Kosten der Bestellmenge, Läufe über viele Datensätze, Zusammenfassung und die Handtabelle in Brüchen.

Newsvendor mit Fehlmengenkosten b = τ je Stück und Überschusskosten h = 1 − τ je Stück (die Skala ist gleichgültig, nur τ = b/(b + h) zählt):
Kosten(q) = τ·E[(D − q)⁺] + (1 − τ)·E[(q − D)⁺]. Die beste Bestellmenge ist das Quantil τ der wahren Nachfrage. Die erwarteten Kosten einer Bestellmenge
werden **geschlossen gegen die wahre Verteilung** gerechnet (nicht an einer Stichprobe): für das Lognormal über die Teilerwartung, für das Gamma über die
regularisierte obere unvollständige Gamma-Funktion. Die Mehrkosten sind Kosten(q)/Kosten(q*) − 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.special import gammaincc, gammaincinv, ndtr, ndtri

import znf_constants as C
import znf_core as K
import znf_estimators as E
import znf_scenario as S

METHODS = ("sales", "km", "km_fallback", "lognormal")

# ------------------------------------------------------------------------------------------------ Kosten gegen die wahre Verteilung


def true_quantile(form: str, tau: float) -> float:
    """Quantil τ der wahren Nachfrage (die beste Bestellmenge): lognormal exp(μ + σ·Φ⁻¹(τ)), gamma θ·Q⁻¹(k, τ)."""
    if form == "lognormal":
        return math.exp(math.log(C.DEMAND_MEDIAN) + C.DEMAND_SIGMA * float(ndtri(tau)))
    if form == "gamma":
        return S.gamma_scale() * float(gammaincinv(S.gamma_shape(), tau))
    raise K.SurvivalError("Die Verteilung der Nachfrage muss „lognormal“ oder „gamma“ sein.")


def expected_shortage(q, form: str):
    """E[(D − q)⁺] gegen die wahre Verteilung, geschlossen. q darf ein Array sein."""
    q = np.maximum(np.asarray(q, dtype=float), 1e-12)
    if form == "lognormal":
        mu, sg = math.log(C.DEMAND_MEDIAN), C.DEMAND_SIGMA
        mean = math.exp(mu + sg * sg / 2.0)
        return mean * ndtr((mu + sg * sg - np.log(q)) / sg) - q * ndtr((mu - np.log(q)) / sg)
    if form == "gamma":
        k, th = S.gamma_shape(), S.gamma_scale()
        return k * th * gammaincc(k + 1.0, q / th) - q * gammaincc(k, q / th)
    raise K.SurvivalError("Die Verteilung der Nachfrage muss „lognormal“ oder „gamma“ sein.")


def demand_mean(form: str) -> float:
    if form == "lognormal":
        return C.DEMAND_MEDIAN * math.exp(C.DEMAND_SIGMA**2 / 2.0)
    return S.gamma_shape() * S.gamma_scale()


def expected_cost(q, tau: float, form: str):
    """Erwartete Kosten je Periode der Bestellmenge q: τ·E[(D − q)⁺] + (1 − τ)·E[(q − D)⁺], mit E[(q − D)⁺] = q − E[D] + E[(D − q)⁺]."""
    q = np.maximum(np.asarray(q, dtype=float), 1e-12)
    short = expected_shortage(q, form)
    excess = q - demand_mean(form) + short
    return tau * short + (1.0 - tau) * excess


def extra_cost(q, tau: float, form: str):
    """Mehrkosten der Bestellmenge q gegen die beste (Quantil τ der wahren Nachfrage): Kosten(q)/Kosten(q*) − 1; nan bleibt nan."""
    q = np.asarray(q, dtype=float)
    best = float(expected_cost(true_quantile(form, tau), tau, form))
    out = expected_cost(np.where(np.isfinite(q), q, 1.0), tau, form) / best - 1.0
    return np.where(np.isfinite(q), out, np.nan)


# ------------------------------------------------------------------------------------------------ Läufe


@dataclass(frozen=True)
class Study:
    """Ergebnis von `run_study`: je Lauf (Zeile) und Quantil (Spalte) die vier Schätzungen der Bestellmenge, dazu Kennzahlen je Lauf."""

    world: S.World
    taus: tuple
    q_true: np.ndarray                 # (ntaus,) beste Bestellmenge je τ
    q: dict                            # Methode -> (reps, ntaus); nan: nicht schätzbar (nur km, lognormal)
    sold_out: np.ndarray               # (reps,) Anteil ausverkaufter Perioden
    ln_ok: np.ndarray | None           # (reps,) Anpassung des Lognormals gelungen; None, wenn es nicht angepasst wurde
    leftover: np.ndarray               # (reps,) mittlerer Überschuss je Periode E[(Bestand − Nachfrage)⁺]

    @property
    def reps(self) -> int:
        return int(self.sold_out.size)


def run_study(world: S.World, taus, reps: int = C.REPS, with_lognormal: bool = True) -> Study:
    """`reps` Datensätze der Welt; je Datensatz die Bestellmenge zu jedem τ aus den vier Verfahren. Das Lognormal wird je Datensatz einmal angepasst."""
    taus = tuple(float(t) for t in taus)
    q = {name: np.full((reps, len(taus)), np.nan) for name in METHODS}
    sold_out, leftover = np.zeros(reps), np.zeros(reps)
    ln_ok = np.zeros(reps, bool) if with_lognormal else None
    for r in range(reps):
        smp = S.make_sample(world, r)
        sold_out[r] = 1.0 - float(np.mean(smp.observed))
        leftover[r] = float(np.mean(np.maximum(smp.stock - smp.demand, 0.0)))
        fit = E.lognormal_fit(smp.sales, smp.observed) if with_lognormal else None
        if fit is not None:
            ln_ok[r] = fit["ok"]
        for j, tau in enumerate(taus):
            q["sales"][r, j] = E.sales_quantile(smp.sales, tau)
            qk = E.km_quantile_fast(smp.sales, smp.observed, tau)
            q["km"][r, j] = qk
            q["km_fallback"][r, j] = qk if math.isfinite(qk) else float(np.max(smp.sales))
            if fit is not None:
                q["lognormal"][r, j] = E.lognormal_quantile(fit, tau)
    q_true = np.array([true_quantile(world.form, t) for t in taus])
    return Study(world, taus, q_true, q, sold_out, ln_ok, leftover)


# ------------------------------------------------------------------------------------------------ Zusammenfassung


@dataclass(frozen=True)
class Cell:
    """Kennzahlen einer Methode bei einem τ: Mittel und Standardfehler der Bestellmenge und der Mehrkosten über die Läufe, in denen die Methode eine Zahl liefert."""

    method: str
    tau: float
    defined: int           # Läufe mit Zahl
    reps: int
    q_mean: float
    q_se: float
    cost_mean: float       # Mehrkosten als Anteil (0,05 = 5 %)
    cost_se: float


def _mean_se(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")
    return float(np.mean(x)), se


def summarize(study: Study, tau: float, method: str) -> Cell:
    j = study.taus.index(float(tau))
    qs = study.q[method][:, j]
    cost = extra_cost(qs, tau, study.world.form)
    qm, qse = _mean_se(qs)
    cm, cse = _mean_se(cost)
    return Cell(method, float(tau), int(np.sum(np.isfinite(qs))), study.reps, qm, qse, cm, cse)


def defined_share(study: Study, tau: float, method: str = "km") -> float:
    """Anteil der Läufe, in denen die Methode zu τ eine Zahl liefert."""
    j = study.taus.index(float(tau))
    return float(np.mean(np.isfinite(study.q[method][:, j])))


# ------------------------------------------------------------------------------------------------ Handtabelle in Brüchen


def hand_table(rows) -> dict:
    """Kaplan–Meier der Nachfrage aus Paaren (Bestand, Verkauf) in exakten Brüchen. Ausverkauft heißt Verkauf = Bestand (Nachfrage ≥ Bestand); sonst ist die Nachfrage genau der Verkauf.

    Gibt {"rows": [...], "sales": sortierte Verkäufe, "events": Zahl genauer Nachfragewerte} zurück; je verschiedenem Wert eine Zeile mit
    value, exact (genau beobachtet), censored (ausverkauft, „≥“), at_risk (alle mit Wert ≥ value), s_prev, s (nur wo exact > 0 ändert sich S)."""
    pairs = [(int(b), int(v)) for b, v in rows]
    if not pairs or any(v <= 0 or v > b for b, v in pairs):
        raise K.SurvivalError("Jede Periode braucht einen Bestand und einen Verkauf zwischen 1 und dem Bestand.")
    exact = [v for b, v in pairs if v < b]
    cens = [v for b, v in pairs if v == b]
    s_prev = Fraction(1)
    out = []
    for value in sorted(set(exact) | set(cens)):
        d = sum(1 for v in exact if v == value)
        c = sum(1 for v in cens if v == value)
        at_risk = sum(1 for v in exact + cens if v >= value)
        s_now = s_prev * Fraction(at_risk - d, at_risk) if d else s_prev
        out.append({"value": value, "exact": d, "censored": c, "at_risk": at_risk, "s_prev": s_prev, "s": s_now})
        s_prev = s_now
    return {"rows": out, "sales": sorted(v for _, v in pairs), "events": len(exact)}


def hand_quantile(table: dict, tau: Fraction):
    """Kleinster genau beobachteter Wert mit S ≤ 1 − τ in Brüchen; None, wenn die Kurve das Niveau nicht erreicht („nicht schätzbar“)."""
    level = 1 - Fraction(tau)
    for r in table["rows"]:
        if r["exact"] and r["s"] <= level:
            return r["value"]
    return None


def hand_sales_quantile(table: dict, tau: Fraction) -> Fraction:
    """Quantil der Verkäufe mit linearer Interpolation in exakten Brüchen (derselbe Wert wie numpy mit der Voreinstellung)."""
    v = table["sales"]
    pos = Fraction(tau) * (len(v) - 1)
    lo = int(pos)            # abrunden (pos >= 0)
    frac = pos - lo
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + frac * (v[hi] - v[lo])
