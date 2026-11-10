"""Die drei Schätzer des Quantils der Nachfrage aus Verkäufen, dazu die schnelle Fassung des Kaplan–Meier-Quantils.

* `sales_quantile`: das Quantil der Verkäufe, als wären sie die Nachfrage (die naheliegende, verzerrte Regel).
* `km_quantile_fast`: Kaplan–Meier-Quantil der Nachfrage, ausverkaufte Perioden als rechtszensiert; nan, wenn die Kurve das Niveau nicht erreicht.
  Dasselbe wie `znf_core.quantile(znf_core.kaplan_meier(...), tau)`, in einem sortierten Durchgang statt einer Schleife über die Ereigniszeiten
  (tests/test_model.py hält beide gegeneinander, auch mit Gleichständen).
* `lognormal_fit`: Maximum-Likelihood-Anpassung eines Lognormals an Daten mit Rechtszensierung; BFGS mit analytischem Gradienten (das kleine BFGS aus
  znf_core), Startwerte bei Bedarf in mehreren Fassungen, Konvergenzprüfung.

Zensierte Likelihood: ein beobachteter Wert geht mit seiner Dichte ein, ein ausverkaufter Wert nur mit der Wahrscheinlichkeit, mindestens so groß zu sein."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import log_ndtr, ndtri

import znf_core as K

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_GTOL = 1e-7   # Abbruch, wenn der Gradient je Beobachtung darunter fällt; feiner liegt im Rundungsrauschen der Zielfunktion (das Verfahren liefe ohne Gewinn weiter)


def sales_quantile(sales, tau: float) -> float:
    """Quantil der Verkäufe mit linearer Interpolation (so wie ein Median von Hand: bei gerader Zahl der Mittelwert der beiden mittleren Werte)."""
    s = np.asarray(sales, dtype=float)
    if s.size == 0 or not np.all(np.isfinite(s)):
        raise K.SurvivalError("Es braucht mindestens einen endlichen Verkauf.")
    return float(np.quantile(s, tau))


def km_quantile_fast(time, event, tau: float) -> float:
    """Kleinste Ereigniszeit mit Ŝ ≤ 1 − τ (Kaplan–Meier, bei gleicher Zeit zählt das Ereignis vor der Zensierung); nan („nicht schätzbar“), wenn Ŝ nie so tief fällt."""
    t, e, _ = K.check_data(time, event)
    ts, d = np.unique(t[e], return_counts=True)
    if ts.size == 0:
        return float("nan")
    n = t.size - np.searchsorted(np.sort(t), ts, side="left")     # im Risiko: alle mit Zeit >= ts
    s = np.cumprod(1.0 - d / n)
    hit = np.where(s <= 1.0 - tau + 1e-12)[0]
    return float(ts[hit[0]]) if hit.size else float("nan")


def _lognormal_nll(params, logt: np.ndarray, ev: np.ndarray):
    """Negative Log-Likelihood und Gradient über (m, log σ) für Lognormal mit Rechtszensierung; Argumente: log der Zeiten, Ereignisanzeige."""
    m, ls = params
    if abs(ls) > 8.0 or abs(m) > 60.0:      # außerhalb jedes sinnvollen Bereichs: die Schrittweitensuche halbiert den Schritt
        return float("inf"), np.zeros(2)
    s = math.exp(ls)
    z = (logt - m) / s
    ze, zc = z[ev], z[~ev]
    # Ereignis: log f = −log σ − log t − ½ log 2π − ½ z²;  zensiert: log S = log Φ(−z)
    log_c = log_ndtr(-zc)
    ll = float(np.sum(-ls - logt[ev] - _HALF_LOG_2PI - 0.5 * ze * ze) + np.sum(log_c))
    lam = np.exp(-0.5 * zc * zc - _HALF_LOG_2PI - log_c)               # Hazard der Standardnormalverteilung φ(z)/Φ(−z), stabil über Logarithmen
    g_m = float(np.sum(ze) / s + np.sum(lam) / s)
    g_ls = float(np.sum(ze * ze - 1.0) + np.sum(lam * zc))
    if not (math.isfinite(ll) and math.isfinite(g_m) and math.isfinite(g_ls)):
        return float("inf"), np.zeros(2)
    return -ll, -np.array([g_m, g_ls])


def lognormal_fit(time, event) -> dict:
    """Maximum-Likelihood-Lognormal mit Rechtszensierung: ln X ~ Normal(m, σ²). Gibt mu (= m), sigma, loglik, ok, message.
    Ohne mindestens zwei verschiedene beobachtete Werte gibt es keinen Schätzer (ok = False)."""
    t, e, _ = K.check_data(time, event)
    if np.unique(t[e]).size < 2:
        return {"mu": float("nan"), "sigma": float("nan"), "loglik": float("nan"), "ok": False,
                "message": "Für die Lognormal-Anpassung braucht es mindestens zwei verschiedene beobachtete Nachfragewerte (nicht ausverkauft)."}
    logt = np.log(t)
    n = float(t.size)

    def scaled(p):
        """Die Zielfunktion je Beobachtung (durch n geteilt): Gradient und Schrittweite bleiben von der Stichprobengröße unabhängig, das BFGS braucht dann wenige Schritte."""
        f, g = _lognormal_nll(p, logt, e)
        return f / n, g / n

    spread = max(float(np.std(logt)), 0.05)
    best = None
    # Die Log-Likelihood ist in (m/σ, 1/σ) konkav, ein Startwert genügt im Regelfall; weitere nur, wenn der erste nicht konvergiert.
    for m0, s0 in ((float(np.mean(logt)), spread), (float(np.median(logt)), 0.5), (float(np.mean(logt)) + 0.5, 1.0)):
        x, f, ok = K._bfgs(scaled, np.array([m0, math.log(s0)]), iters=100, gtol=_GTOL)
        if ok and np.isfinite(f):
            best = (x, f)
            break
    if best is None:
        return {"mu": float("nan"), "sigma": float("nan"), "loglik": float("nan"), "ok": False,
                "message": "Die Lognormal-Anpassung ist nicht konvergiert."}
    return {"mu": float(best[0][0]), "sigma": math.exp(float(best[0][1])), "loglik": -float(best[1]) * n, "ok": True, "message": ""}


def lognormal_quantile(fit: dict, tau: float) -> float:
    """Quantil τ des angepassten Lognormals: exp(m + σ·Φ⁻¹(τ)); nan, wenn die Anpassung fehlschlug."""
    if not fit["ok"]:
        return float("nan")
    return math.exp(fit["mu"] + fit["sigma"] * float(ndtri(tau)))
