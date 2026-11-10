"""Welt der Daten: Nachfrage und Bestand je Periode, Verkauf = Minimum, Ausverkauf zensiert die Nachfrage. Zufall nur über SplitMix64.

Nachfrage lognormal oder gamma (gleiche Streuung), Bestand lognormal und unabhängig von der Nachfrage. Gezogen wird immer in derselben Reihenfolge
(Nachfrage, Bestand, Wahl der Testperioden): Welten mit verschiedenem Bestandsniveau oder verschiedenem Anteil an Testperioden teilen sich bei
gleichem Zufalls-Seed dieselben Zufallszahlen, ein Vergleich über diese Einstellungen ist also gepaart."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaincinv

import znf_constants as C
import znf_core as K

_U_MIN = 2.0**-53   # kleinste Gleichverteilte über 0, damit eine gamma-verteilte Nachfrage nie genau 0 wird (die Rechnung verlangt Zeiten > 0)


@dataclass(frozen=True)
class World:
    """Einstellungen einer Welt. Immer mit Schlüsselwörtern aufrufen (ein eingefügtes Feld verschöbe sonst still die Werte)."""

    m: float = C.DEFAULT_M                 # Median des Bestands
    n: int = C.DEFAULT_N                   # Perioden je Datensatz
    form: str = C.DEFAULT_FORM             # Verteilung der Nachfrage: "lognormal" oder "gamma"
    boost: float = C.DEFAULT_BOOST         # Anteil der Perioden mit sehr hohem Bestand (Testperioden), zufällig und unabhängig von der Nachfrage
    seed: int = C.DEFAULT_SEED             # Zufalls-Seed des ersten Laufs; Lauf r nutzt run_seed(seed, r)


@dataclass(frozen=True)
class Sample:
    """Ein Datensatz. `sales`, `observed` und `stock` sieht man; `demand` ist die latente Nachfrage und dient nur Tests und der Anzeige der Wahrheit."""

    demand: np.ndarray
    stock: np.ndarray
    sales: np.ndarray       # min(Nachfrage, Bestand)
    observed: np.ndarray    # True: Nachfrage genau bekannt (Verkauf < Bestand); False: ausverkauft, Nachfrage nur „mindestens so viel“


def run_seed(seed: int, r: int) -> int:
    """Zufalls-Seed des Laufs r: feste, voneinander getrennte Ströme (Abstand 1000003 zwischen den Läufen)."""
    return int(seed) + 1000003 * int(r)


def gamma_shape() -> float:
    """Form k der Gamma-Nachfrage: gleiche Streuung wie das Lognormal (Variationskoeffizient² = e^σ² − 1, also k = 1/(e^σ² − 1))."""
    return 1.0 / math.expm1(C.DEMAND_SIGMA**2)


def gamma_scale() -> float:
    """Skala θ der Gamma-Nachfrage, so dass ihr Median gleich dem Median der Nachfrage ist."""
    return C.DEMAND_MEDIAN / float(gammaincinv(gamma_shape(), 0.5))


def draw_demand(rng: K.SplitMix64, n: int, form: str) -> np.ndarray:
    """n Nachfragewerte: lognormal (Inversion über die Normalverteilung) oder gamma (Inversion über die umgekehrte regularisierte Gamma-Funktion)."""
    if form == "lognormal":
        return rng.lognormal(n, C.DEMAND_MEDIAN, C.DEMAND_SIGMA)
    if form == "gamma":
        u = np.maximum(rng.uniform(n), _U_MIN)
        return gamma_scale() * gammaincinv(gamma_shape(), u)
    raise K.SurvivalError("Die Verteilung der Nachfrage muss „lognormal“ oder „gamma“ sein.")


def simulate(n: int, m: float, form: str, boost: float, seed: int) -> Sample:
    """n Perioden. Die Testperioden (Anteil `boost`) haben denselben Zufallsanteil im Bestand, aber den Median C.BOOST_STOCK_MEDIAN statt m."""
    if n < 1:
        raise K.SurvivalError("Es braucht mindestens eine Periode.")
    if not m > 0:
        raise K.SurvivalError("Der Median des Bestands muss größer als 0 sein.")
    if not 0.0 <= boost <= 1.0:
        raise K.SurvivalError("Der Anteil der Testperioden muss zwischen 0 und 1 liegen.")
    rng = K.SplitMix64(seed)
    demand = draw_demand(rng, n, form)
    z = rng.normal(n)
    pick = rng.uniform(n)
    median = np.where(pick < boost, C.BOOST_STOCK_MEDIAN, float(m))
    stock = median * np.exp(C.STOCK_SIGMA * z)
    sales = np.minimum(demand, stock)
    return Sample(demand, stock, sales, demand < stock)


def make_sample(world: World, r: int = 0) -> Sample:
    return simulate(world.n, world.m, world.form, world.boost, run_seed(world.seed, r))


# --- Handbeispiele (Experiment 1) ------------------------------------------------------------------------------------------------------------------
# Je Muster sechs Perioden (Bestand, Verkauf), ganze Stück. Ausverkauft heißt Verkauf = Bestand. Die Werte der Tabellen stehen in tests/test_model.py von Hand gerechnet.
PATTERNS = {
    "Sechs Perioden (Handbeispiel)": ((10, 6), (10, 10), (8, 8), (12, 7), (12, 12), (10, 9)),
    "Nie ausverkauft": ((10, 6), (10, 7), (8, 5), (12, 9), (12, 11), (10, 8)),
    "Immer ausverkauft": ((10, 10), (8, 8), (12, 12), (9, 9), (10, 10), (11, 11)),
    "Gleichstand: ein Wert ist genau und zugleich Bestand": ((12, 9), (9, 9), (10, 6), (10, 10), (8, 7), (12, 12)),
}
DEFAULT_PATTERN = next(iter(PATTERNS))
