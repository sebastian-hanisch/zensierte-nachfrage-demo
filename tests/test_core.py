"""Tests der kopierten Kernfunktionen (aus ueberlebenszeit-planung/kern/test_uz_core.py übernommen, soweit die Demo die Funktionen braucht)."""

from fractions import Fraction as F

import numpy as np
import pytest

import znf_core as K


def test_km_handbeispiel_sechs_perioden():
    """Verkauf 6, 10, 8, 7, 12, 9 mit Bestand 10, 10, 8, 12, 12, 10: genau sind 6, 7, 9; zensiert sind 10, 8, 12. S = 5/6, 2/3, 4/9 (von Hand)."""
    km = K.kaplan_meier([6, 10, 8, 7, 12, 9], [1, 0, 0, 1, 0, 1])
    assert km["t"].tolist() == [6, 7, 9]
    assert km["n"].tolist() == [6, 5, 3]
    for got, want in zip(km["s"], [F(5, 6), F(2, 3), F(4, 9)]):
        assert got == pytest.approx(float(want), abs=1e-15)
    assert K.quantile(km, 0.5) == 9.0
    assert np.isnan(K.quantile(km, 0.8))      # S fällt nur bis 4/9 > 0,2: nicht schätzbar


def test_km_ohne_zensierung_ist_eins_minus_verteilungsfunktion():
    rng = K.SplitMix64(1)
    t = np.round(rng.exponential(50) * 3 + 0.5, 1)  # mit Gleichständen
    km = K.kaplan_meier(t, np.ones(50, bool))
    for x in (0.5, 1.0, 2.0, 4.0, 8.0):
        assert K.survival_at(km, x) == pytest.approx(float(np.mean(t > x)), abs=1e-12)


def test_km_alle_zensiert():
    km = K.kaplan_meier([1, 2, 3], [0, 0, 0])
    assert km["t"].size == 0 and K.survival_at(km, 5.0) == 1.0
    assert np.isnan(K.quantile(km, 0.5))


def test_eingabepruefung():
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([3, 3, 5], [1, 1, 1], [3, 3, 0])  # Eintritt = Zeit
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([0, 1], [1, 1])  # Zeit 0
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1, 2])  # Ereignis 2
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([1, 2], [1])  # Länge
    with pytest.raises(K.SurvivalError):
        K.kaplan_meier([], [])


def test_splitmix_referenzwert_und_determinismus():
    # erster Ausgabewert von SplitMix64 bei Seed 0 (Referenz: Vigna, splitmix64.c)
    assert int(K.SplitMix64(0).u64(1)[0]) == 0xE220A8397B1DCDAF
    a, b = K.SplitMix64(5), K.SplitMix64(5)
    assert np.array_equal(a.uniform(10), b.uniform(10))
    u = K.SplitMix64(9).uniform(50000)
    assert 0.0 <= u.min() and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01
    z = K.SplitMix64(9).normal(50000)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03
    x = K.SplitMix64(9).lognormal(50000, 100.0, 0.5)
    assert abs(np.median(x) - 100.0) < 2.0


def test_bfgs_findet_das_minimum_einer_quadratischen_funktion():
    def f(p):
        d = p - np.array([1.5, -2.0])
        return float(d @ np.diag([1.0, 25.0]) @ d), 2 * np.diag([1.0, 25.0]) @ d

    x, val, ok = K._bfgs(f, np.zeros(2))
    assert ok and val == pytest.approx(0.0, abs=1e-12) and x == pytest.approx([1.5, -2.0], abs=1e-8)


def test_kopie_stimmt_mit_der_quelle_ueberein_falls_vorhanden():
    """Die Funktionen dieser Datei sind wörtlich aus ueberlebenszeit-planung/kern/uz_core.py kopiert; gibt es die Quelle, stimmen die Texte der Funktionen überein."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "ueberlebenszeit-planung" / "kern" / "uz_core.py"
    if not src.exists():
        pytest.skip("ueberlebenszeit-planung/kern/uz_core.py nicht vorhanden")
    mine = Path(K.__file__).read_text(encoding="utf-8")
    theirs = src.read_text(encoding="utf-8")

    def definitions(text):
        return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    a, b = definitions(mine), definitions(theirs)
    assert set(a) <= set(b)
    assert {name for name in a if a[name] != b[name]} == set()
