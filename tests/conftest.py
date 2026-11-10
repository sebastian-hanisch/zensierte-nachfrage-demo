"""Gemeinsame Fixtures: das Handbeispiel, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

HAND_PAIRS = ((10, 6), (10, 10), (8, 8), (12, 7), (12, 12), (10, 9))   # (Bestand, Verkauf); ausverkauft heißt Verkauf = Bestand


@pytest.fixture
def hand_pairs():
    """Sechs Perioden. Handrechnung: genau beobachtet sind 6, 7, 9; ausverkauft (≥) sind 8, 10, 12. S = 5/6, 2/3, 4/9; Median 9; Quantil 0,8 nicht schätzbar;
    aus den Verkäufen Median 8,5 und Mittel 26/3."""
    return HAND_PAIRS
