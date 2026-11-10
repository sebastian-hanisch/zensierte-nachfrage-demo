"""Konstanten: Regler-Grenzen, feste Annahmen der Welt, Farben, Presets und der Annahmen-Text (App und README)."""

# --- Welt: Nachfrage je Periode, Bestand je Periode, Verkauf = Minimum --------------------------------------------------------------------------------
# Nachfrage D: lognormal (oder gamma mit derselben Streuung), Median 100, Streuung σ = 0,5 auf der Log-Skala (Variationskoeffizient² = e^σ² − 1).
# Bestand B: lognormal, Median wählbar, σ = 0,3, UNABHÄNGIG von der Nachfrage. Verkauf = min(D, B). Ausverkauft heißt D ≥ B (Gleichstand zählt als ausverkauft).
DEMAND_MEDIAN = 100.0
DEMAND_SIGMA = 0.5
STOCK_SIGMA = 0.3
M_MIN, M_MAX, DEFAULT_M, M_STEP = 60, 200, 100, 10          # Median des Bestands (Bestandsniveau)
TAU_OPTIONS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)                # Quantil der Bestellregel (kritisches Verhältnis b/(b+h))
DEFAULT_TAU = 0.9
N_MIN, N_MAX, DEFAULT_N, N_STEP = 100, 1000, 500, 100        # Perioden je Datensatz
FORMS = ("lognormal", "gamma")
FORM_LABELS = {"lognormal": "Lognormal", "gamma": "Gamma (gleiche Streuung)"}
DEFAULT_FORM = "lognormal"
SEED_MAX, DEFAULT_SEED = 999999, 35

REPS = 1000                                  # Läufe je Zelle (feste Zufalls-Seeds), in der App und in tools/measure_zensierte_nachfrage.py
GRID_M = (70, 100, 150)                      # Bestandsniveaus der Kostentabelle (Experiment 4)
GRID_TAU = (0.5, 0.8, 0.9)                   # Quantile der Kostentabelle
BOOST_SHARES = (0.0, 0.05, 0.10, 0.20, 0.30)  # Anteil der Perioden mit sehr hohem Bestand (Experiment 3)
BOOST_STOCK_MEDIAN = 300.0                   # Median des Bestands in diesen Testperioden (dreifacher Median der Nachfrage)
DEFAULT_BOOST = 0.0

COLOR_SALES = "#c0504d"
COLOR_KM = "#2F6B65"
COLOR_LOGN = "#e08a2b"
COLOR_TRUTH = "#7f7f7f"
COLOR_REST = "rgba(150,150,150,0.55)"
COLOR_LINE = "#4c78a8"

# --- Presets ------------------------------------------------------------------------------------------------------------------------------------
# Schlüssel: m, tau, n, form, seed (Welt und Frage); die Auswahl des Handbeispiels und der Anteil der Testperioden stehen im Hauptbereich beim Experiment.
PRESETS = {
    "Standardwelt (Bestand 100)": {"m": 100, "tau": 0.9, "n": 500, "form": "lognormal", "seed": DEFAULT_SEED},
    "Starker Ausverkauf (Bestand 70)": {"m": 70, "tau": 0.9, "n": 500, "form": "lognormal", "seed": DEFAULT_SEED},
    "Wenig Ausverkauf (Bestand 150)": {"m": 150, "tau": 0.9, "n": 500, "form": "lognormal", "seed": DEFAULT_SEED},
    "Mittleres Quantil (τ = 0,5)": {"m": 100, "tau": 0.5, "n": 500, "form": "lognormal", "seed": DEFAULT_SEED},
    "Falsche Form (Gamma)": {"m": 100, "tau": 0.9, "n": 500, "form": "gamma", "seed": DEFAULT_SEED},
    "Kurze Historie (n = 100)": {"m": 100, "tau": 0.9, "n": 100, "form": "lognormal", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardwelt (Bestand 100)": "Nachfrage lognormal (Median 100, σ 0,5), Bestand lognormal (Median 100, σ 0,3), 500 Perioden, Bestellmenge zum Quantil 0,9. Das ist die Welt der Messung dieser Demo: 50,0 % ausverkauft, Kaplan–Meier in 58 % der Läufe schätzbar.",
    "Starker Ausverkauf (Bestand 70)": "Der Bestand liegt unter dem Median der Nachfrage: 73,0 % der Perioden sind ausverkauft. Bestellen aus den Verkäufen kostet im Mittel 128,9 % mehr als die beste Menge (τ = 0,9, 500 Perioden, 1000 Läufe), das Kaplan–Meier-Quantil 0,9 ist nur in 25 % der Läufe schätzbar.",
    "Wenig Ausverkauf (Bestand 150)": "Der Bestand liegt über dem Median der Nachfrage: 24,3 % der Perioden sind ausverkauft. Kaplan–Meier ist in 99,5 % der Läufe schätzbar; die Verkaufsregel bleibt mit 13,7 % Mehrkosten (τ = 0,9, 500 Perioden, 1000 Läufe) die teuerste der vier.",
    "Mittleres Quantil (τ = 0,5)": "Bestellmenge zum Median der Nachfrage. Auch hier liegt die Verkaufsregel unter der Wahrheit (81,2 gegen 100,0, −19 %); Kaplan–Meier erreicht den Median in allen 1000 Läufen.",
    "Falsche Form (Gamma)": "Die Nachfrage ist gamma-verteilt statt lognormal, bei gleichem Median und gleicher Streuung. Das zensierte Lognormal ist dann die falsche Form: bei τ = 0,9 kostet es 10,6 % Mehrkosten (500 Perioden, 1000 Läufe), Kaplan–Meier mit Rückfall 3,2 %.",
    "Kurze Historie (n = 100)": "Nur 100 Perioden statt 500: Kaplan–Meier schätzt das Quantil 0,9 nur in 42 % der Läufe, und die Mehrkosten steigen (Kaplan–Meier mit Rückfall 8,3 % statt 3,0 %, zensiertes Lognormal 3,1 % statt 0,6 %).",
}

# --- Annahmen, Grenzen und bekannte Probleme (derselbe Text steht in der README; ein Test vergleicht beide) -------------------------------------
ASSUMPTIONS_MARKDOWN = """\
**Annahmen**
- Der Bestand je Periode ist unabhängig von der Nachfrage dieser Periode (hier erzeugt: Nachfrage und Bestand werden getrennt gezogen). Dann ist die Zensierung unabhängig vom Ereignis, und Kaplan–Meier ist gültig. Hängt der Bestand an einer Prognose der Nachfrage, ist die Zensierung informativ (Stück „Informatives Zensieren“ der Linie).
- Ausverkauft heißt, der Verkauf gleicht dem Bestand; dann ist die Nachfrage nur als „mindestens so viel wie der Bestand“ bekannt. Ist der Verkauf kleiner als der Bestand, ist die Nachfrage genau bekannt. Einen Verkauf von 0 gibt es nicht (die Rechnung verlangt Werte über 0).
- Die Nachfrage ist von Periode zu Periode unabhängig und gleich verteilt (keine Saison, kein Trend), und Nachfrage, die bei Ausverkauf verloren geht, kommt nicht als Nachbestellung wieder.
- Kosten: Fehlmenge τ je Stück, Überschuss 1 − τ je Stück, sonst nichts; die beste Bestellmenge ist dann das Quantil τ der Nachfrage. Die Mehrkosten sind relativ zu den Kosten dieser besten Bestellmenge, gerechnet gegen die wahre Verteilung.

**Grenzen**
- Alle Zahlen stammen aus simulierten Welten mit einer Nachfrage-Verteilung (lognormal, Median 100, σ 0,5, oder gamma mit gleicher Streuung) und einem lognormalen Bestand (σ 0,3); andere Bestandsmuster, andere Streuungen und andere Kostenformen ändern die Größen, nicht unbedingt die Richtung.
- Kaplan–Meier schätzt das Quantil nur, wenn die Kurve bis 1 − τ fällt; sonst gibt es keine Zahl („nicht schätzbar“). Der Rückfall auf den größten beobachteten Wert ist eine einfache Regel, keine Empfehlung.
- Das zensierte Lognormal ist ein Verfahren mit Formannahme: In den gemessenen Lognormal-Zellen ist es am günstigsten, bei gamma-verteilter Nachfrage ist es bei τ = 0,8 und 0,9 teurer als Kaplan–Meier mit Rückfall (Experiment 4). Andere Formen sind nicht gemessen.
- Der Anteil der Testperioden mit hohem Bestand (Experiment 3) zählt nur den Nutzen für das Quantil; der Überschuss in diesen Perioden steht nicht in den Mehrkosten.

**Bekannte Probleme und Unsicherheiten**
- Die Läufe sind Stichproben mit festen Zufalls-Seeds: Zahlen wie „in 583 von 1000 Läufen“ gelten für diese Seeds, andere Seeds ändern einzelne Zählungen um einige Läufe.
- Die Mittelwerte der Mehrkosten sind von einzelnen schlechten Läufen mitbestimmt (Mittel, nicht Median).
- Die Seite ist im Browser nur am Rechner geprüft (Seitenaufruf, Regler über AppTest), nicht auf Mobilgeräten.
"""
