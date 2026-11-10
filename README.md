# Zensierte Nachfrage: Ausverkauft heißt mindestens so viel – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-zensierte-nachfrage-demo.streamlit.app/)** -->

Teil der Linie **Überlebenszeit** (Zeit bis zum Ereignis bei Beobachtungen, die irgendwann abbrechen) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Hier ist die Brücke zur Optimierung: Dieselbe Mathematik wie bei der Verweilzeit von Containern, aber die Zensierung kommt vom **Bestand** statt vom Ende eines Beobachtungsfensters. Ist ein Artikel in einer Periode **ausverkauft**,
kennt man die Nachfrage nur als „mindestens so viel wie der Bestand“. Gefragt ist die **Bestellmenge** nach der Newsvendor-Regel, das Quantil $\tau$ der Nachfrage. Die Demo hält drei Wege gegeneinander: die Verkäufe als Nachfrage lesen,
**Kaplan–Meier** auf die zensierte Nachfrage und ein **zensiertes Lognormal**; gemessen werden das Quantil, die **Grenze** von Kaplan–Meier (die Kurve erreicht ein hohes Niveau nicht, wenn der größte Wert ein Ausverkauf ist) und die **Mehrkosten**
der Bestellmenge gegen die beste.

Nachbarn: Die Demo löst den Ausblick „Zensierte Nachfrage schätzen“ aus [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo) ein (dort ist die Nachfrage vollständig beobachtet). Wer die Bestellregel direkt auf den Kosten lernt,
findet das in [newsvendor-direkt-demo](https://github.com/sebastian-hanisch/newsvendor-direkt-demo); von dort wird hier keine Zahl übernommen. Hängt der Bestand an einer Prognose der Nachfrage, ist die Zensierung informativ, siehe
[informatives-zensieren-demo](https://github.com/sebastian-hanisch/informatives-zensieren-demo). Alle Zahlen sind erzeugt, keine stammt aus einem echten Lager.

**Annahme in dieser Demo, sichtbar in der App und hier: Der Bestand einer Periode ist unabhängig von ihrer Nachfrage.** Dann ist die Zensierung unabhängig vom Ereignis, und Kaplan–Meier ist gültig. Die simulierte Welt erfüllt die Annahme; die Demo prüft sie nicht.

## Warum dieses Problem

Wer wissen will, wie viel er bestellen soll, hat Verkäufe, aber keine Nachfrage. In einer ausverkauften Periode ist der Verkauf nur eine Untergrenze. Die naheliegende Rechnung liest die Verkäufe trotzdem als Nachfrage und nimmt daraus das Quantil.
In der Welt dieser Demo (Nachfrage lognormal mit Median 100 und σ 0,5, Bestand lognormal mit Median 100 und σ 0,3, unabhängig von der Nachfrage, 500 Perioden je Datensatz, 1000 Datensätze mit Zufalls-Seed 35) sind 50,0 % der Perioden ausverkauft;
das Quantil 0,9 der Nachfrage ist 189,8. Aus den Verkäufen kommen im Mittel 119,2 heraus (−37 %). Kaplan–Meier liefert nur in 583 von 1000 Läufen (58 %) überhaupt eine Zahl und kommt in diesen Läufen im Mittel auf 179,9 (−5 %).
Die Zahl in Geld: Mit Fehlmengenkosten $\tau$ und Überschusskosten $1 - \tau$ je Stück kostet die Bestellmenge aus den Verkäufen im Mittel 56,0 % mehr als die beste Menge (das 1,6-Fache der besten Kosten); bei Bestand 70, wo 73,0 % der Perioden
ausverkauft sind, sind es 128,9 % (das 2,3-Fache), beide bei $\tau = 0{,}9$, 500 Perioden und 1000 Läufen. Die Kernfrage ist: Wie weit liegt das Quantil aus den Verkäufen daneben, was leistet Kaplan–Meier, wo ist seine Grenze, und was kostet
die Formannahme des zensierten Lognormals, wenn sie falsch ist?

Die „−37 %“ gelten nur für diese Welt (Bestand lognormal mit σ 0,3, 50,0 % ausverkauft, $\tau = 0{,}9$); über die Quantile 0,5 bis 0,95 liegt die Verkaufsregel zwischen −19 % und −42 % (Tabelle unten), und die 58 % schätzbarer Läufe gehören immer dazu.

## Befunde und Korrekturen gegenüber dem Plan

(Plan: `PLAN-11-zensierte-nachfrage.md` der Linien-Planung; Messung dort: 200 Läufe mit numpy-Zufall, Rohausgaben `out_uz_v1_bis_v8.txt` (Messung V8) und `out_uz_v16_bis_v19.txt` (Messungen V19 und V19b).)

1. **Zahlen weichen ab, es gilt die Zahl dieser Demo (1000 Läufe, SplitMix64).** Vormessung V8: aus Verkäufen 119,5 ± 0,2 (−37 %), aus Kaplan–Meier 183,0 ± 2,0 (−4 %), Kaplan–Meier in 54 % der Läufe definiert. Diese Demo: 119,2 (−37 %),
   179,9 (−5 %) in 58 % der Läufe (583 von 1000). Die Vormessung V19 selbst gab für dieselbe Welt 62 % an; die Anteile der Vormessungen und dieser Messung schwanken um einige Punkte (200 Läufe haben bei 58 % einen Standardfehler von 3,5 Punkten).
   Über drei Zufalls-Seeds (je 1000 Läufe) liegt der Anteil bei 58 % bis 60 % (Tabelle „Stabilität“).
2. **Die Tabelle V19 wird bis auf kleine Abstände reproduziert.** In den neun Zellen (Bestand 70, 100, 150 mal $\tau$ = 0,5, 0,8, 0,9) weicht jede Zahl der Vormessung für die Mehrkosten um höchstens 1,9 Prozentpunkte von dieser Messung ab
   (größte Abweichung bei „Kaplan–Meier, nur wo schätzbar“, Bestand 70, $\tau$ = 0,9: 16,9 gegen 18,8), der Anteil schätzbarer Läufe um höchstens 4,6 Punkte (Bestand 70, $\tau$ = 0,8: 42 % gegen 47 %). Die Zahlen dieser Demo stehen in den Tabellen unten.
3. **„Das zensierte Lognormal gewinnt nur bei richtiger Form, bei Gamma nicht“ ist zu grob und wird genauer gesagt.** In der Lognormal-Welt ist das zensierte Lognormal in allen 9 Zellen am günstigsten. In der Gamma-Welt ist es bei $\tau$ = 0,8 und 0,9
   teurer als Kaplan–Meier mit Rückfall (3,4 % gegen 2,2 % und 10,6 % gegen 3,2 % Mehrkosten), bleibt aber weit unter der Verkaufsregel (31,9 % und 62,5 %); beim Median ($\tau$ = 0,5) liegen beide unter 0,3 % und gleichauf (0,19 % gegen 0,22 %, Unterschied etwa 3 Standardfehler).
   Gilt für diese eine Form (gamma mit gleicher Streuung) und einen Bestand.
4. **Der Rückfall „größter Wert“ ist neu mitgemessen** (die Vormessung V19b für Gamma hatte ihn nicht): bei Gamma, Bestand 100, 500 Perioden liegt er bei 2,2 % ($\tau$ = 0,8) und 3,2 % ($\tau$ = 0,9) Mehrkosten, Kaplan–Meier nur wo schätzbar
   bei 1,5 % und 3,1 %. Er ist eine einfache Regel, keine Empfehlung.
5. **Experiment 3 („Bestand zeitweise hoch ansetzen“) hatte im Plan keine Messung und ist hier gemessen.** In einem Anteil der Perioden wird der Bestand zufällig und unabhängig von der Nachfrage auf den Median 300 gesetzt: Schon 5 % Testperioden
   machen das Quantil 0,9 in 97 % der Läufe schätzbar (vorher 58 %), die Mehrkosten von Kaplan–Meier mit Rückfall sinken aber erst ab 10 % Testperioden (4,0 % bei 5 %, 2,5 % bei 10 %, 1,1 % bei 30 %; vorher 3,0 %). Der Preis, der Überschuss je Periode
   (20,9 Stück ohne, 75,3 Stück bei 30 % Testperioden), steht in keiner Mehrkosten-Zahl. Das ist eine Eigenmessung in einer Welt.
6. **Eine Kostenform, ein Bestandsmuster.** Die Kosten sind Fehlmenge $\tau$ und Überschuss $1 - \tau$ je Stück (nur das Verhältnis zählt); die Mehrkosten sind relativ zu den Kosten der besten Menge. Der Bestand ist unabhängig von der Nachfrage (lognormal, σ 0,3);
   andere Bestandsmuster sind nicht gemessen (die Messlücke des Plans bleibt offen). Entscheidung Z4 des Plans (Kosten nur mit ihren Bedingungen nennen) gilt: Jede Kostenzahl nennt Bestand, $\tau$, Perioden und Läufe.
7. **Mehr Handmuster als im Plan:** zusätzlich zum Handbeispiel eine Reihe ohne Ausverkauf, eine mit lauter Ausverkäufen und ein Gleichstand (genauer und ausverkaufter Wert bei derselben Zahl), alle in Brüchen getestet.
8. **Orakel:** Der Plan nennt Kaplan–Meier gegen Aufzählung, das Lognormal gegen `scipy.optimize` und eine Simulation. Hinzu kommen eine Rastersuche und der geschlossene Fall, die erwarteten Kosten gegen numerische Integration und gegen Simulation sowie die beste Menge
   gegen numerische Minimierung. **Kein Fremdcode-Orakel** (lifelines, statsmodels fehlen im Test-venv); SciPy ist Referenz für Likelihood, Integration und Minimierung.

## Modell

Nachfrage $D_i$ je Periode: lognormal ($\ln D \sim \mathrm{N}(\ln 100,\ 0{,}5^2)$) oder gamma mit demselben Median und derselben Streuung (Form $k = 1/(e^{0{,}25} - 1) \approx 3{,}52$). Bestand $B_i$: lognormal mit Median $m$ (60 bis 200, Voreinstellung 100) und σ 0,3, unabhängig von $D_i$.
Verkauf $V_i = \min(D_i, B_i)$; ausverkauft heißt $D_i \ge B_i$ (Gleichstand zählt als ausverkauft), dann ist $D_i$ nur als „mindestens $B_i$“ bekannt. Kosten: Fehlmenge $\tau$ je Stück, Überschuss $1 - \tau$ je Stück, erwartete Kosten
$C(q) = \tau\,\mathrm{E}[(D - q)^+] + (1 - \tau)\,\mathrm{E}[(q - D)^+]$; die beste Menge ist das Quantil $\tau$ der Nachfrage (Ableitung $F(q) - \tau$). Die Schätzer:

- **Aus den Verkäufen:** das Quantil der $V_i$ mit linearer Interpolation (bei sechs Werten ist der Median der Mittelwert der beiden mittleren).
- **Kaplan–Meier:** ausverkaufte Perioden als rechtszensiert; das Quantil ist der kleinste genau beobachtete Wert mit $\hat S \le 1 - \tau$, **nicht schätzbar**, wenn die Kurve nie so tief fällt. Mit dem Rückfall „größter Wert“ ist es der größte Verkauf.
- **Zensiertes Lognormal:** Maximum Likelihood mit Dichte für genaue und Überlebensfunktion für ausverkaufte Werte, BFGS mit analytischem Gradienten.

Die erwarteten Kosten einer Bestellmenge werden **geschlossen gegen die wahre Verteilung** gerechnet (Lognormal: Teilerwartung über die Normalverteilung; Gamma: regularisierte obere unvollständige Gamma-Funktion), nicht an einer Stichprobe.
Formeln, Herleitung und Symboltabelle stehen im Expander „📐 Mathematische Formulierung“ der App, mit Lesehilfe. Der Kern (`znf_core.py`) ist eine Kopie aus der Linien-Vorlage (`ueberlebenszeit-planung/kern/uz_core.py`, Stand 2026-11-10):
SplitMix64-Zufall, Eingabeprüfung, Kaplan–Meier mit Quantil („nicht schätzbar“ als nan) und das kleine BFGS; `tests/test_core.py` vergleicht die Funktionstexte mit der Quelle, falls sie vorhanden ist. Das zensierte Lognormal ist neu geschrieben (`znf_estimators.py`).

## Methodik

Die App hat vier Experimente.

1. **Die Kurve von Hand.** Sechs Perioden (Bestand, Verkauf): (10, 6), (10, 10), (8, 8), (12, 7), (12, 12), (10, 9). Genau beobachtet sind 6, 7 und 9; ausverkauft („≥“) sind 8, 10 und 12. Die App zeigt die Tabelle in exakten Brüchen (genau, ausverkauft,
   im Risiko, $S(x^-) \to S(x)$), wächst Zeile für Zeile (Regler) und hat drei weitere Muster: nie ausverkauft, immer ausverkauft, und ein Gleichstand zwischen einem genauen und einem ausverkauften Wert. Das Quantil ist wählbar.
2. **Quantil aus den Verkäufen gegen Kaplan–Meier** (Kernfrage). 1000 Läufe mit festen Zufalls-Seeds; Regler in der Seitenleiste: Bestandsniveau (Median 60 bis 200), Verteilung der Nachfrage, Perioden (100 bis 1000), Quantil $\tau$ (0,5 bis 0,95), Zufalls-Seed.
   Gezeigt werden die Wahrheit, beide Mittel mit Standardfehler, je Lauf ein Kasten und der Anteil der Läufe, in denen Kaplan–Meier eine Zahl liefert.
3. **Die Grenze.** Ein Datensatz mit seiner Kurve und dem Plateau, in dem sie stehen bleibt; der Anteil schätzbarer Läufe gegen $\tau$ für drei Bestandsniveaus (70, 100, 150, dazu die gewählte Welt); als Gegenmittel ein Anteil Testperioden mit sehr hohem Bestand
   (0 bis 30 %, zufällig und unabhängig von der Nachfrage, Median 300).
4. **Mehrkosten der Bestellmenge.** Vier Wege (Verkäufe, Kaplan–Meier nur wo schätzbar, Kaplan–Meier mit Rückfall, zensiertes Lognormal) für die gewählte Welt, dazu die Tabelle über Bestandsniveau und $\tau$; die Verteilung der Nachfrage (Lognormal oder Gamma) wählt die Seitenleiste.

Jede Meldung statt einer Zahl kommt aus dem Kern (`SurvivalError` mit verständlichem Text) oder aus dem Status der Anpassung; die App fängt sie ab und zeigt eine Fehlermeldung, nie einen Traceback und nie eine stille Zahl. Für die vielen Läufe rechnet `km_quantile_fast`
in `znf_estimators.py` in einem sortierten Durchgang; `tests/test_model.py` hält es gegen den Kern (150 Datensätze mit Gleichständen), `tests/test_oracle_zensierte_nachfrage.py` gegen die Umverteilung nach rechts.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus dem Code der Demo (SplitMix64, `znf_model.py`) mit festen Zufalls-Seeds und stehen eingefroren in `tools/measure_zensierte_nachfrage_result.json` (erzeugt mit `tools/measure_zensierte_nachfrage.py`).
Welt, wenn nichts anderes steht: Nachfrage lognormal (Median 100, σ 0,5), Bestand lognormal (Median 100, σ 0,3), 500 Perioden, 1000 Läufe; der Zufalls-Seed der Läufe ist 35, Lauf $r$ hat den Seed 35 + 1 000 003 · $r$. Die Mehrkosten sind Mittel über die Läufe
(bei Kaplan–Meier „nur wo schätzbar“ nur über die Läufe, in denen die Kurve das Niveau erreicht; dieser Wert ist bedingt und mit den anderen nur eingeschränkt vergleichbar). `tests/test_claims.py` rechnet alle Zellen mit dem Code der Demo neu,
vergleicht sie mit der eingefrorenen Datei und prüft, dass die Tabellen unten Wort für Wort aus ihr stammen (siehe „Tests“).

**Quantil der Nachfrage aus den Verkäufen gegen Kaplan–Meier (Messung 1, Standardwelt, 1000 Läufe, 50,0 % ausverkauft).** Mittel über die Läufe; Kaplan–Meier mittelt nur über die Läufe mit Zahl.

| τ | Wahrheit (Quantil der Nachfrage) | aus den Verkäufen (Mittel) | aus Kaplan–Meier (Mittel über die Läufe mit Zahl) | Läufe mit Zahl |
|---|---|---|---|---|
| 0,5 | 100,0 | 81,2 (−19 %) | 100,0 (±0 %) | 100 % |
| 0,6 | 113,5 | 87,9 (−23 %) | 113,8 (±0 %) | 100 % |
| 0,7 | 130,0 | 95,5 (−27 %) | 131,0 (+1 %) | 99,9 % |
| 0,8 | 152,3 | 105,0 (−31 %) | 154,2 (+1 %) | 95 % |
| 0,9 | 189,8 | 119,2 (−37 %) | 179,9 (−5 %) | 58 % |
| 0,95 | 227,6 | 132,2 (−42 %) | 192,7 (−15 %) | 33 % |

Der Standardfehler des Mittels bei $\tau$ = 0,9 ist 0,08 für die Verkäufe und 0,78 für Kaplan–Meier. Der Mittelwert von Kaplan–Meier bei $\tau$ = 0,9 (−5 %) gilt nur für die 58 % der Läufe, in denen die Kurve das Niveau erreicht; diese Läufe sind durch die Bedingung ausgewählt. Ob Kaplan–Meier bei $\tau$ = 0,9
im Ganzen verzerrt ist, ist mit dieser Messung nicht entschieden.

**Mehrkosten der Bestellmenge in % (Messung 2, Lognormal-Nachfrage, 1000 Läufe).** Mehrkosten = erwartete Kosten der geschätzten Menge gegen die wahre Verteilung geteilt durch die der besten Menge, minus 1; Fehlmenge $\tau$, Überschuss $1 - \tau$ je Stück.

| Bestand (Median) | ausverkauft | τ | aus den Verkäufen | Kaplan–Meier (nur wo schätzbar) | Kaplan–Meier mit Rückfall „größter Wert“ | zensiertes Lognormal |
|---|---|---|---|---|---|---|
| 70 | 73,0 % | 0,5 | 25,6 | 0,7 (99,8 % schätzbar) | 0,8 | 0,3 |
| 70 | 73,0 % | 0,8 | 73,3 | 3,9 (47 % schätzbar) | 3,0 | 0,9 |
| 70 | 73,0 % | 0,9 | 128,9 | 18,8 (25 % schätzbar) | 17,9 | 1,3 |
| 100 | 50,0 % | 0,5 | 6,9 | 0,2 (100 % schätzbar) | 0,2 | 0,1 |
| 100 | 50,0 % | 0,8 | 29,0 | 1,5 (95 % schätzbar) | 2,2 | 0,4 |
| 100 | 50,0 % | 0,9 | 56,0 | 3,3 (58 % schätzbar) | 3,0 | 0,6 |
| 150 | 24,3 % | 0,5 | 0,5 | 0,2 (100 % schätzbar) | 0,2 | 0,1 |
| 150 | 24,3 % | 0,8 | 5,3 | 0,3 (100 % schätzbar) | 0,3 | 0,2 |
| 150 | 24,3 % | 0,9 | 13,7 | 1,3 (99,5 % schätzbar) | 1,4 | 0,3 |

In diesen 9 Zellen ist das zensierte Lognormal in allen 9 am günstigsten (Standardfehler seiner Mehrkosten höchstens 0,06 Prozentpunkte). Die Verkaufsregel ist in allen 9 Zellen die teuerste der drei Verfahren, die in jedem Lauf eine Zahl liefern,
und teurer als jede andere Spalte; ihre Mehrkosten wachsen mit dem Ausverkauf und mit $\tau$ (bei Bestand 70 von 25,6 % bei $\tau$ = 0,5 auf 128,9 % bei $\tau$ = 0,9). Bei $\tau$ = 0,5 und Bestand 100 sind es 6,9 %.

**Falsche Form (Messung 3, Nachfrage gamma-verteilt mit gleichem Median und gleicher Streuung, Bestand 100, 1000 Läufe, 49,2 % ausverkauft).**

| τ | aus den Verkäufen | Kaplan–Meier (nur wo schätzbar) | Kaplan–Meier mit Rückfall „größter Wert“ | zensiertes Lognormal (falsche Form) |
|---|---|---|---|---|
| 0,5 | 6,5 | 0,2 (100 % schätzbar) | 0,2 | 0,2 |
| 0,8 | 31,9 | 1,5 (96 % schätzbar) | 2,2 | 3,4 |
| 0,9 | 62,5 | 3,1 (65 % schätzbar) | 3,2 | 10,6 |

Bei $\tau$ = 0,8 und 0,9 ist das Lognormal in dieser Welt teurer als Kaplan–Meier mit Rückfall, bei $\tau$ = 0,5 liegen beide gleichauf (0,19 % gegen 0,22 %, Unterschied etwa 3 Standardfehler). In allen drei Zeilen bleibt es weit unter der Verkaufsregel (höchstens ein Fünftel ihrer Mehrkosten). Die Formannahme ist Gewinn und Risiko zugleich:
In den gemessenen Lognormal-Zellen ist es der günstigste Weg (Messung 2); in dieser Gamma-Welt sind seine Mehrkosten bei $\tau$ = 0,9 das 3,3-Fache der Mehrkosten von Kaplan–Meier mit Rückfall (10,6 % gegen 3,2 %). Andere Formen sind nicht gemessen.

**Wie oft ist das Kaplan–Meier-Quantil schätzbar? (Messung 4, $\tau$ = 0,9, Bestand 100, Perioden als Reihe, 1000 Läufe je Zeile).**

| Perioden n | Läufe mit Kaplan–Meier-Quantil | aus den Verkäufen | Kaplan–Meier mit Rückfall | zensiertes Lognormal |
|---|---|---|---|---|
| 100 | 42 % | 57,3 | 8,3 | 3,1 |
| 250 | 46 % | 56,2 | 4,1 | 1,1 |
| 500 | 58 % | 56,0 | 3,0 | 0,6 |
| 1000 | 70 % | 55,8 | 2,4 | 0,3 |

Mit der Zahl der Perioden steigt der Anteil schätzbarer Läufe; die Mehrkosten der Verkaufsregel bleiben dagegen in diesen vier Zeilen bei 55,8 % bis 57,3 %: Mehr Perioden beheben ihren Fehler hier nicht. Je Bestandsniveau ($\tau$ = 0,9, 500 Perioden) ist das Quantil in 251 von 1000 Läufen (Bestand 70),
583 (Bestand 100) und 995 (Bestand 150) schätzbar; bei $\tau$ = 0,95 in 226, 330 und 813 von 1000.

**Gegenmittel: Testperioden mit sehr hohem Bestand (Messung 5, $\tau$ = 0,9, Bestand 100, 1000 Läufe).** Ein Anteil der Perioden bekommt den Bestandsmedian 300 statt 100 (Auswahl zufällig und unabhängig von der Nachfrage; gepaart mit der Standardwelt, gleiche Nachfrage und gleicher Zufall im Bestand).
Der Überschuss je Periode ist der mittlere Bestand über der Nachfrage, $\mathrm{E}[(B - D)^+]$ in der Simulation.

| Testperioden (Anteil) | ausverkauft | Läufe mit Kaplan–Meier-Quantil | Mehrkosten, Kaplan–Meier mit Rückfall (%) | Überschuss je Periode (Stück) |
|---|---|---|---|---|
| 0 % | 50,0 % | 58 % | 3,0 | 20,9 |
| 5 % | 47,6 % | 97 % | 4,0 | 29,9 |
| 10 % | 45,3 % | 99,8 % | 2,5 | 39,0 |
| 20 % | 40,6 % | 100 % | 1,6 | 57,2 |
| 30 % | 35,9 % | 100 % | 1,1 | 75,3 |

Der Anteil schätzbarer Läufe springt schon bei 5 % Testperioden von 58 % auf 97 %, die Mehrkosten fallen aber erst ab 10 % unter den Wert ohne Testperioden (Deutung, nicht gemessen: Bei 5 % stützt sich die Schätzung des Tails auf wenige genau beobachtete Werte). Der Überschuss je Periode
wächst von 20,9 auf 75,3 Stück; dessen Kosten stehen in den Mehrkosten nicht. Eine Aussage über den Gesamtnutzen des Gegenmittels folgt daraus nicht.

**Stabilität über Zufalls-Seeds (Messung 6, $\tau$ = 0,9, Bestand 100, je 1000 Läufe).**

| Zufalls-Seed | ausverkauft | aus den Verkäufen | Läufe mit Kaplan–Meier-Quantil | Kaplan–Meier mit Rückfall | zensiertes Lognormal |
|---|---|---|---|---|---|
| 35 | 50,0 % | 56,0 | 58 % | 3,0 | 0,6 |
| 101 | 50,0 % | 55,9 | 59 % | 2,9 | 0,6 |
| 202 | 49,9 % | 55,9 | 60 % | 2,9 | 0,6 |

In allen 9 000 verschiedenen angepassten Datensätzen dieser Messungen (9 verschiedene Zellen je 1000: Standard, Bestand 70 und 150, Gamma, zwei weitere Zufalls-Seeds, drei weitere Perioden-Reihen; die Standardzelle kommt als Zufalls-Seed 35 und als 500 Perioden noch zweimal vor) ist die Anpassung des Lognormals gelungen (0 Fehlschläge, eine Zählung für diese Zufalls-Seeds).

**Presets.** Die Hilfetexte der Presets zitieren Zahlen aus den Tabellen oben: „Standardwelt“ (50,0 %, 58 %), „Starker Ausverkauf“ (73,0 %, 128,9 %, 25 %), „Wenig Ausverkauf“ (24,3 %, 99,5 %, 13,7 %), „Mittleres Quantil“ (81,2 gegen 100,0, −19 %),
„Falsche Form“ (10,6 % gegen 3,2 %), „Kurze Historie“ (42 %, 8,3 % statt 3,0 %, 3,1 % statt 0,6 %); `tests/test_claims.py` prüft jede gegen die Messung.

**Die Kurve von Hand (Experiment 1, Handbeispiel; kein Zufall).** Dieselbe Tabelle zeigt die App in Brüchen; Prüfung gegen die Umverteilung nach rechts und gegen den Kern in `tests/`.

| Nachfrage x | genau beobachtet | ausverkauft („≥ x“) | im Risiko | S(x−) → S(x) |
|---|---|---|---|---|
| 6 | 1 | 0 | 6 | 1 → 5/6 |
| 7 | 1 | 0 | 5 | 5/6 → 2/3 |
| 8 | 0 | 1 | 4 | 2/3 (bleibt) |
| 9 | 1 | 0 | 3 | 2/3 → 4/9 |
| 10 | 0 | 1 | 2 | 4/9 (bleibt) |
| 12 | 0 | 1 | 1 | 4/9 (bleibt) |

Das Quantil 0,5 ist **9** (der kleinste genaue Wert mit $S \le 1/2$). Das Quantil 0,8 ist **nicht schätzbar**: Die Kurve bleibt bei 4/9 stehen, erreichbar sind nur Quantile bis $\tau = 5/9$. Aus den Verkäufen ergibt sich ein Median von 8,5 (Mittel 26/3 ≈ 8,67).

## Ehrliche Grenzen

- **Eine Welt:** eine Nachfrage-Verteilung mit Streuung σ 0,5 (lognormal oder gamma mit gleicher Streuung), ein lognormaler Bestand mit σ 0,3, unabhängig von der Nachfrage, eine Kostenform. Andere Bestandsmuster, andere Streuungen, andere Kosten und andere Formen ändern die Größen, nicht unbedingt die Richtung.
- **Die Mehrkosten beziehen sich auf die Bestellmenge, nicht auf Gesamtkosten eines Betriebs.** Fehlmenge $\tau$ und Überschuss $1 - \tau$ je Stück sind die einzigen Kosten; feste Bestellkosten, Mindestmengen und Servicegrad-Ziele fehlen.
- **Kaplan–Meier „nur wo schätzbar“ ist ein bedingter Wert** (nur Läufe, in denen die Kurve das Niveau erreicht). Der Rückfall „größter Wert“ ist eine einfache Regel zur Anschauung, keine Empfehlung; eine andere Regel würde die Spalte ändern.
- **Das zensierte Lognormal ist an zwei Formen gemessen** (Lognormal-Welt: in 9 von 9 Zellen am günstigsten; Gamma-Welt: bei $\tau$ = 0,8 und 0,9 teurer als Kaplan–Meier mit Rückfall). Andere Formen sind nicht gemessen; welche Form echte Nachfrage hat, ist hier nicht Thema.
- **Die Testperioden** (Experiment 3) sind eine Messung in einer Welt; ihr Preis, der Überschuss, steht nicht in den Mehrkosten. Die Aussage „Schon 5 % Testperioden machen das Quantil in 97 % der Läufe schätzbar“ gilt für diese Welt ($\tau$ = 0,9, Bestand 100, Median 300 in den Testperioden).
- **Ein Verkauf von 0 gibt es nicht.** Die Eingabeprüfung des Kerns verlangt Werte über 0; Perioden ohne Verkauf gehören nicht in diese Rechnung (die Handtabelle verlangt Verkauf von mindestens 1).
- **Nur am Rechner geprüft** (Seitenaufruf im Browser, Regler über AppTest), nicht auf Mobilgeräten; Plotly-Diagramme nur angesehen, nicht mit Mausklicks durchgespielt.
- **Literatur:** Kaplan und Meier (1958) ist aus dem Gedächtnis genannt, das Original ist nicht eingesehen; das Verfahren der Umverteilung nach rechts (Efron) ist als Orakel verwendet, seine Quelle nicht eingesehen. Es wird keine Aussage der Literatur wiedergegeben.

## Annahmen, Grenzen und bekannte Probleme

(Derselbe Text steht in der App im Expander „⚠️ Annahmen, Grenzen und bekannte Probleme“; ein Test vergleicht beide.)

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

## Tests

Die Suite hat **136 Tests** (`pytest --collect-only -q tests`); Orakel-, Eigenschafts- und Modell-Tests laufen in wenigen Sekunden; den größeren Teil der Laufzeit bestimmen der AppTest und die Neuberechnung der Messzellen in `tests/test_claims.py`.

| Datei | Prüft |
|---|---|
| `tests/test_core.py` | kopierte Kernfunktionen (Kaplan–Meier, Quantil mit „nicht schätzbar“, Eingabeprüfung, SplitMix64, BFGS) gegen das Handbeispiel; Funktionstexte gegen die Quelle |
| `tests/test_model.py` | Handtabellen aller vier Muster in Brüchen, schnelles Kaplan–Meier-Quantil gegen den Kern (150 Datensätze mit Gleichständen), Gradient des Lognormals gegen finite Differenz, Kosten von Hand und gegen SciPy, Welt, Läufe |
| `tests/test_oracle_zensierte_nachfrage.py` | Kaplan–Meier gegen Umverteilung nach rechts in Brüchen (250 gezogene Zufallsdatensätze mit Gleichständen, davon 237 mit mindestens einem genauen Wert), Lognormal gegen `scipy.optimize` auf selbst geschriebener Likelihood (21 Datensätze), Rastersuche und geschlossener Fall, erwartete Kosten gegen Integration und Simulation, beste Menge gegen numerische Minimierung, große Datensätze mit bekannter Wahrheit |
| `tests/test_eigenschaften_zensierte_nachfrage.py` | ohne Ausverkauf gleich der empirischen Verteilung, Quantil nie über dem größten beobachteten Wert, Verkaufsquantil nie über dem der Nachfrage, Monotonie in τ, Skalieren, Umsortieren, gepaarte Bestandsniveaus, Mehrkosten nie negativ |
| `tests/test_claims.py` | jede Zahl dieser README: Messzellen neu gerechnet gegen die eingefrorene Datei, Tabellen und Prosa-Zahlen gegen die README, Aussagen der Preset-Hilfen und der Grenzen-Tabelle der App, Abstand zur Vormessung |
| `tests/test_verdicts.py` | jedes Verdikt der App an handgesetzten Werten (Schwellen, jede Zahl im Text) |
| `tests/test_edge_cases.py` | n = 1 und 2, alle ausverkauft, nie ausverkauft, Gleichstand, Eingabeprüfung, Lognormal ohne genug Werte, Reglergrenzen, Meldungen |
| `tests/test_presets.py`, `test_layout.py` | Presets und Reglergrenzen; Schnellstart oben in der Seitenleiste, fester Kopf, Hervorhebung des aktiven Presets, jeder Regler hebt sie auf |
| `tests/test_visualization.py` | alle Abbildungen, Achsen gesperrt |
| `tests/test_app.py`, `test_footer.py`, `test_runtime_imports.py` | AppTest-Rauchtest (jeder Button, Regler an Minimum und Maximum, jede Auswahl, jedes Preset), Footer, Laufzeit-Importe gegen `requirements.txt` |

**Zahl → Test.** Zahlen aus den Messungen 1 bis 6 und aus den Presets: eingefroren in der JSON-Datei und in `tests/test_claims.py` mit dem Code der Demo neu gerechnet (Band 1e-6, Zählungen exakt); die Tabellen der README stammen Wort für Wort aus der Datei. Die Brüche der
Handtabelle: exakt in `tests/test_model.py` und `tests/test_oracle_zensierte_nachfrage.py`. Die Kostenformeln (Wahrheit, beste Menge): `tests/test_model.py` (Handwerte) und `tests/test_oracle_zensierte_nachfrage.py` (Integration, Simulation mit Band). Weitere Zahlen
im Text sind Handwerte (Brüche), Rechnungen aus der JSON-Datei oder die Zahlen der Vormessung, die nur im Abschnitt „Befunde und Korrekturen gegenüber dem Plan“ stehen und in `tests/test_claims.py` gegen diese Messung gehalten werden.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Annahmen, Formeln |
| `znf_core.py` | Kopie aus `ueberlebenszeit-planung/kern/uz_core.py` (SplitMix64, Eingabeprüfung, Kaplan–Meier, Quantil, BFGS); wird nicht verändert |
| `znf_estimators.py` | Quantil aus den Verkäufen, schnelles Kaplan–Meier-Quantil, zensiertes Lognormal (Likelihood, Gradient, Anpassung) |
| `znf_model.py` | Kosten gegen die wahre Verteilung, Läufe und Zusammenfassung, Handtabelle in Brüchen |
| `znf_scenario.py` | Welt (World), Datenerzeugung mit SplitMix64, die vier Muster von Experiment 1 |
| `znf_evaluation.py` | Formatierung, Verdikte als reine Funktionen |
| `znf_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets, Annahmen-Text |
| `znf_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `znf_layout.py` | Fester Kopf (Stylesheet) |
| `znf_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/measure_zensierte_nachfrage.py` | Messskript; schreibt `tools/measure_zensierte_nachfrage_result.json` (eingefroren) und bildet die Tabellen dieser README |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, SciPy, Plotly |
| `tests/` | siehe „Tests“ |

## Bewusst nicht umgesetzt

- **Andere Bestandsmuster** (Bestand, der an einer Prognose hängt: informative Zensierung), **Saison und Trend** der Nachfrage, nachgelieferte Fehlmengen.
- **Andere Kostenformen** (feste Bestellkosten, Mindestmengen, Servicegrad-Ziele) und die Bestellregel selbst (Bestandspolitiken, Lagerfortschreibung).
- **Konfidenzbänder** von Kaplan–Meier und Unsicherheit des angepassten Lognormals für einen einzelnen Datensatz; gezeigt wird die Streuung über Läufe.
- **Weitere Formen** (Weibull, Poisson, negative Binomialverteilung) und eine Auswahl der Form aus den Daten.
- **Echte Lagerdaten:** Alle Zahlen sind erzeugt.
- **PDF-Export.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
