"""Zensierte Nachfrage: Ausverkauft heißt mindestens so viel - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ist ein Artikel ausverkauft, kennt man die Nachfrage der Periode nur als „mindestens so viel wie der Bestand“. Vier Experimente: das Verfahren von Hand an sechs Perioden (Brüche),
das Quantil der Nachfrage aus den Verkäufen gegen Kaplan–Meier, die Grenze von Kaplan–Meier (das Plateau) und die Mehrkosten der Bestellmenge (Newsvendor) bei vier Verfahren.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace
from fractions import Fraction

import streamlit as st

import znf_constants as C
import znf_core as K
import znf_evaluation as ev
import znf_model as M
import znf_scenario as S
import znf_visualization as V
from znf_layout import dunkles_design, kopf_css
from znf_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Zensierte Nachfrage – Sebastian Hanisch", layout="wide")

HAND_TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)


@st.cache_data(show_spinner=False)
def _study(world: S.World, with_lognormal: bool = True):
    """Alle Quantile der Auswahl in einem Durchgang: ein Wechsel von τ rechnet nichts neu."""
    return M.run_study(world, C.TAU_OPTIONS, C.REPS, with_lognormal=with_lognormal)


def md_table(headers, rows) -> str:
    """Markdown-Tabelle aus Texten (kein | in den Zellen)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in znf_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📦 Zensierte Nachfrage: Ausverkauft heißt mindestens so viel")
st.markdown(
    """
Ist ein Artikel **ausverkauft**, kennt man die Nachfrage der Periode nur als **„mindestens so viel wie der Bestand“**: Die Verkäufe sind dann **zensiert**. Wer die Verkäufe als Nachfrage liest und daraus die **Bestellmenge** (das Quantil nach der Newsvendor-Regel) ableitet,
bestellt zu wenig. **Kaplan–Meier**, auf die zensierte Nachfrage angewendet, schätzt das Quantil ohne Formannahme, hat aber eine Grenze: Ist der größte Wert ein Ausverkauf, bleibt die Kurve auf einem Plateau stehen, und jedes Quantil darüber hinaus ist **nicht schätzbar**.
Die Demo prüft das an erzeugten Daten, in Brüchen von Hand und über die **Mehrkosten** der Bestellmenge.

**Annahme in dieser Demo: Der Bestand einer Periode ist unabhängig von ihrer Nachfrage** (dann ist die Zensierung unabhängig vom Ereignis, und Kaplan–Meier ist gültig); die Welt unten erfüllt sie, geprüft wird sie hier nicht.
Hängt der Bestand an einer Prognose der Nachfrage, ist die Zensierung informativ: dazu [informatives-zensieren-demo](https://github.com/sebastian-hanisch/informatives-zensieren-demo). Alle Zahlen sind erzeugt, keine Zahlen aus echten Lagern.
Die Demo löst den Ausblick „Zensierte Nachfrage schätzen“ aus [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo) ein; wer die Bestellregel direkt auf den Kosten lernt, findet das in [newsvendor-direkt-demo](https://github.com/sebastian-hanisch/newsvendor-direkt-demo).

Weiter unten: **🔬 Experiment 1** (von Hand, in Brüchen), **❓ Kernfrage mit Experiment 2** (Verkäufe gegen Kaplan–Meier), **🔬 Experiment 3** (die Grenze), **🔬 Experiment 4** (Mehrkosten der Bestellmenge),
**🚧 Wo die Annahmen enden** sowie die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Kaplan–Meier bei Ausverkauf", expanded=True):
    st.markdown(
        """
1. **Genau oder mindestens.** Bleibt Bestand übrig, ist der Verkauf die Nachfrage (**genau beobachtet**). Gleicht der Verkauf dem Bestand, war die Nachfrage **mindestens** so groß (**ausverkauft**, zensiert); Gleichstand zählt als ausverkauft.
2. **Im Risiko.** Bei einem genau beobachteten Wert $x$ stehen alle Perioden **im Risiko**, deren Wert $\\ge x$ ist, auch die ausverkauften mit Bestand $\\ge x$: Sie haben gezeigt, dass ihre Nachfrage mindestens $x$ war.
3. **Kurve.** $S(x)$ sinkt bei jedem genau beobachteten Wert um den Anteil $d/n$ der Perioden im Risiko; ausverkaufte Werte lassen $S$ stehen und verkleinern nur die Risikomenge.
4. **Quantil.** Das Quantil $\\tau$ ist der kleinste genau beobachtete Wert mit $S \\le 1 - \\tau$. Fällt die Kurve nie so tief, ist es **nicht schätzbar**. Der naheliegende Rückfall (größter Wert) ist eine einfache Regel, keine Empfehlung.
5. **Zensiertes Lognormal.** Alternativ legt man eine Form fest (Lognormal) und passt sie per Maximum Likelihood an: Ein genau beobachteter Wert geht mit seiner Dichte ein, ein Ausverkauf nur mit der Wahrscheinlichkeit, mindestens so groß zu sein.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt der Daten** (Experimente 2 bis 4)")
    m = st.slider("Bestandsniveau (Median des Bestands)", *bounds("m_slider"), step=C.M_STEP, key="m_slider",
                  help="Median des Bestands je Periode (lognormal, σ 0,3, unabhängig von der Nachfrage). Die Nachfrage hat den Median 100: Je weiter der Bestand darunter liegt, desto öfter ist ausverkauft.")
    form = st.radio("Verteilung der Nachfrage", options=list(C.FORMS), format_func=C.FORM_LABELS.get, key="form_radio",
                    help="Lognormal (Median 100, σ 0,5) oder gamma-verteilt mit demselben Median und derselben Streuung. Das zensierte Lognormal ist bei Gamma die falsche Form.")
    n = st.slider("Perioden n je Datensatz", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="Wie viele Perioden ein Datensatz hat. Jede Zelle rechnet 1000 Datensätze mit festen Zufalls-Seeds.")
    st.markdown("**Frage**")
    tau = st.select_slider("Quantil τ der Bestellmenge", options=list(C.TAU_OPTIONS), format_func=ev.tau_text, key="tau_select",
                           help="Kritisches Verhältnis b/(b + h) aus Fehlmengenkosten b und Überschusskosten h: Die beste Bestellmenge ist das Quantil τ der Nachfrage. Je höher τ, desto weiter im Tail der Nachfrage.")
    seed = st.number_input("Zufalls-Seed der Daten", *bounds("seed_input"), key="seed_input", step=1,
                           help="Jeder der 1000 Läufe hat einen eigenen, festen Zufallsstrom, der aus diesem Zufalls-Seed abgeleitet ist.")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"m_slider": int(m), "tau_select": float(tau), "n_slider": int(n), "form_radio": str(form), "seed_input": int(seed)})
world = S.World(m=float(m), n=int(n), form=str(form), boost=0.0, seed=int(seed))

# --- Experiment 1: von Hand ------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🧮 Das Verfahren von Hand: sechs Perioden")
st.subheader("🔬 Experiment 1: Die Kurve in Brüchen")
st.markdown(
    """
Sechs Perioden mit **Bestand** und **Verkauf** (ganze Stück). Ist der Verkauf kleiner als der Bestand, ist er die Nachfrage; ist er gleich dem Bestand, war die Nachfrage **mindestens** so groß (**ausverkauft**, zensiert).
Die Tabelle rechnet bei jedem genau beobachteten Wert $x$: **im Risiko** sind alle Perioden mit Wert $\\ge x$, und $S$ sinkt um den Anteil der genau beobachteten Perioden darunter. Wähle ein Muster und das Quantil.
"""
)
c1, c2, c3 = st.columns(3)
pattern = c1.selectbox("Muster der sechs Perioden", list(S.PATTERNS), key="pattern_select",
                       help="Jedes Muster sind sechs Paare (Bestand, Verkauf). Sie zeigen das Handbeispiel, eine Reihe ohne Ausverkauf, eine mit lauter Ausverkäufen und einen Gleichstand zwischen einem genauen und einem ausverkauften Wert.")
hand_tau = c2.select_slider("Quantil τ der Handrechnung", options=list(HAND_TAUS), value=0.8, format_func=ev.tau_text, key="hand_tau_select",
                            help="Im Handbeispiel ist das Quantil 0,5 schätzbar (Median), das Quantil 0,8 nicht: Die Kurve bleibt bei 4/9 stehen.")
pairs = S.PATTERNS.get(pattern, S.PATTERNS[S.DEFAULT_PATTERN])
table = M.hand_table(pairs)
st.markdown("**Die Perioden (Bestand, Verkauf)**: " + ", ".join(f"({b}, {v})" for b, v in pairs))
shown = c3.slider("Zeilen der Tabelle zeigen", 1, len(table["rows"]), len(table["rows"]), key=f"rows_slider_{list(S.PATTERNS).index(pattern) if pattern in S.PATTERNS else 0}",
                  help="Die Tabelle wächst Wert für Wert; das Verdikt rechnet immer mit allen Zeilen.") if len(table["rows"]) > 1 else 1
st.markdown(md_table(
    ["Nachfrage x", "genau beobachtet", "ausverkauft („≥ x“)", "im Risiko", "S(x−) → S(x)"],
    [[r["value"], r["exact"], r["censored"], r["at_risk"], f"{ev.frac(r['s_prev'])} → {ev.frac(r['s'])}" if r["exact"] else f"{ev.frac(r['s'])} (bleibt)"] for r in table["rows"][:shown]],
))
st.plotly_chart(V.build_hand(table, float(hand_tau), shown), width="stretch", key="hand_chart")
kind, text = ev.verdict_hand(table, Fraction(str(hand_tau)))
getattr(st, kind)(text)

st.markdown("---")

# --- Kernfrage und Experiment 2: Verkäufe gegen Kaplan-Meier --------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie weit liegt das Quantil aus den Verkäufen daneben, und was leistet Kaplan–Meier?")
st.subheader("🔬 Experiment 2: Quantil aus den Verkäufen gegen Kaplan–Meier")
st.markdown(
    f"""
Welt: Nachfrage {C.FORM_LABELS[world.form].lower()} (Median 100, σ 0,5), Bestand lognormal (Median {int(world.m)}, σ 0,3), unabhängig von der Nachfrage, {world.n} Perioden je Datensatz; Verkauf ist das Minimum von beiden.
Es laufen **{C.REPS} Datensätze** mit festen Zufalls-Seeds; jeder schätzt das Quantil {ev.tau_text(tau)} der Nachfrage zweimal: aus den Verkäufen, als wären sie die Nachfrage, und mit Kaplan–Meier (ausverkaufte Perioden als zensiert).
"""
)
try:
    with st.spinner(f"Rechnet {C.REPS} Datensätze …"):
        study = _study(world)
except K.SurvivalError as err:
    st.error(f"Die Rechnung ist nicht möglich: {err}")
    st.stop()
j = study.taus.index(float(tau))
sales_c, km_c = M.summarize(study, tau, "sales"), M.summarize(study, tau, "km")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Wahrheit (Quantil der Nachfrage)", ev.num(float(study.q_true[j])), help="Quantil τ der wahren Verteilung der Nachfrage; zugleich die beste Bestellmenge.")
k2.metric("Aus den Verkäufen", ev.num(sales_c.q_mean), delta=ev.signed_pct(sales_c.q_mean / float(study.q_true[j]) - 1.0), delta_color="inverse", help="Mittel über die Läufe.")
k3.metric("Aus Kaplan–Meier", ev.num(km_c.q_mean) if km_c.defined else "–", delta=f"in {ev.share_pct(km_c.defined, km_c.reps)} der Läufe schätzbar", delta_color="off",
          help="Mittel über die Läufe, in denen die Kurve das Niveau erreicht; in den übrigen Läufen gibt es keine Zahl.")
k4.metric("Ausverkauft", ev.pct(float(study.sold_out.mean()), 0), help="Mittlerer Anteil der Perioden, in denen der Verkauf dem Bestand gleicht.")
st.caption(f"Mittel über {study.reps} Läufe. Standardfehler des Mittels: aus den Verkäufen {ev.num(sales_c.q_se, 2)}" + (f", Kaplan–Meier {ev.num(km_c.q_se, 2)}." if km_c.defined else "."))
st.plotly_chart(V.build_quantile_runs(study, tau), width="stretch", key="quantile_chart")
st.caption("Kasten: mittlere Hälfte der Läufe, gestrichelte Linie im Kasten: Mittel. Grau: die Wahrheit. Kaplan–Meier zeigt nur die Läufe, in denen die Kurve das Niveau 1 − τ erreicht.")
kind, text = ev.verdict_quantile(study, tau)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3: die Grenze ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Die Grenze von Kaplan–Meier")
st.markdown(
    """
Die Kurve fällt nur bei **genau beobachteten** Werten. Ist der größte Wert ein Ausverkauf, bleibt sie auf einem **Plateau** stehen, und jedes Quantil mit $1 - \\tau$ unter dem Plateau ist **nicht schätzbar**.
Oben ein Datensatz mit seinem Plateau, darunter der Anteil der Läufe, in denen das Quantil schätzbar ist, gegen $\\tau$ und das Bestandsniveau. Ein Gegenmittel: in einem Anteil der Perioden den Bestand **zufällig und unabhängig von der Nachfrage** sehr hoch ansetzen
(Median 300 statt des gewählten), damit die Nachfrage dort genau beobachtet wird.
"""
)
boost = st.select_slider("Anteil der Testperioden mit sehr hohem Bestand", options=list(C.BOOST_SHARES), format_func=lambda b: f"{round(100 * b)} %", key="boost_select",
                         help="In diesem Anteil der Perioden hat der Bestand den Median 300 (dreifacher Median der Nachfrage). Die Perioden werden zufällig gewählt, unabhängig von der Nachfrage. Der Überschuss in diesen Perioden steht in keiner Mehrkosten-Zahl.")
world3 = replace(world, boost=float(boost))
try:
    run = ev.example_run(world3)
    sample = S.make_sample(world3, run)
    km = K.kaplan_meier(sample.sales, sample.observed)
except K.SurvivalError as err:
    st.error(f"Der Datensatz ist nicht auswertbar: {err}")
    st.stop()
info = ev.plateau_info(km, sample.sales, sample.observed)
st.plotly_chart(V.build_km_example(km, float(tau), float(max(sample.sales.max(), 1.0)) * 1.02), width="stretch", key="plateau_chart")
st.caption(f"Ein Datensatz (Lauf {run + 1} der Welt, {round(100 * boost)} % Testperioden; gezeigt wird der erste der ersten 60 Läufe, in dem die Kurve auf einem Plateau stehen bleibt, sonst Lauf 1): "
           "durchgezogen Kaplan–Meier, gepunktet das Niveau 1 − τ, gestrichelt das Plateau.")
kind, text = ev.verdict_plateau(info, float(tau))
getattr(st, kind)(text)

try:
    with st.spinner(f"Rechnet {C.REPS} Datensätze je Bestandsniveau und je Anteil Testperioden …"):
        series = {f"Bestand {mm}": _study(replace(world, m=float(mm))) for mm in C.GRID_M}
        if int(world.m) not in C.GRID_M:
            series[f"Ihre Welt (Bestand {int(world.m)})"] = study
        boost_studies = {b: _study(replace(world, boost=b), False) for b in C.BOOST_SHARES}
except K.SurvivalError as err:
    st.error(f"Die Rechnung ist nicht möglich: {err}")
    st.stop()
st.plotly_chart(V.build_defined_series(series), width="stretch", key="defined_chart")
st.caption(f"Anteil der Läufe (je {C.REPS} Datensätze, {world.n} Perioden, Nachfrage {C.FORM_LABELS[world.form].lower()}), in denen die Kaplan–Meier-Kurve das Niveau 1 − τ erreicht.")

boost_rows = [{"boost": b, "share": M.defined_share(sb, float(tau)), "label": ev.share_pct(M.summarize(sb, float(tau), "km").defined, sb.reps), "leftover": float(sb.leftover.mean()), "cost_fb": M.summarize(sb, float(tau), "km_fallback").cost_mean}
              for b, sb in boost_studies.items()]
st.markdown(f"**Gegenmittel: Testperioden mit hohem Bestand** (Quantil {ev.tau_text(tau)}, Bestandsniveau {int(world.m)})")
st.plotly_chart(V.build_boost(boost_rows, float(tau)), width="stretch", key="boost_chart")
st.markdown(md_table(
    ["Testperioden (Anteil)", "Läufe mit Quantil", "Mehrkosten, Kaplan–Meier mit Rückfall (%)", "Überschuss je Periode (Mittel, Stück)"],
    [[ev.pct(r["boost"], 0), r["label"], ev.num(100 * r["cost_fb"]), ev.num(r["leftover"])] for r in boost_rows],
))
st.caption("Der Überschuss je Periode ist der mittlere Bestand, der über die Nachfrage hinaus übrig bleibt: der Preis des Gegenmittels, in keiner Mehrkosten-Zahl enthalten. "
           "Die Mehrkosten gelten für die Bestellmenge, die aus diesen Daten geschätzt wird.")
kind, text = ev.verdict_boundary(boost_studies[float(boost)], float(tau), float(boost))
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4: Kosten --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Mehrkosten der Bestellmenge")
st.markdown(
    f"""
**Kosten.** Fehlmenge kostet $\\tau$ je Stück, Überschuss $1 - \\tau$ je Stück (nur das Verhältnis zählt); die beste Bestellmenge ist das Quantil $\\tau$ der wahren Nachfrage. Die **Mehrkosten** einer Bestellmenge sind die erwarteten Kosten gegen die **wahre** Verteilung
(geschlossen gerechnet, keine Stichprobe) geteilt durch die der besten Menge, minus 1. Verglichen werden vier Wege zur Bestellmenge: die Verkäufe als Nachfrage, Kaplan–Meier (nur wo schätzbar), Kaplan–Meier mit dem Rückfall „größter beobachteter Wert“ und das zensierte Lognormal.
Nachfrage hier: **{C.FORM_LABELS[world.form]}**; die Verteilung wählst du in der Seitenleiste.
"""
)
st.plotly_chart(V.build_costs(study, float(tau)), width="stretch", key="cost_chart")
st.caption(f"Mittel über {study.reps} Läufe, Fehlerbalken ± 1 Standardfehler des Mittels; Bestandsniveau {int(world.m)}, Quantil {ev.tau_text(tau)}, {world.n} Perioden. Kaplan–Meier allein mittelt nur über die Läufe mit Zahl.")
kind, text = ev.verdict_costs(study, float(tau))
getattr(st, kind)(text)

st.markdown(f"**Dieselbe Rechnung über Bestandsniveau und τ** (Mehrkosten in %, Mittel über {C.REPS} Läufe, Nachfrage {C.FORM_LABELS[world.form].lower()}, {world.n} Perioden):")
rows = []
for mm in C.GRID_M:
    sm = series[f"Bestand {mm}"]
    for t in C.GRID_TAU:
        cells = {meth: M.summarize(sm, t, meth) for meth in M.METHODS}
        km_cell = cells["km"]
        rows.append([mm, ev.pct(float(sm.sold_out.mean())), ev.tau_text(t), ev.num(100 * cells["sales"].cost_mean),
                     (f"{ev.num(100 * km_cell.cost_mean)} ({ev.share_pct(km_cell.defined, km_cell.reps)} schätzbar)" if km_cell.defined else "keine Zahl (0 % schätzbar)"),
                     ev.num(100 * cells["km_fallback"].cost_mean), ev.num(100 * cells["lognormal"].cost_mean)])
st.markdown(md_table(["Bestand (Median)", "ausverkauft", "τ", "aus den Verkäufen", "Kaplan–Meier (nur wo schätzbar)", "Kaplan–Meier mit Rückfall „größter Wert“", "zensiertes Lognormal"], rows))
st.markdown(
    "In den gemessenen Lognormal-Zellen (Bestand 70/100/150, τ 0,5/0,8/0,9) ist das zensierte Lognormal am günstigsten. Wähle in der Seitenleiste „Gamma“ (andere Form, gleiche Streuung), und es liegt bei τ = 0,8 und 0,9 hinter Kaplan–Meier mit Rückfall, das keine Form braucht. "
    "Die Formannahme ist Gewinn und Risiko zugleich; andere Formen sind hier nicht gemessen. Die Rückfallregel „größter Wert“ ist eine einfache Regel zur Anschauung, keine Empfehlung."
)

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was gilt, wenn sie verletzt ist |
|---|---|
| **Bestand unabhängig von der Nachfrage** | Nicht gemessen: Die Welt dieser Demo zieht Bestand und Nachfrage getrennt. Hängt der Bestand an einer Prognose der Nachfrage, ist die Zensierung informativ, und Kaplan–Meier kann verzerrt sein; Richtung und Größe hängen am Mechanismus (Stück „Informatives Zensieren“ der Linie, [informatives-zensieren-demo](https://github.com/sebastian-hanisch/informatives-zensieren-demo)). |
| **Nachfrage unabhängig und gleich verteilt** | Nicht gemessen: Saison, Trend und Aktionen sind nicht erzeugt; die Zahlen sagen darüber nichts. |
| **Ausverkauf ist erkennbar, entgangene Nachfrage kommt nicht wieder** | Nicht gemessen: Kommt entgangene Nachfrage später als Nachbestellung, ist sie nicht zensiert, sondern verschoben; ein Ausverkauf kurz vor Ladenschluss gilt hier als genau beobachtet, wenn Bestand übrig ist. |
| **Form der Nachfrage (Lognormal)** | Gemessen (Experiment 4 und README): bei gamma-verteilter Nachfrage ist das zensierte Lognormal bei τ = 0,9 deutlich teurer als Kaplan–Meier mit Rückfall (10,6 % gegen 3,2 % Mehrkosten, Bestand 100, 500 Perioden, 1000 Läufe). |
| **Genug Perioden im Risiko** | Gemessen (README): Mit 100 Perioden ist das Kaplan–Meier-Quantil 0,9 bei Bestand 100 in 42 % der Läufe schätzbar, mit 1000 Perioden in 70 %. |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* je Periode $i$ der Bestand $B_i$ und der Verkauf $V_i = \min(D_i, B_i)$; die Nachfrage $D_i$ selbst ist nur zu sehen, wenn $V_i < B_i$. *Gesucht:* das Quantil $q_\tau = F_D^{-1}(\tau)$ der Nachfrage (die beste Bestellmenge) und die Mehrkosten einer geschätzten Bestellmenge.

| Symbol | Bedeutung |
|---|---|
| $D_i,\ B_i,\ V_i$ | Nachfrage, Bestand und Verkauf der Periode $i$; $V_i = \min(D_i, B_i)$ |
| $\delta_i$ | $1$, wenn die Nachfrage genau bekannt ist ($V_i < B_i$), $0$ bei Ausverkauf ($V_i = B_i$, Nachfrage $\ge B_i$) |
| $x_{(1)} < \dots < x_{(K)}$ | die verschiedenen genau beobachteten Werte |
| $n_k,\ d_k$ | Zahl der Perioden im Risiko bei $x_{(k)}$ (alle mit $V_i \ge x_{(k)}$) und Zahl der genau beobachteten bei $x_{(k)}$ |
| $\hat S(x)$ | Kaplan–Meier-Schätzer der Überlebensfunktion $P(D > x)$ |
| $\tau$ | Quantil der Bestellregel, $\tau = b/(b + h)$ |
| $b,\ h$ | Kosten je fehlendes und je überschüssiges Stück |
| $q_\tau,\ \hat q_\tau$ | wahres und geschätztes Quantil |
| $\Phi,\ \varphi$ | Verteilungs- und Dichtefunktion der Standardnormalverteilung |
| $\mu,\ \sigma$ | Parameter des Lognormals: $\ln D \sim \mathrm{N}(\mu, \sigma^2)$ |

**Verkäufe als Nachfrage.** In jeder Periode gilt $V_i \le D_i$, also liegt jedes Quantil der Verkäufe höchstens beim Quantil der Nachfrage. Bei unabhängigem Bestand gilt $P(V > x) = P(D > x)\,P(B > x)$, die Verkäufe sind „kürzer“ als die Nachfrage, genau um den Faktor $P(B > x)$.

**Kaplan–Meier.**

$$\hat S(x) = \prod_{x_{(k)} \le x} \Big( 1 - \frac{d_k}{n_k} \Big), \qquad \hat q_\tau = \min\{\, x_{(k)} : \hat S(x_{(k)}) \le 1 - \tau \,\}$$

Das Quantil ist **nicht schätzbar**, wenn $\hat S(x_{(K)}) > 1 - \tau$: Die Kurve bleibt nach dem größten genau beobachteten Wert stehen. Erreichbar sind nur $\tau \le 1 - \hat S(x_{(K)})$. Ausverkaufte Perioden fehlen in $d_k$ und bleiben bis zu ihrem Wert in $n_k$.

**Zensiertes Lognormal.** Mit $z_i = (\ln V_i - \mu)/\sigma$ ist die Log-Likelihood

$$\ell(\mu, \sigma) = \sum_{\delta_i = 1} \Big[ -\ln \sigma - \ln V_i - \tfrac12 \ln 2\pi - \tfrac12 z_i^2 \Big] + \sum_{\delta_i = 0} \ln \Phi(-z_i), \qquad \hat q_\tau = \exp\big(\hat\mu + \hat\sigma\,\Phi^{-1}(\tau)\big).$$

Mit $\lambda_i = \varphi(z_i)/\Phi(-z_i)$ lauten die Ableitungen (nach $\mu$ und $\ln\sigma$): $\ \sum_{\delta=1} z_i/\sigma + \sum_{\delta=0} \lambda_i/\sigma\ $ und $\ \sum_{\delta=1} (z_i^2 - 1) + \sum_{\delta=0} \lambda_i z_i$. Gelöst wird per BFGS mit diesem Gradienten.

**Newsvendor.** Mit Fehlmengenkosten $\tau$ und Überschusskosten $1 - \tau$ je Stück sind die erwarteten Kosten der Bestellmenge $q$

$$C(q) = \tau\,\mathrm{E}[(D - q)^+] + (1 - \tau)\,\mathrm{E}[(q - D)^+], \qquad C'(q) = F_D(q) - \tau,$$

also ist $q_\tau$ die beste Menge. Wegen $\mathrm{E}[(q - D)^+] = q - \mathrm{E}[D] + \mathrm{E}[(D - q)^+]$ genügt $\mathrm{E}[(D - q)^+]$. Für das Lognormal mit $\mathrm{E}[D] = e^{\mu + \sigma^2/2}$:

$$\mathrm{E}[(D - q)^+] = e^{\mu + \sigma^2/2}\,\Phi\!\Big(\frac{\mu + \sigma^2 - \ln q}{\sigma}\Big) - q\,\Phi\!\Big(\frac{\mu - \ln q}{\sigma}\Big),$$

für das Gamma mit Form $k$ und Skala $\theta$ (regularisierte obere unvollständige Gamma-Funktion $Q$):

$$\mathrm{E}[(D - q)^+] = k\theta\, Q(k + 1, q/\theta) - q\, Q(k, q/\theta).$$

Die **Mehrkosten** einer geschätzten Menge $\hat q$ sind $C(\hat q)/C(q_\tau) - 1$, gerechnet gegen die wahre Verteilung.

**Welt.** Nachfrage lognormal mit Median 100 und $\sigma = 0{,}5$; gamma-verteilt mit demselben Median und derselben Streuung, also Form $k = 1/(e^{\sigma^2} - 1)$. Bestand lognormal mit $\sigma = 0{,}3$, unabhängig. Ausverkauft heißt $D_i \ge B_i$.

**Voraussetzungen und Konvention.** Bestand unabhängig von der Nachfrage; Risikomenge $\{ i : V_i \ge x_{(k)} \}$, bei gleichem Wert zählt das genaue Ereignis vor dem ausverkauften; Gleichstand $V_i = B_i$ gilt als ausverkauft.

**Literatur.** E. L. Kaplan und P. Meier (1958), Nonparametric Estimation from Incomplete Observations, Journal of the American Statistical Association 53, 457–481: aus dem Gedächtnis genannt, nicht eingesehen. Die Formeln oben sind gegen unabhängige Rechnungen geprüft (`tests/test_oracle_zensierte_nachfrage.py`).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
