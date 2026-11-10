"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import znf_constants as C
import znf_model as M


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.28), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _tau_label(tau: float) -> str:
    return f"{tau:g}".replace(".", ",")


# ------------------------------------------------------------------------------------------------ Experiment 1


def build_hand(table: dict, tau: float, shown: int):
    """Kaplan–Meier-Treppe der Handtabelle (genaue Werte als Knicke, ausverkaufte Werte als Kreuze auf der Kurve) und die Linie 1 − τ.
    `shown`: Zahl der Tabellenzeilen, bis zu denen die Kurve gezeichnet wird."""
    fig = go.Figure()
    rows = table["rows"][:shown]
    if not rows:
        return _base(fig)
    xs = [0.0] + [float(r["value"]) for r in rows]
    ys = [1.0] + [float(r["s"]) for r in rows]
    x_end = float(rows[-1]["value"]) + 1.0
    fig.add_trace(go.Scatter(x=xs + [x_end], y=ys + [ys[-1]], mode="lines", line=dict(color=C.COLOR_KM, width=3, shape="hv"), name="Kaplan–Meier: Überleben S(x) der Nachfrage",
                             hovertemplate="Nachfrage %{x:g}: S = %{y:.4f}<extra></extra>"))
    cx = [float(r["value"]) for r in rows if r["censored"]]
    cy = [float(r["s"]) for r in rows if r["censored"]]
    if cx:
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", marker=dict(symbol="x", size=11, color=C.COLOR_SALES, line=dict(width=2)), name="ausverkauft (Nachfrage ≥ Wert)",
                                 hovertemplate="ausverkauft bei %{x:g}<extra></extra>"))
    fig.add_hline(y=1.0 - tau, line=dict(color=C.COLOR_LINE, dash="dot"), annotation_text=f"1 − τ = {_tau_label(1.0 - tau)}", annotation_position="top left")
    fig.update_xaxes(title_text="Nachfrage (Stück)", range=[0, x_end])
    fig.update_yaxes(title_text="Anteil der Perioden mit größerer Nachfrage", range=[0, 1.05])
    return _base(fig)


# ------------------------------------------------------------------------------------------------ Experiment 2


def build_quantile_runs(study: M.Study, tau: float):
    """Je Lauf ein Punkt: das Quantil der Nachfrage aus den Verkäufen und aus Kaplan–Meier (nur Läufe mit Zahl), dazu die Wahrheit als Linie."""
    j = study.taus.index(float(tau))
    fig = go.Figure()
    sales = study.q["sales"][:, j]
    km = study.q["km"][:, j]
    km = km[np.isfinite(km)]
    for values, name, color in ((sales, "aus den Verkäufen", C.COLOR_SALES), (km, f"Kaplan–Meier ({km.size} von {study.reps} Läufen mit Zahl)", C.COLOR_KM)):
        fig.add_trace(go.Box(y=np.round(values, 3), name=name, boxpoints="outliers", marker=dict(color=color, size=4, opacity=0.6), line=dict(color=color), fillcolor="rgba(0,0,0,0)",
                             boxmean=True, hovertemplate="%{y:.1f}<extra>" + name + "</extra>"))
    q_true = float(study.q_true[j])
    fig.add_hline(y=q_true, line=dict(color=C.COLOR_TRUTH, width=2))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=C.COLOR_TRUTH, width=2), name=f"Wahrheit {q_true:.1f}".replace(".", ","), hoverinfo="skip"))
    fig.update_yaxes(title_text=f"Quantil {_tau_label(tau)} der Nachfrage (Stück)")
    return _base(fig, height=420)


# ------------------------------------------------------------------------------------------------ Experiment 3


def build_km_example(km: dict, tau: float, x_max: float):
    """Die Kurve eines Datensatzes: sie fällt bis zum größten genau beobachteten Wert und bleibt dort stehen; die Linie 1 − τ zeigt, ob sie das Niveau erreicht."""
    fig = go.Figure()
    xs = [0.0] + [float(x) for x in km["t"]]
    ys = [1.0] + [float(s) for s in km["s"]]
    fig.add_trace(go.Scatter(x=xs + [x_max], y=ys + [ys[-1]], mode="lines", line=dict(color=C.COLOR_KM, width=3, shape="hv"), name="Kaplan–Meier: S(x)", hovertemplate="Nachfrage %{x:.1f}: S = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0 - tau, line=dict(color=C.COLOR_LINE, dash="dot"), annotation_text=f"1 − τ = {_tau_label(1.0 - tau)}", annotation_position="top left")
    if ys[-1] > 0.0:        # ein Plateau gibt es nur, wenn die Kurve nicht bei 0 endet
        fig.add_hline(y=ys[-1], line=dict(color=C.COLOR_SALES, dash="dash"), annotation_text=f"Plateau S = {ys[-1]:.3f}".replace(".", ","), annotation_position="top right")
    fig.update_xaxes(title_text="Nachfrage (Stück)", range=[0, x_max])
    fig.update_yaxes(title_text="Anteil der Perioden mit größerer Nachfrage", range=[0, 1.05])
    return _base(fig)


def build_defined_series(series: dict[str, M.Study]):
    """Anteil der Läufe, in denen das Kaplan–Meier-Quantil schätzbar ist, gegen τ; eine Linie je Welt (Beschriftung = Schlüssel)."""
    fig = go.Figure()
    colors = [C.COLOR_SALES, C.COLOR_KM, C.COLOR_LOGN, C.COLOR_LINE]
    for (label, study), color in zip(series.items(), colors):
        fig.add_trace(go.Scatter(x=list(study.taus), y=[M.defined_share(study, t) for t in study.taus], mode="lines+markers", line=dict(color=color, width=3), marker=dict(size=8), name=label,
                                 hovertemplate="τ = %{x}: %{y:.1%} der Läufe<extra>" + label + "</extra>"))
    fig.update_xaxes(title_text="Quantil τ der Bestellregel", tickvals=list(C.TAU_OPTIONS))
    fig.update_yaxes(title_text="Läufe mit Kaplan–Meier-Quantil", tickformat=".0%", range=[0, 1.05])
    return _base(fig, height=400)


def build_boost(rows: list[dict], tau: float):
    """Anteil der Läufe mit Kaplan–Meier-Quantil gegen den Anteil der Testperioden mit hohem Bestand."""
    fig = go.Figure(go.Bar(x=[f"{round(100 * r['boost'])} %" for r in rows], y=[r["share"] for r in rows], marker_color=C.COLOR_KM,
                           text=[r["label"] for r in rows], textposition="outside", hovertemplate="%{x} Testperioden: %{y:.1%}<extra></extra>"))
    fig.update_xaxes(title_text="Anteil der Perioden mit sehr hohem Bestand", type="category")
    fig.update_yaxes(title_text=f"Läufe mit Quantil {_tau_label(tau)}", tickformat=".0%", range=[0, 1.15])
    return _base(fig, height=340)


# ------------------------------------------------------------------------------------------------ Experiment 4


def build_costs(study: M.Study, tau: float):
    """Mehrkosten der Bestellmenge gegen die beste (Mittel über die Läufe, Fehlerbalken ± 1 Standardfehler) für die vier Verfahren."""
    cells = {m: M.summarize(study, tau, m) for m in M.METHODS}
    colors = {"sales": C.COLOR_SALES, "km": C.COLOR_KM, "km_fallback": C.COLOR_KM, "lognormal": C.COLOR_LOGN}
    labels = {   # kurze Zeilen: bei schmalem Fenster überlappen lange Beschriftungen
        "sales": "aus den<br>Verkäufen",
        "km": f"Kaplan–Meier<br>nur wo schätzbar<br>({cells['km'].defined} von {cells['km'].reps} Läufen)",
        "km_fallback": "Kaplan–Meier<br>mit Rückfall<br>„größter Wert“",
        "lognormal": "zensiertes<br>Lognormal",
    }
    fig = go.Figure()
    for m in M.METHODS:
        c = cells[m]
        y = 100.0 * c.cost_mean if c.defined else 0.0
        fig.add_trace(go.Bar(x=[labels[m]], y=[y], error_y=dict(type="data", array=[100.0 * c.cost_se if c.defined else 0.0], visible=True), marker_color=colors[m],
                             marker_pattern_shape="/" if m == "km_fallback" else "", showlegend=False,
                             text=[f"{y:.1f} %".replace(".", ",") if c.defined else "keine Zahl"], textposition="outside",
                             hovertemplate="%{x}: %{y:.2f} %<extra></extra>"))
    fig.update_yaxes(title_text="Mehrkosten gegen die beste Bestellmenge (%)", rangemode="tozero")
    fig.update_xaxes(tickangle=0)
    return _base(fig, height=420)
