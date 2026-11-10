"""Abbildungen: jede wird gebaut, die Achsen sind gesperrt (Touch-Scrollen), Inhalt passt zu den Daten."""

import numpy as np
import pytest

import znf_core as K
import znf_model as M
import znf_scenario as S
import znf_visualization as V


def _locked(fig):
    layout = fig.to_dict()["layout"]
    axes = [v for k, v in layout.items() if k.startswith(("xaxis", "yaxis"))]
    assert axes and all(a.get("fixedrange") is True for a in axes)


@pytest.fixture(scope="module")
def study():
    return M.run_study(S.World(m=100.0, n=200), (0.5, 0.9), reps=12)


def test_hand_alle_muster_und_zeilenzahlen():
    for pairs in S.PATTERNS.values():
        table = M.hand_table(pairs)
        for shown in range(0, len(table["rows"]) + 1):
            fig = V.build_hand(table, 0.8, shown)
            _locked(fig)
            if shown:
                assert fig.data[0].y[-1] == pytest.approx(float(table["rows"][shown - 1]["s"]))
    fig = V.build_hand(M.hand_table(S.PATTERNS["Sechs Perioden (Handbeispiel)"]), 0.8, 6)
    assert len(fig.data) == 2 and list(fig.data[1].x) == [8.0, 10.0, 12.0]          # Kreuze bei den ausverkauften Werten 8, 10, 12
    assert fig.data[0].y[-1] == pytest.approx(4 / 9)


def test_quantile_runs_zeigt_nur_schaetzbare_laeufe_von_kaplan_meier(study):
    fig = V.build_quantile_runs(study, 0.9)
    _locked(fig)
    km = study.q["km"][:, 1]
    assert len(fig.data[1].y) == int(np.isfinite(km).sum()) and len(fig.data[0].y) == study.reps
    assert f"{int(np.isfinite(km).sum())} von {study.reps} Läufen" in fig.data[1].name


def test_km_example_plateau_linie_steht_auf_dem_letzten_wert():
    smp = S.make_sample(S.World(m=100.0, n=300), 0)
    km = K.kaplan_meier(smp.sales, smp.observed)
    fig = V.build_km_example(km, 0.9, float(smp.sales.max()) * 1.02)
    _locked(fig)
    assert fig.data[0].y[0] == 1.0 and fig.data[0].y[-1] == pytest.approx(float(km["s"][-1]))


def test_defined_series_eine_linie_je_welt(study):
    other = M.run_study(S.World(m=70.0, n=200), (0.5, 0.9), reps=12)
    fig = V.build_defined_series({"Bestand 100": study, "Bestand 70": other})
    _locked(fig)
    assert len(fig.data) == 2 and list(fig.data[0].x) == [0.5, 0.9] and all(0.0 <= y <= 1.0 for tr in fig.data for y in tr.y)


def test_boost_und_costs(study):
    rows = [{"boost": b, "share": s, "label": f"{round(100 * s)} %"} for b, s in ((0.0, 0.5), (0.1, 1.0))]
    fig = V.build_boost(rows, 0.9)
    _locked(fig)
    assert list(fig.data[0].x) == ["0 %", "10 %"] and list(fig.data[0].y) == [0.5, 1.0]
    fig = V.build_costs(study, 0.9)
    _locked(fig)
    assert len(fig.data) == 4
    assert fig.data[0].y[0] == pytest.approx(100 * M.summarize(study, 0.9, "sales").cost_mean)


def test_costs_ohne_schaetzbaren_lauf_zeigt_keine_zahl():
    st = M.run_study(S.World(m=1.0, n=100), (0.9,), reps=3)
    fig = V.build_costs(st, 0.9)
    _locked(fig)
    assert fig.data[1].text[0] == "keine Zahl" and fig.data[1].y[0] == 0.0
