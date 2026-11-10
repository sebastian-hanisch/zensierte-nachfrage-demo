"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_select_slider_radio_and_selectbox_option_does_not_raise():
    """Jede Auswahl (Quantil τ, Verteilung, Anteil der Testperioden, Handmuster, Quantil der Handrechnung) mit jeder Option: kein Python-Fehler.
    AppTest liefert die Optionen schon als Anzeigetext; gesetzt werden die rohen Werte (hier aus den Konstanten und den Mustern der App)."""
    import znf_constants as C
    from znf_scenario import PATTERNS

    raw = {"tau_select": C.TAU_OPTIONS, "hand_tau_select": (0.5, 0.6, 0.7, 0.8, 0.9), "boost_select": C.BOOST_SHARES, "form_radio": C.FORMS, "pattern_select": tuple(PATTERNS)}
    probe = _fresh_app()
    keys = {w.key for w in list(probe.select_slider) + list(probe.radio) + list(probe.selectbox)}
    assert keys == set(raw), "eine Auswahl der App fehlt in diesem Test"
    kinds = {"tau_select": "select_slider", "hand_tau_select": "select_slider", "boost_select": "select_slider", "form_radio": "radio", "pattern_select": "selectbox"}
    for key, values in raw.items():
        for value in values:
            at = _fresh_app()
            getattr(at, kinds[key])(key=key).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_standardseite_nennt_die_annahme_und_zeigt_alle_verdikte_und_expander():
    at = _fresh_app()
    text = " ".join(m.value for m in at.markdown)
    assert "Der Bestand einer Periode ist unabhängig von ihrer Nachfrage" in text and "informatives-zensieren-demo" in text
    assert "forecast-inventory-demo" in text and "newsvendor-direkt-demo" in text
    labels = [e.label for e in at.expander]
    assert "📐 Mathematische Formulierung" in labels and "⚠️ Annahmen, Grenzen und bekannte Probleme" in labels
    kinds = len(at.success) + len(at.warning) + len(at.info) + len(at.error)
    assert kinds >= 4           # Experiment 1 bis 4 haben je ein Verdikt (Experiment 3 hat zwei)
    assert not at.exception


def test_handrechnung_quantil_wechselt_zwischen_schaetzbar_und_nicht_schaetzbar():
    at = _fresh_app()
    verdicts = [v.value for v in list(at.warning) + list(at.info) + list(at.success) + list(at.error)]
    assert any("Das Quantil 0,8 ist **nicht schätzbar**" in v and "4/9 ≈ 0,44" in v for v in verdicts)
    at.select_slider(key="hand_tau_select").set_value(0.5).run(timeout=TIMEOUT)
    verdicts = [v.value for v in list(at.warning) + list(at.info) + list(at.success) + list(at.error)]
    assert any("Kaplan–Meier schätzt das Quantil 0,5 auf **9**" in v for v in verdicts)


def test_jedes_preset_zeigt_seine_welt_in_der_kernfrage():
    from znf_constants import PRESETS

    for name, p in PRESETS.items():
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        text = " ".join(m.value for m in at.markdown)
        assert f"Bestand lognormal (Median {p['m']}" in text and f"{p['n']} Perioden je Datensatz" in text
