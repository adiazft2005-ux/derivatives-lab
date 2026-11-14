"""
DERIVATIVES LAB - Financial Derivatives Valuation Platform
Ejecutar con:  streamlit run app.py
Todo el cálculo financiero vive en finance.py; este archivo solo contiene la interfaz.
"""
import html
import inspect
import io
import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots

import finance as fin
from finance import FinanceError

st.set_page_config(page_title="DERIVATIVES LAB", page_icon=":material/query_stats:", layout="wide",
                   initial_sidebar_state="expanded")

# ============================================================================
# Estilo: terminal financiera (azules marino y grises)
# ============================================================================
BG, PANEL, PANEL2, LINE = "#0A1424", "#0F1D33", "#13243F", "#22385C"
TXT, MUTED = "#E5ECF6", "#8CA0BD"
BLUE, CYAN, GREEN, RED, AMBER = "#3B82F6", "#38BDF8", "#34D399", "#F87171", "#FBBF24"

CSS = """
<style>
.stApp { background: __BG__; color: __TXT__; }
.block-container { padding-top: 1.4rem; max-width: 1320px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: __PANEL__; border-right: 1px solid __LINE__; }
h1,h2,h3,h4,h5, label, p, li, span, div { color: __TXT__; }
[data-testid="stCaptionContainer"], .stCaption, small { color: __MUTED__ !important; }
.brand { font-weight: 800; letter-spacing: .12em; font-size: 1.05rem; color: __TXT__; }
.brand span { color: __CYAN__; }
.brand-sub { color: __MUTED__; font-size: .74rem; margin: 2px 0 12px 0; }
.ptitle { font-size: 1.55rem; font-weight: 700; margin: 0; letter-spacing: .01em; }
.psub { color: __MUTED__; font-size: .9rem; margin: 2px 0 14px 0; }
.rule { height: 2px; width: 56px; background: __BLUE__; margin: 2px 0 14px 0; border-radius: 2px; }
.grp { font-size: .7rem; font-weight: 700; letter-spacing: .14em; color: __CYAN__; margin: 14px 0 4px 0;
       border-bottom: 1px solid __LINE__; padding-bottom: 3px; }
.hero { background: linear-gradient(135deg, #12306066 0%, #0F1D33 100%); border: 1px solid __BLUE__; border-radius: 12px;
        padding: 16px 20px; height: 100%; }
.hero .l { color: __MUTED__; font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
.hero .v { font-size: 2.05rem; font-weight: 800; margin: 2px 0; line-height: 1.15; color: #fff; }
.hero .s { color: __MUTED__; font-size: .8rem; line-height: 1.35; }
.kpi { background: __PANEL__; border: 1px solid __LINE__; border-radius: 12px; padding: 12px 14px; height: 100%; }
.kpi .l { color: __MUTED__; font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.kpi .v { font-size: 1.3rem; font-weight: 700; margin: 2px 0; color: __TXT__; }
.kpi .s { color: __MUTED__; font-size: .74rem; line-height: 1.3; }
.pill { display:inline-block; border-radius: 999px; padding: 1px 10px; font-size: .72rem; font-weight: 700; margin-right: 6px; }
.ok { background: #064E3B; color: #6EE7B7; } .warn { background: #78350F; color: #FCD34D; }
.note { background: __PANEL2__; border-left: 3px solid __BLUE__; border-radius: 6px; padding: 8px 12px; font-size: .84rem;
        color: __MUTED__; margin: 6px 0; }
.step { background: __PANEL__; border: 1px solid __LINE__; border-radius: 8px; padding: 8px 12px; margin: 5px 0;
        font-size: .86rem; }
.step b { color: __CYAN__; }
[data-testid="stVerticalBlockBorderWrapper"] { background: __PANEL__; border-color: __LINE__ !important; border-radius: 12px; }
[data-testid="stExpander"] { background: __PANEL__; border: 1px solid __LINE__; border-radius: 10px; }
[data-testid="stExpander"] summary p { color: __TXT__; font-weight: 600; }
input, textarea { background: __PANEL2__ !important; color: __TXT__ !important; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] { background: __PANEL2__ !important; border-color: __LINE__ !important; }
[data-baseweb="select"] * { color: __TXT__ !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] { background: __PANEL2__ !important; color: __TXT__ !important; }
.stButton > button, .stDownloadButton > button { background: __PANEL2__; color: __TXT__; border: 1px solid __LINE__; border-radius: 8px; }
.stButton > button:hover { border-color: __CYAN__; color: __CYAN__; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { background: __PANEL2__; color: __TXT__; }
.stTabs [data-baseweb="tab"] { color: __MUTED__; }
.stTabs [aria-selected="true"] { color: __CYAN__ !important; }
[data-testid="stDataFrame"] { border: 1px solid __LINE__; border-radius: 8px; overflow: hidden;
                              filter: invert(.9) hue-rotate(180deg); }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 3px 4px; }

/* ---- st.selectbox: caja cerrada y lista desplegable legibles sobre fondo oscuro ----
   (Streamlit nuevo ya no usa data-baseweb="select": la caja es role="group"/"combobox" y la lista vive en
   un contenedor propio con fondo blanco; por eso el texto claro no se veía) */
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"] { background: __PANEL2__ !important; border-color: __LINE__ !important; }
[data-testid="stSelectbox"] [role="group"] *, [data-testid="stMultiSelect"] [role="group"] * { color: __TXT__ !important; }
[data-testid="stSelectbox"] [role="group"] svg path:not([fill="none"]) { fill: __TXT__ !important; }
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
[data-baseweb="popover"] [role="listbox"] { background: __PANEL2__ !important; border-radius: 8px; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"], [data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
[data-baseweb="popover"] [role="option"], [data-baseweb="popover"] [role="option"] * { color: __TXT__ !important; background-color: transparent; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover, [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [role="option"][aria-selected="true"] { background-color: __LINE__ !important; }
[data-testid="stSelectbox"] [role="group"] button, [data-testid="stMultiSelect"] [role="group"] button { background: transparent !important; }

/* ---- menús desplegables de las celdas SelectboxColumn dentro de st.data_editor (overlay #portal, react-select) ---- */
#portal [class*="-menu"] { background: __PANEL2__ !important; border: 1px solid __LINE__; border-radius: 8px; }
#portal [role="option"], #portal [role="option"] * { color: __TXT__ !important; }
#portal [role="option"]:hover, #portal [role="option"][aria-selected="true"] { background-color: __LINE__ !important; }
</style>
"""
for _k, _v in {"__BG__": BG, "__PANEL2__": PANEL2, "__PANEL__": PANEL, "__LINE__": LINE, "__TXT__": TXT, "__MUTED__": MUTED,
               "__BLUE__": BLUE, "__CYAN__": CYAN}.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)


def _kw(fn):
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


_PLOT_KW, _DF_KW, _BTN_KW = _kw(st.plotly_chart), _kw(st.dataframe), _kw(st.button)


def style_fig(fig, title=None, height=360, xt=None, yt=None):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
                      title=dict(text=title or "", x=0.01, font=dict(size=14, color=TXT)),
                      margin=dict(l=10, r=10, t=44 if title else 16, b=10), hovermode="x unified",
                      colorway=[CYAN, BLUE, AMBER, GREEN, RED], font=dict(color=TXT),
                      legend=dict(orientation="h", y=-0.22))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    if xt:
        fig.update_xaxes(title_text=xt)
    if yt:
        fig.update_yaxes(title_text=yt)
    return fig


_plot_n = [0]


def plot(fig):
    _plot_n[0] += 1
    st.plotly_chart(fig, key=f"pl{_plot_n[0]}", config={"displaylogo": False}, **_PLOT_KW)


# ============================================================================
# Estado y valores por defecto (EJEMPLOS, no cotizaciones de mercado)
# ============================================================================
TODAY = date.today()
FREQS = list(fin.FREQ_PER_YEAR)
CCYS = ["USD", "MXN", "EUR", "GBP", "JPY", "CAD", "CHF", "BRL"]

DEFAULTS = {
    "fwd": dict(S0=50.0, T=3.0, source="Manual", rate=5.0, comp="Continuous", dmode="Discrete dividends", q=4.0,
                k_mode="Fair forward price", k_custom=50.0, reval_on=False, reval_t=1.5, reval_spot=50.0,
                gen_every=0.5, gen_amt=2.0, gen_n=6,
                fx_on=False, ccy_a="USD", ccy_b="MXN", fx=18.0, fx_cip=False, r_base=8.0),
    "irs": dict(method="Par swaps (bootstrapping)", freq="Annual", notional=1.0, sw_T=2.0, fixed_mode="Par rate", fixed_rate=10.0,
                val_date=TODAY, comp_out="Continuous", def_dc="ACT/360", def_bdc="Modified Following"),
    "bin": dict(S0=100.0, K=100.0, sigma=20.0, T=1.0, N=50, opt="Call", style="European", q=0.0, source="Manual", rate=5.0,
                what="Underlying price", depth=6, fx_on=False, ccy_a="USD", ccy_b="MXN", fx=18.0),
    "bsm": dict(S0=100.0, K=100.0, sigma=20.0, T=1.0, source="Manual", rate=5.0, q=0.0, opt="Call", pos="Long",
                contracts=1.0, mult=100.0, new_S=105.0, days=10.0, mkt_on=False, mkt=5.0,
                fx_on=False, ccy_a="USD", ccy_b="MXN", fx=18.0),
    "cmp": dict(kind="Binomial vs Black-Scholes", S0=100.0, K=100.0, sigma=20.0, T=1.0, r=5.0, q=0.0, opt="Call",
                steps="5, 10, 25, 50, 100, 250, 500, 1000", vmin=10.0, vmax=60.0, vpts=11, vN=200),
}

ss = st.session_state
ss.setdefault("P", {})
ss.setdefault("curve_pack", None)
ss.setdefault("nav", "Dashboard")
if "tables" not in ss:
    ss.tables = {
        "div": pd.DataFrame({"t (years)": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], "Amount": [2.0] * 6}),
        "quotes": pd.DataFrame({"Maturity (years)": [1.0, 2.0], "Swap rate (%)": [9.0, 10.0]}),
        "mkt": pd.DataFrame([["IRS", "1Y", 8.00, None, None, "Annual", "ACT/360", "Modified Following", None],
                             ["IRS", "2Y", 8.20, None, None, "Annual", "ACT/360", "Modified Following", None],
                             ["IRS", "5Y", 8.60, None, None, "Semiannual", "ACT/360", "Modified Following", None]],
                            columns=fin.INSTR_COLS).astype({"Start Date": "datetime64[ns]", "Maturity Date": "datetime64[ns]",
                                                            "Market Rate (%)": float}),
    }

PAGES = ["Dashboard", "Equity Forward", "IRS Curve", "Binomial Trees", "Black-Scholes-Merton", "Greeks Analysis", "Comparison Tool"]
BLURB = {"Equity Forward": "Forward price with discrete dividends or a continuous dividend yield.",
         "IRS Curve": "Bootstrapping of discount factors from swap rates; floating and fixed leg valuation.",
         "Binomial Trees": "CRR tree for European and American options.",
         "Black-Scholes-Merton": "Closed-form option price with continuous dividend yield.",
         "Greeks Analysis": "Delta, Gamma, Theta, Vega and Rho: meaning, value and sensitivity charts.",
         "Comparison Tool": "Tree vs Black-Scholes, volatility sensitivity, discrete vs continuous dividends."}


def goto(page):
    ss.nav = page


def _get(ns, name):
    return ss.P.get(ns, {}).get(name, DEFAULTS[ns][name])


def P(ns):
    return {**DEFAULTS[ns], **ss.P.get(ns, {})}


def _store(ns, name, v):
    ss.P.setdefault(ns, {})[name] = v
    return v


def _key(ns, name):
    return f"{ss.nav}|{ns}.{name}"


def w_num(ns, name, label, **kw):
    return _store(ns, name, st.number_input(label, value=_get(ns, name), key=_key(ns, name), **kw))


def w_sel(ns, name, label, options, **kw):
    cur = _get(ns, name)
    return _store(ns, name, st.selectbox(label, options, index=options.index(cur) if cur in options else 0, key=_key(ns, name), **kw))


def w_radio(ns, name, label, options, **kw):
    cur = _get(ns, name)
    return _store(ns, name, st.radio(label, options, index=options.index(cur) if cur in options else 0, key=_key(ns, name),
                                     horizontal=True, **kw))


def w_check(ns, name, label, **kw):
    return _store(ns, name, st.checkbox(label, value=_get(ns, name), key=_key(ns, name), **kw))


def w_date(ns, name, label, **kw):
    return _store(ns, name, st.date_input(label, value=_get(ns, name), key=_key(ns, name), **kw))


# ============================================================================
# Componentes de interfaz
# ============================================================================
def e(x):
    return html.escape(str(x))


def money(x, d=4):
    return f"{x:,.{d}f}"


def header(title, sub):
    st.markdown(f'<div class="ptitle">{e(title)}</div><div class="psub">{e(sub)}</div><div class="rule"></div>', unsafe_allow_html=True)


def group(title):
    st.markdown(f'<div class="grp">{e(title)}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def result_row(hero_label, hero_value, hero_sub, items):
    """Resultado principal (tarjeta grande) + hasta 3 resultados secundarios."""
    cols = st.columns([1.6] + [1] * len(items))
    cols[0].markdown(f'<div class="hero"><div class="l">{e(hero_label)}</div><div class="v">{e(hero_value)}</div>'
                     f'<div class="s">{e(hero_sub)}</div></div>', unsafe_allow_html=True)
    for c, (l, v, s) in zip(cols[1:], items):
        c.markdown(f'<div class="kpi"><div class="l">{e(l)}</div><div class="v">{e(v)}</div><div class="s">{e(s)}</div></div>',
                   unsafe_allow_html=True)
    st.write("")


def kpi_row(items):
    cols = st.columns(len(items))
    for c, (l, v, s) in zip(cols, items):
        c.markdown(f'<div class="kpi"><div class="l">{e(l)}</div><div class="v">{e(v)}</div><div class="s">{e(s)}</div></div>',
                   unsafe_allow_html=True)
    st.write("")


def checks_box(checks):
    for name, ok, detail in checks:
        st.markdown(f'<div style="font-size:.86rem;margin:2px 0"><span class="pill {"ok" if ok else "warn"}">{"PASS" if ok else "REVIEW"}</span>'
                    f'{e(name)} <span style="color:{MUTED}">({e(detail)})</span></div>', unsafe_allow_html=True)


def to_excel(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def export_button(name, sheets):
    st.download_button("Download results (Excel)", data=to_excel(sheets), file_name=f"derivatives_lab_{name}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_{name}")


def table(df, fmt=None, height=None):
    cfg = {c: st.column_config.NumberColumn(format=f) for c, f in (fmt or {}).items() if c in df.columns}
    kw = {"height": height} if height else {}
    st.dataframe(df, hide_index=True, column_config=cfg, **kw, **_DF_KW)


def numcol(label, **kw):
    """Columna numérica editable: 4 decimales y paso 0.0001 para poder teclear cualquier valor (0.75, 3.0650...)."""
    return st.column_config.NumberColumn(label, step=0.0001, format="%.4f", **kw)


def editable_table(name, config):
    """st.data_editor con estado estable.

    El DataFrame que recibe el editor (ss.editor_base[name]) NO se reasigna en cada rerun: antes se hacía
    `ss.tables[x] = st.data_editor(ss.tables[x], key=...)`, y como el editor recibía otra vez los datos ya
    editados, su estado interno (ediciones por posición) se volvía a aplicar y la primera captura se perdía.
    Ahora las ediciones viven en el estado del widget (key); su resultado se copia a ss.tables[name], que es
    lo que usan los cálculos y lo que sobrevive al cambiar de módulo. Para reemplazar la tabla desde código
    basta con asignar ss.tables[name] y hacer ss.pop(f"ed_{name}") (ver 'gen' en Equity Forward)."""
    key = f"ed_{name}"
    base = ss.setdefault("editor_base", {})
    if key not in ss or name not in base:           # primera vez, regreso desde otro módulo o tras reemplazar la tabla
        base[name] = ss.tables[name].copy()
    edited = st.data_editor(base[name], num_rows="dynamic", hide_index=True, key=key, column_config=config, **_DF_KW)
    ss.tables[name] = edited.reset_index(drop=True)


def pct_view(df, cols):
    """Copia con columnas de tasas (decimales) convertidas a % y renombradas '<col> (%)'."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] * 100.0
    return out.rename(columns={c: f"{c} (%)" for c in cols})


def pct(x, d=4):
    return f"{x * 100:.{d}f}%"


# ---- curva IRS compartida --------------------------------------------------
def default_par_curve():
    p = P("irs")
    q = [(float(r["Maturity (years)"]), float(r["Swap rate (%)"]) / 100.0) for _, r in ss.tables["quotes"].dropna().iterrows()]
    boot = fin.bootstrap_par_swaps(q, fin.FREQ_PER_YEAR[p["freq"]])
    return fin.curve_from_par_bootstrap(boot, TODAY)


def need_curve():
    if ss.curve_pack is None:
        try:
            ss.curve_pack = {"curve": default_par_curve(), "label": "Par-swap bootstrap"}
        except FinanceError:
            raise FinanceError("No valid IRS curve is available. Build one in the 'IRS Curve' module first.")
    return ss.curve_pack["curve"]


def rate_source_inputs(ns, label_rate="Risk-free rate r (% per year, continuous)", comp_note=True):
    """Fuente de la tasa: manual (plana) o curva IRS. Devuelve (curve|None, r|None)."""
    src = w_radio(ns, "source", "Interest rate source", ["Manual", "IRS curve"],
                  help="Manual: one flat rate. IRS curve: discount factors from the curve built in the IRS module.")
    if src == "Manual":
        w_num(ns, "rate", label_rate, step=0.25, format="%.4f", help="Enter in %: 5 means 5%.")
        return None, _get(ns, "rate") / 100.0
    pk = ss.curve_pack
    st.caption(f"Using: {pk['label'] if pk else 'default par-swap curve (1Y 9%, 2Y 10%)'}. No extrapolation beyond the last pillar.")
    return need_curve(), None


def fx_box(ns, cip=False):
    """Conversión de moneda opcional (tipo de cambio spot constante; convención: moneda BASE por 1 unidad del activo)."""
    with st.expander("Currency conversion (optional)"):
        if not w_check(ns, "fx_on", "Also show values in a base currency"):
            return None
        a, b, c = st.columns(3)
        with a:
            ca = w_sel(ns, "ccy_a", "Asset currency", CCYS)
        with b:
            cb = w_sel(ns, "ccy_b", "Base currency", CCYS)
        with c:
            x = w_num(ns, "fx", f"1 {ca} = ? {cb}", min_value=0.0, step=0.01, format="%.4f")
        fin.validate_fx(x, ca, cb)
        out = {"asset": ca, "base": cb, "rate": float(x), "cip": False}
        if cip:
            out["cip"] = w_check(ns, "fx_cip", "Convert at the FX forward (covered interest parity)")
            if out["cip"]:
                out["r_base"] = w_num(ns, "r_base", "Base-currency rate (% per year, continuous)", step=0.25, format="%.4f") / 100.0
        st.caption("Spot FX held constant; no quanto adjustment. Direct quote: base currency per 1 unit of asset currency.")
        return out


# ============================================================================
# DASHBOARD
# ============================================================================
def page_dashboard():
    st.markdown('<div class="ptitle" style="font-size:2rem">DERIVATIVES <span style="color:%s">LAB</span></div>'
                '<div class="psub">Valuation of forwards, swaps and options · inputs on the left, results on the right in every module.</div>'
                '<div class="rule"></div>' % CYAN, unsafe_allow_html=True)
    mods = PAGES[1:]
    for row in (mods[:3], mods[3:]):
        cols = st.columns(3)
        for c, m in zip(cols, row):
            with c.container(border=True):
                st.markdown(f"**{m}**")
                st.caption(BLURB[m])
                st.button("Open", key=f"open_{m}", on_click=goto, args=(m,), **_BTN_KW)
    pk = ss.curve_pack
    st.caption(f"Shared IRS curve: {pk['label'] if pk else 'default par-swap example (1Y 9%, 2Y 10%)'} · "
               "All default inputs are illustrative examples, not market quotes.")


# ============================================================================
# EQUITY FORWARD
# ============================================================================
def fwd_inputs():
    with st.container(border=True):
        group("MARKET")
        w_num("fwd", "S0", "Spot price S₀ (currency units)", min_value=0.0, step=1.0, format="%.4f", help="Current price of the underlying.")
        w_num("fwd", "T", "Time to maturity T (years)", min_value=0.0, step=0.25, format="%.4f", help="Delivery date measured in years from today.")
        src = w_radio("fwd", "source", "Interest rate source", ["Manual", "IRS curve"], help="Manual: flat rate. IRS curve: discount factors from the IRS module.")
        if src == "Manual":
            w_num("fwd", "rate", "Interest rate r (% per year)", step=0.25, format="%.4f", help="Enter in %: 5 means 5%.")
        else:
            st.caption(f"Using: {ss.curve_pack['label'] if ss.curve_pack else 'default par-swap curve'}. No extrapolation beyond its last pillar.")
        group("DIVIDENDS")
        dmode = w_radio("fwd", "dmode", "Dividend model", ["Discrete dividends", "Continuous yield"],
                        help="Discrete: known cash amounts at known times. Continuous: a yield q paid continuously.")
        if dmode == "Discrete dividends":
            st.caption("One row per dividend: payment time t in years (0 < t ≤ T) and cash amount.")
            editable_table("div", {"t (years)": numcol("t (years)", min_value=0.0), "Amount": numcol("Amount", min_value=0.0)})
            with st.expander("Generate a regular schedule"):
                g1, g2, g3 = st.columns(3)
                every = g1.number_input("Every (years)", min_value=0.01, value=float(_get("fwd", "gen_every")), step=0.25, key="gen_every")
                amt = g2.number_input("Amount each", min_value=0.0, value=float(_get("fwd", "gen_amt")), step=0.1, key="gen_amt")
                n = g3.number_input("Number", min_value=1, max_value=200, value=int(_get("fwd", "gen_n")), step=1, key="gen_n")

                def gen():
                    ss.tables["div"] = pd.DataFrame({"t (years)": [round(every * k, 10) for k in range(1, int(n) + 1)], "Amount": [amt] * int(n)})
                    ss.pop("ed_div", None)

                st.button("Replace table with this schedule", on_click=gen, key="gen_btn")
        else:
            w_num("fwd", "q", "Dividend yield q (% per year, continuous)", min_value=0.0, step=0.25, format="%.4f",
                  help="Enter in %: 4 means q = 4%. The forward uses F = S₀·e^(−qT)/DF(T).")
        with st.expander("Advanced settings"):
            if src == "Manual":
                w_sel("fwd", "comp", "Compounding of r", fin.COMPOUNDINGS,
                      help="Continuous (default): DF = e^(−rT). Other choices only change the discount factor formula.")
            w_radio("fwd", "k_mode", "Delivery price K", ["Fair forward price", "Custom"],
                    help="Fair: K = F, so the contract starts with zero value.")
            if _get("fwd", "k_mode") == "Custom":
                w_num("fwd", "k_custom", "Custom K (currency units)", min_value=0.0, step=1.0, format="%.4f")
            if w_check("fwd", "reval_on", "Also value the contract at a later time"):
                w_num("fwd", "reval_t", "Re-valuation time t (years, 0 < t < T)", min_value=0.0, step=0.25, format="%.4f")
                w_num("fwd", "reval_spot", "Spot price at time t", min_value=0.0, step=1.0, format="%.4f")
        return fx_box("fwd", cip=True)


def fwd_compute(p):
    if p["source"] == "IRS curve":
        disc = need_curve()
        warns = []
    else:
        warns = fin.validate_rate(p["rate"] / 100.0, "Interest rate")
        disc = fin.FlatCurve(TODAY, p["rate"] / 100.0, p["comp"])
    discrete = p["dmode"] == "Discrete dividends"
    divs = [(r["t (years)"], r["Amount"]) for r in ss.tables["div"].to_dict("records")] if discrete else None
    K = None if p["k_mode"] == "Fair forward price" else p["k_custom"]
    rt, rs = (p["reval_t"], p["reval_spot"]) if p["reval_on"] else (None, None)
    res = fin.equity_forward_t(p["S0"], p["T"], disc, "discrete" if discrete else "continuous", divs, p["q"] / 100.0, K, rt, rs)
    res["warnings"] = warns
    return res


def fwd_results(d, fx):
    for w in d["warnings"]:
        st.warning(w)
    cont = d["mode"] == "continuous"
    if cont:
        sub = f"F = S₀·e^(−qT)/DF(T) = {d['S0']:,.4f} × {math.exp(-d['q'] * d['T']):.6f} / {d['DF_T']:.6f}"
    else:
        sub = f"F = (S₀ − PV dividends)/DF(T) = ({d['S0']:,.4f} − {d['PV_divs']:,.4f}) / {d['DF_T']:.6f}"
    result_row("Forward price F(0,T)", money(d["F"]), sub,
               [("Contract value today (long)", money(d["value0"], 6), "Zero when K is the fair forward price."),
                ("PV of dividends" if not cont else "Dividend value given up", money(d["PV_divs"]),
                 "Σ D·DF(0,t)" if not cont else "S₀·(1 − e^(−qT))"),
                ("Discount factor DF(0,T)", f"{d['DF_T']:.6f}", f"Equivalent r = {pct(d['r_cont'], 3)} (cont.)")])
    st.caption(f"Forward without dividends: {money(d['F_nodiv'])} · dividends change it by {d['F'] - d['F_nodiv']:+,.4f} · "
               f"equivalent continuous yield q ≈ {pct(d['q_equiv'], 3)}. The forward price is a delivery price, not a premium.")
    tabs = st.tabs(["Breakdown", "Chart", "Formulas & checks"])
    with tabs[0]:
        if not cont and len(d["table"]):
            table(d["table"], {"t (years)": "%.4f", "Dividend": "%.4f", "DF(0,t)": "%.6f", "PV": "%.6f", "Value at T (D / DF(t,T))": "%.6f"})
        elif cont:
            table(pd.DataFrame({"Item": ["Spot S₀", "Yield q", "e^(−qT)", "Adjusted spot S₀·e^(−qT)", "DF(0,T)", "Forward F"],
                                "Value": [d["S0"], d["q"], math.exp(-d["q"] * d["T"]), d["S_adj"], d["DF_T"], d["F"]]}), {"Value": "%.6f"})
        else:
            st.info("No dividends: F = S₀ / DF(0,T).")
        if d["reval"]:
            r = d["reval"]
            kpi_row([(f"Value at t = {r['t']:g}y (long)", money(r["value_long"]), f"Spot {r['spot']:,.4f}, K = {d['K']:,.4f}"),
                     ("Fair forward at t", money(r["fair_forward"]), "Price that would make the value zero"),
                     ("PV of remaining dividends", money(r["pv_remaining_divs"]), "Discounted to t")])
        if fx:
            x = fx["rate"]
            items = [("Spot in base", f"{money(d['S0'] * x)} {fx['base']}", f"FX {x:,.4f}"), ("Forward in base (spot FX)", f"{money(d['F'] * x)} {fx['base']}", "FX exposure not hedged")]
            if fx["cip"]:
                xT = fin.fx_forward_cip(x, d["DF_T"], math.exp(-fx["r_base"] * d["T"]))
                items.append(("Forward in base (FX forward)", f"{money(d['F'] * xT)} {fx['base']}", f"X(0,T) = {xT:,.4f}"))
            kpi_row(items)
    with tabs[1]:
        wf = go.Figure(go.Waterfall(x=["Spot S₀", "Financing (carry)", "Dividends", "Forward F"], measure=["absolute", "relative", "relative", "total"],
                                    y=[d["S0"], d["F_nodiv"] - d["S0"], d["F"] - d["F_nodiv"], 0],
                                    increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
                                    totals=dict(marker=dict(color=BLUE)), connector=dict(line=dict(color=LINE)),
                                    text=[money(d["S0"], 2), f"{d['F_nodiv'] - d['S0']:+.2f}", f"{d['F'] - d['F_nodiv']:+.2f}", money(d["F"], 2)],
                                    textposition="outside"))
        plot(style_fig(wf, "From spot to forward price", yt="Price"))
    with tabs[2]:
        if cont:
            st.latex(r"F(0,T)=\frac{S_0\,e^{-qT}}{DF(0,T)}\;=\;S_0\,e^{(r-q)T}\ \ (\text{continuous } r)")
        else:
            st.latex(r"F(0,T)=\frac{S_0-\sum_i D_i\,DF(0,t_i)}{DF(0,T)}=S_0e^{rT}-\sum_i D_i\,e^{r(T-t_i)}\ \ (\text{continuous } r)")
        st.latex(r"V_0^{long}=S_0^{adj}-K\,DF(0,T)\qquad V_t^{long}=S_t^{adj}-K\,\tfrac{DF(0,T)}{DF(0,t)}")
        checks_box(d["checks"])
        export_button("forward", {"Summary": pd.DataFrame({"Metric": ["S0", "T", "Mode", "DF(0,T)", "PV dividends", "Forward", "Forward no dividends", "K", "Initial value"],
                                                          "Value": [d["S0"], d["T"], d["mode"], d["DF_T"], d["PV_divs"], d["F"], d["F_nodiv"], d["K"], d["value0"]]}),
                                  "Dividends": d["table"]})


def page_forward():
    header("Equity Forward", "Forward price on a stock with discrete dividends or a continuous dividend yield q")
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        try:
            fx = fwd_inputs()
        except FinanceError as ex:
            fx = None
            st.error(str(ex))
    with right:
        try:
            fwd_results(fwd_compute(P("fwd")), fx)
        except FinanceError as ex:
            st.error(f"**Cannot calculate:** {ex}", icon=":material/error:")


# ============================================================================
# IRS CURVE
# ============================================================================
def irs_par_inputs():
    with st.container(border=True):
        group("SWAP QUOTES")
        st.caption("Par swap rate for each maturity, in % (9 means 9%). Fixed payments at the end of each period.")
        editable_table("quotes", {"Maturity (years)": numcol("Maturity (years)", min_value=0.0), "Swap rate (%)": numcol("Swap rate (%)")})
        w_sel("irs", "freq", "Fixed payment frequency", FREQS, help="Payments per year: Annual = 1, Semiannual = 2, Quarterly = 4, Monthly = 12. Each period lasts exactly 1/frequency years.")
        w_num("irs", "notional", "Notional principal", min_value=0.0, step=1.0, format="%.4f", help="Amount on which interest is computed (1 = per unit of notional).")
        group("SWAP TO VALUE")
        w_num("irs", "sw_T", "Swap maturity (years)", min_value=0.0, step=0.5, format="%.4f", help="Must be a payment date of the curve (multiple of 1/frequency).")
        w_radio("irs", "fixed_mode", "Fixed rate of the swap", ["Par rate", "Custom"], help="Par: the rate that makes the swap worth zero today.")
        if _get("irs", "fixed_mode") == "Custom":
            w_num("irs", "fixed_rate", "Custom fixed rate (%)", step=0.05, format="%.4f")


def irs_par_results():
    p = P("irs")
    q = [(float(r["Maturity (years)"]), float(r["Swap rate (%)"]) / 100.0) for _, r in ss.tables["quotes"].dropna().iterrows()]
    freq = fin.FREQ_PER_YEAR[p["freq"]]
    boot = fin.bootstrap_par_swaps(q, freq)
    curve = fin.curve_from_par_bootstrap(boot, TODAY)
    ss.curve_pack = {"curve": curve, "label": "Par-swap bootstrap"}
    K = None if p["fixed_mode"] == "Par rate" else p["fixed_rate"] / 100.0
    v = fin.value_swap_par_curve(boot, p["sw_T"], p["notional"], K)
    tb = boot["table"]
    N = p["notional"]
    result_row("PV of floating-leg interest (excl. notional)", money(v["pv_float_interest"], 6),
               f"N·(1 − DF_T) = {N:g} × (1 − {v['DF_T']:.6f}) · swap of {v['maturity']:g} years",
               [("Discount factor DF_T", f"{v['DF_T']:.6f}", f"At t = {v['maturity']:g} years"),
                ("Par swap rate", pct(v["par_rate"], 4), "(1 − DF_T) / (τ·ΣDF)"),
                ("Swap value (pay fixed)", money(v["value_payer"], 6), f"Fixed rate {pct(v['fixed_rate'], 4)}: PV float − PV fixed")])
    tabs = st.tabs(["Bootstrapping step by step", "Swap legs", "Curve"])
    with tabs[0]:
        note("For each payment date t<sub>k</sub> the par swap satisfies  S<sub>k</sub>·τ·Σ<sub>i≤k</sub> DF<sub>i</sub> + DF<sub>k</sub> = 1,  so  "
             "<b>DF<sub>k</sub> = (1 − S<sub>k</sub>·τ·Σ<sub>i&lt;k</sub> DF<sub>i</sub>) / (1 + S<sub>k</sub>·τ)</b>.  τ = 1/frequency.")
        shown = 0
        for _, r in tb.iterrows():
            if shown >= 12:
                st.caption(f"… {len(tb) - shown} more periods in the table below.")
                break
            S, A = r["Swap rate S"], r["τ·Σ DF(prior)"]
            st.markdown(f'<div class="step"><b>t = {r["t (years)"]:g}y</b> ({r["Source"]}) &nbsp; S = {S:.4%} &nbsp;→&nbsp; '
                        f'DF = (1 − {S:.6f} × {A:.6f}) / (1 + {S:.6f} × {boot["tau"]:.6f}) = {r["Numerator 1 − S·τ·ΣDF"]:.6f} / {r["Denominator 1 + S·τ"]:.6f} '
                        f'= <b>{r["Discount factor DF"]:.6f}</b></div>', unsafe_allow_html=True)
            shown += 1
        st.markdown(f'<div class="step"><b>Floating leg:</b> Σ (DF<sub>i−1</sub> − DF<sub>i</sub>) = 1 − DF<sub>T</sub> = 1 − {v["DF_T"]:.6f} = '
                    f'<b>{1 - v["DF_T"]:.6f}</b> per unit of notional (the notional is excluded because it is paid at both ends).</div>', unsafe_allow_html=True)
        table(pct_view(tb, ["Swap rate S", "Zero rate (annual)", "Forward rate (period, annualised)"]),
              {"t (years)": "%.4f", "Swap rate S (%)": "%.4f", "τ·Σ DF(prior)": "%.6f", "Numerator 1 − S·τ·ΣDF": "%.6f", "Denominator 1 + S·τ": "%.6f",
               "Discount factor DF": "%.6f", "Zero rate (annual) (%)": "%.4f", "Forward rate (period, annualised) (%)": "%.4f", "PV floating interest": "%.6f"})
    with tabs[1]:
        kpi_row([("PV floating interest", money(v["pv_float_interest"], 6), "N·(1 − DF_T)"),
                 ("PV fixed interest", money(v["pv_fixed_interest"], 6), f"N·K·τ·ΣDF at K = {pct(v['fixed_rate'], 4)}"),
                 ("PV floating incl. notional", money(v["pv_float_total"], 6), "Interest + N·DF_T = N (floater at par)"),
                 ("PV fixed incl. notional", money(v["pv_fixed_total"], 6), "Interest + N·DF_T")])
        table(pct_view(v["table"], ["Projected floating rate"]), {"t (years)": "%.4f", "DF": "%.6f", "Projected floating rate (%)": "%.4f",
                                                                  "Floating payment": "%.6f", "PV floating": "%.6f", "Fixed payment": "%.6f", "PV fixed": "%.6f"})
        st.caption("Projected floating rate = forward rate of each period implied by the curve: (DF(i−1)/DF(i) − 1)/τ. Receiver value = −(payer value).")
    with tabs[2]:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=[0] + list(boot["times"]), y=[1] + list(boot["dfs"]), name="Discount factor", mode="lines+markers",
                                 line=dict(color=CYAN, width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=tb["t (years)"], y=tb["Zero rate (annual)"] * 100, name="Zero rate (%)", mode="lines+markers",
                                 line=dict(color=AMBER, width=2)), secondary_y=True)
        fig.add_trace(go.Scatter(x=tb["t (years)"], y=tb["Swap rate S"] * 100, name="Swap rate (%)", mode="markers",
                                 marker=dict(color=GREEN, size=9, symbol="diamond")), secondary_y=True)
        fig.update_yaxes(title_text="Discount factor", secondary_y=False)
        fig.update_yaxes(title_text="Rate (%)", secondary_y=True)
        plot(style_fig(fig, "Discount factors and rates from the bootstrap", xt="Years"))
        export_button("irs_par", {"Bootstrap": tb, "Swap legs": v["table"]})


def irs_mkt_inputs():
    with st.container(border=True):
        group("MARKET INSTRUMENTS")
        st.caption("Date-based bootstrap with day count and business-day conventions (deposits, FRAs and swaps). "
                   "Tenor like 6M, 1Y, 18M; Start Date blank = valuation date.")
        w_date("irs", "val_date", "Valuation date")
        editable_table("mkt", {"Type": st.column_config.SelectboxColumn("Type", options=fin.INSTRUMENT_TYPES, required=True),
                           "Tenor": st.column_config.TextColumn("Tenor"),
                           "Market Rate (%)": numcol("Market rate (%)"),
                           "Start Date": st.column_config.DateColumn("Start", format="YYYY-MM-DD"),
                           "Maturity Date": st.column_config.DateColumn("Maturity", format="YYYY-MM-DD"),
                           "Frequency": st.column_config.SelectboxColumn("Frequency", options=list(fin.FREQ_MONTHS)),
                           "Day Count": st.column_config.SelectboxColumn("Day count", options=fin.DAY_COUNTS),
                           "Business Day": st.column_config.SelectboxColumn("Business day", options=fin.BDCS),
                           "Compounding": st.column_config.SelectboxColumn("Compounding", options=["Simple", "Annual", "Continuous"])})
        w_sel("irs", "comp_out", "Zero / forward rates compounding", fin.COMPOUNDINGS)
        st.caption("Accrual factors use each row's day count (e.g. ACT/360 gives τ ≈ 1.014 for one year), "
                   "so results differ from the idealised par-swap bootstrap.")


def irs_mkt_results():
    p = P("irs")
    insts = fin.instruments_from_table(ss.tables["mkt"], p["val_date"])
    curve, insts, warns = fin.bootstrap(p["val_date"], insts)
    ss.curve_pack = {"curve": curve, "label": "Market-instrument bootstrap"}
    t = fin.curve_table(curve, insts, p["comp_out"])
    for w in warns:
        st.warning(w)
    err = t["Repricing Error (bp)"].abs().max()
    result_row("Last discount factor", f"{t['Discount Factor'].iloc[-1]:.6f}", f"Maturity {t['Maturity'].iloc[-1]} ({t['Years (ACT/365F)'].iloc[-1]:.2f} years)",
               [("Pillars", f"{len(t)}", "Instruments bootstrapped"), ("Last zero rate", pct(t["Zero Rate"].iloc[-1], 4), f"{p['comp_out']} compounding"),
                ("Max repricing error", f"{err:.1e} bp", "0 means every quote is reproduced")])
    tabs = st.tabs(["Curve table", "Chart", "Swap detail"])
    with tabs[0]:
        show = pd.DataFrame({"Instrument": t["Instrument"], "Maturity": t["Maturity"], "Years": t["Years (ACT/365F)"], "Market rate (%)": t["Market Rate"] * 100,
                             "Discount factor": t["Discount Factor"], "Zero rate (%)": t["Zero Rate"] * 100, "Forward rate (%)": t["Forward Rate"] * 100,
                             "Repricing error (bp)": t["Repricing Error (bp)"]})
        table(show, {"Years": "%.4f", "Market rate (%)": "%.4f", "Discount factor": "%.8f", "Zero rate (%)": "%.4f", "Forward rate (%)": "%.4f", "Repricing error (bp)": "%.2e"})
        st.caption("Forward rate = rate between consecutive pillars. Time axis: ACT/365F from the valuation date; log-linear interpolation of DF.")
        export_button("irs_market", {"Curve": t})
    with tabs[1]:
        grid = np.arange(1, int(round(curve.max_time * 12)) + 1) / 12.0
        grid = grid[grid <= curve.max_time + 1e-9]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=[fin.rate_from_df(curve.df_t(x), x, p["comp_out"]) * 100 for x in grid], name="Zero rate", line=dict(color=CYAN, width=3)))
        fig.add_trace(go.Scatter(x=t["Years (ACT/365F)"], y=t["Market Rate"] * 100, name="Market quote", mode="markers", marker=dict(color=AMBER, size=9, symbol="diamond")))
        plot(style_fig(fig, f"Zero curve ({p['comp_out']})", xt="Years", yt="Rate (%)"))
    with tabs[2]:
        swaps = [i for i in insts if i.kind == "IRS"]
        if not swaps:
            st.info("There are no swaps in the table.")
        else:
            names = [f"{i.label} ({i.maturity})" for i in swaps]
            inst = swaps[names.index(st.selectbox("Swap", names, key="irs_detail"))]
            sched = fin.payment_schedule(inst.start, inst.maturity, inst.freq_months, inst.bdc)
            rows = []
            for a, b in zip(sched[:-1], sched[1:]):
                tau, dfb = fin.yearfrac(a, b, inst.dc), curve.df(b)
                rows.append({"Period start": a, "Payment date": b, "τ": tau, "DF": dfb, "Fixed cash flow": inst.rate * tau, "PV": inst.rate * tau * dfb})
            det = pd.DataFrame(rows)
            table(det, {"τ": "%.6f", "DF": "%.8f", "Fixed cash flow": "%.6f", "PV": "%.6f"})
            ann = (det["τ"] * det["DF"]).sum()
            st.caption(f"Annuity = {ann:.8f} · par rate = (DF(start) − DF(T))/annuity = {(curve.df(inst.start) - curve.df(inst.maturity)) / ann:.6%} vs market {inst.rate:.6%}")


def page_irs():
    header("IRS Curve", "Bootstrapping of discount factors from swap rates and valuation of the swap legs")
    method = w_radio("irs", "method", "Method", ["Par swaps (bootstrapping)", "Market instruments (dates)"],
                     help="Par swaps: idealised periods τ = 1/frequency (textbook). Market instruments: dates, day counts and business-day rules.")
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        (irs_par_inputs if method.startswith("Par") else irs_mkt_inputs)()
    with right:
        try:
            (irs_par_results if method.startswith("Par") else irs_mkt_results)()
        except FinanceError as ex:
            st.error(f"**Cannot calculate:** {ex}", icon=":material/error:")


# ============================================================================
# BINOMIAL TREES
# ============================================================================
TREE_STORE_MAX = 200


def bin_inputs():
    with st.container(border=True):
        group("MARKET")
        w_num("bin", "S0", "Underlying price S₀ (currency units)", min_value=0.0, step=1.0, format="%.4f")
        w_num("bin", "sigma", "Volatility σ (% per year)", min_value=0.0, step=1.0, format="%.4f", help="20 means 20%.")
        w_num("bin", "q", "Dividend yield q (% per year, continuous)", min_value=0.0, step=0.25, format="%.4f")
        curve, r = rate_source_inputs("bin")
        group("OPTION")
        a, b = st.columns(2)
        with a:
            w_sel("bin", "opt", "Option type", ["Call", "Put"])
            w_num("bin", "K", "Strike K (currency units)", min_value=0.0, step=1.0, format="%.4f")
        with b:
            w_sel("bin", "style", "Exercise style", ["European", "American"], help="European: only at maturity. American: any time.")
            w_num("bin", "T", "Maturity T (years)", min_value=0.0, step=0.25, format="%.4f")
        w_num("bin", "N", "Number of steps N", min_value=1, max_value=5000, step=1, help="More steps → closer to Black-Scholes, slower.")
        with st.expander("Display settings"):
            w_num("bin", "depth", "Steps drawn in the tree plot", min_value=2, max_value=15, step=1)
        return fx_box("bin"), curve, r


def tree_figure(tr, depth, what, show_ex):
    n = min(tr.N, depth)
    fig = go.Figure()
    ex, ey = [], []
    for i in range(n):
        for j in range(i + 1):
            for jj in (j, j + 1):
                fig.add_trace(go.Scatter(x=[i, i + 1], y=[tr.stock[j, i], tr.stock[jj, i + 1]], mode="lines", line=dict(color=LINE, width=1),
                                         hoverinfo="skip", showlegend=False))
    xs, ys, txt, col = [], [], [], []
    for i in range(n + 1):
        for j in range(i + 1):
            v = tr.stock[j, i] if what == "stock" else tr.value[j, i]
            xs.append(i); ys.append(tr.stock[j, i]); txt.append(f"{v:.2f}"); col.append(v)
            if what == "option" and show_ex and tr.exercised[j, i]:
                ex.append(i); ey.append(tr.stock[j, i])
    show_text = n <= 6
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text" if show_text else "markers", text=txt if show_text else None, textposition="top center",
                             customdata=txt, hovertemplate="Value: %{customdata}<br>Step %{x}<extra></extra>", showlegend=False,
                             marker=dict(size=12 if show_text else 8, color=col, colorscale="Blues", line=dict(color=BG, width=1))))
    if ex:
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers", name="Early exercise", marker=dict(color=AMBER, size=11, symbol="diamond")))
    return style_fig(fig, f"{'Underlying price' if what == 'stock' else 'Option value'} tree (first {n} of {tr.N} steps)", height=420,
                     xt="Step", yt="Underlying price")


def bin_results(fx, curve, r):
    p = P("bin")
    if curve is None:
        fin.validate_rate(r, "Interest rate")
    S0, K, sg, T, N, q = p["S0"], p["K"], p["sigma"] / 100.0, p["T"], int(p["N"]), p["q"] / 100.0
    keep = N <= TREE_STORE_MAX
    tr = fin.crr_option(S0, K, sg, T, N, p["opt"], p["style"], r=r, q=q, curve=curve, keep_tree=keep)
    eu = tr if p["style"] == "European" else fin.crr_option(S0, K, sg, T, N, p["opt"], "European", r=r, q=q, curve=curve)
    am = tr if p["style"] == "American" else fin.crr_option(S0, K, sg, T, N, p["opt"], "American", r=r, q=q, curve=curve)
    bs = fin.bsm_with_curve(S0, K, sg, T, curve, q, p["opt"]) if curve else fin.bsm(S0, K, sg, T, r, q, p["opt"])
    cu = f" {fx['asset']}" if fx else ""
    result_row(f"{p['style']} {p['opt']} · tree value", f"{money(tr.price)}{cu}", f"CRR with N = {tr.N} steps, Δt = {tr.dt:.5f} y",
               [("Black-Scholes (European)", money(bs["price"]), "Same inputs, closed form"),
                ("Intrinsic · time value", f"{money(tr.intrinsic, 2)} · {money(tr.time_value, 2)}", "Payoff now · premium − payoff"),
                ("Early-exercise premium", money(am.price - eu.price), "American − European (tree)")])
    st.caption(f"u = {tr.u:.6f} · d = {tr.d:.6f} · risk-neutral p = "
               + (f"{tr.p[0]:.6f}" if np.allclose(tr.p, tr.p[0]) else f"{tr.p.min():.4f}–{tr.p.max():.4f} (varies by step with the IRS curve)")
               + (f" · premium in {fx['base']}: {money(tr.price * fx['rate'])}" if fx else ""))
    tabs = st.tabs(["Tree", "Terminal distribution", "Formulas & checks"])
    with tabs[0]:
        if tr.stock is None:
            st.info(f"Node-level trees are kept for N ≤ {TREE_STORE_MAX}. Reduce N to inspect nodes; the price above is exact for N = {tr.N}.")
        else:
            what = w_radio("bin", "what", "Show", ["Underlying price", "Option value"])
            plot(tree_figure(tr, int(p["depth"]), "stock" if what == "Underlying price" else "option", p["style"] == "American"))
            with st.expander("Node tables (first 12 steps)"):
                n = min(tr.N, 12)
                for name, arr in (("Underlying", tr.stock), ("Option value", tr.value)):
                    df = pd.DataFrame(arr[:n + 1, :n + 1], columns=[f"Step {i}" for i in range(n + 1)])
                    df.index = [f"{j} down" for j in range(n + 1)]
                    st.markdown(f"**{name}**")
                    st.dataframe(df, **_DF_KW)
    prob = np.array([1.0])
    for i in range(N):
        nxt = np.zeros(i + 2)
        nxt[:i + 1] += prob * tr.p[i]
        nxt[1:] += prob * (1 - tr.p[i])
        prob = nxt
    j = np.arange(N + 1)
    ST = S0 * np.exp((N - 2 * j) * sg * np.sqrt(tr.dt))
    sign = 1.0 if p["opt"] == "Call" else -1.0
    payoff = np.maximum(sign * (ST - K), 0)
    with tabs[1]:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ST, y=prob, name="Risk-neutral probability", marker_color=BLUE, opacity=.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=ST, y=payoff, name="Payoff", line=dict(color=AMBER, width=3)), secondary_y=True)
        fig.update_yaxes(title_text="Probability", secondary_y=False)
        fig.update_yaxes(title_text="Payoff", secondary_y=True)
        plot(style_fig(fig, "Terminal distribution and payoff", xt="Underlying price at maturity"))
    with tabs[2]:
        st.latex(r"\Delta t=\tfrac TN,\ u=e^{\sigma\sqrt{\Delta t}},\ d=\tfrac1u,\ p=\frac{e^{(r-q)\Delta t}-d}{u-d},\ V=e^{-r\Delta t}[pV_u+(1-p)V_d]")
        st.caption("American: V = max(intrinsic, continuation) at every node. With the IRS curve, the discount per step is DF(tᵢ₊₁)/DF(tᵢ) and "
                   "p = (e^(−qΔt)/DF_step − d)/(u − d). The model stops with an error if p is outside (0, 1).")
        dfT = float(np.prod(tr.step_df))
        lower = max(S0 * np.exp(-q * T) - K * dfT, 0.0) if p["opt"] == "Call" else max(K * dfT - S0 * np.exp(-q * T), 0.0)
        rel = abs(eu.price - bs["price"]) / max(bs["price"], 1e-12)
        checks_box([("Probabilities in (0, 1) at every step", True, f"min {tr.p.min():.4f}, max {tr.p.max():.4f}"),
                    ("American ≥ European", am.price >= eu.price - 1e-12, f"{am.price:.4f} ≥ {eu.price:.4f}"),
                    ("European tree ≈ Black-Scholes (converges as N grows)", rel < 0.01, f"diff {eu.price - bs['price']:+.4f} ({rel:.2%})"),
                    ("European price ≥ no-arbitrage lower bound", eu.price >= lower - 1e-9, f"bound {lower:.4f}")])
        export_button("binomial", {"Terminal nodes": pd.DataFrame({"Downs": j, "Terminal price": ST, "Payoff": payoff, "Probability": prob})})


def page_binomial():
    header("Binomial Trees", "Cox-Ross-Rubinstein pricing of European and American options")
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        try:
            fx, curve, r = bin_inputs()
        except FinanceError as ex:
            st.error(str(ex))
            return
    with right:
        try:
            bin_results(fx, curve, r)
        except FinanceError as ex:
            st.error(f"**Cannot calculate:** {ex}", icon=":material/error:")


# ============================================================================
# BLACK-SCHOLES-MERTON y GREEKS (inputs compartidos)
# ============================================================================
def bsm_inputs():
    with st.container(border=True):
        group("MARKET")
        w_num("bsm", "S0", "Underlying price S₀ (currency units)", min_value=0.0, step=1.0, format="%.4f")
        w_num("bsm", "sigma", "Volatility σ (% per year)", min_value=0.0, step=1.0, format="%.4f", help="20 means 20%.")
        w_num("bsm", "q", "Dividend yield q (% per year, continuous)", min_value=0.0, step=0.25, format="%.4f")
        rate_source_inputs("bsm")
        group("OPTION")
        a, b = st.columns(2)
        with a:
            w_sel("bsm", "opt", "Option type", ["Call", "Put"])
            w_num("bsm", "K", "Strike K (currency units)", min_value=0.0, step=1.0, format="%.4f")
        with b:
            w_num("bsm", "T", "Maturity T (years)", min_value=0.0, step=0.25, format="%.4f")
        with st.expander("Position and market quote (optional)"):
            c1, c2, c3 = st.columns(3)
            with c1:
                w_sel("bsm", "pos", "Position", ["Long", "Short"])
            with c2:
                w_num("bsm", "contracts", "Contracts", min_value=0.0, step=1.0, format="%.2f")
            with c3:
                w_num("bsm", "mult", "Multiplier", min_value=0.0, step=1.0, format="%.2f")
            if w_check("bsm", "mkt_on", "I have a market price for this option"):
                w_num("bsm", "mkt", "Market option price (per unit)", min_value=0.0, step=0.1, format="%.4f")
        return fx_box("bsm")


def bsm_any(p, S, K, sg, T, q, opt):
    if p["source"] == "IRS curve":
        return fin.bsm_with_curve(S, K, sg, T, need_curve(), q, opt)
    fin.validate_rate(p["rate"] / 100.0, "Interest rate")
    return fin.bsm(S, K, sg, T, p["rate"] / 100.0, q, opt)


GREEKS = [("delta", "Δ", "Delta", "delta", 1.0), ("gamma", "Γ", "Gamma", "gamma", 1.0), ("theta", "Θ", "Theta (per day)", "theta_daily", 1.0),
          ("vega", "ν", "Vega (per 1 vol pt)", "vega_1pct", 1.0), ("rho", "ρ", "Rho (per 1 rate pt)", "rho_1pct", 1.0)]
GREEK_DOC = {
    "delta": ("How much the option value changes when the underlying rises by +1 unit; also the hedge ratio.",
              "Call Δ ∈ (0, 1) and rises with S; Put Δ ∈ (−1, 0). Short positions flip the sign."),
    "gamma": ("How much Delta changes when the underlying rises by +1 unit (curvature of the value).",
              "Same for Call and Put. Highest near the strike and close to expiry. Long = long gamma, Short = short gamma."),
    "theta": ("Change in value per calendar day that passes, everything else constant (annual Θ ÷ 365).",
              "Usually negative for a Long option (time decay); a Short earns the decay. Deep in-the-money European puts can have Θ > 0."),
    "vega": ("Change in value when volatility rises by 1 percentage point.",
             "Same for Call and Put. Highest at the money and for long maturities. Long options gain when volatility rises."),
    "rho": ("Change in value when the interest rate rises by 1 percentage point.",
            "Positive for Calls, negative for Puts; grows with maturity. Short flips the sign."),
}


def greek_values(d, sgn):
    return {k: sgn * d[col] for k, _, _, col, _ in GREEKS}


def greeks_table(p, fac):
    S, K, sg, q, T = p["S0"], p["K"], p["sigma"] / 100.0, p["q"] / 100.0, p["T"]
    c, pu = (bsm_any(p, S, K, sg, T, q, o) for o in ("Call", "Put"))
    rows = []
    for _, sym, name, col, _ in GREEKS:
        rows.append({"Greek": f"{sym} {name}", "Call (long)": c[col], "Call (short)": -c[col], "Put (long)": pu[col], "Put (short)": -pu[col],
                     f"Your position ({p['pos']} {p['opt']}, total)": fac * (c if p["opt"] == "Call" else pu)[col]})
    return pd.DataFrame(rows)


def page_bsm():
    header("Black-Scholes-Merton", "European option price with continuous dividend yield, and its Greeks")
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        try:
            fx = bsm_inputs()
        except FinanceError as ex:
            fx = None
            st.error(str(ex))
    with right:
        try:
            p = P("bsm")
            S, K, sg, T, q, opt = p["S0"], p["K"], p["sigma"] / 100.0, p["T"], p["q"] / 100.0, p["opt"]
            d = bsm_any(p, S, K, sg, T, q, opt)
            fac = fin.position_factor(p["pos"], p["contracts"], p["mult"])
            sgn = 1.0 if p["pos"] == "Long" else -1.0
            r = d["r_used"]
            cu = f" {fx['asset']}" if fx else ""
            result_row(f"{opt} premium (per option)", f"{money(d['price'])}{cu}", f"d₁ = {d['d1']:.4f} · d₂ = {d['d2']:.4f} · r = {pct(r, 3)} (cont.) · T = {T:g} y",
                       [("Call premium", money(d["call"]), "European call"), ("Put premium", money(d["put"]), "European put"),
                        ("Intrinsic · time value", f"{money(d['intrinsic'], 2)} · {money(d['time_value'], 2)}", "Payoff now · premium − payoff")])
            if p["mkt_on"]:
                st.caption(f"Model − market = {d['price'] - p['mkt']:+.4f} (positive: the quote is below the model price).")
            if fx:
                st.caption(f"Premium in {fx['base']}: {money(d['price'] * fx['rate'])} at constant spot FX {fx['rate']:,.4f}.")
            gv = greek_values(d, sgn)
            kpi_row([(f"{sym} {name.split(' (')[0]}", f"{gv[k]:+.4f}" if k != "gamma" else f"{gv[k]:+.5f}", f"{p['pos']} {opt}, per option")
                     for k, sym, name, _, _ in GREEKS])
            tabs = st.tabs(["Value vs underlying", "Greeks table", "Formulas & checks"])
            with tabs[0]:
                grid = np.linspace(0.5 * K, 1.5 * K, 101)
                vals = [bsm_any(p, s, K, sg, T, q, opt)["price"] for s in grid]
                pay = np.maximum((grid - K) if opt == "Call" else (K - grid), 0.0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=grid, y=vals, name="Option value today", line=dict(color=CYAN, width=3)))
                fig.add_trace(go.Scatter(x=grid, y=pay, name="Payoff at maturity", line=dict(color=AMBER, width=2, dash="dash")))
                fig.add_trace(go.Scatter(x=[S], y=[d["price"]], name="Current", mode="markers", marker=dict(color="white", size=11, symbol="diamond")))
                plot(style_fig(fig, f"{opt} value vs underlying price", xt="Underlying price S", yt="Option value"))
            with tabs[1]:
                gt = greeks_table(p, fac)
                table(gt, {c: "%.5f" for c in gt.columns if c != "Greek"})
                st.caption(f"Total = sign × {p['contracts']:g} contracts × {p['mult']:g} multiplier × per-option Greek. Θ per calendar day; Vega and Rho per +1 percentage point.")
                export_button("bsm", {"Summary": pd.DataFrame({"Metric": ["S0", "K", "sigma", "T", "r", "q", "Call", "Put", "d1", "d2"],
                                                              "Value": [S, K, sg, T, r, q, d["call"], d["put"], d["d1"], d["d2"]]}), "Greeks": gt})
            with tabs[2]:
                st.latex(r"d_1=\frac{\ln(S/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T},\ d_2=d_1-\sigma\sqrt T")
                st.latex(r"C=Se^{-qT}N(d_1)-Ke^{-rT}N(d_2),\qquad P=Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1)")
                parity = d["call"] - d["put"] - (S * math.exp(-q * T) - K * math.exp(-r * T))
                h = 1e-4 * S
                fd = (bsm_any(p, S + h, K, sg, T, q, opt)["price"] - bsm_any(p, S - h, K, sg, T, q, opt)["price"]) / (2 * h)
                lb = max((S * math.exp(-q * T) - K * math.exp(-r * T)) if opt == "Call" else (K * math.exp(-r * T) - S * math.exp(-q * T)), 0)
                checks_box([("Put-call parity: C − P = S·e^(−qT) − K·e^(−rT)", abs(parity) < 1e-8, f"residual {parity:.1e}"),
                            ("Analytic Delta = finite-difference Delta", abs(fd - d["delta"]) < 1e-5, f"{d['delta']:.6f} vs {fd:.6f}"),
                            ("Price ≥ no-arbitrage lower bound", d["price"] >= lb - 1e-9, f"bound {lb:.4f}")])
                note("C − P is the value of a synthetic forward, not an option premium.")
        except FinanceError as ex:
            st.error(f"**Cannot calculate:** {ex}", icon=":material/error:")


def greek_figure(sw, col, scale, xt, title, cur_x, cur_y, sgn, same):
    fig = go.Figure()
    x = sw["x"] * scale
    if same:
        fig.add_trace(go.Scatter(x=x, y=sgn * sw[f"{col}_call"], name="Call = Put", line=dict(color=CYAN, width=3)))
    else:
        fig.add_trace(go.Scatter(x=x, y=sgn * sw[f"{col}_call"], name="Call", line=dict(color=CYAN, width=3)))
        fig.add_trace(go.Scatter(x=x, y=sgn * sw[f"{col}_put"], name="Put", line=dict(color=AMBER, width=3)))
    fig.add_trace(go.Scatter(x=[cur_x], y=[cur_y], mode="markers", name="Current", marker=dict(color="white", size=11, symbol="diamond")))
    return style_fig(fig, title, height=330, xt=xt, yt="Value per option")


def scenario_block(p, fac):
    c1, c2 = st.columns(2)
    with c1:
        w_num("bsm", "new_S", "New stock price", min_value=0.0, step=1.0, format="%.4f")
    with c2:
        w_num("bsm", "days", "Days passed", min_value=0.0, step=1.0, format="%.2f", help="Calendar days; cannot exceed the remaining life.")
    p = P("bsm")
    curve = need_curve() if p["source"] == "IRS curve" else None
    r = None if curve else p["rate"] / 100.0
    sc = fin.scenario_analysis(p["S0"], p["K"], p["sigma"] / 100.0, p["T"], r, p["q"] / 100.0, p["opt"], p["new_S"], p["days"], p["pos"],
                               p["contracts"], p["mult"], curve=curve)
    for w in sc["warnings"]:
        st.warning(w)
    err = sc["rel_error_pct"]
    kpi_row([("Initial price", money(sc["initial_price"]), "Per option"), ("Estimated new price (Δ+Γ+Θ)", money(sc["est_new_price"]), "Taylor estimate"),
             ("Actual new price", money(sc["actual_new_price"]), "Exact revaluation")])
    kpi_row([("Estimated P&L", money(sc["est_pnl_total"]), f"{p['pos']} × {p['contracts']:g} × {p['mult']:g}"),
             ("Actual P&L", money(sc["actual_pnl_total"]), "Total position"),
             ("Relative error", f"{err:.2f} %" if err is not None else "n/a", "|Est − Act| / |Act|")])
    t = sc["table"]
    table(t, {c: "%.4f" for c in t.columns if c != "Component"})


def page_greeks():
    header("Greeks Analysis", "Delta, Gamma, Theta, Vega and Rho: meaning, current value and sensitivity")
    left, right = st.columns([1, 1.7], gap="large")
    with left:
        try:
            bsm_inputs()
        except FinanceError as ex:
            st.error(str(ex))
    with right:
        try:
            p = P("bsm")
            S, K, sg, T, q, opt = p["S0"], p["K"], p["sigma"] / 100.0, p["T"], p["q"] / 100.0, p["opt"]
            curve = need_curve() if p["source"] == "IRS curve" else None
            rate = None if curve else p["rate"] / 100.0
            if rate is not None:
                fin.validate_rate(rate, "Interest rate")
            d = bsm_any(p, S, K, sg, T, q, opt)
            fac = fin.position_factor(p["pos"], p["contracts"], p["mult"])
            sgn = 1.0 if p["pos"] == "Long" else -1.0
            pos = p["pos"]
            gv = greek_values(d, sgn)
            st.caption(f"{pos} {opt} · S₀ = {S:,.2f} · K = {K:,.2f} · σ = {sg:.1%} · T = {T:g} y · r = {pct(d['r_used'], 3)} · q = {q:.1%} · values per option, sign of the position applied.")
            kpi_row([(f"{sym} {name.split(' (')[0]}", f"{gv[k]:+.4f}" if k != "gamma" else f"{gv[k]:+.5f}", unit) for (k, sym, name, _, _), unit in
                     zip(GREEKS, ["Value per +1 in S", "Δ per +1 in S", "Value per calendar day", "Value per +1 vol pt", "Value per +1 rate pt"])])
            sweeps = {"S": fin.greeks_sweep(S, K, sg, T, rate, q, "S", np.linspace(0.5 * K, 1.5 * K, 81), curve),
                      "T": fin.greeks_sweep(S, K, sg, T, rate, q, "T", np.linspace(max(T / 60, 1 / 365), T, 80), curve),
                      "sigma": fin.greeks_sweep(S, K, sg, T, rate, q, "sigma", np.linspace(0.05, max(sg * 2.5, 0.6), 80), curve)}
            if curve is None:
                sweeps["r"] = fin.greeks_sweep(S, K, sg, T, rate, q, "r", np.linspace(0.0, max(0.10, 2 * rate), 60), None)
            specs = {"delta": ("S", 1.0, "Underlying price S", S, False), "gamma": ("S", 1.0, "Underlying price S", S, True),
                     "theta": ("T", 1.0, "Time to maturity (years)", T, False), "vega": ("sigma", 100.0, "Volatility σ (%)", sg * 100.0, True),
                     "rho": ("r", 100.0, "Interest rate r (%)", d["r_used"] * 100.0, False)}
            texts = {"delta": f"A +1 move in S changes one {pos} {opt} by about {gv['delta']:+.4f}. Your position ({fac:,.0f} units) behaves like {fac * d['delta']:+,.2f} units of the underlying.",
                     "gamma": f"After a +1 move in S, Delta changes by about {gv['gamma']:+.5f}. " + ("Long gamma: large moves in either direction help." if sgn > 0 else "Short gamma: large moves hurt and the hedge needs rebalancing."),
                     "theta": f"Time alone changes one option by {gv['theta']:+.5f} per calendar day (annual {sgn * d['theta_annual']:+.4f}). Position: {fac * d['theta_daily']:+,.2f} per day.",
                     "vega": f"+1 volatility point changes the value by about {gv['vega']:+.4f} per option. Position: {fac * d['vega_1pct']:+,.2f}.",
                     "rho": f"+1 interest-rate point changes the value by about {gv['rho']:+.4f} per option. Position: {fac * d['rho_1pct']:+,.2f}."}
            tabs = st.tabs([f"{sym} {name.split(' (')[0]}" for _, sym, name, _, _ in GREEKS])
            for tab, (k, sym, name, col, _) in zip(tabs, GREEKS):
                with tab:
                    vary, scale, xt, cur_x, same = specs[k]
                    meaning, cp = GREEK_DOC[k]
                    st.markdown(f"**What it measures.** {meaning}  \n**Your value.** {texts[k]}  \n**Call vs Put.** {cp}")
                    if vary in sweeps:
                        plot(greek_figure(sweeps[vary], col, scale, xt, f"{name} vs {xt.split(' (')[0].lower()}", cur_x, gv[k], sgn, same))
                    else:
                        st.info("With the IRS curve the rate is not a free input, so the Rho chart needs a manual rate.")
            with st.expander("Call vs Put, Long vs Short and position totals"):
                gt = greeks_table(p, fac)
                table(gt, {c: "%.5f" for c in gt.columns if c != "Greek"})
            with st.expander("Scenario P&L: Delta-Gamma-Theta estimate vs exact revaluation"):
                scenario_block(p, fac)
            with st.expander("Formulas"):
                st.latex(r"\Delta_C=e^{-qT}N(d_1),\ \Delta_P=-e^{-qT}N(-d_1),\ \Gamma=\frac{e^{-qT}n(d_1)}{S\sigma\sqrt T},\ \nu=Se^{-qT}n(d_1)\sqrt T")
                st.latex(r"\Theta_C=-\frac{Se^{-qT}n(d_1)\sigma}{2\sqrt T}+qSe^{-qT}N(d_1)-rKe^{-rT}N(d_2),\ \ \rho_C=KTe^{-rT}N(d_2)")
        except FinanceError as ex:
            st.error(f"**Cannot calculate:** {ex}", icon=":material/error:")


# ============================================================================
# COMPARISON TOOL
# ============================================================================
def cmp_inputs():
    kind = w_radio("cmp", "kind", "Comparison", ["Binomial vs Black-Scholes", "Volatility sensitivity", "Forward: discrete vs continuous dividends"])
    with st.container(border=True):
        if kind.startswith("Forward"):
            note("Uses the inputs of the <b>Equity Forward</b> module: S₀, T, interest rate, discrete dividend table and yield q.")
            return kind
        group("SHARED PARAMETERS (tree and Black-Scholes use exactly the same)")
        a, b, c, d_ = st.columns(4)
        with a:
            w_num("cmp", "S0", "S₀", min_value=0.0, step=1.0, format="%.4f")
            w_sel("cmp", "opt", "Type", ["Call", "Put"])
        with b:
            w_num("cmp", "K", "Strike K", min_value=0.0, step=1.0, format="%.4f")
            w_num("cmp", "T", "T (years)", min_value=0.0, step=0.25, format="%.4f")
        with c:
            w_num("cmp", "sigma", "σ (%)", min_value=0.0, step=1.0, format="%.4f")
            w_num("cmp", "r", "r (%, cont.)", step=0.25, format="%.4f")
        with d_:
            w_num("cmp", "q", "q (%, cont.)", min_value=0.0, step=0.25, format="%.4f")
        if kind.startswith("Binomial"):
            _store("cmp", "steps", st.text_input("Numbers of steps to compare", value=_get("cmp", "steps"), key=_key("cmp", "steps")))
        else:
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                w_num("cmp", "vmin", "Min σ (%)", min_value=0.0, step=1.0)
            with e2:
                w_num("cmp", "vmax", "Max σ (%)", min_value=0.0, step=1.0)
            with e3:
                w_num("cmp", "vpts", "Points", min_value=2, max_value=100, step=1)
            with e4:
                w_num("cmp", "vN", "Tree steps", min_value=1, max_value=3000, step=10)
    return kind


def cmp_results(kind):
    p = P("cmp")
    S, K, sg, T, r, q, opt = p["S0"], p["K"], p["sigma"] / 100.0, p["T"], p["r"] / 100.0, p["q"] / 100.0, p["opt"]
    if kind.startswith("Binomial"):
        try:
            steps = sorted({int(x) for x in p["steps"].replace(";", ",").split(",") if x.strip()})
        except ValueError:
            raise FinanceError("Steps must be a comma-separated list of integers, e.g. 5, 10, 50, 100.")
        if not steps or min(steps) < 1 or max(steps) > 3000:
            raise FinanceError("Steps must be between 1 and 3000.")
        bs = fin.bsm(S, K, sg, T, r, q, opt)["price"]
        rows = []
        for n in steps:
            eu = fin.crr_option(S, K, sg, T, n, opt, "European", r=r, q=q).price
            am = fin.crr_option(S, K, sg, T, n, opt, "American", r=r, q=q).price
            rows.append({"Steps": n, "CRR European": eu, "CRR American": am, "BSM European": bs, "Eur − BSM": eu - bs,
                         "Abs. error (%)": abs(eu - bs) / max(bs, 1e-12) * 100, "Early-exercise premium": am - eu})
        t = pd.DataFrame(rows)
        best = t.loc[t["Abs. error (%)"].idxmin()]
        result_row("BSM European price (reference)", money(bs, 6), "European tree and BSM share S₀, K, σ, T, r, q (continuous)",
                   [("Best tree error", f"{best['Abs. error (%)']:.4f} %", f"At N = {int(best['Steps'])}"),
                    ("Early-exercise premium", money(t["Early-exercise premium"].iloc[-1], 6), f"American − European, N = {steps[-1]}")])
        tabs = st.tabs(["Chart", "Table"])
        with tabs[0]:
            f = go.Figure()
            f.add_trace(go.Scatter(x=t["Steps"], y=t["CRR European"], mode="lines+markers", name="CRR European", line=dict(color=CYAN, width=3)))
            f.add_trace(go.Scatter(x=t["Steps"], y=t["CRR American"], mode="lines+markers", name="CRR American", line=dict(color=BLUE, width=2)))
            f.add_hline(y=bs, line=dict(color=AMBER, dash="dash"), annotation_text="BSM European")
            f.update_xaxes(type="log")
            plot(style_fig(f, "Tree price vs number of steps", xt="Steps (log)", yt="Price"))
        with tabs[1]:
            table(t, {c: "%.6f" for c in t.columns if c != "Steps"})
            st.caption("American options have no closed form: only the European tree is compared with BSM.")
    elif kind == "Volatility sensitivity":
        if p["vmin"] <= 0 or p["vmax"] <= p["vmin"]:
            raise FinanceError("Volatility range must satisfy 0 < min < max.")
        rows = []
        for v in np.linspace(p["vmin"], p["vmax"], int(p["vpts"])) / 100.0:
            b = fin.bsm(S, K, v, T, r, q, opt)["price"]
            tr = fin.crr_option(S, K, v, T, int(p["vN"]), opt, "European", r=r, q=q).price
            rows.append({"Volatility (%)": v * 100, "BSM price": b, f"CRR price (N={int(p['vN'])})": tr, "Difference": tr - b})
        t = pd.DataFrame(rows)
        result_row("Max |tree − BSM|", money(t["Difference"].abs().max(), 6), f"Over σ ∈ [{p['vmin']:g}%, {p['vmax']:g}%], N = {int(p['vN'])}", [])
        tabs = st.tabs(["Chart", "Table"])
        with tabs[0]:
            f = go.Figure()
            f.add_trace(go.Scatter(x=t["Volatility (%)"], y=t["BSM price"], name="BSM", line=dict(color=CYAN, width=3)))
            f.add_trace(go.Scatter(x=t["Volatility (%)"], y=t.iloc[:, 2], name=t.columns[2], line=dict(color=AMBER, width=2, dash="dot")))
            plot(style_fig(f, "European option price vs volatility", xt="Volatility (%)", yt="Price"))
        with tabs[1]:
            table(t, {c: "%.6f" for c in t.columns})
    else:
        fp = P("fwd")
        if fp["source"] == "IRS curve":
            disc = need_curve()
        else:
            fin.validate_rate(fp["rate"] / 100.0, "Interest rate")
            disc = fin.FlatCurve(TODAY, fp["rate"] / 100.0, fp["comp"])
        divs = [(r_["t (years)"], r_["Amount"]) for r_ in ss.tables["div"].to_dict("records") if not fin._blank(r_["t (years)"]) and not fin._blank(r_["Amount"])]
        times = sorted({float(t_) for t_, _ in divs if 0 < float(t_) <= fp["T"]} | {float(fp["T"])})
        rows = []
        for Tj in times:
            dj = [(t_, a) for t_, a in divs if float(t_) <= Tj + 1e-12]
            dis = fin.equity_forward_t(fp["S0"], Tj, disc, "discrete", dj)
            con = fin.equity_forward_t(fp["S0"], Tj, disc, "continuous", q=fp["q"] / 100.0)
            rows.append({"T (years)": Tj, "No dividends": dis["F_nodiv"], "Discrete dividends": dis["F"], f"Continuous q = {fp['q']:g}%": con["F"],
                         "Equivalent continuous q (%)": dis["q_equiv"] * 100})
        t = pd.DataFrame(rows)
        last = t.iloc[-1]
        result_row("Forward at maturity", money(last["Discrete dividends"]), f"Discrete dividends · T = {fp['T']:g} y, S₀ = {fp['S0']:g}",
                   [("Continuous yield q", money(last.iloc[3]), f"q = {fp['q']:g}%"), ("No dividends", money(last["No dividends"]), "S₀ / DF(T)"),
                    ("Equivalent q", f"{last.iloc[4]:.3f} %", "Yield that reproduces the discrete forward")])
        tabs = st.tabs(["Chart", "Table"])
        with tabs[0]:
            f = go.Figure()
            for c, col in zip(t.columns[1:4], [MUTED, CYAN, AMBER]):
                f.add_trace(go.Scatter(x=t["T (years)"], y=t[c], name=c, mode="lines+markers", line=dict(color=col, width=3)))
            plot(style_fig(f, "Forward price by maturity", xt="T (years)", yt="Forward price"))
        with tabs[1]:
            table(t, {c: "%.4f" for c in t.columns})
            st.caption("Same spot, same discount curve and same maturities on every line.")


def page_compare():
    header("Comparison Tool", "Only comparable quantities are compared: European vs European, forward vs forward")
    kind = cmp_inputs()
    try:
        cmp_results(kind)
    except FinanceError as ex:
        st.error(f"**Cannot calculate:** {ex}", icon=":material/error:")


# ============================================================================
# Navegación
# ============================================================================
with st.sidebar:
    st.markdown('<div class="brand">DERIVATIVES <span>LAB</span></div><div class="brand-sub">Valuation platform</div>', unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    pk = ss.curve_pack
    st.markdown(f'<span class="pill {"ok" if pk else "warn"}">IRS curve: {pk["label"] if pk else "default example"}</span>', unsafe_allow_html=True)
    st.caption("Inputs are kept while you move between modules. Defaults are illustrative examples.")

{"Dashboard": page_dashboard, "Equity Forward": page_forward, "IRS Curve": page_irs, "Binomial Trees": page_binomial,
 "Black-Scholes-Merton": page_bsm, "Greeks Analysis": page_greeks, "Comparison Tool": page_compare}[ss.nav]()
