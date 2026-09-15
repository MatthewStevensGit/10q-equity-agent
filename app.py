"""
10-Q Equity Research Agent -- Streamlit entrypoint.

Enter any US-listed ticker -> pulls its latest real 10-Q from SEC EDGAR
(free, public, no key) -> computes standardized financial ratios from the
filing's actual XBRL data -> runs a 5-step LLM analysis pipeline
(services/pipeline.py) -> renders a quantitative dashboard, risk synthesis,
and a final equity research stance.

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io, pointed at this repo, app.py as
               the entrypoint, GEMINI_API_KEY set in the app's Secrets.
"""

import requests
import streamlit as st
from google.genai.errors import ClientError, ServerError

from core import edgar_client, ratios as ratios_mod
from services.pipeline import run_analysis

st.set_page_config(page_title="10-Q Equity Research Agent", page_icon="\U0001F4C8", layout="wide")

# ── Styling ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
    --accent: #2563eb;
    --bull: #16a34a;
    --bear: #dc2626;
    --neutral: #6b7280;
}
.block-container { padding-top: 2rem; max-width: 1100px; }
.stance-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
.stance-bullish { background: rgba(22,163,74,0.12); color: var(--bull); border: 1px solid rgba(22,163,74,0.35); }
.stance-bearish { background: rgba(220,38,38,0.12); color: var(--bear); border: 1px solid rgba(220,38,38,0.35); }
.stance-neutral { background: rgba(107,114,128,0.12); color: var(--neutral); border: 1px solid rgba(107,114,128,0.35); }
.company-card {
    padding: 1rem 1.25rem;
    border-radius: 0.75rem;
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.15);
    margin-bottom: 1rem;
}
.gap-note {
    font-size: 0.85rem;
    color: var(--neutral);
    border-left: 3px solid var(--neutral);
    padding-left: 0.75rem;
    margin-top: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────
st.title("\U0001F4C8 10-Q Equity Research Agent")
st.caption(
    "Real SEC filings in, a 5-step grounded LLM research pipeline out. "
    "Free-tier end to end — SEC EDGAR (no key) + Gemini free tier. Educational demo, not investment advice."
)

with st.expander("How this actually works", expanded=False):
    st.markdown(
        "1. **SEC EDGAR** — resolves the ticker to a CIK, pulls the latest real 10-Q and its structured XBRL "
        "financial facts (the same standardized data every US public company reports).\n"
        "2. **Ratio engine** — computes profitability, liquidity, leverage, and cash-flow ratios directly from "
        "that XBRL data, with duration- and period-alignment checks so a quarterly figure never gets silently "
        "mixed with an annual one.\n"
        "3. **5-step LLM pipeline** — quantitative snapshot → risk/MD&A synthesis → narrative-vs-numbers "
        "consistency check → capital allocation read → final equity stance. Each step is grounded in the real "
        "prior steps, not five independent prompts against the same raw dump."
    )

# ── Input ────────────────────────────────────────────────────────────────
col_input, col_examples = st.columns([2, 3])
with col_input:
    ticker_input = st.text_input("Ticker", placeholder="e.g. AAPL, MSFT, JPM, BRK.B", label_visibility="collapsed")
with col_examples:
    st.caption("Try:")
    ex_cols = st.columns(5)
    example_tickers = ["AAPL", "MSFT", "JPM", "NVDA", "KO"]
    clicked_example = None
    for c, t in zip(ex_cols, example_tickers):
        if c.button(t, use_container_width=True):
            clicked_example = t

ticker = (clicked_example or ticker_input or "").strip().upper()
run_clicked = st.button("Run analysis", type="primary", disabled=not ticker) or bool(clicked_example)

STANCE_CLASS = {"BULLISH": "stance-bullish", "BEARISH": "stance-bearish"}


def stance_badge(text: str) -> str:
    upper = text.upper()
    cls = next((v for k, v in STANCE_CLASS.items() if k in upper), "stance-neutral")
    label = next((k for k in STANCE_CLASS if k in upper), "NEUTRAL")
    return f'<span class="stance-badge {cls}">{label}</span>'


def fmt_ratio(key: str, value) -> str:
    """Found live: XBRL values come back as plain Python ints whenever the
    filer's own value had no decimal point (most raw dollar figures) -- an
    `isinstance(value, float)` check silently skipped all of those and fell
    through to a raw, unformatted number (e.g. free cash flow rendered as
    literal `5570000000` instead of `$5.57B`). Checking (int, float)
    together fixes every dollar-amount metric, not just the ones that
    happened to come back as floats."""
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if any(w in key for w in ("margin", "yoy", "ratio")) and abs(value) < 3:
            return f"{value:+.1%}" if "yoy" in key else f"{value:.1%}"
        if abs(value) >= 1_000_000:
            return f"${value/1e9:,.2f}B"
        return f"{value:.2f}"
    return str(value)


if run_clicked and ticker:
    try:
        with st.spinner(f"Looking up {ticker} on SEC EDGAR..."):
            filing = edgar_client.latest_10q(ticker)
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't reach SEC EDGAR: {exc}. It may be rate-limiting or temporarily down — try again shortly.")
        st.stop()

    if filing is None:
        st.error(
            f"No 10-Q found for '{ticker}' on SEC EDGAR. Check it's a US-listed filer that reports on Form 10-Q "
            "(foreign private issuers file 6-K/20-F instead, and won't resolve here)."
        )
        st.stop()

    st.markdown(
        f'<div class="company-card"><b>{filing.company_name}</b> ({ticker}) — 10-Q for period ending '
        f'{filing.report_date}, filed {filing.filing_date}. '
        f'<a href="{filing.document_url}" target="_blank">View the real filing on SEC.gov →</a></div>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Fetching structured XBRL financials..."):
            facts_json = edgar_client.company_facts(filing.cik)
            facts = ratios_mod.extract_facts(facts_json)
            computed = ratios_mod.compute_ratios(facts)
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't fetch financial data from SEC EDGAR: {exc}. Try again shortly.")
        st.stop()

    # ── KPI row ──────────────────────────────────────────────────────────
    prof, cf, growth = computed["profitability"], computed["cash_flow"], computed["yoy_growth"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue YoY", fmt_ratio("yoy", growth.get("revenue_yoy")) if growth.get("revenue_yoy") is not None else "n/a")
    k2.metric("Gross Margin", fmt_ratio("margin", prof.get("gross_margin")))
    k3.metric("Operating Margin", fmt_ratio("margin", prof.get("operating_margin")))
    k4.metric("Free Cash Flow", fmt_ratio("fcf", cf.get("free_cash_flow")))

    # ── Trend charts (real multi-quarter history, not just 2 points) ─────
    hist_col1, hist_col2 = st.columns(2)
    try:
        rev_hist = ratios_mod.extract_history(facts_json, "revenue", n_quarters=8)
        ni_hist = ratios_mod.extract_history(facts_json, "net_income", n_quarters=8)
        if rev_hist:
            with hist_col1:
                st.caption("Revenue, last 8 quarters ($)")
                st.line_chart({e: v for e, v in rev_hist})
        if ni_hist:
            with hist_col2:
                st.caption("Net income, last 8 quarters ($)")
                st.line_chart({e: v for e, v in ni_hist})
    except Exception:
        pass  # trend charts are a bonus visualization; never block the core analysis on them

    with st.expander("Full computed ratio breakdown", expanded=False):
        for section, values in computed.items():
            if section == "data_gaps":
                continue
            st.markdown(f"**{section.replace('_', ' ').title()}**")
            st.table({k.replace("_", " "): fmt_ratio(k, v) for k, v in values.items()})
        if computed.get("data_gaps"):
            st.markdown(
                f'<div class="gap-note">Not tagged in this filing\'s XBRL data (reported as unavailable, '
                f'never estimated): {", ".join(computed["data_gaps"])}</div>',
                unsafe_allow_html=True,
            )

    try:
        with st.spinner("Fetching filing text for MD&A / risk factors..."):
            filing_text = edgar_client.fetch_filing_text(filing)
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't fetch the filing text: {exc}. Try again shortly.")
        st.stop()

    # ── 5-step pipeline with live progress ────────────────────────────────
    run = None
    try:
        with st.status("Running 5-step research pipeline...", expanded=True) as status:
            run = run_analysis(
                filing.company_name, ticker, filing.report_date, computed, filing_text,
                on_step=lambda i, title: status.update(label=f"Step {i}/5: {title}"),
            )
            status.update(label="Analysis complete", state="complete")
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    except (ClientError, ServerError) as exc:
        # Every model in the fallback chain rejected the same request --
        # gemini_client already retries individual models through quota
        # exhaustion, overload, and transient bad-request errors, so getting
        # here means it's not one flaky model, it's every one of them.
        st.error(
            f"The free-tier LLM pipeline failed on every fallback model: {exc}. "
            "Free-tier quota resets daily -- try again later, or try a different ticker "
            "(a very long or unusual filing can occasionally trip a request-size limit)."
        )
        st.stop()

    if run is None:
        st.stop()

    st.header("Equity Research Stance")
    st.markdown(stance_badge(run.recommendation), unsafe_allow_html=True)
    st.markdown(run.recommendation)

    st.header("Step-by-Step Analysis")
    tabs = st.tabs([s.title for s in run.steps])
    for tab, step in zip(tabs, run.steps):
        with tab:
            st.markdown(step.output)

    st.caption(
        "Generated by a 5-step LLM pipeline grounded in this filing's real SEC XBRL data and filing text. "
        "Educational demo only — not investment advice."
    )
