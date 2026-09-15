"""
10-Q Equity Research Agent -- Streamlit entrypoint.

Enter any US-listed ticker or company name -> pulls its latest real 10-Q
from SEC EDGAR (free, public, no key) -> computes standardized financial
ratios from the filing's actual XBRL data -> runs a 5-step LLM analysis
pipeline (services/pipeline.py) -> renders a quantitative dashboard, risk
synthesis, and a final equity research stance. Supports comparing two
tickers side by side.

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io, pointed at this repo, app.py as
               the entrypoint, GEMINI_API_KEY set in the app's Secrets.
"""

import hashlib

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
.block-container { padding-top: 2rem; max-width: 1200px; }
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
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    border-radius: 0.75rem;
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.15);
    margin-bottom: 1rem;
}
.company-card-text b { font-size: 1.05rem; }
.gap-note {
    font-size: 0.85rem;
    color: var(--neutral);
    border-left: 3px solid var(--neutral);
    padding-left: 0.75rem;
    margin-top: 0.5rem;
}
.step-recap {
    border-left: 3px solid var(--accent);
    padding: 0.4rem 0 0.4rem 0.9rem;
    margin: 0.5rem 0 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────
st.title("\U0001F4C8 10-Q Equity Research Agent")
st.caption(
    "Real SEC filings in, a 5-step grounded LLM research pipeline out. "
    "Free-tier end to end — SEC EDGAR (no key) + Gemini free tier."
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
        "prior steps, not five independent prompts against the same raw dump, and each step's finding is shown "
        "live as it completes."
    )

# ── Company logos (curated domains for Clearbit's free logo API; anything
#    not in this map falls back to a generated initials badge, never a
#    broken image) ─────────────────────────────────────────────────────────
_LOGO_DOMAINS = {
    "AAPL": "apple.com", "MSFT": "microsoft.com", "GOOGL": "abc.xyz", "GOOG": "abc.xyz",
    "AMZN": "amazon.com", "META": "meta.com", "NVDA": "nvidia.com", "TSLA": "tesla.com",
    "JPM": "jpmorganchase.com", "BAC": "bankofamerica.com", "WFC": "wellsfargo.com",
    "GS": "goldmansachs.com", "MS": "morganstanley.com", "C": "citigroup.com",
    "KO": "coca-colacompany.com", "PEP": "pepsico.com", "MCD": "mcdonalds.com",
    "SBUX": "starbucks.com", "NKE": "nike.com", "DIS": "disney.com",
    "NFLX": "netflix.com", "V": "visa.com", "MA": "mastercard.com",
    "PYPL": "paypal.com", "ADBE": "adobe.com", "CRM": "salesforce.com",
    "ORCL": "oracle.com", "INTC": "intel.com", "AMD": "amd.com",
    "IBM": "ibm.com", "CSCO": "cisco.com", "QCOM": "qualcomm.com",
    "T": "att.com", "VZ": "verizon.com", "XOM": "exxonmobil.com",
    "CVX": "chevron.com", "WMT": "walmart.com", "TGT": "target.com",
    "HD": "homedepot.com", "COST": "costco.com", "UNH": "unitedhealthgroup.com",
    "JNJ": "jnj.com", "PFE": "pfizer.com", "MRK": "merck.com",
    "BA": "boeing.com", "GE": "ge.com", "F": "ford.com", "GM": "gm.com",
    "UBER": "uber.com", "ABNB": "airbnb.com", "SHOP": "shopify.com",
    "SQ": "squareup.com", "COIN": "coinbase.com", "SPOT": "spotify.com",
    "SNAP": "snap.com", "PINS": "pinterest.com", "RIVN": "rivian.com",
    "BRK-B": "berkshirehathaway.com",
}


def _avatar_color(ticker: str) -> str:
    """Deterministic color from the ticker so the same symbol always gets
    the same fallback badge color across runs (not random per render)."""
    digest = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
    hue = digest % 360
    return f"hsl({hue}, 62%, 46%)"


def _avatar_badge(ticker: str, size: int) -> str:
    initials = ticker[:2]
    color = _avatar_color(ticker)
    return (
        f'<div style="display:flex;width:{size}px;height:{size}px;border-radius:14px;'
        f"background:{color};color:white;align-items:center;justify-content:center;"
        f'font-weight:800;font-size:{size * 0.34}px;letter-spacing:-0.02em;">{initials}</div>'
    )


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _logo_url(domain: str) -> str | None:
    """Google's favicon service (no key, no signup) -- found live: Clearbit's
    free logo API is dead (logo.clearbit.com doesn't even resolve in DNS
    anymore, discontinued after the 2023 HubSpot acquisition), and Streamlit's
    markdown sanitizer strips inline onerror handlers, so a client-side <img
    onerror> fallback can't paper over a dead source either way. Checked
    server-side once per domain per day; only returns a URL confirmed to
    actually resolve, so a broken image never reaches the page."""
    url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    try:
        r = requests.get(url, timeout=3, stream=True)
        ok = r.status_code == 200 and (r.headers.get("content-type") or "").startswith("image")
        r.close()
        return url if ok else None
    except requests.exceptions.RequestException:
        return None


def company_avatar_html(ticker: str, size: int = 56) -> str:
    """A real company favicon (Google's favicon service, keyed off a curated
    domain map), pre-checked server-side so an unresolvable logo never
    reaches the page as a broken image -- falls back to a generated
    initials badge instead."""
    ticker = ticker.upper()
    domain = _LOGO_DOMAINS.get(ticker)
    url = _logo_url(domain) if domain else None
    if url:
        img = (
            f'<img src="{url}" '
            f'style="width:{size}px;height:{size}px;border-radius:14px;object-fit:contain;'
            f'background:white;padding:8px;box-sizing:border-box;" />'
        )
        return f'<div style="display:inline-block;">{img}</div>'
    return _avatar_badge(ticker, size)


# ── Ticker / company-name search (real autocomplete over SEC's own ~10k-
#    company directory, not a small hardcoded list) ─────────────────────────
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _load_ticker_options() -> list[str]:
    directory = edgar_client.company_directory()
    seen: dict[str, str] = {}
    for t, name, _cik in directory:
        if t not in seen:
            seen[t] = name
    return [f"{t} — {name}" for t, name in sorted(seen.items())]


def _ticker_from_label(label: str | None) -> str:
    if not label:
        return ""
    return label.split(" — ", 1)[0].strip().upper()


TICKER_OPTIONS = _load_ticker_options()
_LABEL_BY_TICKER = {opt.split(" — ", 1)[0]: opt for opt in TICKER_OPTIONS}
EXAMPLE_TICKERS = ["AAPL", "MSFT", "JPM", "NVDA", "KO"]

# Example-ticker buttons run BEFORE the selectbox is instantiated below --
# Streamlit only allows setting a widget's session_state value before that
# widget renders in a given script run, not after.
clicked_ticker: str | None = None
compare_mode = st.checkbox("Compare with a second ticker")

btn_cols = st.columns(len(EXAMPLE_TICKERS) + 1)
btn_cols[0].caption("Try:")
for c, t in zip(btn_cols[1:], EXAMPLE_TICKERS):
    if c.button(t, use_container_width=True, key=f"ex_{t}"):
        clicked_ticker = t

if clicked_ticker and clicked_ticker in _LABEL_BY_TICKER:
    # Found live: setting session_state["ticker_select"] and letting the
    # selectbox render later in this SAME run left the box empty -- the
    # click registered (button went active/red) but the value never took.
    # A hard rerun, so the selectbox reads the pre-set value from a fresh
    # run instead of fighting same-run widget-instantiation ordering, is
    # the documented-safe version of this pattern.
    st.session_state["ticker_select"] = _LABEL_BY_TICKER[clicked_ticker]
    st.rerun()

col1, col2 = st.columns(2) if compare_mode else (st.container(), None)
with col1:
    label1 = st.selectbox(
        "Ticker or company name" if not compare_mode else "First ticker",
        options=TICKER_OPTIONS,
        index=None,
        placeholder="Type a ticker or company name (e.g. Apple, AAPL, Microsoft)...",
        key="ticker_select",
        label_visibility="collapsed" if not compare_mode else "visible",
    )
ticker1 = _ticker_from_label(label1)

ticker2 = ""
if compare_mode:
    with col2:
        label2 = st.selectbox(
            "Second ticker",
            options=TICKER_OPTIONS,
            index=None,
            placeholder="Type a second ticker or company name...",
            key="ticker_select_2",
        )
    ticker2 = _ticker_from_label(label2)

run_clicked = st.button(
    "Run analysis" if not compare_mode else "Compare",
    type="primary",
    disabled=not ticker1 or (compare_mode and not ticker2),
)

STANCE_CLASS = {"BULLISH": "stance-bullish", "BEARISH": "stance-bearish"}


def escape_markdown_math(text: str) -> str:
    """Found live: the LLM's own prose routinely writes two dollar amounts in
    one sentence ('$34.37 billion... $2.46 billion'), and Streamlit's
    markdown renders a paired '$...$' as inline LaTeX math, silently eating
    the text between them. Escaping every literal '$' turns off math mode
    without changing anything the reader sees."""
    return text.replace("$", "\\$")


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
            return f"${value / 1e9:,.2f}B"
        return f"{value:.2f}"
    return str(value)


def run_and_render(ticker: str, container) -> None:
    """Everything for one ticker: fetch, compute, run the pipeline, render.
    Called once for a single lookup, or twice (once per column) when
    comparing two tickers, so the two modes share one code path.

    Found live: st.spinner()/st.status() aren't available as methods on a
    column's DeltaGenerator in this Streamlit version (only a subset of
    element functions are, like .markdown/.error/.columns/.tabs) --
    StreamlitAPIException at the first container.spinner() call. Wrapping
    the whole function body in `with container:` and using plain,
    unqualified st.X() calls throughout sidesteps that distinction
    entirely: Streamlit routes every element -- spinner and status
    included -- into whichever container is the active context."""
    with container:
        try:
            with st.spinner(f"Looking up {ticker} on SEC EDGAR..."):
                filing = edgar_client.latest_10q(ticker)
        except requests.exceptions.RequestException as exc:
            st.error(f"Couldn't reach SEC EDGAR: {exc}. It may be rate-limiting or temporarily down — try again shortly.")
            return

        if filing is None:
            st.error(
                f"No 10-Q found for '{ticker}' on SEC EDGAR. Check it's a US-listed filer that reports on Form 10-Q "
                "(foreign private issuers file 6-K/20-F instead, and won't resolve here)."
            )
            return

        st.markdown(
            f'<div class="company-card">{company_avatar_html(ticker)}'
            f'<div class="company-card-text"><b>{filing.company_name}</b> ({ticker})<br/>'
            f"10-Q for period ending {filing.report_date}, filed {filing.filing_date}. "
            f'<a href="{filing.document_url}" target="_blank">View the real filing on SEC.gov →</a></div></div>',
            unsafe_allow_html=True,
        )

        try:
            with st.spinner("Fetching structured XBRL financials..."):
                facts_json = edgar_client.company_facts(filing.cik)
                facts = ratios_mod.extract_facts(facts_json)
                computed = ratios_mod.compute_ratios(facts)
        except requests.exceptions.RequestException as exc:
            st.error(f"Couldn't fetch financial data from SEC EDGAR: {exc}. Try again shortly.")
            return

        # ── KPI row ──────────────────────────────────────────────────────
        prof, cf, growth = computed["profitability"], computed["cash_flow"], computed["yoy_growth"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Revenue YoY", fmt_ratio("yoy", growth.get("revenue_yoy")) if growth.get("revenue_yoy") is not None else "n/a")
        k2.metric("Gross Margin", fmt_ratio("margin", prof.get("gross_margin")))
        k3.metric("Operating Margin", fmt_ratio("margin", prof.get("operating_margin")))
        k4.metric("Free Cash Flow", fmt_ratio("fcf", cf.get("free_cash_flow")))

        # ── Trend charts (real multi-quarter history, not just 2 points) ──
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
            return

        # ── 5-step pipeline with live progress AND a live recap per step ──
        run = None
        try:
            with st.status("Running 5-step research pipeline...", expanded=True) as status:
                def _on_step(i, title):
                    status.update(label=f"Step {i}/5: {title}")

                def _on_step_done(i, title, output):
                    preview = output if len(output) <= 500 else output[:500].rsplit(" ", 1)[0] + "…"
                    preview = escape_markdown_math(preview)
                    st.markdown(f"**✓ Step {i}/5 — {title}**")
                    st.markdown(f'<div class="step-recap">{preview}</div>', unsafe_allow_html=True)

                run = run_analysis(
                    filing.company_name, ticker, filing.report_date, computed, filing_text,
                    on_step=_on_step, on_step_done=_on_step_done,
                )
                status.update(label="Analysis complete", state="complete")
        except RuntimeError as exc:
            st.error(str(exc))
            return
        except (ClientError, ServerError) as exc:
            # Every model in the fallback chain rejected the same request --
            # gemini_client already retries individual models through quota
            # exhaustion, overload, and transient bad-request errors, so
            # getting here means it's not one flaky model, it's every one.
            st.error(
                f"The free-tier LLM pipeline failed on every fallback model: {exc}. "
                "Free-tier quota resets daily -- try again later, or try a different ticker "
                "(a very long or unusual filing can occasionally trip a request-size limit)."
            )
            return

        if run is None:
            return

        st.header("Equity Research Stance")
        st.markdown(stance_badge(run.recommendation), unsafe_allow_html=True)
        st.markdown(escape_markdown_math(run.recommendation))

        st.header("Step-by-Step Analysis")
        tabs = st.tabs([s.title for s in run.steps])
        for tab, step in zip(tabs, run.steps):
            with tab:
                st.markdown(escape_markdown_math(step.output))


if run_clicked:
    if compare_mode:
        left, right = st.columns(2)
        run_and_render(ticker1, left)
        run_and_render(ticker2, right)
    else:
        run_and_render(ticker1, st.container())
