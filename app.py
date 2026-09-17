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
from dataclasses import dataclass

import pandas as pd
import requests
import streamlit as st
from google.genai.errors import ClientError, ServerError
from streamlit_searchbox import st_searchbox

from core import edgar_client, ratios as ratios_mod
from services.pipeline import AnalysisRun, compare_stances, run_analysis

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
/* layout="wide" above already tells Streamlit to use the full viewport,
   but this rule was still capping the actual content column at 1200px --
   fine on a laptop, wasteful on a real desktop monitor. This app is used
   mostly at a desk (unlike Kaiser-Stats, which is mostly phone), so widen
   the working area instead of defaulting to a narrow, scroll-heavy column. */
.block-container { padding-top: 2rem; max-width: 1600px; }
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
.kpi-win { color: var(--bull); font-weight: 700; }
.verdict-card {
    background: rgba(37,99,235,0.05);
    border: 1px solid rgba(37,99,235,0.2);
    border-left: 4px solid var(--accent);
    border-radius: 0.5rem;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 1.5rem;
}
h1, h2, h3 { letter-spacing: -0.01em; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────
st.title("\U0001F4C8 10-Q Equity Research Agent")

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


# ── Ticker search (one real combobox, ranked on the ticker itself) ────────
#
# Streamlit's own selectbox filters its options client-side by unranked
# substring match against whatever text is displayed, so typing "BE" could
# surface "ABEO" (contains "be") ahead of the actual exact ticker "BE" --
# and there's no way to fix that from inside a plain selectbox, since its
# filtering runs in the browser, not in this script. A separate text_input
# next to it fixed the ranking but made it two boxes instead of one search
# experience. streamlit_searchbox is a single widget that calls a real
# Python function on every keystroke and renders ITS return value as the
# dropdown, so the same box both takes the input and shows the live-ranked
# results -- and the ranking below only ever looks at the ticker itself:
# exact ticker > ticker starts with the query > ticker contains the query.
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _load_ticker_directory() -> list[tuple[str, str]]:
    directory = edgar_client.company_directory()
    seen: dict[str, str] = {}
    for t, name, _cik in directory:
        if t not in seen:
            seen[t] = name
    return sorted(seen.items())


def _ticker_rank(query: str, ticker: str) -> int:
    """Lower is a better match; 3 means "no match" and gets dropped."""
    if ticker == query:
        return 0
    if ticker.startswith(query):
        return 1
    if query in ticker:
        return 2
    return 3


def _search_tickers(query: str, directory: list[tuple[str, str]], limit: int = 20) -> list[tuple[str, str]]:
    """Returns (display_label, ticker) pairs -- st_searchbox shows the label
    in the dropdown and returns the matching ticker once one is picked."""
    q = query.strip().upper()
    if not q:
        return [(f"{t} — {name}", t) for t, name in directory[:limit]]
    ranked = sorted(
        ((_ticker_rank(q, t), t, name) for t, name in directory if _ticker_rank(q, t) < 3),
        key=lambda r: (r[0], r[1]),
    )
    return [(f"{t} — {name}", t) for _, t, name in ranked[:limit]]


TICKER_DIRECTORY = _load_ticker_directory()

compare_mode = st.checkbox("Compare with a second ticker")

col1, col2 = st.columns(2) if compare_mode else (st.container(), None)
with col1:
    ticker1 = st_searchbox(
        lambda q: _search_tickers(q, TICKER_DIRECTORY),
        placeholder="Type a ticker (e.g. AAPL, BE, MSFT)...",
        label="Ticker" if not compare_mode else "First ticker",
        key="ticker_searchbox_1",
        clear_on_submit=False,
    ) or ""
ticker1 = ticker1.strip().upper()

ticker2 = ""
if compare_mode:
    with col2:
        ticker2 = st_searchbox(
            lambda q: _search_tickers(q, TICKER_DIRECTORY),
            placeholder="Type a second ticker...",
            label="Second ticker",
            key="ticker_searchbox_2",
            clear_on_submit=False,
        ) or ""
    ticker2 = ticker2.strip().upper()

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
    # Found live: scanning the FULL multi-paragraph recommendation let a
    # later hedging phrase ("...rather than a fully bullish read") override
    # the actually-stated stance -- one real case showed a BULLISH badge on
    # a recommendation whose own heading said "Final Equity Research
    # Stance: Neutral". The model always states its stance up front (see
    # the step-5 prompt in services/pipeline.py), so only search that
    # opening window, not the whole text.
    opening = text[:150].upper()
    cls = next((v for k, v in STANCE_CLASS.items() if k in opening), "stance-neutral")
    label = next((k for k in STANCE_CLASS if k in opening), "NEUTRAL")
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


@dataclass
class Bundle:
    ticker: str
    filing: object
    facts_json: dict
    computed: dict
    run: AnalysisRun


def fetch_and_analyze(ticker: str) -> "Bundle | None":
    """Fetch, compute, and run the pipeline for one ticker -- everything up
    to but NOT including rendering. Kept separate from rendering (below) so
    compare mode can bring both tickers to full completion first, then draw
    every section as one aligned row shared by both sides, instead of the
    old approach of running each ticker's entire fetch-render flow straight
    through in its own column. That's what caused the misalignment: two
    independently-flowing columns drift apart the moment one side's content
    (a longer company name, a longer stance paragraph) is taller than the
    other's, and every section below the drift inherits the offset. A
    shared row for each section can't drift, because both sides are built
    from the same Streamlit columns() call for that row alone."""
    try:
        with st.spinner(f"Looking up {ticker} on SEC EDGAR..."):
            filing = edgar_client.latest_10q(ticker)
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't reach SEC EDGAR for {ticker}: {exc}. It may be rate-limiting or temporarily down — try again shortly.")
        return None

    if filing is None:
        st.error(
            f"No 10-Q found for '{ticker}' on SEC EDGAR. Check it's a US-listed filer that reports on Form 10-Q "
            "(foreign private issuers file 6-K/20-F instead, and won't resolve here)."
        )
        return None

    try:
        with st.spinner(f"Fetching {ticker}'s structured XBRL financials..."):
            facts_json = edgar_client.company_facts(filing.cik)
            facts = ratios_mod.extract_facts(facts_json)
            computed = ratios_mod.compute_ratios(facts)
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't fetch financial data for {ticker} from SEC EDGAR: {exc}. Try again shortly.")
        return None

    try:
        with st.spinner(f"Fetching {ticker}'s filing text for MD&A / risk factors..."):
            filing_text = edgar_client.fetch_filing_text(filing)
    except requests.exceptions.RequestException as exc:
        st.error(f"Couldn't fetch {ticker}'s filing text: {exc}. Try again shortly.")
        return None

    # ── 5-step pipeline with live progress AND a live recap per step ──
    run = None
    try:
        with st.status(f"Running 5-step research pipeline for {ticker}...", expanded=True) as status:
            def _on_step(i, title):
                # expanded=True must be re-asserted on every update() call --
                # st.status can silently fall back to collapsed otherwise,
                # which is what was closing the recap after each step.
                status.update(label=f"{ticker} — Step {i}/5: {title}", expanded=True)

            def _on_step_done(i, title, output):
                # Used to hard-truncate this to 500 chars -- verified live
                # against real runs that this silently cut off every step
                # mid-sentence (confirmed: the "…" landed at ~500 chars into
                # EVERY step, every run), which is exactly why a risk table
                # with 3 rows (Risk 2 or 3, e.g. "Management") could render
                # as if there were nothing there -- the real content existed,
                # it just never reached the page. Show the full real output.
                rendered = escape_markdown_math(output)
                st.markdown(f"**✓ Step {i}/5 — {title}**")
                st.markdown(f'<div class="step-recap">{rendered}</div>', unsafe_allow_html=True)

            run = run_analysis(
                filing.company_name, ticker, filing.report_date, computed, filing_text,
                on_step=_on_step, on_step_done=_on_step_done,
            )
            status.update(label=f"{ticker} analysis complete", state="complete", expanded=True)
    except RuntimeError as exc:
        st.error(str(exc))
        return None
    except (ClientError, ServerError) as exc:
        # Every model in the fallback chain rejected the same request --
        # gemini_client already retries individual models through quota
        # exhaustion, overload, and transient bad-request errors, so
        # getting here means it's not one flaky model, it's every one.
        st.error(
            f"The free-tier LLM pipeline failed on every fallback model for {ticker}: {exc}. "
            "Free-tier quota resets daily -- try again later, or try a different ticker "
            "(a very long or unusual filing can occasionally trip a request-size limit)."
        )
        return None

    if run is None:
        return None
    return Bundle(ticker=ticker, filing=filing, facts_json=facts_json, computed=computed, run=run)


# ── Render building blocks -- each renders into whatever container is
#    currently active (the caller opens `with column:` where it matters),
#    shared verbatim by single-ticker mode and compare mode. ────────────────
def render_company_card(bundle: Bundle) -> None:
    filing = bundle.filing
    st.markdown(
        f'<div class="company-card">{company_avatar_html(bundle.ticker)}'
        f'<div class="company-card-text"><b>{filing.company_name}</b> ({bundle.ticker})<br/>'
        f"10-Q for period ending {filing.report_date}, filed {filing.filing_date}. "
        f'<a href="{filing.document_url}" target="_blank">View the real filing on SEC.gov →</a></div></div>',
        unsafe_allow_html=True,
    )


def render_kpis(bundle: Bundle) -> None:
    prof, cf, growth = bundle.computed["profitability"], bundle.computed["cash_flow"], bundle.computed["yoy_growth"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue YoY", fmt_ratio("yoy", growth.get("revenue_yoy")) if growth.get("revenue_yoy") is not None else "n/a")
    k2.metric("Gross Margin", fmt_ratio("margin", prof.get("gross_margin")))
    k3.metric("Operating Margin", fmt_ratio("margin", prof.get("operating_margin")))
    k4.metric("Free Cash Flow", fmt_ratio("fcf", cf.get("free_cash_flow")))


def render_charts(bundle: Bundle) -> None:
    hist_col1, hist_col2 = st.columns(2)
    try:
        rev_hist = ratios_mod.extract_history(bundle.facts_json, "revenue", n_quarters=8)
        ni_hist = ratios_mod.extract_history(bundle.facts_json, "net_income", n_quarters=8)
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


def render_ratio_breakdown(bundle: Bundle) -> None:
    with st.expander(f"{bundle.ticker} — full computed ratio breakdown", expanded=False):
        for section, values in bundle.computed.items():
            if section == "data_gaps":
                continue
            st.markdown(f"**{section.replace('_', ' ').title()}**")
            st.table({k.replace("_", " "): fmt_ratio(k, v) for k, v in values.items()})
        if bundle.computed.get("data_gaps"):
            st.markdown(
                f'<div class="gap-note">Not tagged in this filing\'s XBRL data (reported as unavailable, '
                f'never estimated): {", ".join(bundle.computed["data_gaps"])}</div>',
                unsafe_allow_html=True,
            )


def render_stance(bundle: Bundle) -> None:
    st.subheader("Equity Research Stance")
    st.markdown(stance_badge(bundle.run.recommendation), unsafe_allow_html=True)
    st.markdown(escape_markdown_math(bundle.run.recommendation))


def render_steps(bundle: Bundle) -> None:
    st.subheader("Step-by-Step Analysis")
    tabs = st.tabs([s.title for s in bundle.run.steps])
    for tab, step in zip(tabs, bundle.run.steps):
        with tab:
            st.markdown(escape_markdown_math(step.output))


def render_single(bundle: Bundle) -> None:
    render_company_card(bundle)
    render_kpis(bundle)
    render_charts(bundle)
    render_ratio_breakdown(bundle)
    st.divider()
    render_stance(bundle)
    render_steps(bundle)


# ── Compare-mode-only render blocks: these actually compare, not just
#    lay two independent analyses side by side. ─────────────────────────────
_KPI_COMPARE_DEFS = [
    ("Revenue YoY", "yoy", lambda c: c["yoy_growth"].get("revenue_yoy")),
    ("Gross Margin", "margin", lambda c: c["profitability"].get("gross_margin")),
    ("Operating Margin", "margin", lambda c: c["profitability"].get("operating_margin")),
    ("Free Cash Flow", "fcf", lambda c: c["cash_flow"].get("free_cash_flow")),
]


def render_kpi_comparison(b1: Bundle, b2: Bundle) -> None:
    # st.container(border=True), not a hand-rolled <div>, because a raw
    # unsafe_allow_html div opened in one st.markdown call and closed in a
    # later one does NOT wrap the native st.columns rows in between --
    # Streamlit mounts each st.markdown call as its own sibling DOM node,
    # so the browser just auto-closes the dangling tag. Found live.
    with st.container(border=True):
        header = st.columns([2, 1, 1])
        header[1].markdown(f"**{b1.ticker}**")
        header[2].markdown(f"**{b2.ticker}**")
        for label, fmt_key, getter in _KPI_COMPARE_DEFS:
            v1, v2 = getter(b1.computed), getter(b2.computed)
            # Only highlight a winner when BOTH sides have a real value to
            # compare -- found live: with the old (v2 is None or v1 > v2)
            # form, a lone value on one side (the other reported "n/a")
            # always highlighted green, even a negative YoY figure, which
            # reads as "this side won" when really there was nothing to
            # compare it against.
            both_present = v1 is not None and v2 is not None
            better1 = both_present and v1 > v2
            better2 = both_present and v2 > v1
            s1 = fmt_ratio(fmt_key, v1) if v1 is not None else "n/a"
            s2 = fmt_ratio(fmt_key, v2) if v2 is not None else "n/a"
            row = st.columns([2, 1, 1])
            row[0].markdown(label)
            row[1].markdown(f'<span class="{"kpi-win" if better1 else ""}">{s1}</span>', unsafe_allow_html=True)
            row[2].markdown(f'<span class="{"kpi-win" if better2 else ""}">{s2}</span>', unsafe_allow_html=True)


def render_chart_comparison(b1: Bundle, b2: Bundle) -> None:
    try:
        rev1 = dict(ratios_mod.extract_history(b1.facts_json, "revenue", n_quarters=8))
        rev2 = dict(ratios_mod.extract_history(b2.facts_json, "revenue", n_quarters=8))
        ni1 = dict(ratios_mod.extract_history(b1.facts_json, "net_income", n_quarters=8))
        ni2 = dict(ratios_mod.extract_history(b2.facts_json, "net_income", n_quarters=8))
    except Exception:
        return  # trend charts are a bonus visualization; never block the core comparison on them

    c1, c2 = st.columns(2)
    if rev1 or rev2:
        with c1:
            st.caption("Revenue, last 8 quarters ($) — overlaid")
            st.line_chart(pd.DataFrame({b1.ticker: rev1, b2.ticker: rev2}))
    if ni1 or ni2:
        with c2:
            st.caption("Net income, last 8 quarters ($) — overlaid")
            st.line_chart(pd.DataFrame({b1.ticker: ni1, b2.ticker: ni2}))


def render_verdict(b1: Bundle, b2: Bundle) -> None:
    st.subheader("Head-to-Head: Which Would You Rather Own?")
    try:
        with st.spinner("Weighing both theses against each other..."):
            verdict = compare_stances(
                b1.filing.company_name, b1.ticker, b1.run,
                b2.filing.company_name, b2.ticker, b2.run,
            )
    except RuntimeError as exc:
        st.error(str(exc))
        return
    except (ClientError, ServerError) as exc:
        st.error(f"Couldn't generate a head-to-head verdict: {exc}")
        return
    # Blank lines right after the opening tag matter: found live, a
    # markdown heading (###) glued directly to <div> with no blank line
    # doesn't get parsed as a block-level heading and renders as literal
    # text, even though a table a few lines later parses fine.
    st.markdown(f'<div class="verdict-card">\n\n{escape_markdown_math(verdict)}\n\n</div>', unsafe_allow_html=True)


def render_compare(b1: "Bundle | None", b2: "Bundle | None") -> None:
    if b1 is None or b2 is None:
        return  # a fetch/pipeline error for one side was already shown by fetch_and_analyze
    left, right = st.columns(2, gap="large")
    with left:
        render_company_card(b1)
    with right:
        render_company_card(b2)

    st.subheader("Key Metrics — Head to Head")
    render_kpi_comparison(b1, b2)
    render_chart_comparison(b1, b2)

    left, right = st.columns(2, gap="large")
    with left:
        render_ratio_breakdown(b1)
    with right:
        render_ratio_breakdown(b2)

    st.divider()
    render_verdict(b1, b2)

    st.divider()
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        render_stance(b1)
        st.divider()
        render_steps(b1)
    with right, st.container(border=True):
        render_stance(b2)
        st.divider()
        render_steps(b2)


if run_clicked:
    if compare_mode:
        b1 = fetch_and_analyze(ticker1)
        b2 = fetch_and_analyze(ticker2)
        render_compare(b1, b2)
    else:
        bundle = fetch_and_analyze(ticker1)
        if bundle is not None:
            render_single(bundle)
