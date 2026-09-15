"""
Thin client for SEC EDGAR's free, public, no-key-required data APIs.

Three endpoints, all under https://www.sec.gov or https://data.sec.gov:
  - company_tickers.json  -> ticker -> CIK lookup (one static file, ~800KB)
  - submissions/CIK##########.json -> a company's full filing history
  - api/xbrl/companyfacts/CIK##########.json -> every standardized (us-gaap)
    XBRL fact the company has ever reported, across all filings

SEC's fair-use policy requires a descriptive User-Agent identifying the
requester (name/contact) on every request, or requests get rate-limited or
blocked -- not optional, it's the access requirement, not an oversight.
See https://www.sec.gov/os/webmaster-faq#developers
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from core import _net  # noqa: F401 -- must run before any requests call, see module docstring
import requests

USER_AGENT = "10q-equity-agent (matthewginzburg@gmail.com)"
_HEADERS = {"User-Agent": USER_AGENT}

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# SEC asks for <=10 req/s; we stay well under that since this is a
# single-user interactive tool, not a bulk scraper.
_MIN_REQUEST_GAP_SECONDS = 0.3
_last_request_ts = 0.0


def _get(url: str) -> dict:
    global _last_request_ts
    wait = _MIN_REQUEST_GAP_SECONDS - (time.monotonic() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    _last_request_ts = time.monotonic()
    resp.raise_for_status()
    return resp.json()


_ticker_cik_cache: dict[str, int] | None = None


def ticker_to_cik(ticker: str) -> int | None:
    """Resolve a ticker (e.g. 'AAPL') to its SEC CIK number. Caches the
    ~800KB ticker map in-process for the life of the app -- it changes
    rarely, no need to refetch per lookup.

    Normalizes real paste-shaped input, not just a clean uppercase string:
    stripped whitespace, a leading '$' (cashtag style, e.g. copied from
    Twitter/StockTwits), and share-class tickers written with a dot
    (Yahoo/Google Finance style, 'BRK.B') vs SEC's own dash convention
    ('BRK-B') -- found live: SEC's ticker file only has the dash form, so
    the dot form (the more common way people actually write it) silently
    failed before this normalization existed."""
    global _ticker_cik_cache
    if _ticker_cik_cache is None:
        raw = _get(_TICKER_MAP_URL)
        _ticker_cik_cache = {
            row["ticker"].upper(): row["cik_str"] for row in raw.values()
        }
    cleaned = ticker.strip().upper().lstrip("$")
    if cleaned in _ticker_cik_cache:
        return _ticker_cik_cache[cleaned]
    if "." in cleaned:
        return _ticker_cik_cache.get(cleaned.replace(".", "-"))
    if "-" in cleaned:
        return _ticker_cik_cache.get(cleaned.replace("-", "."))
    return None


@dataclass(frozen=True)
class FilingRef:
    accession_number: str
    filing_date: str
    report_date: str
    primary_document: str
    company_name: str
    cik: int

    @property
    def document_url(self) -> str:
        acc_nodash = self.accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{self.cik}/{acc_nodash}/{self.primary_document}"
        )


def latest_10q(ticker: str) -> FilingRef | None:
    """The most recently filed 10-Q for this ticker, or None if the ticker
    doesn't resolve or has no 10-Q on file (e.g. not a US filer)."""
    cik = ticker_to_cik(ticker)
    if cik is None:
        return None
    data = _get(_SUBMISSIONS_URL.format(cik=cik))
    company_name = data.get("name", ticker.upper())
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-Q":
            return FilingRef(
                accession_number=recent["accessionNumber"][i],
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                primary_document=recent["primaryDocument"][i],
                company_name=company_name,
                cik=cik,
            )
    return None


def fetch_filing_text(filing: FilingRef, max_chars: int = 60_000) -> str:
    """Plain-text MD&A/Risk-Factors content of the filing, truncated to keep
    token usage bounded on the free tier.

    Modern 10-Qs are inline-XBRL HTML: the raw markup's ix:header/ix:hidden
    block (thousands of contextRef/unitRef tags carrying the same facts
    core/ratios.py already reads structurally) can run past 35K+ characters
    BEFORE any human-readable prose starts. Found live: slicing the raw HTML
    by a fixed byte offset returned nothing but tag noise for a large
    filer's 10-Q, and the LLM correctly (and uselessly) reported it had no
    real MD&A text to read. Fixed by stripping tags to plain text first,
    then anchoring on the actual "Management's Discussion and Analysis"
    section heading rather than trusting any fixed offset to land on it."""
    from bs4 import BeautifulSoup

    resp = requests.get(filing.document_url, headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    # requests' charset auto-detection mis-guessed this as non-UTF-8 in
    # testing, mangling curly apostrophes ("Management's" -> "Management�s")
    # and breaking the exact-phrase anchor search below. SEC filings are
    # UTF-8; let BeautifulSoup parse the raw bytes and detect it properly
    # instead of trusting requests' guess.
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    plain = soup.get_text(separator=" ", strip=True)
    plain = " ".join(plain.split())  # collapse repeated whitespace from table-cell spacing

    lower = plain.lower()
    # The FIRST hit on either anchor is the table-of-contents entry, not the
    # actual section -- rfind (last occurrence) lands on the real heading,
    # which is what precedes the actual narrative paragraphs.
    anchor = lower.rfind("management’s discussion and analysis")
    if anchor == -1:
        anchor = lower.rfind("management's discussion and analysis")
    if anchor == -1:
        anchor = lower.rfind("item 2.")  # 10-Q Item 2 is always MD&A
    start = anchor if anchor != -1 else 0
    return plain[start : start + max_chars]


def company_facts(cik: int) -> dict:
    """Every standardized XBRL fact the company has ever reported. Keyed by
    us-gaap taxonomy tag (e.g. 'Revenues', 'NetIncomeLoss') -- the same tags
    every US public filer uses, so this works for any ticker, not just one
    hardcoded company."""
    return _get(_COMPANYFACTS_URL.format(cik=cik))
