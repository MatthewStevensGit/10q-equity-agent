"""
Computes standard financial ratios from a company's SEC XBRL facts
(core/edgar_client.company_facts). Works for any US filer, because every
10-Q/10-K uses the same standardized us-gaap taxonomy tags -- unlike the
original prototype, which read fixed row labels out of one hardcoded
Amazon spreadsheet.

Each metric lists fallback tag names, tried in order, because different
companies (and different filing years) tag the same concept differently
-- e.g. a services company reports "Revenues", a lot of newer filers
report "RevenueFromContractWithCustomerExcludingAssessedTax" instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# metric name -> ordered list of us-gaap tags to try
_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "total_assets": ["Assets"],
    "stockholders_equity": ["StockholdersEquity"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}


@dataclass
class MetricPoint:
    value: float
    end: str
    start: str | None  # None for instant (balance-sheet) facts


@dataclass
class ExtractedFacts:
    latest: dict[str, MetricPoint] = field(default_factory=dict)
    year_ago: dict[str, MetricPoint] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def _span_days(f: dict) -> int | None:
    start = f.get("start")
    if start is None:
        return None  # instant (balance-sheet) fact, not a duration
    from datetime import date
    return (date.fromisoformat(f["end"]) - date.fromisoformat(start)).days


def _all_points(us_gaap: dict, tags: list[str]) -> list[dict]:
    """Every raw fact across every fallback tag for this concept, 10-Q/10-K
    only. Deliberately does NOT stop at the first tag with any data -- a
    company can retire an old tag (e.g. Apple stopped using 'Revenues' for
    'RevenueFromContractWithCustomerExcludingAssessedTax' years ago), and
    that old tag's stale historical points would otherwise silently win
    over the current tag's fresh ones just by being checked first."""
    out = []
    for tag in tags:
        node = us_gaap.get(tag)
        if not node:
            continue
        for unit_facts in node.get("units", {}).values():
            out.extend(f for f in unit_facts if f.get("form") in ("10-Q", "10-K"))
    return out


def _isolate_quarter(raw: list[dict], target_end: str) -> dict | None:
    """Return the single-quarter (~70-100 day) fact ending at target_end if
    one is directly tagged. If only a YTD/cumulative fact exists there (true
    for MOST companies' cash-flow-statement line items, which 10-Qs present
    cumulative-since-fiscal-year-start, not isolated per quarter) derive the
    standalone quarter as YTD_this_period minus YTD_through_the_prior_quarter
    -- the standard way to de-cumulate XBRL duration facts, not a guess."""
    at_end = [f for f in raw if f["end"] == target_end and f.get("start")]
    if not at_end:
        return None
    direct = [f for f in at_end if 70 <= _span_days(f) <= 100]
    if direct:
        return max(direct, key=lambda f: f.get("filed", ""))

    cumulative = min(at_end, key=lambda f: _span_days(f))  # shortest available YTD span at this end-date
    prior = [f for f in raw if f.get("start") == cumulative["start"] and f["end"] != target_end]
    if not prior:
        return None
    prior_ytd = max(prior, key=lambda f: f["end"])  # the YTD fact ending at the start of this quarter
    if not (70 <= (_span_days(cumulative) - _span_days(prior_ytd)) <= 100):
        return None  # the subtraction doesn't land on a plausible single quarter -- don't fabricate one
    return {
        "val": cumulative["val"] - prior_ytd["val"],
        "end": target_end,
        "start": prior_ytd["end"],
    }


def _extract_metric(us_gaap: dict, tags: list[str]) -> tuple[MetricPoint | None, MetricPoint | None]:
    """Latest single-quarter (or instant, for balance-sheet items) point,
    and the one closest to 12 months earlier for YoY comparisons."""
    from datetime import date, timedelta

    raw = _all_points(us_gaap, tags)
    if not raw:
        return None, None

    instant = all(f.get("start") is None for f in raw)
    if instant:
        latest_f = max(raw, key=lambda f: f["end"])
        latest = MetricPoint(value=latest_f["val"], end=latest_f["end"], start=None)
        target = date.fromisoformat(latest.end) - timedelta(days=365)
        candidates = [f for f in raw if f["end"] != latest.end]
        year_ago_f = min(candidates, key=lambda f: abs((date.fromisoformat(f["end"]) - target).days), default=None)
        year_ago = None
        if year_ago_f is not None and abs((date.fromisoformat(year_ago_f["end"]) - target).days) <= 45:
            year_ago = MetricPoint(value=year_ago_f["val"], end=year_ago_f["end"], start=None)
        return latest, year_ago

    # Duration fact: find the most recent quarter-end this concept was
    # reported for at all, then isolate that specific quarter (direct or
    # via YTD subtraction) rather than trusting whatever duration happens
    # to sort last.
    all_ends = sorted({f["end"] for f in raw}, reverse=True)
    latest_iso = next((_isolate_quarter(raw, e) for e in all_ends if _isolate_quarter(raw, e)), None)
    if latest_iso is None:
        return None, None
    latest = MetricPoint(value=latest_iso["val"], end=latest_iso["end"], start=latest_iso["start"])

    target = date.fromisoformat(latest.end) - timedelta(days=365)
    year_ago = None
    for e in all_ends:
        if abs((date.fromisoformat(e) - target).days) <= 45:
            iso = _isolate_quarter(raw, e)
            if iso:
                year_ago = MetricPoint(value=iso["val"], end=iso["end"], start=iso["start"])
                break
    return latest, year_ago


def extract_facts(company_facts_json: dict) -> ExtractedFacts:
    us_gaap = company_facts_json.get("facts", {}).get("us-gaap", {})
    result = ExtractedFacts()
    for name, tags in _TAGS.items():
        latest, year_ago = _extract_metric(us_gaap, tags)
        if latest is None:
            result.missing.append(name)
            continue
        result.latest[name] = latest
        if year_ago is not None:
            result.year_ago[name] = year_ago
    return result


def extract_history(company_facts_json: dict, metric: str, n_quarters: int = 8) -> list[tuple[str, float]]:
    """Up to n_quarters of real, isolated single-quarter values for one
    metric, oldest first -- for an actual trend chart, not just a two-point
    latest-vs-year-ago comparison. Reuses the same tag-merge and YTD-
    subtraction logic as the point-in-time extraction, applied across every
    distinct period this filer has reported, not just the two closest to
    today."""
    us_gaap = company_facts_json.get("facts", {}).get("us-gaap", {})
    tags = _TAGS.get(metric)
    if not tags:
        return []
    raw = _all_points(us_gaap, tags)
    if not raw:
        return []

    if all(f.get("start") is None for f in raw):  # instant/balance-sheet metric
        by_end: dict[str, dict] = {}
        for f in raw:
            by_end[f["end"]] = f  # last one wins if duplicated across filings
        ends = sorted(by_end)[-n_quarters:]
        return [(e, by_end[e]["val"]) for e in ends]

    ends = sorted({f["end"] for f in raw}, reverse=True)
    points: list[tuple[str, float]] = []
    for e in ends:
        if len(points) >= n_quarters:
            break
        iso = _isolate_quarter(raw, e)
        if iso:
            points.append((e, iso["val"]))
    return list(reversed(points))  # oldest first, for a left-to-right chart


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _growth(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / old


def _aligned(facts: ExtractedFacts, *names: str, max_gap_days: int = 10) -> bool:
    """True only if every named metric's latest data point is from the same
    reporting period (within a small tolerance). Prevents silently dividing
    a current-quarter figure by a stale one just because both happened to
    exist somewhere in the company's XBRL history -- e.g. Apple's
    'InterestExpense' tag stopped getting used years before this quarter's
    filing; without this guard, interest_coverage would quietly pair a
    fresh operating_income against a multi-year-old interest figure."""
    from datetime import date
    points = [facts.latest.get(n) for n in names]
    if any(p is None for p in points):
        return False
    ends = [date.fromisoformat(p.end) for p in points]
    return (max(ends) - min(ends)).days <= max_gap_days


def compute_ratios(facts: ExtractedFacts) -> dict:
    L = {k: v.value for k, v in facts.latest.items()}
    Y = {k: v.value for k, v in facts.year_ago.items()}

    revenue = L.get("revenue")
    return {
        "profitability": {
            "gross_margin": _div(revenue - L["cost_of_revenue"], revenue)
                if _aligned(facts, "revenue", "cost_of_revenue") else None,
            "operating_margin": _div(L.get("operating_income"), revenue)
                if _aligned(facts, "revenue", "operating_income") else None,
            "net_profit_margin": _div(L.get("net_income"), revenue)
                if _aligned(facts, "revenue", "net_income") else None,
            "eps_basic": L.get("eps_basic"),
            "eps_diluted": L.get("eps_diluted"),
            "interest_coverage": _div(L.get("operating_income"), L.get("interest_expense"))
                if _aligned(facts, "operating_income", "interest_expense") else None,
        },
        "liquidity": {
            "current_ratio": _div(L.get("current_assets"), L.get("current_liabilities"))
                if _aligned(facts, "current_assets", "current_liabilities") else None,
        },
        "leverage": {
            "debt_to_equity": _div(L.get("long_term_debt"), L.get("stockholders_equity"))
                if _aligned(facts, "long_term_debt", "stockholders_equity") else None,
            "debt_ratio": _div(L["total_assets"] - L["stockholders_equity"], L.get("total_assets"))
                if _aligned(facts, "total_assets", "stockholders_equity") else None,
            "equity_ratio": _div(L.get("stockholders_equity"), L.get("total_assets"))
                if _aligned(facts, "stockholders_equity", "total_assets") else None,
        },
        "cash_flow": {
            "operating_cash_flow": L.get("operating_cash_flow"),
            "capex": L.get("capex"),
            "free_cash_flow": (L["operating_cash_flow"] - L["capex"])
                if _aligned(facts, "operating_cash_flow", "capex") else None,
            "ocf_to_capex": _div(L.get("operating_cash_flow"), L.get("capex"))
                if _aligned(facts, "operating_cash_flow", "capex") else None,
        },
        "yoy_growth": {
            "revenue_yoy": _growth(Y.get("revenue"), L.get("revenue")),
            "operating_income_yoy": _growth(Y.get("operating_income"), L.get("operating_income")),
            "net_income_yoy": _growth(Y.get("net_income"), L.get("net_income")),
        },
        "data_gaps": facts.missing,  # honest about what this filer didn't tag / we couldn't find
    }
