"""
The multi-step analysis pipeline: real SEC filing data -> five sequential
LLM steps, each one grounded in the previous steps' actual output (not five
independent prompts against the same raw context) -> a final equity
recommendation.

Adapted from the original amazon-10q-agent's 5-prompt structure, but
generalized off standardized XBRL ratios instead of one hardcoded Excel
file, and extended with an explicit final recommendation step the original
never had.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from services.gemini_client import generate

_FMT_SYSTEM = (
    "You are an equity research analyst. Be concrete and numbers-grounded. "
    "Never invent a figure that isn't in the provided context -- if something "
    "needed isn't available, say so explicitly rather than guessing. "
    "This is an educational research demo, not investment advice; do not use "
    "the words 'buy' or 'sell' as a directive -- use a qualitative stance "
    "(Bullish / Neutral / Bearish) with reasoning instead."
)


@dataclass
class StepResult:
    title: str
    output: str


@dataclass
class AnalysisRun:
    company: str
    ticker: str
    report_date: str
    steps: list[StepResult] = field(default_factory=list)
    recommendation: str = ""


def _fmt_ratios(ratios: dict) -> str:
    lines = []
    for section, values in ratios.items():
        if section == "data_gaps":
            continue
        lines.append(f"\n{section.upper()}:")
        for k, v in values.items():
            if v is None:
                lines.append(f"  {k}: not available in this filing's XBRL data")
            elif isinstance(v, float) and abs(v) < 5:
                lines.append(f"  {k}: {v:.2%}" if "margin" in k or "yoy" in k or "ratio" in k and abs(v) < 3 else f"  {k}: {v:.3f}")
            else:
                lines.append(f"  {k}: {v:,.1f}")
    if ratios.get("data_gaps"):
        lines.append(f"\nNOTE -- these metrics were not found tagged in this filing's XBRL data: {', '.join(ratios['data_gaps'])}")
    return "\n".join(lines)


def run_analysis(company: str, ticker: str, report_date: str, ratios: dict, filing_text: str) -> AnalysisRun:
    run = AnalysisRun(company=company, ticker=ticker, report_date=report_date)
    ratio_block = _fmt_ratios(ratios)
    # edgar_client.fetch_filing_text already returns plain text anchored on
    # the MD&A section heading (not raw HTML from a fixed byte offset), so
    # no further slicing is needed here -- just bound it for the prompt.
    filing_excerpt = filing_text[:30_000]

    # Step 1 -- quantitative snapshot from the real, structured ratios.
    s1 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Company: {company} ({ticker}), 10-Q for period ending {report_date}.\n\n"
            f"Computed financial ratios (from this filing's actual SEC XBRL data):\n{ratio_block}\n\n"
            "Write a quantitative snapshot: profitability trend, liquidity/leverage posture, "
            "and the most notable YoY moves. Cite the actual numbers. If a metric is marked "
            "unavailable, say plainly that this filer didn't tag it rather than estimating it."
        ),
        max_output_tokens=800,
    )
    run.steps.append(StepResult("1. Quantitative Snapshot", s1))

    # Step 2 -- MD&A / risk-factor synthesis from the real filing text.
    s2 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Below is an excerpt of {company}'s actual 10-Q filing text (HTML, may include markup noise "
            f"-- read through it for the MD&A and Risk Factors content):\n\n{filing_excerpt}\n\n"
            "Identify and rank the top 3 risks disclosed, and summarize management's stated priorities "
            "or mitigation strategy for each, in a markdown table (Risk | Evidence from filing | Priority)."
        ),
        max_output_tokens=800,
    )
    run.steps.append(StepResult("2. Risk & MD&A Synthesis", s2))

    # Step 3 -- consistency check: does the qualitative story match the numbers?
    s3 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Quantitative snapshot (step 1):\n{s1}\n\n"
            f"Risk/MD&A synthesis (step 2):\n{s2}\n\n"
            "Cross-check these two: does management's narrative in the filing match what the actual "
            "numbers show? Flag any place where the tone of the disclosure seems more optimistic or "
            "more cautious than the ratios support. If nothing material is available to check, say so."
        ),
        max_output_tokens=600,
    )
    run.steps.append(StepResult("3. Narrative-vs-Numbers Consistency Check", s3))

    # Step 4 -- capital allocation / sustainability read.
    s4 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Quantitative snapshot:\n{s1}\n\n"
            "Focusing only on cash flow, capex, and leverage figures in that snapshot: assess whether "
            "the company's capital allocation looks sustainable at current free cash flow generation, "
            "and whether leverage is a real constraint. If cash flow or capex data wasn't available, "
            "say so explicitly instead of assessing this blind."
        ),
        max_output_tokens=500,
    )
    run.steps.append(StepResult("4. Capital Allocation & Sustainability", s4))

    # Step 5 -- final recommendation, synthesizing steps 1-4, not raw data.
    s5 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"You have four prior analysis steps for {company} ({ticker}):\n\n"
            f"1) Quantitative snapshot:\n{s1}\n\n"
            f"2) Risk/MD&A synthesis:\n{s2}\n\n"
            f"3) Consistency check:\n{s3}\n\n"
            f"4) Capital allocation read:\n{s4}\n\n"
            "Synthesize a final equity research stance: Bullish / Neutral / Bearish, with a one-paragraph "
            "thesis, the single biggest supporting factor, and the single biggest risk to that thesis. "
            "Ground every claim in the four steps above -- do not introduce new figures. "
            "Remind the reader this is an educational demo, not investment advice."
        ),
        max_output_tokens=500,
    )
    run.recommendation = s5
    return run
