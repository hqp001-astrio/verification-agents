"""End-to-end verification against the buggy.diff fixture.

Uses the specialist pipeline (``verify_diff``) — the same path as ``POST /api/analyze``.
The LangChain orchestrator is intentionally not exercised here: it makes dozens of
sequential LLM calls and can run 5–10+ minutes with no output.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from verification_agents.models import SolverStatus
from verification_agents.specialists.verify import verify
from verification_agents.tools import parse_diff as parse_diff_mod

load_dotenv()

FIXTURES = Path(__file__).parent / "fixtures"
WEAVE_PROJECT = os.environ.get("WEAVE_PROJECT", "astrio/verification-agents")


def _init_weave() -> bool:
    """Best-effort Weave init (uses WANDB_API_KEY or prior ``wandb login``)."""
    try:
        import weave

        weave.init(WEAVE_PROJECT)
        return True
    except Exception as exc:
        print(f"[weave] disabled: {exc}", flush=True)
        return False


def _run_verify(analysis, *, selected_ids: list[str], api_key: str, model: str, weave_on: bool):
    kwargs = dict(
        selected_ids=selected_ids,
        api_key=api_key,
        model=model,
        z3_timeout_ms=10_000,
    )
    if weave_on and hasattr(verify, "call"):
        report, call = verify.call(analysis, **kwargs)
        report.weave_trace_url = getattr(call, "ui_url", "") or ""
        return report
    return verify(analysis, **kwargs)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_full_verification():
    diff = (FIXTURES / "buggy.diff").read_text()

    print("\n[test] parsing diff…", flush=True)
    analysis = parse_diff_mod.run(diff)
    all_ids = [p.id for p in analysis.properties]
    print(f"[test] verifying all {len(all_ids)} properties via specialist pipeline…", flush=True)
    for p in analysis.properties:
        print(f"  [{p.kind.value}] {p.unit_name}", flush=True)

    weave_on = _init_weave()
    if weave_on:
        print(f"[weave] tracing enabled ({WEAVE_PROJECT})", flush=True)
    else:
        print("[weave] tracing off", flush=True)

    report = _run_verify(
        analysis,
        selected_ids=all_ids,
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("VERIFY_MODEL", "gpt-4o-mini"),
        weave_on=weave_on,
    )

    print("\n=== Verification Report ===", flush=True)
    print(f"Decision: {report.decision}", flush=True)

    findings = [
        o for o in (*report.bugs, *report.inconclusive, *report.proposed)
        if o.status == SolverStatus.SAT
    ]
    print(f"\nFindings ({len(findings)} potential bug(s)):", flush=True)
    for o in findings:
        kind = o.concern.value.replace("_", " ")
        tag = "CONFIRMED" if o.execution_confirmed is True else "UNCONFIRMED"
        print(f"  • {o.unit_name} — {kind} [{tag}]", flush=True)
        if o.execution_detail:
            print(f"    {o.execution_detail}", flush=True)
        elif o.counterexample:
            print(f"    Counterexample: {o.counterexample}", flush=True)

    print(f"\nConfirmed bugs (in report): {len(report.bugs)}", flush=True)
    print(f"Clean: {len(report.clean)}  Inconclusive: {len(report.inconclusive)}", flush=True)
    print(f"Elapsed: {report.elapsed_s:.1f}s", flush=True)
    if report.weave_trace_url:
        print(f"Weave trace: {report.weave_trace_url}", flush=True)

    assert report.total_constraints >= 6, (
        "Pipeline should verify most fixture properties "
        "(integer overflow skipped for Python; see verify._applicable)"
    )
    assert report.summary, "Report should include a non-empty summary"

    sat_units = {
        o.unit_name
        for o in (*report.bugs, *report.inconclusive, *report.proposed)
        if o.status == SolverStatus.SAT
    }
    assert "get_username" in sat_units or "process_items" in sat_units, (
        "Expected SAT on get_username (null deref) or process_items (array bounds)"
    )
    assert report.decision in ("no", "unsure"), f"Unexpected decision: {report.decision}"
