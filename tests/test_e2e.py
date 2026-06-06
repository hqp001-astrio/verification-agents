from __future__ import annotations

import os
from pathlib import Path

import pytest

from verification_agents.models import Severity, UserSelection, VerifiableProperty
from verification_agents.orchestrator import OrchestratorAgent

FIXTURES = Path(__file__).parent / "fixtures"


def auto_select_all(question: str, options: list[VerifiableProperty]) -> UserSelection:
    """Clarification handler that auto-selects all proposed properties."""
    print(f"\n[ask_user] {question}")
    for opt in options:
        print(f"  [{opt.kind}] {opt.unit_name}: {opt.description}")
    return UserSelection(selected_ids=[o.id for o in options], extra_notes="")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
def test_full_verification():
    diff = (FIXTURES / "buggy.diff").read_text()

    agent = OrchestratorAgent(
        api_key=os.environ["OPENAI_API_KEY"],
        clarification_handler=auto_select_all,
    )

    report = agent.run(diff, user_intent="Check for null dereferences and array bounds bugs")

    print("\n=== Verification Report ===")
    print(f"Bugs found: {len(report.bugs)}")
    for bug in report.bugs:
        print(f"  [{bug.severity}] {bug.unit_name}:{bug.start_line} — {bug.description}")
        if bug.counterexample:
            print(f"    Counterexample: {bug.counterexample}")
    print(f"\nClean properties: {len(report.clean)}")
    print(f"Summary:\n{report.summary}")
    print(f"Elapsed: {report.elapsed_s:.1f}s")

    assert report.total_properties_checked > 0, "Agent should have checked at least one property"
    assert report.summary, "Report should include a non-empty summary"
    assert len(report.bugs) > 0, "The fixture contains intentional bugs; at least one should be found"
    assert any(
        b.severity in (Severity.CRITICAL, Severity.HIGH) for b in report.bugs
    ), "At least one critical or high severity bug expected"
