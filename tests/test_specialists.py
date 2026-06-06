"""Offline tests for the specialist-column pipeline.

These run with NO OPENAI_API_KEY and NO Redis server — the encoder falls back to
the heuristic path and Redis falls back to the in-memory shim — so the three
verdict classes are deterministic in CI.
"""

from __future__ import annotations

from verification_agents.models import PropertyKind, SolverStatus
from verification_agents.specialists.examples import EXAMPLES
from verification_agents.specialists.verify import verify_diff


def _verify(name: str):
    return verify_diff(EXAMPLES[name])  # offline: heuristic encoder + memory backend


def test_bug_is_no_and_execution_confirmed():
    r = _verify("bug")
    assert r.decision == "no"
    assert r.bugs, "off-by-one should produce a violation"
    bug = next(b for b in r.bugs if b.concern == PropertyKind.ARRAY_BOUNDS)
    assert bug.status == SolverStatus.SAT
    assert bug.execution_confirmed is True, "counterexample should reproduce on the real function"
    assert bug.counterexample, "a SAT result must carry a counterexample"


def test_safe_is_yes():
    r = _verify("safe")
    assert r.decision == "yes"
    assert r.clean and not r.bugs
    assert all(o.status == SolverStatus.UNSAT for o in r.clean)


def test_unsure_when_cannot_ground():
    r = _verify("unsure")
    assert r.decision == "unsure"
    assert not r.bugs, "an unconfirmable violation must not be reported as a hard bug"
    assert r.inconclusive


def test_report_shape_and_streaming():
    r = _verify("bug")
    assert r.summary and r.github_comment
    assert r.columns_run, "specialist columns that ran should be recorded"
    assert r.total_constraints >= 1
    # Streaming events were captured (start -> ... -> done) for UI replay.
    stages = [e.get("stage") for e in r.events]
    assert "start" in stages and "done" in stages
    assert any(e.get("stage") == "property" for e in r.events)


def test_verification_report_adapter():
    r = _verify("bug")
    vr = r.to_verification_report()
    assert vr.bugs and vr.total_properties_checked >= 1
    assert "execution-confirmed" in vr.bugs[0].description
