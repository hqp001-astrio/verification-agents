"""Offline tests for the generalist's safe AST->Z3 compiler and generic executor.

The compiler is security-critical: it must turn LLM predicates into Z3 *without*
executing arbitrary code, rejecting anything outside the whitelist.
"""

from __future__ import annotations

import z3

from verification_agents.models import CodeUnit, Language
from verification_agents.specialists.executor import reproduce_generic
from verification_agents.specialists.generalist import (
    TranslationError,
    compile_predicate,
)


def _check(pred: str):
    ctx = z3.Context()
    env: dict = {}
    return compile_predicate(pred, env, ctx), env, ctx


def test_compiles_arithmetic_and_logic():
    expr, env, ctx = _check("count != 0 and total >= 0")
    s = z3.Solver(ctx=ctx)
    s.add(expr)
    assert s.check() == z3.sat  # satisfiable predicate

def test_division_by_zero_property_is_sat():
    # safety `count != 0`, negated -> count == 0 is reachable
    expr, env, ctx = _check("count != 0")
    s = z3.Solver(ctx=ctx)
    s.add(z3.Not(expr))
    assert s.check() == z3.sat
    assert str(s.model()[env["count"]]) == "0"


def test_min_max_abs_supported():
    expr, _, ctx = _check("abs(x) >= 0 and min(a, b) <= max(a, b)")
    s = z3.Solver(ctx=ctx)
    s.add(z3.Not(expr))
    assert s.check() == z3.unsat  # always true


def test_rejects_arbitrary_calls():
    for malicious in ("__import__('os').system('x')", "open('f')", "eval('1')", "x.attr"):
        try:
            _check(malicious)
            raised = False
        except TranslationError:
            raised = True
        assert raised, f"compiler must reject: {malicious}"


def test_generic_executor_confirms_div_by_zero():
    unit = CodeUnit(
        name="average", filename="m.py", language=Language.PYTHON,
        start_line=1, end_line=2, source="def average(total, count):\n    return total / count",
    )
    confirmed, detail = reproduce_generic(unit, {"count": "0", "total": "0"})
    assert confirmed is True, detail


def test_generic_executor_refutes_spurious():
    unit = CodeUnit(
        name="average", filename="m.py", language=Language.PYTHON,
        start_line=1, end_line=4,
        source="def average(total, count):\n    if count == 0:\n        return 0\n    return total / count",
    )
    confirmed, detail = reproduce_generic(unit, {"count": "-1", "total": "0"})
    assert confirmed is False, detail  # runs clean -> spurious counterexample
