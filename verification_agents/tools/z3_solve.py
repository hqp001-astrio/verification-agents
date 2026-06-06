from __future__ import annotations

import time

import z3

from verification_agents.models import SolverResult, SolverStatus, Z3Constraint
from verification_agents.tools import formalize


def run(constraints: list[dict], timeout_ms: int = 30_000) -> list[dict]:
    results: list[SolverResult] = []

    for raw in constraints:
        c = Z3Constraint(**raw)
        expr = formalize.get_z3_object(c.property_id)

        if expr is None:
            results.append(SolverResult(
                constraint=c,
                status=SolverStatus.UNKNOWN,
                elapsed_s=0.0,
                error="Z3 expression not found in store (formalize may have failed)",
            ))
            continue

        solver = z3.Solver()
        solver.set("timeout", timeout_ms)

        start = time.monotonic()
        try:
            # Negate the property: SAT means the property CAN be violated → bug
            solver.add(z3.Not(expr))
            check = solver.check()
        except Exception as exc:
            results.append(SolverResult(
                constraint=c,
                status=SolverStatus.UNKNOWN,
                elapsed_s=time.monotonic() - start,
                error=str(exc),
            ))
            continue

        elapsed = time.monotonic() - start

        if check == z3.sat:
            model = solver.model()
            counterexample = {
                str(decl): str(model[decl])
                for decl in model.decls()
            }
            results.append(SolverResult(
                constraint=c,
                status=SolverStatus.SAT,
                counterexample=counterexample,
                elapsed_s=elapsed,
            ))
        elif check == z3.unsat:
            results.append(SolverResult(
                constraint=c,
                status=SolverStatus.UNSAT,
                elapsed_s=elapsed,
            ))
        else:
            results.append(SolverResult(
                constraint=c,
                status=SolverStatus.UNKNOWN,
                elapsed_s=elapsed,
                error="Z3 returned unknown (timeout or undecidable)",
            ))

    return [r.model_dump() for r in results]
