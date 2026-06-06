"""Parallel solve + execution + aggregation — the specialist pipeline.

Flow (each stage a Weave op, each writing through Redis):

    code_analysis
      -> per property, in parallel:  encode (specialist) -> Z3 solve -> execute CE
      -> aggregate: coverage + execution -> decision (yes/no/unsure)
      -> ColumnsReport

The per-property work runs concurrently (the "parallel solvers"), and every
property publishes a streaming event to Redis the moment it resolves. Repeat runs
on unchanged functions are served from the Redis abstraction memory.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import z3

from verification_agents.models import (
    CodeAnalysis,
    CodeUnit,
    PropertyKind,
    SolverStatus,
    VerifiableProperty,
)
from verification_agents.specialists import encoders, executor, generalist
from verification_agents.specialists.redis_store import JobStore
from verification_agents.specialists.types import (
    DECISION_NO,
    DECISION_UNSURE,
    DECISION_YES,
    ColumnOutcome,
    ColumnsReport,
    ExprStore,
    SpecialistColumn,
)

try:
    import weave

    _HAS_WEAVE = True
except Exception:  # pragma: no cover
    weave = None
    _HAS_WEAVE = False


def _op(name: str):
    def deco(fn):
        return weave.op(fn, name=name) if _HAS_WEAVE else fn
    return deco


_MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# Solve one constraint (the stored expr is already `reachable ∧ ¬safety`)
# ---------------------------------------------------------------------------

@_op("specialist.solve")
def solve_constraint(constraint, store: ExprStore, timeout_ms: int) -> ColumnOutcome:
    base = dict(
        constraint_id=constraint.constraint_id,
        concern=constraint.concern,
        property_id=constraint.property_id,
        description=constraint.description,
        unit_name=constraint.unit_name,
        filename=constraint.filename,
        start_line=constraint.start_line,
        reachable=constraint.reachable,
        safety=constraint.safety,
        provenance=constraint.provenance,
    )
    expr = store.get(constraint.constraint_id)
    if expr is None:
        # No expression: either a translation error (generalist) or encoder miss.
        return ColumnOutcome(status=SolverStatus.UNKNOWN,
                             error=constraint.note or "no z3 expr", **base)

    # Reuse the expression's own context (each property has a private one) so this
    # solve is thread-safe alongside the other parallel solves.
    solver = z3.Solver(ctx=expr.ctx)
    solver.set("timeout", timeout_ms)
    solver.add(expr)  # SAT => a reachable state violates safety => bug

    start = time.monotonic()
    try:
        check = solver.check()
    except Exception as exc:
        return ColumnOutcome(status=SolverStatus.UNKNOWN,
                             runtime_ms=(time.monotonic() - start) * 1000, error=str(exc), **base)
    runtime_ms = (time.monotonic() - start) * 1000

    if check == z3.sat:
        model = solver.model()
        ce = {str(d.name()): str(model[d]) for d in model.decls()}
        return ColumnOutcome(status=SolverStatus.SAT, counterexample=ce, runtime_ms=runtime_ms, **base)
    if check == z3.unsat:
        return ColumnOutcome(status=SolverStatus.UNSAT, runtime_ms=runtime_ms, **base)
    return ColumnOutcome(status=SolverStatus.UNKNOWN, runtime_ms=runtime_ms,
                         error="z3 unknown (timeout/undecidable)", **base)


# ---------------------------------------------------------------------------
# Per-property unit of work: memory -> encode -> solve -> execute
# ---------------------------------------------------------------------------

@_op("specialist.process_property")
def _process_property(
    prop: VerifiableProperty,
    unit: CodeUnit | None,
    job: JobStore,
    api_key: str | None,
    model: str,
    timeout_ms: int,
) -> list[ColumnOutcome]:
    concern_label = encoders.CONCERN_LABEL.get(prop.kind, str(prop.kind))
    fn_hash = JobStore.hash_code(unit.source if unit else prop.description,
                                 f"{prop.kind.value}:{prop.start_line}")

    # 1. Abstraction memory: identical (function, concern) seen before -> reuse.
    cached = job.memory_get(fn_hash, prop.kind.value)
    if cached:
        outcome = ColumnOutcome(**cached)
        job.publish({"stage": "property", "concern": concern_label, "unit": prop.unit_name,
                     "status": outcome.status.value, "cached": True})
        return [outcome]

    # 2. Fresh: encode with the specialist, solve, and (for SAT) execute the CE.
    store = ExprStore()
    constraint = encoders.encode_property(prop, unit, store, api_key, model)
    if constraint is None:
        return [ColumnOutcome(concern=prop.kind, property_id=prop.id, description=prop.description,
                              unit_name=prop.unit_name, filename=prop.filename,
                              start_line=prop.start_line, constraint_id="",
                              status=SolverStatus.UNKNOWN, error="encoder produced no constraint")]

    outcome = solve_constraint(constraint, store, timeout_ms)

    if outcome.status == SolverStatus.SAT:
        confirmed, detail = executor.reproduce(prop.kind, unit, constraint.var_binding,
                                               outcome.counterexample)
        outcome.execution_confirmed = confirmed
        outcome.execution_detail = detail

    job.memory_put(fn_hash, prop.kind.value, outcome.model_dump())
    job.publish({"stage": "property", "concern": concern_label, "unit": prop.unit_name,
                 "status": outcome.status.value,
                 "execution_confirmed": outcome.execution_confirmed, "cached": False})
    return [outcome]


@_op("generalist.process_unit")
def _process_generalist(
    unit: CodeUnit,
    job: JobStore,
    api_key: str | None,
    model: str,
    timeout_ms: int,
) -> list[ColumnOutcome]:
    """Run the generalist on a whole function: one model -> many checks -> solves.

    Covers any concern the specialists don't. Findings are tagged ``generalist`` so
    the aggregator treats an unexecuted SAT as proposed/translation-unverified."""
    if not api_key:
        return []
    store = ExprStore()
    constraints = generalist.encode_generalist(unit, store, api_key, model)
    outcomes: list[ColumnOutcome] = []
    for c in constraints:
        outcome = solve_constraint(c, store, timeout_ms)
        # Confirm/refute the generalist's finding by running the function on the
        # counterexample (maps CE vars to params by name). This is what catches a
        # mistranslated property: a spurious CE runs clean -> refuted.
        if outcome.status == SolverStatus.SAT:
            confirmed, detail = executor.reproduce_generic(unit, outcome.counterexample)
            outcome.execution_confirmed = confirmed
            outcome.execution_detail = detail
        job.publish({"stage": "property", "concern": "generalist", "unit": unit.name,
                     "status": outcome.status.value, "cached": False})
        outcomes.append(outcome)
    return outcomes


# ---------------------------------------------------------------------------
# Aggregator: coverage + execution -> decision + narrative
# ---------------------------------------------------------------------------

# Concerns whose counterexamples we can actually run. For these we *require*
# execution to confirm a SAT before calling it a bug; an unconfirmed violation is
# reported as unsure (couldn't ground it — e.g. an external dependency), not a hard
# no. Concerns without an executor (overflow, termination) report SAT as a finding.
_EXECUTABLE_CONCERNS = {PropertyKind.ARRAY_BOUNDS, PropertyKind.NULL_DEREFERENCE}


@_op("specialist.aggregate")
def aggregate(outcomes: list[ColumnOutcome], job_id: str, elapsed_s: float) -> ColumnsReport:
    bugs: list[ColumnOutcome] = []          # confirmed or template-sound -> drives NO
    proposed: list[ColumnOutcome] = []      # generalist, unverified translation -> drives UNSURE
    clean: list[ColumnOutcome] = []
    inconclusive: list[ColumnOutcome] = []

    for o in outcomes:
        if o.status == SolverStatus.SAT:
            if o.execution_confirmed is True:
                bugs.append(o)                       # reproduced on the real function -> sound bug
            elif o.execution_confirmed is False:
                inconclusive.append(o)               # spurious -> a mistranslation we CAUGHT
            elif o.provenance == "generalist":
                proposed.append(o)                   # free-form: flag, don't over-claim
            elif o.concern in _EXECUTABLE_CONCERNS:
                inconclusive.append(o)               # executable but couldn't ground -> unsure
            else:
                bugs.append(o)                       # template-sound specialist (overflow/termination)
        elif o.status == SolverStatus.UNSAT:
            clean.append(o)
        else:
            inconclusive.append(o)                   # UNKNOWN / translation error

    if bugs:
        decision = DECISION_NO
    elif proposed or inconclusive:
        decision = DECISION_UNSURE
    elif clean:
        decision = DECISION_YES
    else:
        decision = DECISION_UNSURE

    # Columns that ran, grouped by concern.
    by_concern: dict[PropertyKind, SpecialistColumn] = {}
    for o in outcomes:
        col = by_concern.get(o.concern)
        if col is None:
            col = SpecialistColumn(concern=o.concern,
                                   label=encoders.CONCERN_LABEL.get(o.concern, str(o.concern)))
            by_concern[o.concern] = col
        col.n_properties += 1
        col.n_constraints += 1

    report = ColumnsReport(
        job_id=job_id,
        decision=decision,
        bugs=bugs,
        proposed=proposed,
        clean=clean,
        inconclusive=inconclusive,
        columns_run=list(by_concern.values()),
        total_constraints=len(outcomes),
        elapsed_s=elapsed_s,
    )
    report.summary = _summarize(report)
    report.github_comment = _github_comment(report)
    return report


def _confirmed_word(o: ColumnOutcome) -> str:
    if o.execution_confirmed is True:
        return "execution-confirmed"
    if o.execution_confirmed is None:
        return "unconfirmed by execution"
    return ""


def _summarize(r: ColumnsReport) -> str:
    cols = ", ".join(f"{c.label}×{c.n_constraints}" for c in r.columns_run) or "none"
    lines = [f"Decision: {r.decision.upper()}.  Specialists run: {cols}."]
    for o in r.bugs:
        ce = ", ".join(f"{k}={v}" for k, v in (o.counterexample or {}).items())
        lines.append(f"  [BUG/{o.concern.value}] {o.unit_name} ({o.filename}:{o.start_line}) — "
                     f"{o.safety} can be violated ({_confirmed_word(o)}); counterexample: {ce or 'n/a'}.")
    for o in r.proposed:
        ce = ", ".join(f"{k}={v}" for k, v in (o.counterexample or {}).items())
        lines.append(f"  [PROPOSED/{o.concern.value}] {o.unit_name} — generalist flags "
                     f"`{o.safety}` may fail (counterexample: {ce or 'n/a'}); translation unverified.")
    for o in r.clean:
        lines.append(f"  [SAFE/{o.concern.value}] {o.unit_name} — proved {o.safety} (UNSAT).")
    for o in r.inconclusive:
        why = o.execution_detail or o.error or "no decisive result"
        lines.append(f"  [UNSURE/{o.concern.value}] {o.unit_name} — {why}.")
    return "\n".join(lines)


def _github_comment(r: ColumnsReport) -> str:
    icon = {"no": "🟥", "yes": "🟩", "unsure": "🟨"}[r.decision]
    head = {"no": "Changes requested — verification found a violation",
            "yes": "Looks safe — properties verified",
            "unsure": "Inconclusive — could not soundly verify"}[r.decision]
    out = [f"{icon} **Formal verification: {r.decision.upper()}** — {head}", ""]
    if r.bugs:
        out.append("**Violations:**")
        for o in r.bugs:
            ce = ", ".join(f"`{k}={v}`" for k, v in (o.counterexample or {}).items())
            out.append(f"- `{o.unit_name}` ({o.filename}:{o.start_line}): `{o.safety}` can fail "
                       f"({_confirmed_word(o)}). Counterexample: {ce or 'n/a'}")
    if r.proposed:
        out.append("")
        out.append("**Proposed by generalist (verify the translation):**")
        for o in r.proposed:
            ce = ", ".join(f"`{k}={v}`" for k, v in (o.counterexample or {}).items())
            out.append(f"- `{o.unit_name}`: `{o.safety}` may fail — counterexample: {ce or 'n/a'}")
    if r.clean:
        out.append("")
        out.append("**Verified clean:** " + ", ".join(
            f"`{o.unit_name}`·{o.concern.value}" for o in r.clean))
    if r.inconclusive:
        out.append("")
        out.append("**Inconclusive:** " + ", ".join(
            f"`{o.unit_name}`·{o.concern.value}" for o in r.inconclusive))
    out.append("")
    out.append(f"_Columns: {', '.join(c.label for c in r.columns_run) or 'none'} · "
               f"{r.total_constraints} checks · {r.elapsed_s:.2f}s_")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

@_op("specialist.pipeline")
def verify(
    code_analysis: CodeAnalysis | dict,
    selected_ids: list[str] | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o",
    job_id: str | None = None,
    z3_timeout_ms: int = 10_000,
    redis_url: str | None = None,
) -> ColumnsReport:
    """Run the full specialist-column pipeline over a parsed code analysis."""
    if isinstance(code_analysis, dict):
        code_analysis = CodeAnalysis(**code_analysis)

    job_id = job_id or uuid.uuid4().hex[:12]
    job = JobStore(job_id, url=redis_url)
    start = time.monotonic()

    job.set_status("queued")
    job.publish({"stage": "start", "backend": job.backend,
                 "properties": len(code_analysis.properties)})

    unit_by_name = {u.name: u for u in code_analysis.units}
    selected = set(selected_ids) if selected_ids else None
    prop_work = [
        (p, unit_by_name.get(p.unit_name))
        for p in code_analysis.properties
        if p.kind in encoders.CONCERN_LABEL and (selected is None or p.id in selected)
    ]
    # The generalist runs once per changed function, covering everything the
    # specialists don't. Skipped when there's no API key (no offline heuristic).
    gen_units = list(code_analysis.units) if (api_key and not selected) else []

    job.set_status("solving")
    job.publish({"stage": "fanout", "specialist_tasks": len(prop_work),
                 "generalist_tasks": len(gen_units)})
    outcomes: list[ColumnOutcome] = []
    total_tasks = len(prop_work) + len(gen_units)
    if total_tasks:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, total_tasks)) as pool:
            futures = [
                pool.submit(_process_property, p, u, job, api_key, model, z3_timeout_ms)
                for p, u in prop_work
            ] + [
                pool.submit(_process_generalist, u, job, api_key, model, z3_timeout_ms)
                for u in gen_units
            ]
            for f in futures:
                outcomes.extend(f.result())

    job.set_status("aggregating")
    report = aggregate(outcomes, job_id, time.monotonic() - start)
    report.backend = job.backend

    job.set_status("done")
    job.publish({"stage": "done", "decision": report.decision,
                 "bugs": len(report.bugs), "clean": len(report.clean)})
    report.events = job.events()           # replayable stream of what happened
    job.set_context(report.model_dump())
    return report


def verify_diff(
    diff: str,
    selected_ids: list[str] | None = None,
    api_key: str | None = None,
    model: str = "gpt-4o",
    job_id: str | None = None,
    z3_timeout_ms: int = 10_000,
    redis_url: str | None = None,
) -> ColumnsReport:
    """Convenience: parse a unified diff, then verify."""
    from verification_agents.tools import parse_diff as _parse_diff

    analysis = _parse_diff.run(diff)
    return verify(analysis, selected_ids=selected_ids, api_key=api_key, model=model,
                  job_id=job_id, z3_timeout_ms=z3_timeout_ms, redis_url=redis_url)
