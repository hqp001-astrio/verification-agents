from __future__ import annotations

from verification_agents.models import (
    Bug,
    CodeAnalysis,
    PropertyKind,
    Severity,
    SolverResult,
    SolverStatus,
    VerificationReport,
    VerifiableProperty,
)

_SEVERITY_MAP: dict[PropertyKind, Severity] = {
    PropertyKind.NULL_DEREFERENCE: Severity.CRITICAL,
    PropertyKind.ARRAY_BOUNDS: Severity.CRITICAL,
    PropertyKind.INTEGER_OVERFLOW: Severity.HIGH,
    PropertyKind.LOOP_TERMINATION: Severity.HIGH,
    PropertyKind.POSTCONDITION: Severity.HIGH,
    PropertyKind.PRECONDITION: Severity.MEDIUM,
    PropertyKind.TYPE_CONSTRAINT: Severity.MEDIUM,
    PropertyKind.CUSTOM: Severity.LOW,
}


def run(
    results: list[dict],
    summary: str,
    code_analysis: dict,
    elapsed_s: float,
) -> dict:
    analysis = CodeAnalysis(**code_analysis)
    prop_by_id: dict[str, VerifiableProperty] = {p.id: p for p in analysis.properties}

    bugs: list[Bug] = []
    clean: list[VerifiableProperty] = []

    for raw in results:
        r = SolverResult(**raw)
        prop = prop_by_id.get(r.constraint.property_id)
        kind = prop.kind if prop else PropertyKind.CUSTOM
        unit_name = prop.unit_name if prop else "unknown"
        filename = prop.filename if prop else "unknown"
        start_line = prop.start_line if prop else 0

        if r.status == SolverStatus.SAT:
            bugs.append(Bug(
                severity=_SEVERITY_MAP.get(kind, Severity.LOW),
                unit_name=unit_name,
                filename=filename,
                start_line=start_line,
                description=r.constraint.description,
                counterexample=r.counterexample,
                property_kind=kind,
            ))
        elif r.status == SolverStatus.UNSAT and prop:
            clean.append(prop)

    bugs.sort(key=lambda b: list(Severity).index(b.severity))

    report = VerificationReport(
        bugs=bugs,
        clean=clean,
        summary=summary,
        total_properties_checked=len(results),
        elapsed_s=elapsed_s,
    )
    return report.model_dump()
