"""Structured contracts for the specialist-column pipeline.

Z3 expression objects are *not* stored on these models (they aren't serializable
and would pollute the Weave trace). The live ``z3.BoolRef`` for each constraint
lives in an :class:`ExprStore` keyed by ``constraint_id``; these models carry only
the human-readable string forms plus the metadata the aggregator needs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from verification_agents.models import (
    Bug,
    PropertyKind,
    Severity,
    SolverStatus,
    VerifiableProperty,
    VerificationReport,
)

# Final verdict vocabulary, matching the product spec.
DECISION_YES = "yes"      # everything checked is provably safe
DECISION_NO = "no"        # at least one violation (ideally execution-confirmed)
DECISION_UNSURE = "unsure"  # nothing decisive — only UNKNOWN / spurious results


class ExprStore:
    """In-process map ``constraint_id -> z3.BoolRef`` (the violation condition).

    Kept off the pydantic models so live solver objects never get serialized.
    One store is created per pipeline run and threaded through encode -> solve.
    """

    def __init__(self) -> None:
        self._exprs: dict[str, Any] = {}

    def put(self, constraint_id: str, expr: Any) -> None:
        self._exprs[constraint_id] = expr

    def get(self, constraint_id: str) -> Any | None:
        return self._exprs.get(constraint_id)


class ColumnConstraint(BaseModel):
    """One specialist's formalization of one property: the violation condition
    ``reachable ∧ ¬safety`` that Z3 will be asked to satisfy."""

    constraint_id: str
    concern: PropertyKind
    property_id: str
    description: str = ""
    unit_name: str = ""
    filename: str = ""
    start_line: int = 0

    reachable: str = ""        # modeled reachable-state predicate (string form)
    safety: str = ""           # property that must always hold (string form)
    var_binding: dict[str, str] = Field(default_factory=dict)  # symbolic var -> how to build input
    source: str = "llm"        # "llm" or "heuristic" — how the spec was filled
    provenance: str = "specialist"  # "specialist" (template-sound) or "generalist" (free-form)
    note: str = ""


class ProofWitness(BaseModel):
    """Evidence that Z3 refuted the violation condition (UNSAT path).

    When Z3 cannot satisfy ``reachable ∧ ¬safety`` it has proven that no
    reachable state can violate the safety property — this object packages
    that certificate for the UI.
    """
    safety: str       # the predicate proven to always hold
    reachable: str    # assumption (reachable-state envelope) under which it holds
    runtime_ms: float # wall-clock proof time


class ColumnOutcome(BaseModel):
    """A constraint after solving (and, for SAT, after execution)."""

    constraint_id: str
    concern: PropertyKind
    property_id: str
    description: str = ""
    unit_name: str = ""
    filename: str = ""
    start_line: int = 0

    status: SolverStatus = SolverStatus.UNKNOWN
    counterexample: dict[str, Any] | None = None
    runtime_ms: float = 0.0
    error: str | None = None
    provenance: str = "specialist"  # "specialist" or "generalist"

    reachable: str = ""
    safety: str = ""
    proof_witness: ProofWitness | None = None  # set on UNSAT: Z3 proof certificate
    # Execution of the counterexample against the real function:
    #   True  -> reproduced (confirmed bug)
    #   False -> ran clean (spurious counterexample; abstraction over-approximated)
    #   None  -> could not execute (no ground truth)
    execution_confirmed: bool | None = None
    execution_detail: str = ""

    # Challenger agent verdict (only set when status == SAT):
    #   True  -> challenger confirms this is a real, faithfully-formalized concern
    #   False -> challenger flagged it as a mistranslation / false positive
    #   None  -> challenger was not run (no api_key, or non-SAT result)
    challenger_valid: bool | None = None
    challenger_issue: str = ""


class SpecialistColumn(BaseModel):
    """Bookkeeping for one specialist that ran in this pipeline."""

    concern: PropertyKind
    label: str
    n_properties: int = 0
    n_constraints: int = 0


class ColumnsReport(BaseModel):
    """The aggregator's output — the whole run, structured for UI + CLI."""

    job_id: str = ""
    decision: str = DECISION_UNSURE

    bugs: list[ColumnOutcome] = Field(default_factory=list)         # confirmed / template-sound
    proposed: list[ColumnOutcome] = Field(default_factory=list)     # generalist, translation-unverified
    clean: list[ColumnOutcome] = Field(default_factory=list)
    inconclusive: list[ColumnOutcome] = Field(default_factory=list)

    columns_run: list[SpecialistColumn] = Field(default_factory=list)
    total_constraints: int = 0

    summary: str = ""
    github_comment: str = ""
    elapsed_s: float = 0.0
    weave_trace_url: str = ""
    cached: bool = False
    backend: str = "memory"                      # "redis" or "memory"
    events: list[dict[str, Any]] = Field(default_factory=list)  # streamed stage events (replay)

    # --- severity mapping reused from the existing report path ---
    _SEVERITY: dict[PropertyKind, Severity] = {}

    def to_verification_report(self) -> VerificationReport:
        """Adapt to the existing ``VerificationReport`` so the current Rich CLI
        (``run.py:_print_report``) renders specialist results unchanged."""
        sev_map = {
            PropertyKind.NULL_DEREFERENCE: Severity.CRITICAL,
            PropertyKind.ARRAY_BOUNDS: Severity.CRITICAL,
            PropertyKind.INTEGER_OVERFLOW: Severity.HIGH,
            PropertyKind.LOOP_TERMINATION: Severity.HIGH,
        }
        bugs = [
            Bug(
                severity=sev_map.get(o.concern, Severity.MEDIUM),
                unit_name=o.unit_name,
                filename=o.filename,
                start_line=o.start_line,
                description=(
                    o.description
                    + (" [execution-confirmed]" if o.execution_confirmed else "")
                ),
                counterexample=o.counterexample,
                property_kind=o.concern,
            )
            for o in self.bugs
        ]
        clean = [
            VerifiableProperty(
                id=o.property_id,
                kind=o.concern,
                unit_name=o.unit_name,
                filename=o.filename,
                start_line=o.start_line,
                description=o.description,
            )
            for o in self.clean
        ]
        return VerificationReport(
            bugs=bugs,
            clean=clean,
            summary=self.summary,
            total_properties_checked=self.total_constraints,
            elapsed_s=self.elapsed_s,
        )
