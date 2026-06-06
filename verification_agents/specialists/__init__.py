"""Specialist-column verification pipeline.

A deterministic fan-out of *specialist* encoder agents — one per safety concern
(array bounds, null dereference, integer overflow, loop termination) — each of
which formalizes only its concern (plus the guards that make it safe) into a Z3
constraint. All constraints are solved in parallel; any counterexample is then
*executed against the real function* to confirm it reproduces; finally an
aggregator agent summarizes coverage and findings.

Design notes
------------
* No "rows" / over-vs-under approximation — just columns (one specialist per
  concern), as decided. Confidence scoring was intentionally dropped; the report
  speaks in terms of *coverage* (which concerns were checked) and *execution*
  (which bugs reproduced).
* Each specialist models the code's **guards**, so safe code verifies as UNSAT
  instead of the trivially-SAT behaviour of the generic template.
* Every stage is a ``weave.op`` so a whole run reads as one trace:
  code -> per-concern abstractions -> parallel Z3 -> execution -> decision.
"""

from verification_agents.specialists.types import (
    ColumnConstraint,
    ColumnOutcome,
    ColumnsReport,
    SpecialistColumn,
)
from verification_agents.specialists.verify import verify, verify_diff

__all__ = [
    "ColumnConstraint",
    "ColumnOutcome",
    "ColumnsReport",
    "SpecialistColumn",
    "verify",
    "verify_diff",
]
