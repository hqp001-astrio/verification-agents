from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    UNKNOWN = "unknown"


class CodeUnit(BaseModel):
    name: str
    filename: str
    language: Language
    start_line: int
    end_line: int
    source: str


class CallEdge(BaseModel):
    caller: str
    callee: str
    filename: str


class PropertyKind(str, Enum):
    NULL_DEREFERENCE = "null_dereference"
    ARRAY_BOUNDS = "array_bounds"
    INTEGER_OVERFLOW = "integer_overflow"
    LOOP_TERMINATION = "loop_termination"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    TYPE_CONSTRAINT = "type_constraint"
    CUSTOM = "custom"


class VerifiableProperty(BaseModel):
    id: str
    kind: PropertyKind
    unit_name: str
    filename: str
    start_line: int
    description: str


class CodeAnalysis(BaseModel):
    units: list[CodeUnit]
    call_edges: list[CallEdge]
    properties: list[VerifiableProperty]


class UserSelection(BaseModel):
    selected_ids: list[str]
    extra_notes: str = ""


class Z3Constraint(BaseModel):
    property_id: str
    description: str
    z3_code: str


class SolverStatus(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


class SolverResult(BaseModel):
    constraint: Z3Constraint
    status: SolverStatus
    counterexample: dict[str, Any] | None = None
    elapsed_s: float
    error: str | None = None


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Bug(BaseModel):
    severity: Severity
    unit_name: str
    filename: str
    start_line: int
    description: str
    counterexample: dict[str, Any] | None = None
    property_kind: PropertyKind


class VerificationReport(BaseModel):
    bugs: list[Bug] = Field(default_factory=list)
    clean: list[VerifiableProperty] = Field(default_factory=list)
    summary: str = ""
    total_properties_checked: int = 0
    elapsed_s: float = 0.0
