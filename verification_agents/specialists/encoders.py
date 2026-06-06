"""Specialist encoders — one per safety concern.

Each specialist extracts a small, *guard-aware* structured spec from the code and
a deterministic Z3 template turns it into a violation condition
``reachable ∧ ¬safety``. Modeling the guards is the whole point: a bounds-checked
access verifies as UNSAT (safe), while an off-by-one verifies as SAT (bug) — unlike
a generic template that ignores guards and reports SAT for every access.

Two extraction paths:
* **LLM** (``openai`` structured output) — the real path; fills the spec from code.
* **Heuristic** (string/regex) — offline fallback so the pipeline (and the demo)
  runs with no API key. Approximate, but enough for the toy benchmark.

No arbitrary code is ``exec``'d: the LLM only fills a typed spec; the Z3 object is
always built by our own template.
"""

from __future__ import annotations

import os
import re
import uuid

import z3
from pydantic import BaseModel, Field

from verification_agents.models import CodeUnit, PropertyKind, VerifiableProperty
from verification_agents.specialists.types import ColumnConstraint, ExprStore

try:
    import weave

    _HAS_WEAVE = True
except Exception:  # pragma: no cover
    weave = None
    _HAS_WEAVE = False


def _op(name: str):
    """weave.op if available, else identity — keeps this module import-safe."""
    def deco(fn):
        return weave.op(fn, name=name) if _HAS_WEAVE else fn
    return deco


# ---------------------------------------------------------------------------
# Concern-specific structured specs (what the LLM fills in)
# ---------------------------------------------------------------------------

class BoundsSpec(BaseModel):
    index_var: str = Field(default="index", description="Name for the symbolic index.")
    length_var: str = Field(default="length", description="Name for the symbolic sequence length.")
    guarded_lower: bool = Field(description="Is 0 <= index guaranteed on every path to the access?")
    guarded_upper: bool = Field(description="Is index < len guaranteed on every path to the access?")
    off_by_one: bool = Field(description="Can the index reach exactly len (e.g. range(len+1), <=)?")
    seq_param: str = Field(default="", description="Function parameter that is the sequence, if any.")
    index_param: str = Field(default="", description="Function parameter that flows into the index, if any.")


class NullSpec(BaseModel):
    object_var: str = Field(default="obj", description="Name of the object being dereferenced.")
    guarded: bool = Field(description="Is there a not-None / truthiness check guarding the dereference?")
    object_param: str = Field(default="", description="Function parameter that is the object, if any.")


class OverflowSpec(BaseModel):
    operand_vars: list[str] = Field(default_factory=lambda: ["a", "b"])
    operator: str = Field(default="+", description="Arithmetic operator: + - *")
    bounded_inputs: bool = Field(description="Are the operands clamped to a safe sub-range before the op?")
    width_bits: int = Field(default=32, description="Integer width in bits (32 or 64).")


class TerminationSpec(BaseModel):
    variant_var: str = Field(default="i", description="The loop variant / decreasing quantity.")
    decreases: bool = Field(description="Does the variant strictly decrease each iteration?")
    bounded_below: bool = Field(description="Is the variant bounded below (e.g. >= 0)?")


_SPEC_BY_CONCERN: dict[PropertyKind, type[BaseModel]] = {
    PropertyKind.ARRAY_BOUNDS: BoundsSpec,
    PropertyKind.NULL_DEREFERENCE: NullSpec,
    PropertyKind.INTEGER_OVERFLOW: OverflowSpec,
    PropertyKind.LOOP_TERMINATION: TerminationSpec,
}

CONCERN_LABEL: dict[PropertyKind, str] = {
    PropertyKind.ARRAY_BOUNDS: "bounds",
    PropertyKind.NULL_DEREFERENCE: "null",
    PropertyKind.INTEGER_OVERFLOW: "overflow",
    PropertyKind.LOOP_TERMINATION: "termination",
    PropertyKind.CUSTOM: "generalist",
}

_SYSTEM_PROMPTS: dict[PropertyKind, str] = {
    PropertyKind.ARRAY_BOUNDS: (
        "You are the ARRAY-BOUNDS specialist. Look ONLY at array/sequence index "
        "accesses and the control-flow guards that reach them. Decide whether the "
        "index is guarded below (0 <= index) and above (index < len), and whether an "
        "off-by-one lets the index reach exactly len (e.g. range(len+1), <=). Ignore "
        "every other concern. Fill the spec from the actual variable names in the code."
    ),
    PropertyKind.NULL_DEREFERENCE: (
        "You are the NULL-DEREFERENCE specialist. Look ONLY at object dereferences "
        "(attribute/member/index on a possibly-None object) and whether a not-None or "
        "truthiness check guards them on every path. Ignore every other concern."
    ),
    PropertyKind.INTEGER_OVERFLOW: (
        "You are the INTEGER-OVERFLOW specialist. Look ONLY at the arithmetic "
        "operation. Identify the operands and operator, the integer width, and whether "
        "the operands are clamped to a safe sub-range before the operation. Ignore "
        "every other concern."
    ),
    PropertyKind.LOOP_TERMINATION: (
        "You are the LOOP-TERMINATION specialist. Look ONLY at the loop. Identify the "
        "variant (decreasing quantity), whether it strictly decreases each iteration, "
        "and whether it is bounded below. Ignore every other concern."
    ),
}


# ---------------------------------------------------------------------------
# Deterministic Z3 builders: spec -> (violation_expr, reachable_str, safety_str, var_binding)
# ---------------------------------------------------------------------------

def _int_range(width_bits: int) -> tuple[int, int]:
    half = 1 << (width_bits - 1)
    return -half, half - 1


# Each builder takes its OWN z3.Context so the pipeline can solve in parallel
# threads without tripping z3's non-thread-safe global context. The returned
# expression carries its context (``expr.ctx``), so the solver reuses it.

def _build_bounds(spec: BoundsSpec, ctx: z3.Context):
    index = z3.Int(spec.index_var, ctx)
    length = z3.Int(spec.length_var, ctx)

    safety = z3.And(index >= 0, index < length)

    parts = [length >= 0]
    if spec.guarded_lower:
        parts.append(index >= 0)
    if spec.off_by_one:
        # the off-by-one is at the top end: index reaches exactly len. Pin index >= 0
        # so the counterexample is the illustrative index == length, not a negative.
        parts.append(index >= 0)
        parts.append(index <= length)
        parts.append(length >= 1)  # nicer counterexample than the empty-list degenerate case
    elif spec.guarded_upper:
        parts.append(index < length)
    # else: index is unconstrained above -> can exceed length
    reachable = z3.And(*parts)

    violation = z3.And(reachable, z3.Not(safety))
    var_binding = {
        "seq_param": spec.seq_param,
        "index_param": spec.index_param,
        "index_var": spec.index_var,
        "length_var": spec.length_var,
    }
    return violation, str(reachable), f"0 <= {spec.index_var} < {spec.length_var}", var_binding


def _build_null(spec: NullSpec, ctx: z3.Context):
    is_null = z3.Bool(f"{spec.object_var}_is_null", ctx)
    safety = z3.Not(is_null)
    reachable = z3.Not(is_null) if spec.guarded else z3.BoolVal(True, ctx)
    violation = z3.And(reachable, z3.Not(safety))
    var_binding = {"object_param": spec.object_param, "object_var": spec.object_var}
    return violation, str(reachable), f"{spec.object_var} is not None", var_binding


def _build_overflow(spec: OverflowSpec, ctx: z3.Context):
    lo, hi = _int_range(spec.width_bits)
    names = spec.operand_vars or ["a", "b"]
    a = z3.Int(names[0], ctx)
    b = z3.Int(names[1], ctx) if len(names) > 1 else z3.IntVal(0, ctx)
    result = {"+": a + b, "-": a - b, "*": a * b}.get(spec.operator, a + b)

    safety = z3.And(result >= lo, result <= hi)
    if spec.bounded_inputs:
        # code clamps operands so the op cannot leave range
        reachable = z3.And(a >= lo // 2, a <= hi // 2, b >= lo // 2, b <= hi // 2)
    else:
        reachable = z3.And(a >= lo, a <= hi, b >= lo, b <= hi)
    violation = z3.And(reachable, z3.Not(safety))
    var_binding = {"operands": ",".join(names), "operator": spec.operator}
    return violation, str(reachable), f"{lo} <= ({names[0]} {spec.operator} ...) <= {hi}", var_binding


def _build_termination(spec: TerminationSpec, ctx: z3.Context):
    variant = z3.Int(spec.variant_var, ctx)
    nxt = z3.Int(spec.variant_var + "_next", ctx)

    step = (nxt == variant - 1) if spec.decreases else (nxt == variant)
    floor = (variant >= 0) if spec.bounded_below else z3.BoolVal(True, ctx)
    reachable = z3.And(step, floor, variant > 0)

    safety = z3.And(nxt < variant, nxt >= 0)  # progress + bounded below => terminates
    violation = z3.And(reachable, z3.Not(safety))
    var_binding = {"variant": spec.variant_var}
    return violation, str(reachable), f"{spec.variant_var} strictly decreases and is bounded below", var_binding


_BUILDERS = {
    PropertyKind.ARRAY_BOUNDS: _build_bounds,
    PropertyKind.NULL_DEREFERENCE: _build_null,
    PropertyKind.INTEGER_OVERFLOW: _build_overflow,
    PropertyKind.LOOP_TERMINATION: _build_termination,
}


# ---------------------------------------------------------------------------
# Spec extraction: LLM (real) and heuristic (offline fallback)
# ---------------------------------------------------------------------------

def _extract_llm(concern: PropertyKind, prop: VerifiableProperty, unit: CodeUnit | None,
                 api_key: str, model: str) -> BaseModel | None:
    import openai

    spec_cls = _SPEC_BY_CONCERN[concern]
    client = openai.OpenAI(api_key=api_key)
    code = f"\n\nFunction under review:\n```\n{unit.source}\n```" if unit else ""
    user = (
        f"Property: {prop.description}\n"
        f"Function: {prop.unit_name} ({prop.filename}:{prop.start_line}){code}\n\n"
        "Fill the spec from the ACTUAL code. Only your concern matters."
    )
    # Retry a couple of times: a transient API blip shouldn't silently drop us to
    # the heuristic path (which hides that the LLM was even attempted).
    last_exc: Exception | None = None
    for _attempt in range(3):
        try:
            resp = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPTS[concern]},
                    {"role": "user", "content": user},
                ],
                response_format=spec_cls,
                max_tokens=256,
                temperature=0,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if last_exc is not None and os.environ.get("VERIFY_DEBUG"):
        print(f"[encoder] LLM extraction failed for {concern.value}: {last_exc!r}")
    return None


def _extract_heuristic(concern: PropertyKind, prop: VerifiableProperty,
                       unit: CodeUnit | None) -> BaseModel:
    """Cheap string-based spec inference for offline runs. Approximate by design."""
    src = (unit.source if unit else "") or ""

    if concern == PropertyKind.ARRAY_BOUNDS:
        off_by_one = bool(re.search(r"len\([^)]*\)\s*\+\s*1", src)) or "<= len(" in src
        guarded_upper = bool(re.search(r"<\s*len\(", src)) and not off_by_one
        guarded_lower = bool(re.search(r"0\s*<=|>=\s*0", src))
        m = re.search(r"def\s+\w+\(([^)]*)\)", src)
        params = [p.strip().split(":")[0].split("=")[0].strip() for p in (m.group(1).split(",") if m else [])]
        seq_param = next((p for p in params if p not in ("self", "")), "")
        index_param = next((p for p in params if p in ("index", "i", "idx", "n", "k")), "")
        return BoundsSpec(
            guarded_lower=guarded_lower, guarded_upper=guarded_upper, off_by_one=off_by_one,
            seq_param=seq_param, index_param=index_param,
        )

    if concern == PropertyKind.NULL_DEREFERENCE:
        guarded = ("is not None" in src) or bool(re.search(r"if\s+\w+\s*:", src))
        m = re.search(r"def\s+\w+\(([^)]*)\)", src)
        params = [p.strip().split(":")[0].split("=")[0].strip() for p in (m.group(1).split(",") if m else [])]
        obj = next((p for p in params if p not in ("self", "")), "obj")
        return NullSpec(guarded=guarded, object_var=obj, object_param=obj)

    if concern == PropertyKind.INTEGER_OVERFLOW:
        bounded = ("min(" in src and "max(" in src) or "clamp" in src.lower()
        return OverflowSpec(bounded_inputs=bounded)

    if concern == PropertyKind.LOOP_TERMINATION:
        decreases = ("range(" in src) or "-= 1" in src or "- 1" in src
        bounded = "range(" in src or ">= 0" in src
        return TerminationSpec(decreases=decreases, bounded_below=bounded)

    raise ValueError(f"No heuristic for concern {concern}")


# ---------------------------------------------------------------------------
# Public: encode one property and a whole batch
# ---------------------------------------------------------------------------

@_op("specialist.encode")
def encode_property(
    prop: VerifiableProperty,
    unit: CodeUnit | None,
    store: ExprStore,
    api_key: str | None,
    model: str,
) -> ColumnConstraint | None:
    """Run the matching specialist on one property. Returns the constraint and
    registers the live Z3 violation expression in ``store``."""
    concern = prop.kind
    if concern not in _BUILDERS:
        return None

    spec = None
    source = "heuristic"
    if api_key:
        spec = _extract_llm(concern, prop, unit, api_key, model)
        if spec is not None:
            source = "llm"
    if spec is None:
        spec = _extract_heuristic(concern, prop, unit)

    ctx = z3.Context()  # private context -> safe to build/solve in a worker thread
    violation, reachable_str, safety_str, var_binding = _BUILDERS[concern](spec, ctx)

    constraint_id = uuid.uuid4().hex[:12]
    store.put(constraint_id, violation)

    return ColumnConstraint(
        constraint_id=constraint_id,
        concern=concern,
        property_id=prop.id,
        description=prop.description,
        unit_name=prop.unit_name,
        filename=prop.filename,
        start_line=prop.start_line,
        reachable=reachable_str,
        safety=safety_str,
        var_binding=var_binding,
        source=source,
    )


def encode_columns(
    properties: list[VerifiableProperty],
    units: list[CodeUnit],
    store: ExprStore,
    api_key: str | None,
    model: str,
) -> list[ColumnConstraint]:
    """Route every property to its specialist column and encode it."""
    unit_by_name = {u.name: u for u in units}
    constraints: list[ColumnConstraint] = []
    for prop in properties:
        if prop.kind not in _BUILDERS:
            continue
        unit = unit_by_name.get(prop.unit_name)
        c = encode_property(prop, unit, store, api_key, model)
        if c is not None:
            constraints.append(c)
    return constraints
