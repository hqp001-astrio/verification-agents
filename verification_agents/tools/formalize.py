from __future__ import annotations

import openai
from pydantic import BaseModel, Field

from verification_agents.models import (
    CodeAnalysis,
    CodeUnit,
    PropertyKind,
    UserSelection,
    VerifiableProperty,
    Z3Constraint,
)

_z3_store: dict[str, object] = {}


def get_z3_object(property_id: str) -> object | None:
    return _z3_store.get(property_id)


def clear_store() -> None:
    _z3_store.clear()


class _FormalConstraintSpec(BaseModel):
    kind: PropertyKind
    description: str
    index_var: str = Field(default="index", description="Symbolic array index variable name")
    length_var: str = Field(default="length", description="Symbolic array length variable name")
    nullable_var: str = Field(default="not_null", description="Symbolic nullable object variable name")
    operand_vars: list[str] = Field(
        default_factory=lambda: ["a", "b"],
        description="Symbolic operand variable names for the arithmetic expression",
    )
    operator: str = Field(default="+", description="Arithmetic operator: +, -, *")
    int_min: int = Field(default=-(2**31), description="Minimum safe integer value")
    int_max: int = Field(default=2**31 - 1, description="Maximum safe integer value")
    variant_var: str = Field(default="variant", description="Symbolic loop variant variable name")


def _build_z3_constraint(spec: _FormalConstraintSpec):
    import z3

    match spec.kind:
        case PropertyKind.ARRAY_BOUNDS:
            index = z3.Int(spec.index_var)
            length = z3.Int(spec.length_var)
            return z3.And(index >= 0, index < length, length > 0)

        case PropertyKind.NULL_DEREFERENCE:
            return z3.Bool(spec.nullable_var)

        case PropertyKind.INTEGER_OVERFLOW:
            lo, hi = spec.int_min, spec.int_max
            if len(spec.operand_vars) >= 2:
                a = z3.Int(spec.operand_vars[0])
                b = z3.Int(spec.operand_vars[1])
                result = {"+": a + b, "-": a - b, "*": a * b}.get(spec.operator, a + b)
                precond = z3.And(a >= lo, a <= hi, b >= lo, b <= hi)
            elif spec.operand_vars:
                a = z3.Int(spec.operand_vars[0])
                result, precond = a, z3.And(a >= lo, a <= hi)
            else:
                return None
            return z3.Implies(precond, z3.And(result >= lo, result <= hi))

        case PropertyKind.LOOP_TERMINATION:
            variant = z3.Int(spec.variant_var)
            next_v = z3.Int(spec.variant_var + "_next")
            return z3.Implies(variant > 0, z3.And(next_v < variant, next_v >= 0))

        case _:
            return None


_FORMALIZE_SYSTEM = """\
You are a formal verification expert.
Given a code property to verify, fill in the constraint specification for the Z3 SMT solver.

Use meaningful variable names drawn from the actual code:
- array_bounds: index_var = the index expression, length_var = the array/sequence length
- null_dereference: nullable_var = the object being dereferenced
- integer_overflow: operand_vars = operand names, operator = +/-/*, int_min/int_max = bounds (default: 32-bit signed)
- loop_termination: variant_var = the loop counter or decreasing quantity
- For other kinds use sensible defaults.

Return only the JSON object matching the schema. No explanation.
"""


def _formalize_one(
    prop: VerifiableProperty,
    unit: CodeUnit | None,
    api_key: str,
    model: str,
) -> Z3Constraint | None:
    client = openai.OpenAI(api_key=api_key)

    code_section = f"\n\nRelevant code:\n```\n{unit.source}\n```" if unit else ""
    prompt = (
        f"Property to formalize:\n"
        f"- Kind: {prop.kind}\n"
        f"- Description: {prop.description}\n"
        f"- Function: {prop.unit_name}\n"
        f"- File: {prop.filename}, line {prop.start_line}"
        f"{code_section}"
    )

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _FORMALIZE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format=_FormalConstraintSpec,
        max_tokens=256,
        temperature=0,
    )

    spec = response.choices[0].message.parsed
    if spec is None:
        return None

    constraint_obj = _build_z3_constraint(spec)
    if constraint_obj is None:
        return None

    _z3_store[prop.id] = constraint_obj

    return Z3Constraint(
        property_id=prop.id,
        description=spec.description,
        z3_code=f"# template-built for {spec.kind.value}",
    )


def run(
    selected: dict,
    code_analysis: dict,
    api_key: str,
    model: str,
) -> list[dict]:
    selection = UserSelection(**selected)
    analysis = CodeAnalysis(**code_analysis)

    unit_by_name: dict[str, CodeUnit] = {u.name: u for u in analysis.units}
    prop_by_id: dict[str, VerifiableProperty] = {p.id: p for p in analysis.properties}

    constraints: list[Z3Constraint] = []
    for pid in selection.selected_ids:
        prop = prop_by_id.get(pid)
        if prop is None:
            continue
        unit = unit_by_name.get(prop.unit_name)
        result = _formalize_one(prop, unit, api_key, model)
        if result:
            constraints.append(result)

    return [c.model_dump() for c in constraints]
