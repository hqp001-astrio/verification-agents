from __future__ import annotations

import openai

from verification_agents.models import (
    CodeAnalysis,
    CodeUnit,
    UserSelection,
    VerifiableProperty,
    Z3Constraint,
)

# Module-level store: property_id → z3.BoolRef (live Z3 objects across tool calls)
_z3_store: dict[str, object] = {}


def get_z3_object(property_id: str) -> object | None:
    return _z3_store.get(property_id)


def clear_store() -> None:
    _z3_store.clear()


_FORMALIZE_SYSTEM = """\
You are a formal verification expert specializing in Z3 SMT solver.
Given a code snippet and a property to verify, produce a valid Python code snippet \
that uses the z3 Python library to express the property as a constraint.

Rules:
- Output only the Python code snippet, no markdown fences, no explanation.
- The snippet must define a variable named `constraint` of type `z3.BoolRef`.
- Start with `import z3`
- Keep it concise. Use z3.Int, z3.Bool, z3.ForAll, z3.Implies, z3.And, z3.Or, z3.Not as needed.
- Model the property abstractly — use symbolic variables for function parameters.
- For array bounds: use a symbolic integer index and array length, assert 0 <= index < length.
- For null dereference: use a z3.Bool to represent nullability, assert it is True (not null).
- For integer overflow: assert the arithmetic result stays within a safe integer range.
- For loop termination: encode a decreasing variant (loop counter strictly decreases each step).
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
        f"{code_section}\n\n"
        f"Produce a z3 Python snippet (no markdown) that defines `constraint` as a z3.BoolRef."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _FORMALIZE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0,
    )

    code = response.choices[0].message.content.strip()
    # Strip markdown fences if the model added them anyway
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    local_scope: dict = {}
    try:
        import builtins
        exec(code, {"__builtins__": builtins, "z3": __import__("z3")}, local_scope)  # noqa: S102
    except Exception:
        return None

    import z3
    constraint_obj = local_scope.get("constraint")
    if not isinstance(constraint_obj, z3.ExprRef):
        return None

    _z3_store[prop.id] = constraint_obj

    return Z3Constraint(
        property_id=prop.id,
        description=prop.description,
        z3_code=code,
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
