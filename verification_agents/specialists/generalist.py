"""Generalist column — total coverage for any checkable safety property.

The specialists are template-bound (the LLM only fills a typed form; we build the
Z3), so they are sound but only cover concerns we wrote templates for. The
generalist is the complement: it proposes the *entire* abstraction — typed
variables, reachable-state assumptions, and one or more safety predicates — for
*any* concern the specialists miss (division-by-zero, custom invariants, ...).

The catch the product accepts: the generalist's only failure mode is a **wrong
translation**. We contain that two ways:

1. **No arbitrary code execution.** The LLM's predicates are parsed by a small
   AST→Z3 compiler with a strict node whitelist. A malformed/unsupported
   predicate raises ``TranslationError`` and is dropped — it never runs as code.
2. **Provenance.** Generalist findings are tagged ``generalist`` so the aggregator
   treats an unexecuted generalist SAT as *proposed / translation-unverified*
   rather than a hard bug — and the executor downgrades any that it can refute.
"""

from __future__ import annotations

import ast
import uuid

import z3
from pydantic import BaseModel, Field

from verification_agents.models import CodeUnit, PropertyKind
from verification_agents.specialists.types import ColumnConstraint, ExprStore

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


class TranslationError(ValueError):
    """Raised when a generalist predicate cannot be compiled to Z3 — i.e. the
    abstraction was mistranslated. Surfaced, never executed."""


# ---------------------------------------------------------------------------
# Structured model the generalist LLM produces
# ---------------------------------------------------------------------------

class GenVar(BaseModel):
    name: str = Field(description="Identifier used in the predicates below.")
    sort: str = Field(default="Int", description="One of: Int, Bool, Real.")
    description: str = ""


class GenCheck(BaseModel):
    name: str = Field(description="Short id for this safety property.")
    safety: str = Field(description="Predicate that must ALWAYS hold (Python expr over the variables).")
    rationale: str = ""


class GeneralModel(BaseModel):
    variables: list[GenVar] = Field(default_factory=list)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Reachable-state predicates true on every path to the operation.",
    )
    checks: list[GenCheck] = Field(default_factory=list)


GENERALIST_SYSTEM = (
    "You are the GENERALIST verification agent. The specialists already cover array "
    "bounds, null dereference, integer overflow and loop termination — do NOT repeat "
    "those. Find ANY OTHER checkable safety/correctness property in the function: "
    "division/modulo by zero, dictionary key presence, value-range invariants, "
    "monotonicity, sign constraints, postconditions, etc.\n\n"
    "Output a small formal model: typed variables, reachable-state `assumptions`, and "
    "one or more `checks` whose `safety` predicate must ALWAYS hold.\n\n"
    "CRITICAL — assumptions: list ONLY conditions the code ITSELF enforces before the "
    "operation (an explicit `if` guard, an early return/raise, a validated argument, a "
    "range from a literal loop). NEVER assume the very condition you are checking, and "
    "NEVER assume a caller precondition the code does not enforce. If the code has no "
    "guard protecting the operation, list NO assumption — that is how a real bug is "
    "found. Example: `return total / count` with no guard -> assumptions: [] and check "
    "`count != 0` (this is SAT = a real division-by-zero).\n\n"
    "Write predicates as plain Python expressions over your variables using only: "
    "+ - * / % , comparisons (< <= > >= == !=), and/or/not, abs/min/max, and "
    "integer/boolean/real literals. Keep it minimal and faithful to the code. If there "
    "is genuinely nothing else to check, return empty `checks`."
)


# ---------------------------------------------------------------------------
# Safe AST -> Z3 compiler (strict whitelist; no exec)
# ---------------------------------------------------------------------------

_CMP = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}
_ALLOWED_CALLS = {"abs", "min", "max"}


def _z3_var(name: str, sort: str, ctx: z3.Context):
    s = (sort or "Int").lower()
    if s.startswith("bool"):
        return z3.Bool(name, ctx)
    if s.startswith("real") or s.startswith("float"):
        return z3.Real(name, ctx)
    return z3.Int(name, ctx)


def _compile(node, env: dict, ctx: z3.Context):
    if isinstance(node, ast.Expression):
        return _compile(node.body, env, ctx)
    if isinstance(node, ast.BoolOp):
        vals = [_compile(v, env, ctx) for v in node.values]
        return z3.And(*vals) if isinstance(node.op, ast.And) else z3.Or(*vals)
    if isinstance(node, ast.UnaryOp):
        v = _compile(node.operand, env, ctx)
        if isinstance(node.op, ast.Not):
            return z3.Not(v)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise TranslationError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _compile(node.left, env, ctx)
        right = _compile(node.right, env, ctx)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            return left / right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow) and isinstance(node.right, ast.Constant) \
                and isinstance(node.right.value, int) and 0 <= node.right.value <= 4:
            # expand small constant powers to repeated multiplication
            result = None
            for _ in range(node.right.value):
                result = left if result is None else result * left
            return result if result is not None else z3.IntVal(1, ctx)
        raise TranslationError(f"unsupported binary operator {type(op).__name__}")
    if isinstance(node, ast.Compare):
        parts, cur = [], _compile(node.left, env, ctx)
        for op, comp in zip(node.ops, node.comparators):
            fn = _CMP.get(type(op))
            if fn is None:
                raise TranslationError("unsupported comparison")
            right = _compile(comp, env, ctx)
            parts.append(fn(cur, right))
            cur = right
        return z3.And(*parts) if len(parts) > 1 else parts[0]
    if isinstance(node, ast.Name):
        if node.id not in env:
            env[node.id] = z3.Int(node.id, ctx)  # lenient: auto-declare unknown as Int
        return env[node.id]
    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            return z3.BoolVal(val, ctx)
        if isinstance(val, int):
            return z3.IntVal(val, ctx)
        if isinstance(val, float):
            return z3.RealVal(val, ctx)
        raise TranslationError("unsupported literal")
    if isinstance(node, ast.Call):
        fn = node.func.id if isinstance(node.func, ast.Name) else None
        if fn not in _ALLOWED_CALLS:
            raise TranslationError(f"unsupported call {fn!r}")
        args = [_compile(a, env, ctx) for a in node.args]
        if fn == "abs":
            return z3.If(args[0] >= 0, args[0], -args[0])
        red = args[0]
        for a in args[1:]:
            red = z3.If(red <= a, red, a) if fn == "min" else z3.If(red >= a, red, a)
        return red
    raise TranslationError(f"unsupported syntax {type(node).__name__}")


def compile_predicate(expr: str, env: dict, ctx: z3.Context):
    """Parse a Python-expression predicate into a Z3 expression (whitelist only)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise TranslationError(f"could not parse: {expr!r}") from exc
    return _compile(tree, env, ctx)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def _extract_model(unit: CodeUnit, api_key: str, model: str) -> GeneralModel | None:
    import openai

    client = openai.OpenAI(api_key=api_key)
    user = (
        f"Function under review ({unit.filename}:{unit.start_line}):\n"
        f"```\n{unit.source}\n```\n\nProduce the formal model."
    )
    for _ in range(3):
        try:
            resp = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": GENERALIST_SYSTEM},
                    {"role": "user", "content": user},
                ],
                response_format=GeneralModel,
                max_tokens=700,
                temperature=0,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception:
            continue
    return None


@_op("generalist.encode")
def encode_generalist(
    unit: CodeUnit,
    store: ExprStore,
    api_key: str | None,
    model: str,
) -> list[ColumnConstraint]:
    """Ask the generalist for a model of ``unit`` and compile each check to Z3.

    Returns one constraint per check. Checks that fail to compile are dropped
    (a recorded translation error) rather than executed. Needs an API key — there
    is no offline heuristic for open-ended property discovery."""
    if not api_key:
        return []
    gm = _extract_model(unit, api_key, model)
    if gm is None or not gm.checks:
        return []

    constraints: list[ColumnConstraint] = []
    for check in gm.checks:
        ctx = z3.Context()
        env = {v.name: _z3_var(v.name, v.sort, ctx) for v in gm.variables}
        try:
            assumptions = [compile_predicate(a, env, ctx) for a in gm.assumptions]
            safety = compile_predicate(check.safety, env, ctx)
        except TranslationError as exc:
            # Surface the mistranslation as an inconclusive marker, don't run code.
            cid = uuid.uuid4().hex[:12]
            constraints.append(ColumnConstraint(
                constraint_id=cid, concern=PropertyKind.CUSTOM, property_id=cid,
                description=f"[generalist] {check.name}: {check.rationale}",
                unit_name=unit.name, filename=unit.filename, start_line=unit.start_line,
                reachable="; ".join(gm.assumptions), safety=check.safety,
                provenance="generalist", source="llm",
                note=f"translation error: {exc}",
            ))
            continue

        reachable = z3.And(*assumptions) if assumptions else z3.BoolVal(True, ctx)
        violation = z3.And(reachable, z3.Not(safety))
        cid = uuid.uuid4().hex[:12]
        store.put(cid, violation)
        constraints.append(ColumnConstraint(
            constraint_id=cid, concern=PropertyKind.CUSTOM, property_id=cid,
            description=f"[generalist] {check.name}: {check.rationale}",
            unit_name=unit.name, filename=unit.filename, start_line=unit.start_line,
            reachable=str(reachable), safety=check.safety,
            provenance="generalist", source="llm",
        ))
    return constraints
