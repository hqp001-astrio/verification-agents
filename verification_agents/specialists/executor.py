"""Counterexample executor — the soundness anchor.

When Z3 returns SAT it claims the property *can* be violated. That claim is only
as sound as the LLM's abstraction, so before reporting a bug we try to *reproduce*
it: build concrete inputs from the counterexample and run the real function in a
sandboxed subprocess.

Returns, per outcome:
* ``True``  — the function actually raised the matching error -> confirmed bug.
* ``False`` — it ran clean -> the counterexample was spurious (abstraction was an
  over-approximation); the aggregator demotes this from "bug" to "inconclusive".
* ``None``  — could not build/execute a faithful harness -> no ground truth.

Only self-contained pure-ish Python functions are reproducible; anything with
external dependencies returns ``None`` (honestly "we couldn't decide by running").
"""

from __future__ import annotations

import re
import subprocess
import sys

from verification_agents.models import CodeUnit, PropertyKind

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


_TARGET_ERRORS = {
    PropertyKind.ARRAY_BOUNDS: ("IndexError", "KeyError"),
    PropertyKind.NULL_DEREFERENCE: ("AttributeError", "TypeError"),
    PropertyKind.DIVISION_BY_ZERO: ("ZeroDivisionError",),
}

_HARNESS = '''\
import sys
{source}

try:
    _ret = {call}
except ({targets}):
    print("REPRODUCED")
    sys.exit(0)
except BaseException as exc:
    print("INDETERMINATE:" + type(exc).__name__)
    sys.exit(0)
print("CLEAN")
'''


def _parse_def(source: str) -> tuple[str, list[str]] | None:
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", source)
    if not m:
        return None
    name = m.group(1)
    params = []
    for raw in m.group(2).split(","):
        p = raw.strip().split(":")[0].split("=")[0].strip()
        if p and p not in ("*", "/"):
            params.append(p.lstrip("*"))
    return name, params


def _literal(value: int | str) -> str:
    return repr(value)


def _build_call(concern: PropertyKind, fn_name: str, params: list[str],
                var_binding: dict, counterexample: dict) -> str | None:
    if params and params[0] == "self":
        return None  # can't cheaply instantiate a method receiver

    args: list[str] = []
    if concern == PropertyKind.ARRAY_BOUNDS:
        seq_param = var_binding.get("seq_param") or ""
        index_param = var_binding.get("index_param") or ""
        length = int(counterexample.get(var_binding.get("length_var", "length"), 1) or 1)
        index = int(counterexample.get(var_binding.get("index_var", "index"), length) or 0)
        length = max(0, min(length, 10_000))  # avoid pathological allocations
        if not seq_param:
            return None
        for p in params:
            if p == seq_param:
                args.append(f"[0]*{length}")
            elif index_param and p == index_param:
                args.append(_literal(index))
            else:
                args.append("0")
        # If the index never flows through a parameter (e.g. internal loop), the
        # sequence alone still drives the access, so the call is still meaningful.
        return f"{fn_name}({', '.join(args)})"

    if concern == PropertyKind.NULL_DEREFERENCE:
        object_param = var_binding.get("object_param") or ""
        if not object_param:
            return None
        for p in params:
            args.append("None" if p == object_param else "0")
        return f"{fn_name}({', '.join(args)})"

    if concern == PropertyKind.DIVISION_BY_ZERO:
        denom_param = var_binding.get("denom_param") or ""
        if not denom_param:
            return None  # divisor is an expression, not a bare param -> can't ground simply
        # set the divisor parameter to 0, others to 1 (avoid unrelated div-by-zero)
        for p in params:
            args.append("0" if p == denom_param else "1")
        return f"{fn_name}({', '.join(args)})"

    return None


@_op("specialist.execute")
def reproduce(
    concern: PropertyKind,
    unit: CodeUnit | None,
    var_binding: dict,
    counterexample: dict | None,
    timeout_s: float = 5.0,
) -> tuple[bool | None, str]:
    """Try to reproduce a counterexample. See module docstring for the contract."""
    if unit is None or not counterexample or concern not in _TARGET_ERRORS:
        return None, "not reproducible (no source / unsupported concern)"

    parsed = _parse_def(unit.source)
    if parsed is None:
        return None, "could not parse function signature"
    fn_name, params = parsed

    call = _build_call(concern, fn_name, params, var_binding, counterexample)
    if call is None:
        return None, "could not map counterexample to concrete inputs"

    script = _HARNESS.format(
        source=unit.source,
        call=call,
        targets=", ".join(_TARGET_ERRORS[concern]),
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, "execution timed out"
    except Exception as exc:  # pragma: no cover
        return None, f"execution failed to launch: {exc}"

    out = (proc.stdout or "").strip().splitlines()
    marker = out[-1] if out else ""
    if marker == "REPRODUCED":
        return True, f"ran `{call}` -> raised {_TARGET_ERRORS[concern][0]} (confirmed)"
    if marker == "CLEAN":
        return False, f"ran `{call}` -> no error (counterexample is spurious)"
    if marker.startswith("INDETERMINATE"):
        return None, f"ran `{call}` -> unrelated error: {marker.split(':', 1)[-1]}"
    return None, "execution produced no verdict"


# Broad set of correctness errors the generalist might expose (div-by-zero, bad
# key, bad type, ...). Used to confirm/refute generalist findings by execution.
_GENERIC_ERRORS = (
    "ZeroDivisionError", "IndexError", "KeyError", "ValueError",
    "TypeError", "OverflowError", "AttributeError", "AssertionError",
)


def _coerce(value) -> str:
    try:
        return repr(int(value))
    except (TypeError, ValueError):
        try:
            return repr(float(value))
        except (TypeError, ValueError):
            return "0"


@_op("generalist.execute")
def reproduce_generic(
    unit: CodeUnit | None,
    counterexample: dict | None,
    timeout_s: float = 5.0,
) -> tuple[bool | None, str]:
    """Confirm/refute a generalist finding by running the function on the CE.

    Maps counterexample variables to parameters BY NAME (the generalist is told to
    use the code's real names). Reproduced any common error -> True; ran clean ->
    False (spurious / mistranslation); can't map or unrelated failure -> None."""
    if unit is None or not counterexample:
        return None, "not reproducible (no source / counterexample)"
    parsed = _parse_def(unit.source)
    if parsed is None:
        return None, "could not parse function signature"
    fn_name, params = parsed
    if params and params[0] == "self":
        return None, "method receiver not constructable"
    if not any(p in counterexample for p in params):
        return None, "counterexample variables do not map to parameters"

    args = [_coerce(counterexample[p]) if p in counterexample else "0" for p in params]
    call = f"{fn_name}({', '.join(args)})"
    script = _HARNESS.format(source=unit.source, call=call, targets=", ".join(_GENERIC_ERRORS))
    try:
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, "execution timed out"
    except Exception as exc:  # pragma: no cover
        return None, f"execution failed to launch: {exc}"

    marker = ((proc.stdout or "").strip().splitlines() or [""])[-1]
    if marker == "REPRODUCED":
        return True, f"ran `{call}` -> raised a correctness error (confirmed)"
    if marker == "CLEAN":
        return False, f"ran `{call}` -> no error (counterexample is spurious)"
    return None, f"ran `{call}` -> {marker or 'no verdict'}"
