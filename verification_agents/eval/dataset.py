"""Labeled yes/no/unsure benchmark of small Python functions.

Contract (so the gold labels are defensible): a function is judged over ALL inputs
of its parameter types.
  "no"     -> some typed input triggers a crash/violation (out-of-bounds, division
              by zero, etc.) -> the code has a bug.
  "yes"    -> safe for every typed input (a guard rules out the violation).
  "unsure" -> safety depends on state NOT derivable from the inputs (a global, a
              network/IO call) -> cannot be decided from the function alone.

Integer overflow is intentionally excluded: Python ints are arbitrary-precision,
so it is not a real safety property here.
"""

from __future__ import annotations

from pydantic import BaseModel


class Case(BaseModel):
    name: str
    code: str
    label: str  # yes | no | unsure


CASES: list[Case] = [
    # --- bugs (no): a valid-typed input crashes ---
    Case(name="bounds_off_by_one", label="no", code=(
        "def get(items, i):\n"
        "    for i in range(len(items) + 1):\n"
        "        v = items[i]\n"
        "    return v\n")),
    Case(name="div_by_zero", label="no", code=(
        "def average(total, count):\n"
        "    return total / count\n")),
    Case(name="modulo_zero", label="no", code=(
        "def wrap(x, n):\n"
        "    return x % n\n")),
    Case(name="first_of_empty", label="no", code=(
        "def first(items):\n"
        "    return items[0]\n")),

    # --- safe (yes): a guard rules out every violation ---
    Case(name="bounds_guarded", label="yes", code=(
        "def get(items, i):\n"
        "    if 0 <= i < len(items):\n"
        "        return items[i]\n"
        "    return None\n")),
    Case(name="div_guarded", label="yes", code=(
        "def average(total, count):\n"
        "    if count == 0:\n"
        "        return 0\n"
        "    return total / count\n")),
    Case(name="modulo_guarded", label="yes", code=(
        "def wrap(x, n):\n"
        "    if n == 0:\n"
        "        return x\n"
        "    return x % n\n")),
    Case(name="safe_passthrough", label="yes", code=(
        "def clamp(x):\n"
        "    return x if x < 1000 else 1000\n")),

    # --- unsure: depends on state not in the inputs ---
    Case(name="external_length", label="unsure", code=(
        "def get(i):\n"
        "    items = load_remote_items()\n"
        "    return items[i]\n")),
    Case(name="external_global", label="unsure", code=(
        "def scale(x):\n"
        "    return CONFIG['factor'] * x\n")),
]
