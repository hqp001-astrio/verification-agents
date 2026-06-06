"""Adversarial benchmark: cases that require real computation, not pattern-matching.

Whether the code is safe depends on solving an arithmetic question exactly — can a
quadratic / modular / product denominator hit zero within the guarded input range?
An LLM answers by intuition and often gets the arithmetic wrong; Z3 solves it
exactly and the executor reproduces the concrete crash. Division is used (not array
bounds) because a zero divisor is an unambiguous Python crash with no negative-index
wraparound, and the counterexample maps straight to a parameter.
"""

from __future__ import annotations

from verification_agents.eval.dataset import Case

CASES_HARD: list[Case] = [
    # --- bugs: denominator CAN hit zero in range (no) ---
    Case(name="quad_div_bug", label="no", code=(
        "def f(x):\n"
        "    if 0 <= x < 10:\n"
        "        return 100 / (x * x - 5 * x + 6)\n"   # roots 2, 3 are in range
        "    return 0\n")),
    Case(name="mod_div_bug", label="no", code=(
        "def f(x):\n"
        "    if 0 <= x < 20:\n"
        "        return 50 / ((x % 7) - 3)\n"          # zero at x % 7 == 3 (x=3,10,17)
        "    return 0\n")),
    Case(name="product_div_bug", label="no", code=(
        "def f(a, b):\n"
        "    if 0 <= a < 5 and 0 <= b < 5:\n"
        "        return 1 / (a * b - 6)\n"             # zero at (2,3),(3,2)
        "    return 0\n")),
    Case(name="linear_div_bug", label="no", code=(
        "def f(x):\n"
        "    if 0 <= x <= 10:\n"
        "        return 1 / (3 * x - 12)\n"            # zero at x == 4
        "    return 0\n")),

    # --- safe: denominator never zero in range (yes) ---
    Case(name="quad_div_safe", label="yes", code=(
        "def f(x):\n"
        "    if 0 <= x < 2:\n"
        "        return 100 / (x * x - 5 * x + 6)\n"   # x in {0,1} -> 6, 2 ; never 0
        "    return 0\n")),
    Case(name="mod_div_safe", label="yes", code=(
        "def f(x):\n"
        "    if 0 <= x < 3:\n"
        "        return 50 / ((x % 7) - 3)\n"          # x in {0,1,2} -> -3,-2,-1
        "    return 0\n")),
    Case(name="linear_div_safe", label="yes", code=(
        "def f(x):\n"
        "    if 1 <= x <= 5:\n"
        "        return 1 / (2 * x + 1)\n"             # 3,5,7,9,11 ; never 0
        "    return 0\n")),

    # --- genuinely context-dependent (unsure) ---
    Case(name="div_external", label="unsure", code=(
        "def f(a):\n"
        "    return a / get_divisor()\n")),
]
