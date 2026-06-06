"""Adversarial benchmark: division safety that hinges on exact computation.

Each case asks: can the denominator hit zero within the guarded input range? The
arithmetic is deliberately non-obvious — larger coefficients, bigger moduli, two
variables, modular squares — so even a strong model occasionally slips on the
mental math, while Z3 solves it exactly and the executor reproduces the crash.

Cases come in near-identical safe/bug PAIRS that differ only by a range bound or a
constant, so surface pattern-matching can't separate them — only the computation
can. All labels are objectively correct (verify the factorization yourself).
"""

from __future__ import annotations

from verification_agents.eval.dataset import Case

CASES_HARD: list[Case] = [
    # quadratic, roots 11 and 12 (x^2 - 23x + 132)
    Case(name="quad_a_bug", label="no", code=(
        "def f(x):\n"
        "    if 0 <= x < 30:\n"
        "        return 1 / (x * x - 23 * x + 132)\n"   # 0 at x=11,12
        "    return 0\n")),
    Case(name="quad_a_safe", label="yes", code=(
        "def f(x):\n"
        "    if 0 <= x < 11:\n"                          # x<=10 -> roots out of range
        "        return 1 / (x * x - 23 * x + 132)\n"
        "    return 0\n")),

    # non-monic quadratic 3x^2 - 17x + 10, integer root x=5
    Case(name="quad_b_bug", label="no", code=(
        "def f(x):\n"
        "    if 0 <= x < 50:\n"
        "        return 1 / (3 * x * x - 17 * x + 10)\n"  # 0 at x=5
        "    return 0\n")),
    Case(name="quad_b_safe", label="yes", code=(
        "def f(x):\n"
        "    if 0 <= x < 4:\n"                           # x in 0..3 -> never 0
        "        return 1 / (3 * x * x - 17 * x + 10)\n"
        "    return 0\n")),

    # larger modulus: zero when x % 23 == 17
    Case(name="mod_a_bug", label="no", code=(
        "def f(x):\n"
        "    if 0 <= x < 100:\n"
        "        return 1 / ((x % 23) - 17)\n"           # 0 at x=17,40,63,86
        "    return 0\n")),
    Case(name="mod_a_safe", label="yes", code=(
        "def f(x):\n"
        "    if 5 <= x < 11:\n"                          # x%23 in 5..10 -> never 17
        "        return 1 / ((x % 23) - 17)\n"
        "    return 0\n")),

    # two variables: a^2 - b^2 - 7, zero at (a,b)=(4,3)
    Case(name="twovar_bug", label="no", code=(
        "def f(a, b):\n"
        "    if 0 <= a < 12 and 0 <= b < 12:\n"
        "        return 1 / (a * a - b * b - 7)\n"        # 0 at a=4,b=3
        "    return 0\n")),
    Case(name="twovar_safe", label="yes", code=(
        "def f(a, b):\n"
        "    if 0 <= a < 4 and 0 <= b < 4:\n"            # |a^2-b^2| <= 9, never 7... check: 9-? no =7
        "        return 1 / (a * a - b * b - 7)\n"
        "    return 0\n")),

    # modular square: x^2 % 11 == 3 has solutions (x=5,6,...)
    Case(name="modsq_bug", label="no", code=(
        "def f(x):\n"
        "    if 0 <= x < 40:\n"
        "        return 1 / ((x * x) % 11 - 3)\n"         # 0 when x^2 % 11 == 3
        "    return 0\n")),
    Case(name="modsq_safe", label="yes", code=(
        "def f(x):\n"
        "    if 0 <= x < 3:\n"                           # x^2 in {0,1,4} -> %11 in {0,1,4}, never 3
        "        return 1 / ((x * x) % 11 - 3)\n"
        "    return 0\n")),

    # genuinely context-dependent
    Case(name="div_external", label="unsure", code=(
        "def f(a):\n"
        "    return a / get_divisor()\n")),
]
