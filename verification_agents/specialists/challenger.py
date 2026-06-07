"""Spec challenger — independent validation of generalist safety properties.

Runs only on SAT outcomes (potential bugs). Reads the same source the generalist
saw and decides whether the proposed safety predicate is a faithful formalization:
is this really something that could fail in the real code?

A wrong translation (Z3 says SAT but the function is actually safe) is caught here
before reaching the aggregator. Always uses the fast/cheap model since it only
needs to validate a short predicate against short code.
"""

from __future__ import annotations

from pydantic import BaseModel

from verification_agents.models import CodeUnit

try:
    import weave as _weave
    _HAS_WEAVE = True
except Exception:
    _weave = None  # type: ignore[assignment]
    _HAS_WEAVE = False


def _op(name: str):
    def deco(fn):
        return _weave.op(fn, name=name) if _HAS_WEAVE else fn
    return deco


class ChallengerResult(BaseModel):
    valid: bool
    issue: str = ""


_SYSTEM = (
    "You are the CHALLENGER. A generalist verification agent proposed a safety property "
    "about a function. Decide whether the property faithfully represents the code.\n\n"
    "Answer valid=true if the property correctly describes a real safety concern that "
    "could actually fail in the given function.\n"
    "Answer valid=false (with a concise issue) if:\n"
    "- the property references variables or operations not present in the code,\n"
    "- the property is trivially guaranteed by the code structure (false positive),\n"
    "- the formalization is logically inverted or semantically wrong.\n\n"
    "Be permissive: if the property is approximately correct, answer valid=true."
)


@_op("challenger.challenge")
def challenge(
    safety: str,
    reachable: str,
    unit: CodeUnit | None,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> ChallengerResult:
    """Challenge one safety predicate against the real source. Defaults to valid on any failure."""
    if unit is None:
        return ChallengerResult(valid=True, issue="no source available")

    import openai

    client = openai.OpenAI(api_key=api_key)
    user = (
        f"Function:\n```\n{unit.source}\n```\n\n"
        f"Safety property (must always hold): `{safety}`\n"
        f"Reachable-state assumption: `{reachable}`\n\n"
        "Is this a faithful formalization of a real safety concern in this function?"
    )
    try:
        resp = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format=ChallengerResult,
            max_tokens=128,
            temperature=0,
        )
        result = resp.choices[0].message.parsed
        if result is not None:
            return result
    except Exception:
        pass
    return ChallengerResult(valid=True, issue="challenger unavailable — defaulting to valid")
